"""
检索协调器 (Memory Search Coordinator)。

无需路由决策，每次请求均同时检索图谱和技能库：
- 对图谱：关键词匹配 / 多跳子查询
- 对技能库：HyDE → embedding → 向量检索
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.embedder.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.memory.graph.graphiti_engine import GraphitiEngine
from src.memory.procedural.file_skill_store import FileSkillStore

HYDE_SYSTEM_PROMPT = """\
你是一个「HyDE 假设性回答生成器」。

目标：
- 给定用户查询，为"程序/技能类"意图生成一段 **假设性的理想回答文本（Draft Answer）**。
- 这段草稿回答不会直接返回给用户，而是作为向量检索的"锚点文本"，用来提高召回率。

要求：
1. 假装你已经成功完成了用户想要的任务，用 2-3 句话描述一个合理的解决方案草稿。
2. 文本中应自然包含：可能会调用的工具名称、关键参数名、重要中间产物等关键信息。
3. 保持简洁，聚焦关键实体、步骤和概念，不要展开长篇解释。
4. 不要使用列表或 JSON，只输出连续自然语言段落。

示例（仅用于你在脑中参考，不要原样抄写）：

- 用户查询："帮我写一个脚本，每天凌晨 3 点备份 PostgreSQL 数据库到 S3。"
    假设性回答示例：
    "我为你编写了一个使用 `pg_dump` 的备份脚本，并通过 crontab 配置在每天凌晨 3 点运行。脚本会将生成的备份文件上传到你指定的 S3 bucket，并使用时间戳作为文件名，方便后续检索和清理。"

- 用户查询："搭一个最简单的 FastAPI 服务，并用 Docker 部署。"
    假设性回答示例：
    "我创建了一个包含单个 `/health` 路由的 FastAPI 应用，并编写了一个使用 `python:3.10-slim` 基础镜像的 Dockerfile。通过 `docker build` 构建镜像后，在服务器上使用 `docker run -p 8000:8000` 运行该服务。"
"""

RETRIEVAL_PLAN_PROMPT = """\
你是一个「记忆检索协调器（Search Coordinator）」的规划子模块。

你的任务是：将粗粒度的用户查询，转化为面向不同记忆源的 **多路子查询（Multi-Query）** 检索计划。

针对图谱和技能库两个记忆源，请：
1. 将复杂问题拆解为 1-5 个更具体、可直接检索的子查询；
2. 子查询应比原问题更聚焦，明确实体、时间、任务目标等关键信息；
3. 避免重复或语义等价的子查询；
4. **关键规则：如果用户消息涉及多个不同主题或实体，必须为每个主题/实体分别生成独立的 graph 子查询**。
   例如用户同时问"车库门密码"和"猫能不能吃虾"，你必须生成两条 graph 子查询，
   否则检索系统只能召回其中一个主题的记忆，导致另一个主题的信息丢失。

以 JSON 格式回复（保持字段名与下方完全一致）：
{
    "sub_queries": [
        {"source": "graph", "query": "针对知识图谱的子查询A"},
        {"source": "graph", "query": "针对知识图谱的子查询B"},
        {"source": "skill", "query": "针对技能库的子查询"}
    ],
    "reasoning": "用一两句话解释你为何这样拆分"
}

规则：
- source 只能是 "graph" 或 "skill"；
- 同一 source 可以出现多次——对于包含多个主题的查询，必须为每个主题生成独立的 graph 子查询；
- 如果查询简单不需要分解，可以只返回 1 个子查询；
- 每个子查询应该比原查询更具体，便于检索；
- 不要输出任何 JSON 以外的文字。

示例1（多主题查询——最常见的需要拆分场景）：

用户查询：
    "我刚回家，车库门密码是多少？另外我的猫今晚能不能吃虾？"

期望 JSON 输出示例：
{
    "sub_queries": [
        {
            "source": "graph",
            "query": "车库门 密码"
        },
        {
            "source": "graph",
            "query": "猫 能否吃虾 饮食禁忌"
        }
    ],
    "reasoning": "该查询包含两个完全独立的主题：车库门密码和猫的饮食，需要分别检索图谱以确保两方面信息都被召回。"
}

示例2（跨源查询）：

用户查询：
    "回顾一下上次我们部署订单服务的流程，然后看看知识图谱里有没有和支付相关的错误模式。"

期望 JSON 输出示例：
{
    "sub_queries": [
        {
            "source": "skill",
            "query": "订单服务 部署 流程"
        },
        {
            "source": "graph",
            "query": "支付 错误 模式"
        }
    ],
    "reasoning": "该查询同时涉及复用历史中的订单服务部署流程（技能库）以及与支付相关的错误模式（图谱），因此需要拆分为 skill 与 graph 两路精简子查询。"
}
"""


class SearchCoordinator(BaseAgent):
    """检索协调器：每次请求均同时检索图谱和技能库。"""

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        skill_store: FileSkillStore,
        graphiti_engine: GraphitiEngine,
        graph_top_k: int = 3,
        skill_top_k: int = 3,
        skill_reuse_threshold: float = 0.85,
    ):
        super().__init__(llm=llm, prompt_template=HYDE_SYSTEM_PROMPT)
        self.embedder = embedder
        self.skill_store = skill_store
        self.graphiti_engine = graphiti_engine
        self.graph_top_k = graph_top_k
        self.skill_top_k = skill_top_k
        self.skill_reuse_threshold = skill_reuse_threshold

    def run(
        self,
        user_query: str,
        **kwargs,
    ) -> dict[str, Any]:
        """同时检索图谱和技能库（含结构化检索规划）。

        流程：
        1. LLM 生成面向不同源的子查询
        2. 分别检索图谱和技能库
        3. 合并结果

        Args:
            user_query: 用户查询。

        Returns:
            dict 包含：retrieved_graph_memories, retrieved_skills
        """
        sub_queries = self._plan_retrieval(user_query)

        graph_queries = sub_queries.get("graph") or [user_query]
        skill_queries = sub_queries.get("skill") or [user_query]

        session_id = kwargs.get("session_id")
        if session_id is not None:
            session_id = str(session_id)
        user_id = kwargs.get("user_id", "")

        return {
            "retrieved_graph_memories": self._search_graph(
                graph_queries, session_id=session_id, user_id=user_id
            ),
            "retrieved_skills": self._search_skills(skill_queries[0], user_id=user_id),
        }

    def _plan_retrieval(self, query: str) -> dict[str, list[str]]:
        """LLM 驱动的检索规划：将复杂查询分解为面向不同源的子查询。

        Returns:
            dict 映射 source → [refined_query, ...]。
            同一 source 可对应多条子查询（多主题场景）。
            解析失败时返回空 dict（退回到原始查询）。
        """
        try:
            response = self._call_llm(
                query,
                system_content=RETRIEVAL_PLAN_PROMPT,
                add_time_basis=True,
            )
            parsed = self._parse_json(response)
            sub_queries = parsed.get("sub_queries", [])
            result: dict[str, list[str]] = {}
            for sq in sub_queries:
                if "source" in sq and "query" in sq:
                    result.setdefault(sq["source"], []).append(sq["query"])
            if result:
                return result
        except Exception:
            pass

        return {}

    def _search_graph(
        self,
        queries: list[str],
        *,
        session_id: str | None = None,
        user_id: str = "",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """在知识图谱中检索相关节点和邻居（仅 Graphiti 路径）。

        支持多条子查询（multi-query）：对列表中每条查询独立检索，
        然后合并去重，确保多主题场景下各主题的记忆都能被召回。
        """
        all_contexts: list[str] = []
        all_provenance: list[dict[str, Any]] = []
        seen_fact_ids: set[str] = set()

        graph_top_k = top_k if top_k is not None else self.graph_top_k

        for query in queries:
            query_embedding = self.embedder.embed_query(query)
            r = self.graphiti_engine.search(
                query=query,
                session_id=session_id,
                top_k=graph_top_k,
                query_embedding=query_embedding,
                include_communities=True,
                user_id=user_id,
            )
            if r.context.strip():
                all_contexts.append(r.context.strip())
            for p in r.provenance:
                fid = p.get("fact_id", "")
                if fid not in seen_fact_ids:
                    seen_fact_ids.add(fid)
                    all_provenance.append(p)

        merged_context = "\n\n".join(all_contexts)
        if not merged_context.strip():
            return []
        return [
            {
                "anchor": {
                    "node_id": "graphiti_context",
                    "name": "GraphitiContext",
                    "label": "Context",
                    "score": 1.0,
                },
                "subgraph": {"nodes": [], "edges": []},
                "constructed_context": merged_context,
                "provenance": all_provenance,
            }
        ]

    def _search_skills(
        self,
        query: str,
        user_id: str = "",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """使用 HyDE 策略检索技能库。

        1. 用 LLM 生成假设性回答
        2. 对假设回答做 embedding
        3. 用该 embedding 做向量检索
        4. 标记超过复用阈值的技能为 reusable
        """
        hyde_doc = self._call_llm(
            query,
            system_content=self.prompt_template,
            add_time_basis=True,
        )

        hyde_embedding = self.embedder.embed_query(hyde_doc)

        skill_top_k = top_k if top_k is not None else self.skill_top_k

        results = self.skill_store.search(
            query=query,
            top_k=skill_top_k,
            query_embedding=hyde_embedding,
            user_id=user_id,
        )

        for skill in results:
            skill["reusable"] = skill.get("score", 0) >= self.skill_reuse_threshold

        return results
