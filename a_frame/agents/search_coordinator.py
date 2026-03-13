"""
检索协调器 (Memory Search Coordinator)。

将路由指令转化为对不同记忆存储模块的具体查询：
- 对图谱：关键词匹配 / 多跳子查询
- 对技能库：HyDE → embedding → 向量检索
- 对感觉缓存：直接取最近 N 条
"""

from __future__ import annotations

from typing import Any

from a_frame.agents.base_agent import BaseAgent
from a_frame.embedder.base import BaseEmbedder
from a_frame.llm.base import BaseLLM
from a_frame.memory.graph.graph_store import NetworkXGraphStore
from a_frame.memory.procedural.skill_store import InMemorySkillStore
from a_frame.memory.sensory.buffer import SensoryBuffer

HYDE_SYSTEM_PROMPT = """\
你是一个「HyDE 假设性回答生成器」。

目标：
- 给定用户查询，为“程序/技能类”意图生成一段 **假设性的理想回答文本（Draft Answer）**。
- 这段草稿回答不会直接返回给用户，而是作为向量检索的“锚点文本”，用来提高召回率。

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

针对需要检索的每一种记忆源（图谱/技能/感觉），请：
1. 将复杂问题拆解为 1-3 个更具体、可直接检索的子查询；
2. 子查询应比原问题更聚焦，明确实体、时间、任务目标等关键信息；
3. 避免重复或语义等价的子查询。

以 JSON 格式回复（保持字段名与下方完全一致）：
{
    "sub_queries": [
        {"source": "graph", "query": "针对知识图谱的子查询"},
        {"source": "skill", "query": "针对技能库的子查询"}
    ],
    "reasoning": "用一两句话解释你为何这样拆分和分配到不同源"
}

规则：
- source 只能是 "graph", "skill", "sensory" 之一；
- 如果查询简单不需要分解，可以只返回 1 个子查询；
- 每个子查询应该比原查询更具体，便于检索；
- 不要输出任何 JSON 以外的文字。

示例（仅用于你在脑中参考，不要原样抄写）：

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
    """检索协调器：根据路由决策从各记忆基质检索。"""

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        graph_store: NetworkXGraphStore,
        skill_store: InMemorySkillStore,
        sensory_buffer: SensoryBuffer,
        graph_search_depth: int = 1,
        skill_top_k: int = 3,
        sensory_recent_n: int = 5,
        skill_reuse_threshold: float = 0.85,
    ):
        super().__init__(llm=llm, prompt_template=HYDE_SYSTEM_PROMPT)
        self.embedder = embedder
        self.graph_store = graph_store
        self.skill_store = skill_store
        self.sensory_buffer = sensory_buffer
        self.graph_search_depth = graph_search_depth
        self.skill_top_k = skill_top_k
        self.sensory_recent_n = sensory_recent_n
        self.skill_reuse_threshold = skill_reuse_threshold

    def run(
        self,
        user_query: str,
        route: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """根据路由决策执行多源检索（含结构化检索规划）。

        流程：
        1. 如果查询需要多源检索 → LLM 生成检索计划（子查询分解）
        2. 按子查询分别检索各记忆基质
        3. 合并结果

        Args:
            user_query: 用户查询。
            route: 路由决策 {need_graph, need_skills, need_sensory}。

        Returns:
            dict 包含：retrieved_graph_memories, retrieved_skills, retrieved_sensory
        """
        result: dict[str, Any] = {
            "retrieved_graph_memories": [],
            "retrieved_skills": [],
            "retrieved_sensory": [],
        }

        # 计算需要检索的源数量
        sources_needed = sum([
            bool(route.get("need_graph")),
            bool(route.get("need_skills")),
            bool(route.get("need_sensory")),
        ])

        # 如果需要多源检索，生成结构化检索计划做 query 细化
        sub_queries = self._plan_retrieval(user_query, route) if sources_needed > 1 else {}

        if route.get("need_graph"):
            graph_query = sub_queries.get("graph", user_query)
            result["retrieved_graph_memories"] = self._search_graph(graph_query)

        if route.get("need_skills"):
            skill_query = sub_queries.get("skill", user_query)
            result["retrieved_skills"] = self._search_skills(skill_query)

        if route.get("need_sensory"):
            result["retrieved_sensory"] = self._search_sensory()

        return result

    def _plan_retrieval(self, query: str, route: dict[str, Any]) -> dict[str, str]:
        """LLM 驱动的检索规划：将复杂查询分解为面向不同源的子查询。

        Returns:
            dict 映射 source → refined_query（如 {"graph": "...", "skill": "..."}）。
            解析失败时返回空 dict（退回到原始查询）。
        """
        try:
            response = self._call_llm(query, system_content=RETRIEVAL_PLAN_PROMPT)
            parsed = self._parse_json(response)
            sub_queries = parsed.get("sub_queries", [])
            result = {sq["source"]: sq["query"] for sq in sub_queries if "source" in sq and "query" in sq}
            if result:
                return result
        except Exception:
            # 如果 LLM 规划失败，退回到基于空格的简单拆分逻辑
            pass

        # 降级方案：当需要多源检索且用户主动用空格拼接多个关键词时，
        # 按 source 顺序将空格分隔的片段映射到各自的子查询。
        sources_in_order: list[str] = []
        if route.get("need_graph"):
            sources_in_order.append("graph")
        if route.get("need_skills"):
            sources_in_order.append("skill")
        if route.get("need_sensory"):
            sources_in_order.append("sensory")

        # 只在确实需要多源时才尝试拆分
        if len(sources_in_order) <= 1:
            return {}

        # 按空格拆分 query，过滤掉空片段
        parts = [p.strip() for p in query.split() if p.strip()]
        if len(parts) <= 1:
            return {}

        fallback_result: dict[str, str] = {}

        # 将每个 source 对应到一个片段，剩余片段合并到最后一个 source
        for idx, source in enumerate(sources_in_order):
            if idx >= len(parts):
                break
            if idx == len(sources_in_order) - 1:
                # 最后一个 source 接收剩余所有片段
                fallback_result[source] = " ".join(parts[idx:])
            else:
                fallback_result[source] = parts[idx]

        return fallback_result

    def _search_graph(self, query: str) -> list[dict[str, Any]]:
        """在知识图谱中检索相关节点和邻居。"""
        # 先做关键词搜索找到锚点节点
        hits = self.graph_store.search(query, top_k=3)
        if not hits:
            return []

        # 对每个命中节点展开 N 跳邻居
        results = []
        seen_ids = set()
        for hit in hits:
            node_id = hit.get("node_id", hit.get("id", ""))
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            subgraph = self.graph_store.get_neighbors(node_id, depth=self.graph_search_depth)
            results.append({
                "anchor": hit,
                "subgraph": subgraph,
            })
        return results

    def _search_skills(self, query: str) -> list[dict[str, Any]]:
        """使用 HyDE 策略检索技能库。

        1. 用 LLM 生成假设性回答
        2. 对假设回答做 embedding
        3. 用该 embedding 做向量检索
        4. 标记超过复用阈值的技能为 reusable
        """
        # HyDE: 生成假设文档
        hyde_doc = self._call_llm(query, system_content=self.prompt_template)

        # 对假设文档做 embedding
        hyde_embedding = self.embedder.embed_query(hyde_doc)

        # 向量检索
        results = self.skill_store.search(
            query=query,
            top_k=self.skill_top_k,
            query_embedding=hyde_embedding,
        )

        # 标记可复用技能
        for skill in results:
            skill["reusable"] = skill.get("score", 0) >= self.skill_reuse_threshold

        return results

    def _search_sensory(self) -> list[dict[str, Any]]:
        """获取最近的感觉记忆条目。"""
        items = self.sensory_buffer.get_recent(self.sensory_recent_n)
        return [
            {"content": item.content, "modality": item.modality, "timestamp": item.timestamp}
            for item in items
        ]
