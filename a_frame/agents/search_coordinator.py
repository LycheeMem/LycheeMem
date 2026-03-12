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
你是一个假设文档生成器。给定用户查询，生成一段假设性的"理想回答"文本。
这段文本将被用于向量检索，因此应该包含与查询相关的关键概念和词汇。

注意：你生成的不是真正的回答，而是一个用于语义匹配的"锚点文本"。
保持简洁（2-3 句话），聚焦关键实体和概念。"""

RETRIEVAL_PLAN_PROMPT = """\
你是一个检索规划器。根据用户查询，生成结构化的检索计划。
分析查询需要哪些子问题，以及每个子问题应该查询哪个记忆源。

以 JSON 格式回复：
{
  "sub_queries": [
    {"source": "graph", "query": "针对图谱的子查询"},
    {"source": "skill", "query": "针对技能库的子查询"}
  ],
  "reasoning": "为什么这样分解查询"
}

规则：
- source 只能是 "graph", "skill", "sensory" 之一
- 如果查询简单不需要分解，sub_queries 可以只有一个元素
- 每个子查询应该比原查询更精确"""


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
            return {sq["source"]: sq["query"] for sq in sub_queries if "source" in sq and "query" in sq}
        except Exception:
            return {}

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
