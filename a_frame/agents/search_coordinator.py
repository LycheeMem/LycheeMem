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
    ):
        super().__init__(llm=llm, prompt_template=HYDE_SYSTEM_PROMPT)
        self.embedder = embedder
        self.graph_store = graph_store
        self.skill_store = skill_store
        self.sensory_buffer = sensory_buffer
        self.graph_search_depth = graph_search_depth
        self.skill_top_k = skill_top_k
        self.sensory_recent_n = sensory_recent_n

    def run(
        self,
        user_query: str,
        route: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """根据路由决策执行多源检索。

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

        if route.get("need_graph"):
            result["retrieved_graph_memories"] = self._search_graph(user_query)

        if route.get("need_skills"):
            result["retrieved_skills"] = self._search_skills(user_query)

        if route.get("need_sensory"):
            result["retrieved_sensory"] = self._search_sensory()

        return result

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
        """
        # HyDE: 生成假设文档
        hyde_doc = self._call_llm(query, system_content=self.prompt_template)

        # 对假设文档做 embedding
        hyde_embedding = self.embedder.embed_query(hyde_doc)

        # 向量检索
        return self.skill_store.search(
            query=query,
            top_k=self.skill_top_k,
            query_embedding=hyde_embedding,
        )

    def _search_sensory(self) -> list[dict[str, Any]]:
        """获取最近的感觉记忆条目。"""
        items = self.sensory_buffer.get_recent(self.sensory_recent_n)
        return [
            {"content": item.content, "modality": item.modality, "timestamp": item.timestamp}
            for item in items
        ]
