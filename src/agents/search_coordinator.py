"""
检索协调器 (Memory Search Coordinator)。

语义记忆检索直接调用 CompactSemanticEngine.search()。
技能库检索使用 HyDE → embedding → 向量检索。
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.embedder.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.memory.procedural.sqlite_skill_store import SQLiteSkillStore
from src.memory.semantic.base import BaseSemanticMemoryEngine

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


class SearchCoordinator(BaseAgent):
    """检索协调器：每次请求均同时检索语义记忆和技能库。"""

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        skill_store: SQLiteSkillStore,
        semantic_engine: BaseSemanticMemoryEngine,
        graph_top_k: int = 3,
        skill_top_k: int = 3,
        skill_reuse_threshold: float = 0.85,
    ):
        super().__init__(llm=llm, prompt_template=HYDE_SYSTEM_PROMPT)
        self.embedder = embedder
        self.skill_store = skill_store
        self.semantic_engine = semantic_engine
        self.graph_top_k = graph_top_k
        self.skill_top_k = skill_top_k
        self.skill_reuse_threshold = skill_reuse_threshold

    def run(
        self,
        user_query: str,
        **kwargs,
    ) -> dict[str, Any]:
        """同时检索语义记忆和技能库。

        Returns:
            dict 包含：retrieved_graph_memories, retrieved_skills
        """
        session_id = kwargs.get("session_id")
        if session_id is not None:
            session_id = str(session_id)
        user_id = kwargs.get("user_id", "")
        return self._run_compact(user_query, session_id=session_id, user_id=user_id)

    def _run_compact(
        self,
        user_query: str,
        *,
        session_id: str | None = None,
        user_id: str = "",
    ) -> dict[str, Any]:
        """Compact 后端路径：semantic_engine.search() + 技能库。"""
        result = self.semantic_engine.search(
            query=user_query,
            session_id=session_id,
            top_k=self.graph_top_k,
            user_id=user_id,
        )

        # 将 SemanticSearchResult 转为 pipeline 期望的格式
        graph_memories = []
        if result.context.strip():
            graph_memories = [
                {
                    "anchor": {
                        "node_id": "compact_context",
                        "name": "CompactSemanticMemory",
                        "label": "Context",
                        "score": 1.0,
                    },
                    "subgraph": {"nodes": [], "edges": []},
                    "constructed_context": result.context,
                    "provenance": result.provenance,
                }
            ]

        skill_results = self._search_skills(user_query, user_id=user_id)

        return {
            "retrieved_graph_memories": graph_memories,
            "retrieved_skills": skill_results,
        }

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
