"""语义长期记忆引擎的抽象接口。

所有长期语义记忆后端（Compact / Graphiti adapter）都必须实现此接口。
SearchCoordinator 和 ConsolidatorAgent 通过此接口与后端交互，
不直接依赖具体实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SemanticSearchResult:
    """检索返回的统一结构。"""

    context: str  # 可直接注入 LLM 的格式化文本块
    provenance: list[dict[str, Any]]  # 溯源信息列表


@dataclass(slots=True)
class ConsolidationResult:
    """固化返回的统一结构。"""

    records_added: int
    records_merged: int
    records_expired: int
    steps: list[dict[str, Any]] = field(default_factory=list)


class BaseSemanticMemoryEngine(ABC):
    """所有长期语义记忆引擎必须实现的契约。

    SearchCoordinator 调用 search()，ConsolidatorAgent 调用 ingest_conversation()。
    API router 调用 delete_all_for_user() / export_debug()。
    """

    @abstractmethod
    def search(
        self,
        *,
        query: str,
        session_id: str | None = None,
        top_k: int = 10,
        query_embedding: list[float] | None = None,
        user_id: str = "",
        retrieval_plan: dict[str, Any] | None = None,
    ) -> SemanticSearchResult:
        """检索与 query 相关的长期记忆。

        Args:
            query: 用户查询文本。
            session_id: 当前会话 ID（可选，用于 session-aware 检索）。
            top_k: 返回条目上限。
            query_embedding: 预计算的 query 向量。
            user_id: 用户 ID（多用户隔离）。
            retrieval_plan: Task-Aware Retrieval Plan (SearchPlan) 的 dict 表示（可选）。

        Returns:
            SemanticSearchResult 包含格式化 context 和 provenance。
        """

    @abstractmethod
    def ingest_conversation(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        user_id: str = "",
        retrieved_context: str = "",
        reference_timestamp: str | None = None,
    ) -> ConsolidationResult:
        """将对话固化为长期记忆。

        Args:
            turns: 完整的对话轮次列表。
            session_id: 会话 ID。
            user_id: 用户 ID。
            retrieved_context: 检索阶段已有的记忆上下文（用于新颖性检查）。
            reference_timestamp: 参考时间戳（ISO 格式）。

        Returns:
            ConsolidationResult 包含写入统计和步骤详情。
        """

    @abstractmethod
    def delete_all_for_user(self, user_id: str) -> dict[str, int]:
        """清空指定用户的所有语义记忆。

        Returns:
            dict 包含删除计数，如 {"records_deleted": N, "composites_deleted": M}。
        """

    @abstractmethod
    def export_debug(self, *, user_id: str = "") -> dict[str, Any]:
        """导出全量数据用于调试 / 前端展示。

        Returns:
            dict 包含 records / composites / stats 等。
        """
