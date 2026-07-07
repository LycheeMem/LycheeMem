"""Semantic long-term memory engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SemanticSearchResult:
    """检索返回的统一结构。"""

    context: str  # 可直接注入 LLM 的格式化文本块
    provenance: list[dict[str, Any]]  # 溯源信息列表
    retrieval_plan: dict[str, Any] = field(default_factory=dict)
    action_state: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    usage_log_id: str = ""
    mode: str = "semantic"


@dataclass(slots=True)
class ConsolidationResult:
    """固化返回的统一结构。"""

    records_added: int
    records_merged: int
    records_expired: int
    turns_consumed: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)


class BaseSemanticMemoryEngine(ABC):
    """Contract implemented by semantic memory backends."""

    @abstractmethod
    def search(
        self,
        *,
        query: str,
        session_id: str | None = None,
        top_k: int = 0,
        query_embedding: list[float] | None = None,
        recent_context: str = "",
        action_state: dict[str, Any] | None = None,
        retrieval_plan: dict[str, Any] | None = None,
        reference_time: str | None = None,
    ) -> SemanticSearchResult:
        """检索与 query 相关的长期记忆。

        Args:
            query: 用户查询文本。
            session_id: 当前会话 ID（可选，用于 session-aware 检索）。
            top_k: 检索返回上限；0 表示由检索模式决定。
            query_embedding: 预计算的 query 向量。
            recent_context: 最近几轮对话上下文（用于 planner 消歧）。
            action_state: 兼容旧调用链的任务状态；目标语义检索实现不直接依赖。
            retrieval_plan: Semantic Retrieval Plan 的 dict 表示（可选）。
            reference_time: 当前查询的参考时间（ISO 格式，可选）。

        Returns:
            SemanticSearchResult 包含格式化 context 和 provenance。
        """

    @abstractmethod
    def ingest_conversation(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        turn_index_offset: int = 0,
        reference_timestamp: str | None = None,
        flush_session: bool = False,
    ) -> ConsolidationResult:
        """将对话固化为长期记忆。

        Args:
            turns: 完整的对话轮次列表。
            session_id: 会话 ID。
            turn_index_offset: 当前 turns 在完整 session 中的绝对起始索引。
            reference_timestamp: 参考时间戳（ISO 格式）。
            flush_session: session 结束时强制编码当前 pending chunk。

        Returns:
            ConsolidationResult 包含写入统计和步骤详情。
        """

    @abstractmethod
    def delete_all(self) -> dict[str, int]:
        """清空所有语义记忆。

        Returns:
            dict 包含删除计数，如 {"records_deleted": N, "evidence_nodes_deleted": M}。
        """

    @abstractmethod
    def export_debug(self) -> dict[str, Any]:
        """导出全量数据用于调试 / 前端展示。

        Returns:
            dict 包含 records / evidence_nodes / stats 等。
        """

    def finalize_usage_log(
        self,
        *,
        log_id: str,
        final_response_excerpt: str = "",
    ) -> None:
        """兼容旧 usage log 接口；fielded evidence 检索不维护该日志。"""

    def apply_feedback_from_user_turn(
        self,
        *,
        session_id: str,
        user_turn: str,
    ) -> dict[str, Any]:
        """兼容旧反馈回写接口；fielded evidence 检索不维护该日志。"""
        return {}
