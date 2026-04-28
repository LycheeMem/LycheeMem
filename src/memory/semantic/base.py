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
    retrieval_plan: dict[str, Any] = field(default_factory=dict)
    action_state: dict[str, Any] = field(default_factory=dict)
    usage_log_id: str = ""
    mode: str = "answer"


@dataclass(slots=True)
class ConsolidationResult:
    """固化返回的统一结构。"""

    records_added: int
    records_merged: int
    records_expired: int
    has_novelty: bool | None = None
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
        top_k: int = 0,
        query_embedding: list[float] | None = None,
        recent_context: str = "",
        action_state: dict[str, Any] | None = None,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> SemanticSearchResult:
        """检索与 query 相关的长期记忆。

        Args:
            query: 用户查询文本。
            session_id: 当前会话 ID（可选，用于 session-aware 检索）。
            top_k: 检索返回上限；0 表示由 planner 的 depth 决定。
            query_embedding: 预计算的 query 向量。
            recent_context: 最近几轮对话上下文（用于 state-conditioned retrieval）。
            action_state: 当前决策状态（可选）。
            retrieval_plan: Action-Aware Retrieval Plan 的 dict 表示（可选）。

        Returns:
            SemanticSearchResult 包含格式化 context 和 provenance。
        """

    @abstractmethod
    def ingest_conversation(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        retrieved_context: str = "",
        turn_index_offset: int = 0,
        reference_timestamp: str | None = None,
    ) -> ConsolidationResult:
        """将对话固化为长期记忆。

        Args:
            turns: 完整的对话轮次列表。
            session_id: 会话 ID。
            retrieved_context: 检索阶段已有的记忆上下文（用于新颖性检查）。
            turn_index_offset: 当前 turns 在完整 session 中的绝对起始索引。
            reference_timestamp: 参考时间戳（ISO 格式）。

        Returns:
            ConsolidationResult 包含写入统计和步骤详情。
        """

    @abstractmethod
    def delete_all(self) -> dict[str, int]:
        """清空所有语义记忆。

        Returns:
            dict 包含删除计数，如 {"records_deleted": N, "composites_deleted": M}。
        """

    @abstractmethod
    def export_debug(self) -> dict[str, Any]:
        """导出全量数据用于调试 / 前端展示。

        Returns:
            dict 包含 records / composites / stats 等。
        """

    @abstractmethod
    def finalize_usage_log(
        self,
        *,
        log_id: str,
        final_response_excerpt: str = "",
    ) -> None:
        """在回答生成后补充 usage log 的结果摘要。

        该方法不直接判断 success/fail，只记录本轮最终回复的摘要，
        供后续下一轮用户反馈或显式 outcome 回写使用。
        """

    @abstractmethod
    def apply_feedback_from_user_turn(
        self,
        *,
        session_id: str,
        user_turn: str,
    ) -> dict[str, Any]:
        """利用下一轮用户输入，对最近一次 action/mixed 检索做 outcome 回写。

        Returns:
            dict，通常包含 log_id / outcome / feedback；若未匹配到待回写日志则返回空 dict。
        """
