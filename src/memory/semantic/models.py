"""Compact Semantic Memory 的数据模型。

定义核心对象：
- MemoryRecord：经过去噪、指代消解、时间归一化后的最小自洽记忆记录
- SearchPlan：语义记忆检索计划
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── memory_type 枚举值（字符串常量，不用 Enum，方便 JSON 序列化） ──
MEMORY_TYPE_FACT = "fact"
MEMORY_TYPE_PREFERENCE = "preference"
MEMORY_TYPE_EVENT = "event"
MEMORY_TYPE_CONSTRAINT = "constraint"
MEMORY_TYPE_PROCEDURE = "procedure"
MEMORY_TYPE_FAILURE_PATTERN = "failure_pattern"
MEMORY_TYPE_TOOL_AFFORDANCE = "tool_affordance"

VALID_MEMORY_TYPES = frozenset({
    MEMORY_TYPE_FACT,
    MEMORY_TYPE_PREFERENCE,
    MEMORY_TYPE_EVENT,
    MEMORY_TYPE_CONSTRAINT,
    MEMORY_TYPE_PROCEDURE,
    MEMORY_TYPE_FAILURE_PATTERN,
    MEMORY_TYPE_TOOL_AFFORDANCE,
})

@dataclass
class MemoryRecord:
    """最小自洽记忆记录（Atomic Memory Record）。

    经过去噪、指代消解、时间归一化后的 context-independent 记忆条目。
    每条 record 脱离原始对话上下文也能被完整理解。
    """

    record_id: str  # SHA256(semantic_text)，天然幂等
    memory_type: str  # 取值见 VALID_MEMORY_TYPES

    # ── 文本 ──
    semantic_text: str  # 经过指代消解后的完整语义文本
    normalized_text: str  # 由 Python 生成: f"{memory_type}: {semantic_text}"（用于去重 & FTS）

    # ── 结构化元数据 ──
    entities: list[str] = field(default_factory=list)  # 涉及的实体名
    temporal: dict[str, Any] = field(default_factory=dict)  # {t_ref, t_valid_from, t_valid_to}
    tags: list[str] = field(default_factory=list)  # 统一标签（工具/约束/任务/失败/可供性等）

    # ── 置信度 & 来源 ──
    confidence: float = 1.0
    evidence_turn_range: list[int] = field(default_factory=list)  # [start_turn, end_turn]
    source_session: str = ""
    source_role: str = ""  # "user" | "assistant" | "both" | ""：该 record 主要出自哪一方
    created_at: str = ""  # ISO timestamp
    updated_at: str = ""

    # ── 软删除 ──
    expired: bool = False
    expired_at: str = ""
    expired_reason: str = ""


@dataclass
class EvidenceRoute:
    """One independent evidence need in a semantic retrieval plan."""

    route_id: str = ""
    evidence_goal: str = ""
    queries: list[str] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    temporal_filter: dict[str, str] | None = None


@dataclass
class ActionState:
    """Compatibility state passed by the agent-facing search coordinator."""

    current_subgoal: str = ""
    tentative_action: str = ""
    last_tool_name: str = ""
    last_tool_result: str = ""
    missing_slots: list[str] = field(default_factory=list)
    known_constraints: list[str] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=list)
    failure_signal: str = ""
    token_budget: int = 0
    recent_context_excerpt: str = ""


@dataclass
class SearchPlan:
    """语义记忆检索计划。

    planner 只负责描述用户问题本身：问题类型、改写查询、时间范围和语义约束。
    """

    semantic_queries: list[str] = field(default_factory=list)  # 面向语义内容的检索词
    pragmatic_queries: list[str] = field(default_factory=list)
    mode: str = "answer"
    temporal_filter: dict[str, str] | None = None  # {"since": "YYYY-MM-DD", "until": "YYYY-MM-DD"}
    depth: int = 15  # 统一语义证据召回深度
    question_type: str = "single"  # single|aggregate|temporal|comparison|personalized_advice|prior_assistant_response|other
    evidence_target: str = ""  # 用户问题中需要寻找的事实/事件/集合/属性
    evidence_constraints: list[str] = field(default_factory=list)  # 用户问题显式要求的条件/属性/关系
    constraints: list[dict[str, Any]] = field(default_factory=list)  # 通用证据约束
    evidence_routes: list[EvidenceRoute] = field(default_factory=list)  # 独立证据需求
    reasoning: str = ""  # 规划理由
