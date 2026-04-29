"""Compact Semantic Memory 的数据模型。

定义两类核心对象：
- MemoryRecord：经过去噪、指代消解、时间归一化后的最小自洽记忆记录
- CompositeRecord：由多个 MemoryRecord 聚合形成的高密度记录

以及辅助对象：
- SearchPlan：行动感知检索计划
- UsageLog：检索使用记录（为 RL 阶段准备）
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

SYNTH_TYPE_PREFERENCE = "composite_preference"
SYNTH_TYPE_PATTERN = "composite_pattern"
SYNTH_TYPE_CONSTRAINT = "composite_constraint"
SYNTH_TYPE_USAGE = "usage_pattern"

VALID_SYNTH_TYPES = frozenset({
    SYNTH_TYPE_PREFERENCE,
    SYNTH_TYPE_PATTERN,
    SYNTH_TYPE_CONSTRAINT,
    SYNTH_TYPE_USAGE,
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

    # ── 使用统计（为 RL 阶段准备 next-state signal） ──
    retrieval_count: int = 0
    retrieval_hit_count: int = 0  # 被检索且被 synthesizer 保留的次数
    action_success_count: int = 0  # 关联 action 执行成功的次数
    action_fail_count: int = 0
    last_retrieved_at: str = ""

    # ── 软删除 ──
    expired: bool = False
    expired_at: str = ""
    expired_reason: str = ""


@dataclass
class CompositeRecord:
    """复合记忆记录。

    由多个 MemoryRecord 通过 embedding 聚类形成的聚合节点。
    聚合后不删除 source records（保留细粒度检索能力），
    但在检索排序中 composite record 优先于碎片 records。
    """

    composite_id: str
    memory_type: str  # 取值见 VALID_SYNTH_TYPES

    # ── 文本 ──
    semantic_text: str  # cluster 中距质心最近的 record 的 semantic_text
    normalized_text: str

    # ── 聚合来源 ──
    source_record_ids: list[str] = field(default_factory=list)
    child_composite_ids: list[str] = field(default_factory=list)  # 直接子 composite，形成层级树

    # ── 继承自 source records 的聚合元数据 ──
    entities: list[str] = field(default_factory=list)
    temporal: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)  # 统一标签

    confidence: float = 1.0
    created_at: str = ""
    updated_at: str = ""

    # ── 使用统计 ──
    retrieval_count: int = 0
    retrieval_hit_count: int = 0
    action_success_count: int = 0
    action_fail_count: int = 0
    last_retrieved_at: str = ""


@dataclass
class ActionState:
    """当前决策状态（Decision State）。

    用于把检索从“只看 query”推进到“结合当前动作意图、约束与执行状态”。
    该结构会传入 planner，并随 usage log 一并记录，作为后续 usage-aware / RL
    阶段的状态基座。
    """

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
    """行动感知检索计划。

    planner 分析当前请求后输出此结构，
    指导多通道召回和 scorer 打分。
    """

    mode: str  # "answer" | "action" | "mixed"
    semantic_queries: list[str] = field(default_factory=list)  # 面向语义内容的检索词
    pragmatic_queries: list[str] = field(default_factory=list)  # 面向 action/tool/constraint 的检索词
    temporal_filter: dict[str, str] | None = None  # {since, until}
    tool_hints: list[str] = field(default_factory=list)  # 当前请求可能需要的工具名
    required_constraints: list[str] = field(default_factory=list)  # 当前 action 缺的约束
    required_affordances: list[str] = field(default_factory=list)  # 当前 action 所需的能力/可供性
    missing_slots: list[str] = field(default_factory=list)  # 当前 action 缺的参数/slot
    tree_retrieval_mode: str = "balanced"  # "root_only" | "balanced" | "descend"
    tree_expansion_depth: int = 1  # 树下钻深度；0=不下钻
    include_leaf_records: bool = False  # 是否将叶子 record 纳入最终候选池
    include_episodic_context: bool = False  # 是否补充原始对话上下文
    episodic_turn_window: int = 0  # 原始对话窗口大小（按 evidence turn 向两侧扩展）
    depth: int = 5  # 建议检索深度 (top_k)
    reasoning: str = ""  # 规划理由


@dataclass
class UsageLog:
    """单次检索使用记录（为 RL 阶段准备 next-state signal）。

    每次 search 调用后记录一条，包含检索计划、召回结果、保留结果。
    后续通过 action_outcome / user_feedback 回填使用效果。
    """

    log_id: str
    session_id: str
    timestamp: str  # ISO
    query: str
    retrieval_plan: dict[str, Any] = field(default_factory=dict)
    action_state: dict[str, Any] = field(default_factory=dict)
    retrieved_record_ids: list[str] = field(default_factory=list)  # 被召回的 record IDs
    kept_record_ids: list[str] = field(default_factory=list)  # 被后续融合阶段最终保留的
    final_response_excerpt: str = ""
    user_feedback: str = ""  # "positive" | "negative" | "correction" | ""
    action_outcome: str = ""  # "success" | "fail" | "unknown"
