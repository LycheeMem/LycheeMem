"""Compact Semantic Memory 的数据模型。

定义两类核心对象：
- MemoryUnit：经过去噪、指代消解、时间归一化后的最小自洽记忆单元
- SynthesizedUnit：由多个 MemoryUnit 在线合成的高密度条目

以及辅助对象：
- RetrievalPlan：Action-Aware 检索计划
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

SYNTH_TYPE_PREFERENCE = "synthesized_preference"
SYNTH_TYPE_PATTERN = "synthesized_pattern"
SYNTH_TYPE_CONSTRAINT = "synthesized_constraint"
SYNTH_TYPE_USAGE = "usage_pattern"

VALID_SYNTH_TYPES = frozenset({
    SYNTH_TYPE_PREFERENCE,
    SYNTH_TYPE_PATTERN,
    SYNTH_TYPE_CONSTRAINT,
    SYNTH_TYPE_USAGE,
})


@dataclass
class MemoryUnit:
    """最小自洽记忆单元（Atomic Memory Unit）。

    经过去噪、指代消解、时间归一化后的 context-independent 记忆条目。
    每个 unit 脱离原始对话上下文也能被完整理解。

    对应 idea 模块一（Compact Semantic Encoding）的输出。
    """

    unit_id: str  # SHA256(normalized_text)，天然幂等
    memory_type: str  # 取值见 VALID_MEMORY_TYPES

    # ── 文本 ──
    semantic_text: str  # 经过指代消解后的完整语义文本
    normalized_text: str  # 归一化后的紧凑表述（用于去重 & FTS）

    # ── 结构化元数据（action-facing metadata） ──
    entities: list[str] = field(default_factory=list)  # 涉及的实体名
    temporal: dict[str, Any] = field(default_factory=dict)  # {t_ref, t_valid_from, t_valid_to}
    task_tags: list[str] = field(default_factory=list)  # 适用任务类型
    tool_tags: list[str] = field(default_factory=list)  # 关联工具/API
    constraint_tags: list[str] = field(default_factory=list)  # 约束条件
    failure_tags: list[str] = field(default_factory=list)  # 失败模式标记
    affordance_tags: list[str] = field(default_factory=list)  # 行动可供性

    # ── 置信度 & 来源 ──
    confidence: float = 1.0
    evidence_turn_range: list[int] = field(default_factory=list)  # [start_turn, end_turn]
    source_session: str = ""
    user_id: str = ""
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
class SynthesizedUnit:
    """合成记忆单元（Pragmatic Synthesis 输出）。

    由多个 MemoryUnit 在线合成的高密度条目。
    合成后不删除 source units（保留细粒度检索能力），
    但在检索排序中 synth unit 优先于碎片 units。

    对应 idea 模块二（Pragmatic Memory Synthesis）的输出。
    """

    synth_id: str
    memory_type: str  # 取值见 VALID_SYNTH_TYPES

    # ── 文本 ──
    semantic_text: str
    normalized_text: str

    # ── 合成来源 ──
    source_unit_ids: list[str] = field(default_factory=list)
    synthesis_reason: str = ""  # LLM 给出的合成理由

    # ── 继承自 source units 的聚合元数据 ──
    entities: list[str] = field(default_factory=list)
    temporal: dict[str, Any] = field(default_factory=dict)
    task_tags: list[str] = field(default_factory=list)
    tool_tags: list[str] = field(default_factory=list)
    constraint_tags: list[str] = field(default_factory=list)
    failure_tags: list[str] = field(default_factory=list)
    affordance_tags: list[str] = field(default_factory=list)

    confidence: float = 1.0
    user_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    # ── 使用统计 ──
    retrieval_count: int = 0
    retrieval_hit_count: int = 0
    action_success_count: int = 0
    action_fail_count: int = 0
    last_retrieved_at: str = ""


@dataclass
class RetrievalPlan:
    """Action-Aware Retrieval Plan（模块三 planner 输出）。

    planner 分析当前请求后输出此结构，
    指导多通道召回和 scorer 打分。
    """

    mode: str  # "answer" | "action" | "mixed"
    semantic_queries: list[str] = field(default_factory=list)  # 面向语义内容的检索词
    pragmatic_queries: list[str] = field(default_factory=list)  # 面向 action/tool/constraint 的检索词
    temporal_filter: dict[str, str] | None = None  # {since, until}
    tool_hints: list[str] = field(default_factory=list)  # 当前请求可能需要的工具名
    required_constraints: list[str] = field(default_factory=list)  # 当前 action 缺的约束
    missing_slots: list[str] = field(default_factory=list)  # 当前 action 缺的参数/slot
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
    user_id: str
    timestamp: str  # ISO
    query: str
    retrieval_plan: dict[str, Any] = field(default_factory=dict)
    retrieved_unit_ids: list[str] = field(default_factory=list)  # 被召回的 unit IDs
    kept_unit_ids: list[str] = field(default_factory=list)  # 被 synthesizer 最终保留的
    final_response_excerpt: str = ""
    user_feedback: str = ""  # "positive" | "negative" | "correction" | ""
    action_outcome: str = ""  # "success" | "fail" | "unknown"
