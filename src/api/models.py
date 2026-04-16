"""API 请求/响应数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ─── Chat ───


class ChatRequest(BaseModel):
    """对话请求。"""

    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=100_000)


class WMManagerTrace(BaseModel):
    wm_token_usage: int = 0
    compressed_turn_count: int = 0
    raw_recent_turn_count: int = 0
    compression_happened: bool = False


class GraphMemoryHit(BaseModel):
    node_id: str = ""
    name: str = ""
    label: str = ""
    score: float = 0.0
    neighbor_count: int = 0


class SkillHit(BaseModel):
    skill_id: str = ""
    intent: str = ""
    score: float = 0.0
    reusable: bool = False


class SearchCoordinatorTrace(BaseModel):
    graph_memories: list[GraphMemoryHit] = []
    skills: list[SkillHit] = []
    total_retrieved: int = 0


class ProvenanceItem(BaseModel):
    """溯源条目：携带一条 Fact 的检索评分元数据以及回溯到原始 Episode 的完整引用链。

    Paper §2.1: "Semantic artifacts can be traced to their sources for citation
    or quotation, while episodes can quickly retrieve their relevant entities
    and facts."
    """

    # ── 评分/排名元数据 ──
    source: str = ""          # 来源标识（如 "graphiti_retrieval"）
    index: int = 0            # 在 provenance 列表中的位置（0-based）
    relevance: float = 0.0    # 综合得分（RRF + boosts + cross-encoder）

    # ── Fact 标识 ──
    fact_id: str = ""         # 对应 Fact 节点的 fact_id
    summary: str = ""         # Fact 的 fact_text（人类可读）

    # ── 检索信号细节 ──
    rrf_score: float = 0.0
    bm25_rank: int | None = None
    bfs_rank: int | None = None
    mention_count: int = 0
    graph_distance: int | None = None
    cross_encoder_score: float | None = None

    # ── 双向 Episode 引用链（Paper §2.1）──
    # 每条条目是一个 Episode 快照，包含 episode_id、session_id、role、
    # content（原始文本）、turn_index、t_ref（参考时间戳）。
    # 通过 EVIDENCE_FOR（Fact 直接证据）或 MENTIONS（实体出现）关系
    source_episodes: list[dict[str, Any]] = []


class SynthesizerTrace(BaseModel):
    background_context: str = ""
    provenance: list[ProvenanceItem] = []
    skill_reuse_plan: list[dict[str, Any]] = []
    kept_count: int = 0
    dropped_count: int = 0


class ReasonerTrace(BaseModel):
    response_length: int = 0


class ConsolidatorStepTrace(BaseModel):
    name: str
    status: str = "done"
    detail: str = ""


class ConsolidatorTrace(BaseModel):
    status: str = "pending"
    entities_added: int = 0
    skills_added: int = 0
    facts_added: int = 0
    records_expired: int = 0
    has_novelty: bool | None = None
    skipped_reason: str | None = None
    steps: list[ConsolidatorStepTrace] = []


class PipelineTrace(BaseModel):
    wm_manager: WMManagerTrace = WMManagerTrace()
    search_coordinator: SearchCoordinatorTrace = SearchCoordinatorTrace()
    synthesizer: SynthesizerTrace = SynthesizerTrace()
    reasoner: ReasonerTrace = ReasonerTrace()
    consolidator: ConsolidatorTrace = ConsolidatorTrace()


class ChatResponse(BaseModel):
    """对话响应。"""

    session_id: str
    response: str
    memories_retrieved: int = 0
    wm_token_usage: int = 0
    turn_input_tokens: int = 0   # 本轮主流程消耗的输入 token 总量
    turn_output_tokens: int = 0  # 本轮主流程消耗的输出 token 总量
    trace: PipelineTrace | None = None


# ─── Memory ───


class GraphNode(BaseModel):
    id: str
    label: str = ""
    properties: dict[str, Any] = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str = ""


class GraphResponse(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    tree_roots: list[dict[str, Any]] = []


class FactEdge(BaseModel):
    """Graphiti Fact-node 映射出来的兼容 edge 视图（PR4）。"""

    source: str
    target: str
    relation: str = ""

    # Optional enriched fields (keep defaults to stay backward/forward compatible)
    confidence: float = 1.0
    fact: str = ""
    evidence: str = ""
    source_session: str = ""
    timestamp: str = ""

    t_valid_from: str = ""
    t_valid_to: str = ""
    t_tx_created: str = ""
    t_tx_expired: str = ""

    episode_ids: list[str] = Field(default_factory=list)


class FactEdgesResponse(BaseModel):
    edges: list[FactEdge]
    total: int


class SkillItem(BaseModel):
    id: str
    intent: str = ""
    doc_markdown: str = ""
    metadata: dict[str, Any] = {}


class SkillsResponse(BaseModel):
    skills: list[dict[str, Any]]
    total: int


# ─── Session ───


class TurnItem(BaseModel):
    """单个会话轮次的数据模型。"""

    role: str  # 'user' | 'assistant'
    content: str
    token_count: int = 0
    created_at: str | None = None
    deleted: bool = False


class SessionResponse(BaseModel):
    session_id: str
    turns: list[TurnItem]
    turn_count: int
    summaries: list[dict[str, Any]] = []
    wm_max_tokens: int = 128000
    wm_current_tokens: int = 0  # 当前实际工作记忆 token 占用（摘要 + 近期轮次）


class DeleteResponse(BaseModel):
    message: str


# ─── Sessions List ───


class SessionSummary(BaseModel):
    session_id: str
    turn_count: int
    last_message: str = ""
    title: str = ""  # 优先使用 topic，否则取首条用户消息前40字
    topic: str = ""
    tags: list[str] = []
    created_at: str | None = None
    updated_at: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]
    total: int


class SessionUpdateRequest(BaseModel):
    """会话元数据更新请求。"""

    topic: str | None = None
    tags: list[str] | None = None


# ─── Memory Search ───


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    include_graph: bool = True
    include_skills: bool = True


class MemorySearchResponse(BaseModel):
    query: str
    graph_results: list[dict[str, Any]]
    semantic_results: list[dict[str, Any]]
    skill_results: list[dict[str, Any]]
    novelty_retrieved_context: str = ""
    total: int


class MemorySmartSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    include_graph: bool = True
    include_skills: bool = True
    synthesize: bool = True
    mode: str = Field(default="compact", pattern="^(raw|full|compact)$")


# ─── Memory Synthesize ───


class MemorySynthesizeRequest(BaseModel):
    """记忆合成请求：对检索结果进行 LLM-as-Judge 评分与融合，生成 background_context。

    设计为可直接衔接 /memory/search 的响应：将 graph_results / skill_results 原样传入。
    """

    user_query: str = Field(..., min_length=1, max_length=100_000)
    graph_results: list[dict[str, Any]] = Field(default_factory=list)
    semantic_results: list[dict[str, Any]] = Field(default_factory=list)
    skill_results: list[dict[str, Any]] = Field(default_factory=list)


class MemorySynthesizeResponse(BaseModel):
    """记忆合成响应。"""

    background_context: str
    skill_reuse_plan: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    kept_count: int = 0
    dropped_count: int = 0


class MemorySmartSearchResponse(BaseModel):
    query: str
    mode: str = "compact"
    graph_results: list[dict[str, Any]]
    semantic_results: list[dict[str, Any]]
    skill_results: list[dict[str, Any]]
    novelty_retrieved_context: str = ""
    total: int
    synthesized: bool = False
    background_context: str = ""
    skill_reuse_plan: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    kept_count: int = 0
    dropped_count: int = 0


# ─── Memory Reason ───


class MemoryReasonRequest(BaseModel):
    """最终推理请求：基于合成后的上下文生成回答。

    典型用法：
      1. POST /memory/search         → graph_results / skill_results
      2. POST /memory/synthesize     → background_context / skill_reuse_plan
      3. POST /memory/reason         → response（本端点）
      4. POST /memory/consolidate    → 固化长期记忆
    """

    session_id: str = Field(..., min_length=1, max_length=128)
    user_query: str = Field(..., min_length=1, max_length=100_000)
    background_context: str = ""
    skill_reuse_plan: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_skills: list[dict[str, Any]] = Field(default_factory=list)
    # 是否将本轮 user/assistant 轮次写回会话（供后续 /memory/consolidate 使用）
    append_to_session: bool = True
    # 可选参考时间（ISO 8601 字符串），用于帮助推理器正确解析相对时间表达式。
    # 例如 Locomo 评测中对话发生在 2022-2023 年，注入此字段可避免 LLM 使用系统当前时间。
    reference_time: str | None = None


class MemoryReasonResponse(BaseModel):
    """最终推理响应。"""

    response: str
    session_id: str
    wm_token_usage: int = 0


# ─── Memory Append Turn ───


class MemoryAppendTurnRequest(BaseModel):
    """向 LycheeMem session store 追加外部宿主对话轮次。"""

    session_id: str = Field(..., min_length=1, max_length=128)
    role: str = Field(..., min_length=1, max_length=32)
    content: str = Field(..., min_length=1, max_length=100_000)
    token_count: int = Field(default=0, ge=0, le=1_000_000)


class MemoryAppendTurnResponse(BaseModel):
    status: str = "appended"
    session_id: str
    turn_count: int = 0


# ─── Memory Consolidate ───


class MemoryConsolidateRequest(BaseModel):
    """记忆固化请求：对当前会话进行记忆萃取，写入图谱与技能库。

    传入 retrieved_context 有助于新颖性判断（避免重复固化已有记忆）。
    retrieved_context 应优先取自 search 阶段召回的原始语义记忆片段（pre-synthesis raw context），
    而不是 /memory/synthesize 的 background_context。

    background=True（默认）：在后台线程中异步执行固化，立即返回 status="started"，
        与 Pipeline 内部行为一致，适合生产调用（固化耗时可能超过 60 秒）。
    background=False：同步等待固化完成后返回详细结果，适合调试/验证。
    """

    session_id: str = Field(..., min_length=1, max_length=128)
    # search 阶段召回的原始已有语义记忆片段，用于判断对话是否引入了新信息
    retrieved_context: str = ""
    # 是否在后台线程中异步执行（默认 True，避免 HTTP 超时）
    background: bool = True
    # 强制固化：忽略水位线，从头重新处理所有 turns（用于补跑被跳过的 session）
    force_ingest: bool = False
    # 跳过技能抽取，只写入语义记忆（适合批量摄入场景）
    skip_skills: bool = False
    # 对话发生的日期（自由文本，如 "May 8, 2023"），用于将相对时间（yesterday/last week）
    # 解析为绝对日期。由调用方提供，后端透传给 encoder。
    session_date: str | None = None


class MemoryConsolidateResponse(BaseModel):
    """记忆固化响应。

    background=True 时：status="started"，其余数值字段为 0（结果在后台写入）。
    background=False 时：status="done" 或 "skipped"，包含实际计数和步骤日志。
    """

    # "started"  — 后台异步触发
    # "done"     — 同步执行完毕
    # "skipped"  — 无有效轮次或无新信息
    status: str = "done"
    entities_added: int = 0
    skills_added: int = 0
    facts_added: int = 0
    has_novelty: bool | None = None
    skipped_reason: str | None = None
    steps: list[dict[str, Any]] = []


# ─── Graph Manual Operations ───


class GraphNodeAddRequest(BaseModel):
    id: str = Field(..., min_length=1)
    label: str = "Entity"
    properties: dict[str, Any] = {}


class GraphEdgeAddRequest(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)
    properties: dict[str, Any] = {}


# ─── Health ───


class HealthResponse(BaseModel):
    status: str
    version: str


# ─── Pipeline Status ───


class PipelineStatusResponse(BaseModel):
    """Pipeline 运行状态。"""

    session_count: int
    graph_node_count: int
    graph_edge_count: int
    skill_count: int
