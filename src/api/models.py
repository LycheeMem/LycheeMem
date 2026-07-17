"""API 请求/响应数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求。"""

    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=100_000)
    reference_time: str | None = None
    images: list[str] = Field(
        default=[],
        description="Base64 编码的图片列表（可选，支持多模态输入）",
    )
    image_mime_types: list[str] = Field(
        default=[],
        description="图片 MIME 类型列表（可选，与 images 一一对应）",
    )


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
    skipped_reason: str | None = None
    steps: list[ConsolidatorStepTrace] = []


class PipelineTrace(BaseModel):
    wm_manager: WMManagerTrace = WMManagerTrace()
    search_coordinator: SearchCoordinatorTrace = SearchCoordinatorTrace()
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


class OpenAIChatMessage(BaseModel):
    role: str = Field(..., min_length=1, max_length=32)
    content: str | list[dict[str, Any]] | None = ""
    name: str | None = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str = "lycheemem"
    messages: list[OpenAIChatMessage] = Field(..., min_length=1)
    stream: bool = False
    store: bool | None = None
    consolidate: bool | None = None
    user: str | None = Field(default=None, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    reference_time: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream_options: dict[str, Any] | None = None


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class OpenAIChatChoice(BaseModel):
    index: int = 0
    message: OpenAIChatCompletionMessage
    finish_reason: str = "stop"


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatChoice]
    usage: OpenAIUsage


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


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    include_graph: bool = True
    include_skills: bool = True
    reference_time: str | None = None


class MemorySearchResponse(BaseModel):
    query: str
    graph_results: list[dict[str, Any]]
    semantic_results: list[dict[str, Any]]
    skill_results: list[dict[str, Any]]
    total: int


class MemorySmartSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    include_graph: bool = True
    include_skills: bool = True
    mode: str = Field(default="compact", pattern="^(raw|full|compact)$")
    response_level: str = Field(default="full", pattern="^(minimal|compact|full)$")
    reference_time: str | None = None


class MemorySmartSearchResponse(BaseModel):
    query: str
    mode: str = "compact"
    graph_results: list[dict[str, Any]] = Field(default_factory=list)
    semantic_results: list[dict[str, Any]] = Field(default_factory=list)
    skill_results: list[dict[str, Any]] = Field(default_factory=list)
    total: int
    background_context: str = ""


class MemoryReasonRequest(BaseModel):
    """最终推理请求：基于检索上下文生成回答。

    典型用法：
      1. POST /memory/smart-search   → background_context / skill_reuse_plan
      2. POST /memory/reason         → response（本端点）
      3. POST /memory/consolidate    → 固化长期记忆
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


class MemoryAppendTurnRequest(BaseModel):
    """向 LycheeMem session store 追加外部宿主对话轮次。"""

    session_id: str = Field(..., min_length=1, max_length=128)
    role: str = Field(..., min_length=1, max_length=32)
    # Optional participant identity, distinct from the transport role.  This is
    # useful for imported multi-party conversations where every participant is
    # submitted through a user-role API turn.
    speaker: str | None = Field(default=None, max_length=128)
    content: str = Field(..., min_length=1, max_length=100_000)
    token_count: int = Field(default=0, ge=0, le=1_000_000)
    created_at: str | None = None


class MemoryAppendTurnResponse(BaseModel):
    status: str = "appended"
    session_id: str
    turn_count: int = 0


class MemoryConsolidateRequest(BaseModel):
    """记忆固化请求：对当前会话进行记忆萃取，写入图谱与技能库。

    background=True（默认）：在后台线程中异步执行固化，立即返回 status="started"，
        与 Pipeline 内部行为一致，适合生产调用（固化耗时可能超过 60 秒）。
    background=False：同步等待固化完成后返回详细结果，适合调试/验证。
    """

    session_id: str = Field(..., min_length=1, max_length=128)
    # 是否在后台线程中异步执行（默认 True，避免 HTTP 超时）
    background: bool = True
    # 强制固化：忽略水位线，从头重新处理所有 turns（用于补跑被跳过的 session）
    force_ingest: bool = False
    # 跳过技能抽取，只写入语义记忆（适合批量摄入场景）
    skip_skills: bool = False
    # 强制结束当前 session 的 pending 语义 chunk，适合会话结束或批量导入收尾时使用
    flush_session: bool = True
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
    # "skipped"  — 无有效轮次或无新增轮次
    status: str = "done"
    entities_added: int = 0
    skills_added: int = 0
    facts_added: int = 0
    skipped_reason: str | None = None
    steps: list[dict[str, Any]] = []


class GraphNodeAddRequest(BaseModel):
    id: str = Field(..., min_length=1)
    label: str = "Entity"
    properties: dict[str, Any] = {}


class GraphEdgeAddRequest(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)
    properties: dict[str, Any] = {}




class HealthResponse(BaseModel):
    status: str
    version: str




class PipelineStatusResponse(BaseModel):
    """Pipeline 运行状态。"""

    session_count: int
    graph_node_count: int
    graph_edge_count: int
    skill_count: int
