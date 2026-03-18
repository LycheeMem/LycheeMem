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
    fact_id: str = ""         # 对应 Neo4j Fact 节点的 fact_id
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
    # 从 Neo4j 回溯得到，按 turn_index 升序排列。
    source_episodes: list[dict[str, Any]] = []


class SynthesizerTrace(BaseModel):
    background_context: str = ""
    provenance: list[ProvenanceItem] = []
    skill_reuse_plan: list[dict[str, Any]] = []
    kept_count: int = 0
    dropped_count: int = 0


class ReasonerTrace(BaseModel):
    response_length: int = 0


class ConsolidatorTrace(BaseModel):
    status: str = "pending"
    entities_added: int = 0
    skills_added: int = 0


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
    created_at: str | None = None
    deleted: bool = False


class SessionResponse(BaseModel):
    session_id: str
    turns: list[TurnItem]
    turn_count: int
    summaries: list[dict[str, Any]] = []
    wm_max_tokens: int = 128000


class DeleteResponse(BaseModel):
    message: str


# ─── Sessions List ───


class SessionSummary(BaseModel):
    session_id: str
    turn_count: int
    last_message: str = ""
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
    include_graph: bool = True
    include_skills: bool = True


class MemorySearchResponse(BaseModel):
    query: str
    graph_results: list[dict[str, Any]]
    skill_results: list[dict[str, Any]]
    total: int


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
