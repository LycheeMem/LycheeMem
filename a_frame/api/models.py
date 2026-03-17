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
    source: str = ""
    index: int = 0
    relevance: float = 0.0
    summary: str = ""


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

class SessionResponse(BaseModel):
    session_id: str
    turns: list[dict[str, str]]
    turn_count: int
    summaries: list[dict[str, Any]] = []


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
