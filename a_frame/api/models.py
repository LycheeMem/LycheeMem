"""API 请求/响应数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ─── Chat ───


class ChatRequest(BaseModel):
    """对话请求。"""
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=100_000)


class ChatResponse(BaseModel):
    """对话响应。"""
    session_id: str
    response: str
    route: dict[str, Any] | None = None
    memories_retrieved: int = 0
    wm_token_usage: int = 0


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


class SkillItem(BaseModel):
    id: str
    intent: str = ""
    tool_chain: list[str] = []
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
    include_sensory: bool = True


class MemorySearchResponse(BaseModel):
    query: str
    graph_results: list[dict[str, Any]]
    skill_results: list[dict[str, Any]]
    sensory_results: list[dict[str, Any]]
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


# ─── Sensory ───

class SensoryResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    max_size: int


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
    sensory_buffer_size: int
