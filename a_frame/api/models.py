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


# ─── Health ───

class HealthResponse(BaseModel):
    status: str
    version: str
