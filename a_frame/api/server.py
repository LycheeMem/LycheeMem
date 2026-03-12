"""
FastAPI 服务器。

提供 HTTP 接口对外暴露 A-Frame Pipeline。

端点:
- POST /chat/complete — 非流式完整对话
- POST /chat         — SSE 流式对话
- GET  /memory/graph — 查看知识图谱
- GET  /memory/skills — 查看技能库
- GET  /memory/session/{session_id} — 查看会话
- DELETE /memory/session/{session_id} — 删除会话
- GET  /health — 健康检查
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from a_frame.api.models import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    GraphResponse,
    HealthResponse,
    SessionResponse,
    SkillsResponse,
)

logger = logging.getLogger("a_frame.api")

# ──────────────────────────────────────
# App factory
# ──────────────────────────────────────


def create_app(pipeline=None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        pipeline: AFramePipeline 实例。传 None 时可用于测试（需后续赋值 app.state.pipeline）。
    """
    app = FastAPI(
        title="A-Frame Cognitive Memory API",
        version="0.1.0",
        description="Training-free Agentic Cognitive Memory Framework",
    )

    if pipeline is not None:
        app.state.pipeline = pipeline

    # ── 中间件：trace_id 注入 ──

    @app.middleware("http")
    async def trace_id_middleware(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-ID", uuid.uuid4().hex[:16])
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response

    # ── Health ──

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", version="0.1.0")

    # ── Chat ──

    @app.post("/chat/complete", response_model=ChatResponse)
    async def chat_complete(req: ChatRequest):
        pipeline = _get_pipeline(app)
        result = pipeline.run(user_query=req.message, session_id=req.session_id)
        return _build_chat_response(req.session_id, result)

    @app.post("/chat")
    async def chat_stream(req: ChatRequest):
        """SSE 流式对话。

        流式 chunk 格式 (Server-Sent Events):
          data: {"type": "status", "content": "routing..."}
          data: {"type": "status", "content": "searching..."}
          data: {"type": "answer", "content": "最终回答文本"}
          data: {"type": "done", "session_id": "...", "memories_retrieved": 0}
        """
        pipeline = _get_pipeline(app)

        async def event_stream():
            yield _sse({"type": "status", "content": "processing"})

            result = pipeline.run(user_query=req.message, session_id=req.session_id)

            route = result.get("route", {})
            if route.get("need_graph") or route.get("need_skills") or route.get("need_sensory"):
                yield _sse({"type": "status", "content": "retrieved"})

            yield _sse({
                "type": "answer",
                "content": result.get("final_response", ""),
            })

            memories = (
                len(result.get("retrieved_graph_memories", []))
                + len(result.get("retrieved_skills", []))
                + len(result.get("retrieved_sensory", []))
            )
            yield _sse({
                "type": "done",
                "session_id": req.session_id,
                "memories_retrieved": memories,
                "wm_token_usage": result.get("wm_token_usage", 0),
            })

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ── Memory: Graph ──

    @app.get("/memory/graph", response_model=GraphResponse)
    async def get_graph():
        pipeline = _get_pipeline(app)
        graph_store = pipeline.search_coordinator.graph_store
        nodes = graph_store.get_all()
        # 兼容 NetworkX (内存) 和 Neo4j (持久化) 两种后端
        if hasattr(graph_store, "get_all_edges"):
            edges = graph_store.get_all_edges()
        else:
            edges = [
                {"source": u, "target": v, **d}
                for u, v, d in graph_store.graph.edges(data=True)
            ]
        return GraphResponse(nodes=nodes, edges=edges)

    # ── Memory: Skills ──

    @app.get("/memory/skills", response_model=SkillsResponse)
    async def get_skills():
        pipeline = _get_pipeline(app)
        skill_store = pipeline.search_coordinator.skill_store
        skills = skill_store.get_all()
        return SkillsResponse(skills=skills, total=len(skills))

    # ── Memory: Session ──

    @app.get("/memory/session/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str):
        pipeline = _get_pipeline(app)
        session_store = pipeline.wm_manager.session_store
        log = session_store.get_or_create(session_id)
        return SessionResponse(
            session_id=session_id,
            turns=log.turns,
            turn_count=len(log.turns),
            summaries=log.summaries,
        )

    @app.delete("/memory/session/{session_id}", response_model=DeleteResponse)
    async def delete_session(session_id: str):
        pipeline = _get_pipeline(app)
        session_store = pipeline.wm_manager.session_store
        session_store.delete_session(session_id)
        return DeleteResponse(message=f"Session '{session_id}' deleted.")

    return app


# ──────────────────────────────────────
# Helpers
# ──────────────────────────────────────


def _get_pipeline(app: FastAPI):
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


def _build_chat_response(session_id: str, result: dict[str, Any]) -> ChatResponse:
    memories = (
        len(result.get("retrieved_graph_memories", []))
        + len(result.get("retrieved_skills", []))
        + len(result.get("retrieved_sensory", []))
    )
    return ChatResponse(
        session_id=session_id,
        response=result.get("final_response", ""),
        route=result.get("route"),
        memories_retrieved=memories,
        wm_token_usage=result.get("wm_token_usage", 0),
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
