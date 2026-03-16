"""
FastAPI 服务器。

提供 HTTP 接口对外暴露 A-Frame Pipeline。

端点:
- POST /chat/complete          — 非流式完整对话
- POST /chat                  — SSE 流式对话
- GET  /sessions               — 会话列表
- GET  /memory/graph           — 查看知识图谱
- GET  /memory/graph/search    — 搜索图谱节点
- POST /memory/graph/nodes     — 手动添加节点
- POST /memory/graph/edges     — 手动添加边
- DELETE /memory/graph/nodes/{node_id} — 删除节点
- POST /memory/search          — 统一记忆检索
- GET  /memory/skills          — 查看技能库
- DELETE /memory/skills/{skill_id} — 删除技能
- GET  /memory/sensory         — 感觉缓冲区
- DELETE /memory/sensory       — 清空感觉缓冲区
- GET  /memory/session/{session_id} — 查看会话
- DELETE /memory/session/{session_id} — 删除会话
- GET  /health                 — 健康检查
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from a_frame.api.models import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    GraphEdgeAddRequest,
    GraphNodeAddRequest,
    GraphResponse,
    HealthResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    PipelineStatusResponse,
    SensoryResponse,
    SessionListResponse,
    SessionResponse,
    SessionSummary,
    SessionUpdateRequest,
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

    # ── CORS ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],           # 生产环境建议改为具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Demo UI (optional) ──
    demo_dir = Path(__file__).resolve().parents[1] / "demo"
    if demo_dir.exists():
        app.mount(
            "/demo-static",
            StaticFiles(directory=str(demo_dir), html=False),
            name="demo-static",
        )

        @app.get("/demo")
        async def demo_index():
            return FileResponse(str(demo_dir / "index.html"))

        @app.get("/demo/")
        async def demo_index_slash():
            return RedirectResponse(url="/demo")

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

    # ── Sessions ──

    @app.get("/sessions", response_model=SessionListResponse)
    async def list_sessions(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        """列出所有会话，按最新活动倒序，支持分页。"""
        pipeline = _get_pipeline(app)
        session_store = pipeline.wm_manager.session_store
        raw = session_store.list_sessions(offset=offset, limit=limit)
        sessions = [SessionSummary(**s) for s in raw]
        return SessionListResponse(sessions=sessions, total=len(sessions))

    # ── Memory: Search ──

    @app.post("/memory/search", response_model=MemorySearchResponse)
    async def memory_search(req: MemorySearchRequest):
        """统一记忆检索：同时查询图谱、技能库和感觉缓冲。"""
        pipeline = _get_pipeline(app)
        sc = pipeline.search_coordinator

        graph_results: list[dict[str, Any]] = []
        if req.include_graph:
            graph_results = sc.graph_store.search(req.query, top_k=req.top_k)

        skill_results: list[dict[str, Any]] = []
        if req.include_skills:
            q_emb = sc.embedder.embed_query(req.query)
            skill_results = sc.skill_store.search(
                req.query, top_k=req.top_k, query_embedding=q_emb
            )

        sensory_results: list[dict[str, Any]] = []
        if req.include_sensory:
            items = sc.sensory_buffer.get_recent(req.top_k)
            sensory_results = [
                {"content": i.content, "modality": i.modality, "timestamp": i.timestamp}
                for i in items
            ]

        total = len(graph_results) + len(skill_results) + len(sensory_results)
        return MemorySearchResponse(
            query=req.query,
            graph_results=graph_results,
            skill_results=skill_results,
            sensory_results=sensory_results,
            total=total,
        )

    # ── Memory: Graph ──

    @app.get("/memory/graph", response_model=GraphResponse)
    async def get_graph():
        pipeline = _get_pipeline(app)
        graph_store = pipeline.search_coordinator.graph_store
        nodes = graph_store.get_all()
        if hasattr(graph_store, "get_all_edges"):
            edges = graph_store.get_all_edges()
        else:
            edges = [
                {"source": u, "target": v, **d}
                for u, v, d in graph_store.graph.edges(data=True)
            ]
        return GraphResponse(nodes=nodes, edges=edges)

    @app.get("/memory/graph/search", response_model=GraphResponse)
    async def search_graph(
        q: str = Query(..., min_length=1, description="搜索关键词"),
        top_k: int = Query(default=10, ge=1, le=100),
    ):
        """按关键词搜索图谱节点，返回处国子图。"""
        pipeline = _get_pipeline(app)
        graph_store = pipeline.search_coordinator.graph_store
        matched_nodes = graph_store.search(q, top_k=top_k)
        node_ids = {n["node_id"] for n in matched_nodes}
        if hasattr(graph_store, "get_all_edges"):
            all_edges = graph_store.get_all_edges()
            edges = [e for e in all_edges if e["source"] in node_ids or e["target"] in node_ids]
        else:
            edges = [
                {"source": u, "target": v, **d}
                for u, v, d in graph_store.graph.edges(data=True)
                if u in node_ids or v in node_ids
            ]
        return GraphResponse(nodes=matched_nodes, edges=edges)

    @app.post("/memory/graph/nodes", response_model=DeleteResponse)
    async def add_graph_node(req: GraphNodeAddRequest):
        """手动添加一个实体节点到知识图谱。"""
        pipeline = _get_pipeline(app)
        graph_store = pipeline.search_coordinator.graph_store
        graph_store.add_node(req.id, label=req.label, properties=req.properties)
        return DeleteResponse(message=f"Node '{req.id}' added.")

    @app.post("/memory/graph/edges", response_model=DeleteResponse)
    async def add_graph_edge(req: GraphEdgeAddRequest):
        """手动添加一条关系边到知识图谱。"""
        pipeline = _get_pipeline(app)
        graph_store = pipeline.search_coordinator.graph_store
        graph_store.add_edge(req.source, req.target, relation=req.relation, properties=req.properties)
        return DeleteResponse(message=f"Edge '{req.source}' -{req.relation}-> '{req.target}' added.")

    @app.delete("/memory/graph/nodes/{node_id}", response_model=DeleteResponse)
    async def delete_graph_node(node_id: str):
        """删除图谱中的一个节点（及其所有关联边）。"""
        pipeline = _get_pipeline(app)
        graph_store = pipeline.search_coordinator.graph_store
        graph_store.delete([node_id])
        return DeleteResponse(message=f"Node '{node_id}' deleted.")

    # ── Memory: Skills ──

    @app.get("/memory/skills", response_model=SkillsResponse)
    async def get_skills():
        pipeline = _get_pipeline(app)
        skill_store = pipeline.search_coordinator.skill_store
        skills = skill_store.get_all()
        return SkillsResponse(skills=skills, total=len(skills))

    @app.delete("/memory/skills/{skill_id}", response_model=DeleteResponse)
    async def delete_skill(skill_id: str):
        """删除指定技能条目。"""
        pipeline = _get_pipeline(app)
        skill_store = pipeline.search_coordinator.skill_store
        skill_store.delete([skill_id])
        return DeleteResponse(message=f"Skill '{skill_id}' deleted.")

    # ── Memory: Sensory ──

    @app.get("/memory/sensory", response_model=SensoryResponse)
    async def get_sensory():
        """查看当前感觉记忆缓冲区内容。"""
        pipeline = _get_pipeline(app)
        buf = pipeline.search_coordinator.sensory_buffer
        items = [
            {"content": i.content, "modality": i.modality, "timestamp": i.timestamp}
            for i in buf.get_recent()
        ]
        return SensoryResponse(items=items, total=len(items), max_size=buf.max_size)

    @app.delete("/memory/sensory", response_model=DeleteResponse)
    async def clear_sensory():
        """清空感觉记忆缓冲区。"""
        pipeline = _get_pipeline(app)
        pipeline.search_coordinator.sensory_buffer.clear()
        return DeleteResponse(message="Sensory buffer cleared.")

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

    # ── Session Metadata ──

    @app.patch("/memory/session/{session_id}/meta", response_model=DeleteResponse)
    async def update_session_meta(session_id: str, req: SessionUpdateRequest):
        """更新会话元数据（主题、标签）。"""
        pipeline = _get_pipeline(app)
        session_store = pipeline.wm_manager.session_store
        if hasattr(session_store, "update_session_meta"):
            session_store.update_session_meta(session_id, topic=req.topic, tags=req.tags)
        return DeleteResponse(message=f"Session '{session_id}' meta updated.")

    # ── Consolidation Trigger ──

    @app.post("/memory/consolidate/{session_id}", response_model=DeleteResponse)
    async def trigger_consolidation(session_id: str):
        """手动触发指定会话的固化（实体→图谱，技能→技能库）。"""
        pipeline = _get_pipeline(app)
        result = pipeline.consolidate(session_id)
        entities = result.get("entities_added", 0)
        skills = result.get("skills_added", 0)
        return DeleteResponse(
            message=f"Consolidation done: {entities} entities, {skills} skills extracted."
        )

    # ── Graph: Time & Relation Search ──

    @app.get("/memory/graph/by-relation")
    async def search_graph_by_relation(
        relation: str = Query(..., min_length=1),
        top_k: int = Query(default=10, ge=1, le=100),
    ):
        """按关系类型检索图谱边。"""
        pipeline = _get_pipeline(app)
        graph_store = pipeline.search_coordinator.graph_store
        if hasattr(graph_store, "search_by_relation"):
            edges = graph_store.search_by_relation(relation, top_k=top_k)
        else:
            edges = []
        return {"edges": edges, "total": len(edges)}

    @app.get("/memory/graph/by-time")
    async def search_graph_by_time(
        since: str | None = Query(default=None),
        until: str | None = Query(default=None),
        top_k: int = Query(default=10, ge=1, le=100),
    ):
        """按时间范围检索图谱边。"""
        pipeline = _get_pipeline(app)
        graph_store = pipeline.search_coordinator.graph_store
        if hasattr(graph_store, "search_by_time"):
            edges = graph_store.search_by_time(since=since, until=until, top_k=top_k)
        else:
            edges = []
        return {"edges": edges, "total": len(edges)}

    # ── Pipeline Status ──

    @app.get("/pipeline/status", response_model=PipelineStatusResponse)
    async def pipeline_status():
        """返回 Pipeline 当前各组件状态统计。"""
        pipeline = _get_pipeline(app)
        gs = pipeline.search_coordinator.graph_store
        ss = pipeline.search_coordinator.skill_store
        sb = pipeline.search_coordinator.sensory_buffer
        ws = pipeline.wm_manager.session_store

        node_count = len(gs.get_all())
        edge_count = len(gs.get_all_edges()) if hasattr(gs, "get_all_edges") else 0
        skill_count = len(ss.get_all())
        sensory_size = len(sb.get_recent())
        session_count = len(ws.list_sessions())

        return PipelineStatusResponse(
            session_count=session_count,
            graph_node_count=node_count,
            graph_edge_count=edge_count,
            skill_count=skill_count,
            sensory_buffer_size=sensory_size,
        )

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
