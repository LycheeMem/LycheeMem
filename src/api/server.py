"""
FastAPI 服务器。

提供 HTTP 接口对外暴露 LycheeMemOS Pipeline。

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
- GET  /memory/session/{session_id} — 查看会话
- DELETE /memory/session/{session_id} — 删除会话
- GET  /health                 — 健康检查
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.models import HealthResponse
from src.api.routers.chat import router as chat_router
from src.api.routers.memory import router as memory_router
from src.api.routers.pipeline import router as pipeline_router
from src.api.routers.session import router as session_router

logger = logging.getLogger("src.api")

# ──────────────────────────────────────
# App factory
# ──────────────────────────────────────


def create_app(pipeline=None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        pipeline: LycheePipeline 实例。传 None 时可用于测试（需后续赋值 app.state.pipeline）。
    """
    app = FastAPI(
        title="LycheeMemOS Cognitive Memory API",
        version="0.1.0",
        description="Training-free Agentic Cognitive Memory Framework",
    )

    if pipeline is not None:
        app.state.pipeline = pipeline

    # ── CORS ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境建议改为具体域名
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

    # ── Routers ──
    app.include_router(chat_router)
    app.include_router(session_router)
    app.include_router(memory_router)
    app.include_router(pipeline_router)

    return app

