"""FastAPI 应用入口。"""

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
from src.api.routers.visual import router as visual_router
from src.mcp.server import register_mcp_routes

logger = logging.getLogger("src.api")

def create_app(pipeline=None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        pipeline: LycheePipeline 实例。传 None 时可用于测试（需后续赋值 app.state.pipeline）。
    """
    app = FastAPI(
        title="LycheeMemAPI",
        version="0.1.0",
    )

    if pipeline is not None:
        app.state.pipeline = pipeline

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境建议改为具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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


    @app.middleware("http")
    async def trace_id_middleware(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-ID", uuid.uuid4().hex[:16])
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response


    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", version="0.1.0")

    app.include_router(chat_router)
    app.include_router(session_router)
    app.include_router(memory_router)
    app.include_router(pipeline_router)
    app.include_router(visual_router)
    if pipeline is not None:
        register_mcp_routes(app, app.state.pipeline)

    return app
