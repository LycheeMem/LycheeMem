"""
FastAPI 鏈嶅姟鍣ㄣ€?

鎻愪緵 HTTP 鎺ュ彛瀵瑰鏆撮湶 A-Frame Pipeline銆?

绔偣:
- POST /chat/complete          鈥?闈炴祦寮忓畬鏁村璇?
- POST /chat                  鈥?SSE 娴佸紡瀵硅瘽
- GET  /sessions               鈥?浼氳瘽鍒楄〃
- GET  /memory/graph           鈥?鏌ョ湅鐭ヨ瘑鍥捐氨
- GET  /memory/graph/search    鈥?鎼滅储鍥捐氨鑺傜偣
- POST /memory/graph/nodes     鈥?鎵嬪姩娣诲姞鑺傜偣
- POST /memory/graph/edges     鈥?鎵嬪姩娣诲姞杈?
- DELETE /memory/graph/nodes/{node_id} 鈥?鍒犻櫎鑺傜偣
- POST /memory/search          鈥?缁熶竴璁板繂妫€绱?
- GET  /memory/skills          鈥?鏌ョ湅鎶€鑳藉簱
- DELETE /memory/skills/{skill_id} 鈥?鍒犻櫎鎶€鑳?
- GET  /memory/session/{session_id} 鈥?鏌ョ湅浼氳瘽
- DELETE /memory/session/{session_id} 鈥?鍒犻櫎浼氳瘽
- GET  /health                 鈥?鍋ュ悍妫€鏌?
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from a_frame.api.models import HealthResponse
from a_frame.api.routers.chat import router as chat_router
from a_frame.api.routers.memory import router as memory_router
from a_frame.api.routers.pipeline import router as pipeline_router
from a_frame.api.routers.session import router as session_router

logger = logging.getLogger("a_frame.api")

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# App factory
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


def create_app(pipeline=None) -> FastAPI:
    """鍒涘缓 FastAPI 搴旂敤銆?

    Args:
        pipeline: AFramePipeline 瀹炰緥銆備紶 None 鏃跺彲鐢ㄤ簬娴嬭瘯锛堥渶鍚庣画璧嬪€?app.state.pipeline锛夈€?
    """
    app = FastAPI(
        title="A-Frame Cognitive Memory API",
        version="0.1.0",
        description="Training-free Agentic Cognitive Memory Framework",
    )

    if pipeline is not None:
        app.state.pipeline = pipeline

    # 鈹€鈹€ CORS 鈹€鈹€
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 鐢熶骇鐜寤鸿鏀逛负鍏蜂綋鍩熷悕
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 鈹€鈹€ Demo UI (optional) 鈹€鈹€
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

    # 鈹€鈹€ 涓棿浠讹細trace_id 娉ㄥ叆 鈹€鈹€

    @app.middleware("http")
    async def trace_id_middleware(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-ID", uuid.uuid4().hex[:16])
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response

    # 鈹€鈹€ Health 鈹€鈹€

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", version="0.1.0")

    # 鈹€鈹€ Routers 鈹€鈹€
    app.include_router(chat_router)
    app.include_router(session_router)
    app.include_router(memory_router)
    app.include_router(pipeline_router)

    return app

