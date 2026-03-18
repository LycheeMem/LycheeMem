"""FastAPI 依赖项。"""

from __future__ import annotations

from fastapi import HTTPException, Request


def get_pipeline(request: Request):
    """FastAPI 依赖：从 app.state 获取已初始化的 LycheePipeline 实例。"""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline
