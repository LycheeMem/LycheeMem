"""会话端点。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_optional_user, get_pipeline
from src.api.models import (
    DeleteResponse,
    SessionListResponse,
    SessionResponse,
    SessionSummary,
    SessionUpdateRequest,
)

router = APIRouter()


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """列出当前用户的会话，按最新活动倒序，支持分页。"""
    session_store = pipeline.wm_manager.session_store
    user_id = user.user_id if user else ""
    raw = session_store.list_sessions(offset=offset, limit=limit, user_id=user_id)
    sessions = [SessionSummary(**s) for s in raw]
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/memory/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    session_store = pipeline.wm_manager.session_store
    compressor = pipeline.wm_manager.compressor
    user_id = user.user_id if user else ""
    log = session_store.get_or_create(session_id, user_id=user_id)
    if user_id and getattr(log, "user_id", "") and log.user_id != user_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="No access to this session.")
    wm_max_tokens = compressor.max_tokens
    # 计算当前工作记忆实际 token 占用（摘要 + 近期轮次）
    rendered = compressor.render_context(log.turns, log.summaries)
    wm_current_tokens = compressor.count_tokens(rendered)
    return SessionResponse(
        session_id=session_id,
        turns=log.turns,
        turn_count=len(log.turns),
        summaries=log.summaries,
        wm_max_tokens=wm_max_tokens,
        wm_current_tokens=wm_current_tokens,
    )


@router.delete("/memory/session/{session_id}", response_model=DeleteResponse)
async def delete_session(
    session_id: str,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    session_store = pipeline.wm_manager.session_store
    user_id = user.user_id if user else ""
    log = session_store.get_or_create(session_id, user_id=user_id)
    if user_id and getattr(log, "user_id", "") and log.user_id != user_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="No access to this session.")
    session_store.delete_session(session_id)
    return DeleteResponse(message=f"Session '{session_id}' deleted.")


@router.patch("/memory/session/{session_id}/meta", response_model=DeleteResponse)
async def update_session_meta(
    session_id: str,
    req: SessionUpdateRequest,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """更新会话元数据（主题、标签）。"""
    session_store = pipeline.wm_manager.session_store
    user_id = user.user_id if user else ""
    log = session_store.get_or_create(session_id, user_id=user_id)
    if user_id and getattr(log, "user_id", "") and log.user_id != user_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="No access to this session.")
    if hasattr(session_store, "update_session_meta"):
        session_store.update_session_meta(session_id, topic=req.topic, tags=req.tags)
    return DeleteResponse(message=f"Session '{session_id}' meta updated.")
