"""视觉记忆 API 路由。

提供视觉记忆的查询、检索、统计、删除等端点。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_pipeline

logger = logging.getLogger("src.api.visual")

router = APIRouter(prefix="/visual", tags=["visual-memory"])


# ── Pydantic 模型 ──


class VisualMemoryItem(BaseModel):
    """视觉记忆列表项。"""

    record_id: str
    session_id: str
    timestamp: str
    caption: str
    image_path: str
    image_hash: str
    scene_type: str
    importance_score: float
    retrieval_count: int
    expired: bool
    entities: list[dict[str, Any]] = []


class VisualMemoryDetail(BaseModel):
    """视觉记忆详情。"""

    record_id: str
    session_id: str
    timestamp: str
    caption: str
    image_path: str
    image_url: str
    image_hash: str
    image_mime_type: str
    image_size: int
    entities: list[dict[str, Any]]
    scene_type: str
    structured_data: dict[str, Any]
    importance_score: float
    confidence: float
    source_role: str
    retrieval_count: int
    related_memory_record_ids: list[str]
    expired: bool


class VisualMemoryListResponse(BaseModel):
    """视觉记忆列表响应。"""

    memories: list[VisualMemoryItem]
    total: int


class VisualSearchRequest(BaseModel):
    """视觉记忆检索请求。"""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    session_id: Optional[str] = None


class VisualSearchResponse(BaseModel):
    """视觉记忆检索响应。"""

    query: str
    results: list[dict[str, Any]]
    total: int


class VisualStatsResponse(BaseModel):
    """视觉记忆统计响应。"""

    total: int
    active: int
    expired: int
    by_scene_type: dict[str, int]


# ── 端点 ──


@router.get("/memories", response_model=VisualMemoryListResponse)
async def list_visual_memories(
    session_id: Optional[str] = Query(None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_expired: bool = Query(default=False),
    pipeline=Depends(get_pipeline),
):
    """列出视觉记忆。"""
    from src.memory.visual.visual_store import VisualStore

    visual_store: VisualStore = pipeline.visual_store

    records = visual_store.list_memories(
        session_id=session_id,
        limit=limit,
        offset=offset,
        include_expired=include_expired,
    )

    items = [
        VisualMemoryItem(
            record_id=r.record_id,
            session_id=r.session_id,
            timestamp=r.timestamp,
            caption=r.caption[:200],
            image_path=r.image_path,
            image_hash=r.image_hash,
            scene_type=r.scene_type,
            importance_score=r.importance_score,
            retrieval_count=r.retrieval_count,
            expired=r.expired,
            entities=r.entities,
        )
        for r in records
    ]

    # 获取总数
    stats = visual_store.get_stats()
    total = stats["active"] if not include_expired else stats["total"]

    return VisualMemoryListResponse(memories=items, total=total)


@router.get("/memories/{record_id}", response_model=VisualMemoryDetail)
async def get_visual_memory_detail(record_id: str, pipeline=Depends(get_pipeline)):
    """获取单个视觉记忆详情。"""
    from src.memory.visual.visual_store import VisualStore

    visual_store: VisualStore = pipeline.visual_store
    record = visual_store.get_by_id(record_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Visual memory not found")

    return VisualMemoryDetail(
        record_id=record.record_id,
        session_id=record.session_id,
        timestamp=record.timestamp,
        caption=record.caption,
        image_path=record.image_path,
        image_url=record.image_url,
        image_hash=record.image_hash,
        image_mime_type=record.image_mime_type,
        image_size=record.image_size,
        entities=record.entities,
        scene_type=record.scene_type,
        structured_data=record.structured_data,
        importance_score=record.importance_score,
        confidence=record.confidence,
        source_role=record.source_role,
        retrieval_count=record.retrieval_count,
        related_memory_record_ids=record.related_memory_record_ids,
        expired=record.expired,
    )


@router.get("/memories/{record_id}/image")
async def get_visual_memory_image(record_id: str, pipeline=Depends(get_pipeline)):
    """获取视觉记忆对应的图片文件。"""
    from src.memory.visual.visual_store import VisualStore

    visual_store: VisualStore = pipeline.visual_store
    record = visual_store.get_by_id(record_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Visual memory not found")

    # 构建完整路径
    base_path = Path(visual_store.image_storage_path)
    full_path = base_path / record.image_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    ext = full_path.suffix.lower()
    media_type = media_type_map.get(ext, "image/jpeg")

    return FileResponse(str(full_path), media_type=media_type)


@router.post("/search", response_model=VisualSearchResponse)
async def search_visual_memories(
    req: VisualSearchRequest, pipeline=Depends(get_pipeline)
):
    """检索视觉记忆。"""
    from src.memory.visual.visual_store import VisualStore
    from src.memory.visual.visual_retriever import VisualRetriever

    visual_store: VisualStore = pipeline.visual_store
    retriever = VisualRetriever(visual_store)

    results = retriever.retrieve_by_text(
        query=req.query,
        top_k=req.top_k,
        session_id=req.session_id,
    )

    return VisualSearchResponse(
        query=req.query,
        results=results,
        total=len(results),
    )


@router.get("/stats", response_model=VisualStatsResponse)
async def get_visual_memory_stats(pipeline=Depends(get_pipeline)):
    """获取视觉记忆统计信息。"""
    from src.memory.visual.visual_store import VisualStore

    visual_store: VisualStore = pipeline.visual_store
    stats = visual_store.get_stats()

    return VisualStatsResponse(
        total=stats["total"],
        active=stats["active"],
        expired=stats["expired"],
        by_scene_type=stats.get("by_scene_type", {}),
    )


@router.delete("/memories/{record_id}")
async def delete_visual_memory(record_id: str, pipeline=Depends(get_pipeline)):
    """删除视觉记忆（软删除/过期标记）。"""
    from src.memory.visual.visual_store import VisualStore

    visual_store: VisualStore = pipeline.visual_store
    record = visual_store.get_by_id(record_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Visual memory not found")

    visual_store.mark_expired(record_id)

    return {"status": "deleted", "record_id": record_id}
