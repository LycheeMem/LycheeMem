"""Pipeline 状态与固化端点。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from a_frame.api.dependencies import get_pipeline
from a_frame.api.models import DeleteResponse, PipelineStatusResponse

router = APIRouter()


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def pipeline_status(pipeline=Depends(get_pipeline)):
    """返回 Pipeline 当前各组件状态统计。"""
    gs = pipeline.search_coordinator.graph_store
    ss = pipeline.search_coordinator.skill_store
    ws = pipeline.wm_manager.session_store

    node_count = len(gs.get_all())
    edge_count = len(gs.get_all_edges()) if hasattr(gs, "get_all_edges") else 0
    skill_count = len(ss.get_all())
    session_count = len(ws.list_sessions())

    return PipelineStatusResponse(
        session_count=session_count,
        graph_node_count=node_count,
        graph_edge_count=edge_count,
        skill_count=skill_count,
    )


@router.get("/pipeline/last-consolidation")
async def last_consolidation(pipeline=Depends(get_pipeline)):
    """返回最近一次固化的结果（供前端轮询）。"""
    result = getattr(pipeline, "_last_consolidation", None)
    if result is None:
        return {"status": "pending"}
    return {"status": "done", **result}


@router.post("/memory/consolidate/{session_id}", response_model=DeleteResponse)
async def trigger_consolidation(session_id: str, pipeline=Depends(get_pipeline)):
    """手动触发指定会话的固化（实体→图谱，技能→技能库）。"""
    result = pipeline.consolidate(session_id)
    entities = result.get("entities_added", 0)
    skills = result.get("skills_added", 0)
    return DeleteResponse(
        message=f"Consolidation done: {entities} entities, {skills} skills extracted."
    )
