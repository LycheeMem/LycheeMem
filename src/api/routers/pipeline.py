"""Pipeline 状态与固化端点。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline
from src.api.models import DeleteResponse, PipelineStatusResponse

router = APIRouter()


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def pipeline_status(pipeline=Depends(get_pipeline)):
    """返回 Pipeline 当前各组件状态统计。"""
    ss = pipeline.search_coordinator.skill_store
    ws = pipeline.wm_manager.session_store

    sc = pipeline.search_coordinator
    node_count = 0
    edge_count = 0

    # Compact 后端：从 sqlite_store 计数
    if getattr(sc, "semantic_engine", None) is not None:
        try:
            node_count = sc.semantic_engine._sqlite.count_units()
        except Exception:
            node_count = 0
        # synthesized units 作为"边"的近似
        try:
            debug = sc.semantic_engine.export_debug(user_id="")
            edge_count = len(debug.get("synthesized", []))
        except Exception:
            edge_count = 0
    else:
        # Graphiti 后端
        graphiti = getattr(getattr(pipeline, "consolidator", None), "graphiti_engine", None)
        store = getattr(graphiti, "store", None)
        if store is not None:
            try:
                all_entities = store.list_all_entity_ids() if hasattr(store, "list_all_entity_ids") else []
                node_count = len(all_entities)
            except Exception:
                node_count = 0
            if node_count > 0 and hasattr(store, "export_semantic_subgraph"):
                try:
                    subgraph = store.export_semantic_subgraph(
                        entity_ids=all_entities[:200], edge_limit=5000
                    )
                    edge_count = len(subgraph.get("edges", []))
                except Exception:
                    edge_count = 0

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
    status = "skipped" if result.get("skipped_reason") else "done"
    return {"status": status, **result}


@router.post("/memory/consolidate/{session_id}", response_model=DeleteResponse)
async def trigger_consolidation(session_id: str, pipeline=Depends(get_pipeline)):
    """手动触发指定会话的固化（实体→图谱，技能→技能库）。"""
    result = pipeline.consolidate(session_id)
    entities = result.get("entities_added", 0)
    skills = result.get("skills_added", 0)
    return DeleteResponse(
        message=f"Consolidation done: {entities} entities, {skills} skills extracted."
    )
