"""Pipeline 状态与固化端点。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_pipeline
from src.api.models import DeleteResponse, PipelineStatusResponse
from src.evolve.prompt_registry import select_prompt_versions

router = APIRouter()

_MEMORY_CONSOLIDATE_API_PROMPTS: frozenset[str] = frozenset({
    "novelty_check",
    "compact_encoding",
    "consolidation",
})


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
            node_count = (
                sc.semantic_engine._sqlite.count_records()
                + sc.semantic_engine._sqlite.count_composites()
            )
        except Exception:
            node_count = 0
        # composite records 作为"边"的近似
        try:
            debug = sc.semantic_engine.export_debug()
            edge_count = len(debug.get("composites", []))
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
async def last_consolidation(
    session_id: str | None = Query(default=None, min_length=1, max_length=128),
    pipeline=Depends(get_pipeline),
):
    """返回最近一次固化的结果（支持按会话轮询）。"""
    result = pipeline.get_last_consolidation(session_id=session_id)
    if result is None:
        return {"status": "pending", "session_id": session_id}
    return result


@router.post("/memory/consolidate/{session_id}", response_model=DeleteResponse)
async def trigger_consolidation(session_id: str, pipeline=Depends(get_pipeline)):
    """手动触发指定会话的固化（实体→图谱，技能→技能库）。"""
    prompt_versions_snapshot = select_prompt_versions(_MEMORY_CONSOLIDATE_API_PROMPTS)
    result = pipeline.consolidate(session_id)
    recorder = getattr(pipeline, "record_api_usage", None)
    if callable(recorder) and not result.get("skipped_reason"):
        recorder(
            api_name="memory/consolidate/session",
            prompt_versions_used=prompt_versions_snapshot,
        )
    entities = result.get("entities_added", 0)
    skills = result.get("skills_added", 0)
    return DeleteResponse(
        message=f"Consolidation done: {entities} entities, {skills} skills extracted."
    )


# ──────────────────────────────────────
# Self-Evolve (Prompt Self-Optimization)
# ──────────────────────────────────────


@router.get("/evolve/status")
async def evolve_status(pipeline=Depends(get_pipeline)):
    """返回 Self-Evolve 当前运行状态（用于前端面板展示）。"""
    loop = getattr(pipeline, "evolve_loop", None)
    if loop is None:
        return {"enabled": False}
    try:
        status = loop.get_status()
    except Exception as e:
        return {"enabled": True, "error": str(e)}
    return {"enabled": True, "status": status}


@router.get("/evolve/health")
async def evolve_health(
    prompt_name: str | None = Query(default=None, min_length=1, max_length=128),
    pipeline=Depends(get_pipeline),
):
    """返回单个或全部 prompt 的健康报告。"""
    loop = getattr(pipeline, "evolve_loop", None)
    if loop is None:
        return {"enabled": False}
    try:
        if prompt_name:
            report = loop.get_health_report(prompt_name)
            return {"enabled": True, "report": report.__dict__}
        reports = loop.evaluator.evaluate_all()
        return {"enabled": True, "reports": [r.__dict__ for r in reports]}
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@router.post("/evolve/optimize")
async def evolve_optimize(
    prompt_name: str | None = Query(default=None, min_length=1, max_length=128),
    pipeline=Depends(get_pipeline),
):
    """触发一次优化（可指定 prompt）。"""
    loop = getattr(pipeline, "evolve_loop", None)
    if loop is None:
        return {"enabled": False}
    try:
        results = loop.maybe_optimize(force_prompt=prompt_name)
        return {
            "enabled": True,
            "results": [
                {
                    "prompt_name": r.prompt_name,
                    "success": bool(r.success),
                    "message": r.reason,
                    "review_verdict": r.review_verdict,
                    "candidate_version": (r.candidate_version.version if r.candidate_version else None),
                }
                for r in results
            ],
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@router.post("/evolve/rollback")
async def evolve_rollback(
    prompt_name: str = Query(..., min_length=1, max_length=128),
    version: int = Query(..., ge=0, le=1_000_000),
    pipeline=Depends(get_pipeline),
):
    """手动回滚某个 prompt 版本。"""
    loop = getattr(pipeline, "evolve_loop", None)
    if loop is None:
        return {"enabled": False}
    try:
        loop.rollback_candidate(prompt_name, version)
        return {"enabled": True, "ok": True}
    except Exception as e:
        return {"enabled": True, "ok": False, "error": str(e)}


@router.get("/evolve/events")
async def evolve_events(
    limit: int = Query(default=100, ge=1, le=500),
    prompt_name: str | None = Query(default=None, min_length=1, max_length=128),
    event_type: str | None = Query(default=None, min_length=1, max_length=64),
    pipeline=Depends(get_pipeline),
):
    """返回自进化历史事件时间线（用于前端可视化）。"""
    loop = getattr(pipeline, "evolve_loop", None)
    if loop is None:
        return {"enabled": False, "events": []}
    try:
        store = getattr(loop, "_store", None)
        if store is None:
            return {"enabled": True, "events": []}
        events = store.list_events(limit=int(limit), prompt_name=prompt_name, event_type=event_type)
        return {
            "enabled": True,
            "events": [
                {
                    "id": e.event_id,
                    "created_at": e.created_at,
                    "event_type": e.event_type,
                    "prompt_name": e.prompt_name,
                    "from_version": e.from_version,
                    "to_version": e.to_version,
                    "summary": e.summary,
                    "payload": e.payload,
                }
                for e in events
            ],
        }
    except Exception as e:
        return {"enabled": True, "error": str(e), "events": []}
