"""记忆端点：图谱、技能、检索。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_optional_user, get_pipeline
from src.api.models import (
    DeleteResponse,
    MemoryAppendTurnRequest,
    MemoryAppendTurnResponse,
    GraphResponse,
    MemoryConsolidateRequest,
    MemoryConsolidateResponse,
    MemoryReasonRequest,
    MemoryReasonResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySmartSearchRequest,
    MemorySmartSearchResponse,
    MemorySynthesizeRequest,
    MemorySynthesizeResponse,
    SkillsResponse,
)

logger = logging.getLogger("src.api")

router = APIRouter()


def run_memory_search(
    pipeline,
    req: MemorySearchRequest,
    *,
    user_id: str = "",
) -> MemorySearchResponse:
    """执行统一记忆检索，返回可直接供 Synthesizer 消费的 richer 结构。"""
    sc = pipeline.search_coordinator

    graph_results: list[dict[str, Any]] = []
    skill_results: list[dict[str, Any]] = []

    if req.include_graph:
        search_result = sc.semantic_engine.search(
            query=req.query,
            top_k=req.top_k,
            user_id=user_id,
        )
        if search_result.context.strip():
            graph_results = [
                {
                    "anchor": {
                        "node_id": "compact_context",
                        "name": "CompactSemanticMemory",
                        "label": "Context",
                        "score": 1.0,
                    },
                    "subgraph": {"nodes": [], "edges": []},
                    "constructed_context": search_result.context,
                    "provenance": search_result.provenance,
                }
            ]

    if req.include_skills:
        skill_results = sc._search_skills(req.query, top_k=req.top_k, user_id=user_id)

    graph_total = 0
    for item in graph_results:
        provenance = item.get("provenance")
        if isinstance(provenance, list) and provenance:
            graph_total += len(provenance)
        else:
            graph_total += 1

    total = graph_total + len(skill_results)
    return MemorySearchResponse(
        query=req.query,
        graph_results=graph_results,
        skill_results=skill_results,
        total=total,
    )


def run_memory_synthesize(
    pipeline,
    req: MemorySynthesizeRequest,
) -> MemorySynthesizeResponse:
    """执行检索结果压缩，供 HTTP Router 与 MCP 共享。"""
    result = pipeline.synthesizer.run(
        user_query=req.user_query,
        retrieved_graph_memories=req.graph_results,
        retrieved_skills=req.skill_results,
    )

    provenance_raw = result.get("provenance", [])
    provenance_flat: list[dict] = []
    for item in provenance_raw:
        if isinstance(item, dict) and isinstance(item.get("items"), list):
            provenance_flat.extend(item["items"])
        elif isinstance(item, dict):
            provenance_flat.append(item)

    input_count = len(req.graph_results) + len(req.skill_results)
    kept_count = len(provenance_flat)
    dropped_count = max(0, input_count - kept_count)

    return MemorySynthesizeResponse(
        background_context=result.get("background_context", ""),
        skill_reuse_plan=result.get("skill_reuse_plan", []),
        provenance=provenance_flat,
        kept_count=kept_count,
        dropped_count=dropped_count,
    )


def run_memory_smart_search(
    pipeline,
    req: MemorySmartSearchRequest,
    *,
    user_id: str = "",
) -> MemorySmartSearchResponse:
    """执行 one-shot 检索；可选自动 synthesize，便于宿主快速试验效果。"""
    search_result = run_memory_search(
        pipeline,
        MemorySearchRequest(
            query=req.query,
            top_k=req.top_k,
            include_graph=req.include_graph,
            include_skills=req.include_skills,
        ),
        user_id=user_id,
    )

    if not req.synthesize:
        return MemorySmartSearchResponse(
            query=search_result.query,
            mode=req.mode,
            graph_results=search_result.graph_results,
            skill_results=search_result.skill_results,
            total=search_result.total,
            synthesized=False,
        )

    if req.mode == "raw":
        return MemorySmartSearchResponse(
            query=search_result.query,
            mode=req.mode,
            graph_results=search_result.graph_results,
            skill_results=search_result.skill_results,
            total=search_result.total,
            synthesized=False,
        )

    synth_result = run_memory_synthesize(
        pipeline,
        MemorySynthesizeRequest(
            user_query=req.query,
            graph_results=search_result.graph_results,
            skill_results=search_result.skill_results,
        ),
    )
    if req.mode == "compact":
        return MemorySmartSearchResponse(
            query=search_result.query,
            mode=req.mode,
            graph_results=[],
            skill_results=[],
            total=search_result.total,
            synthesized=True,
            background_context=synth_result.background_context,
            skill_reuse_plan=synth_result.skill_reuse_plan,
            provenance=synth_result.provenance,
            kept_count=synth_result.kept_count,
            dropped_count=synth_result.dropped_count,
        )

    return MemorySmartSearchResponse(
        query=search_result.query,
        mode=req.mode,
        graph_results=search_result.graph_results,
        skill_results=search_result.skill_results,
        total=search_result.total,
        synthesized=True,
        background_context=synth_result.background_context,
        skill_reuse_plan=synth_result.skill_reuse_plan,
        provenance=synth_result.provenance,
        kept_count=synth_result.kept_count,
        dropped_count=synth_result.dropped_count,
    )


def run_memory_append_turn(
    pipeline,
    req: MemoryAppendTurnRequest,
    *,
    user_id: str = "",
) -> MemoryAppendTurnResponse:
    """向 session store 追加单条宿主对话轮次，供后续 consolidate 使用。"""
    pipeline.wm_manager.session_store.append_turn(
        req.session_id,
        req.role,
        req.content,
        token_count=req.token_count,
        user_id=user_id,
    )
    log = pipeline.wm_manager.session_store.get_or_create(req.session_id, user_id=user_id)
    return MemoryAppendTurnResponse(
        status="appended",
        session_id=req.session_id,
        turn_count=len(log.turns),
    )


def run_memory_consolidate(
    pipeline,
    req: MemoryConsolidateRequest,
    *,
    user_id: str = "",
) -> MemoryConsolidateResponse:
    """执行长期记忆固化，供 HTTP Router 与 MCP 共享。"""
    import threading

    log = pipeline.wm_manager.session_store.get_or_create(
        req.session_id,
        user_id=user_id,
    )
    turns = [t for t in log.turns if not t.get("deleted", False)]
    if not turns:
        return MemoryConsolidateResponse(
            status="skipped",
            skipped_reason="session_empty",
            steps=[{"name": "session_check", "status": "skipped", "detail": "会话无有效对话轮次"}],
        )

    if req.background:
        def _run() -> None:
            try:
                pipeline.consolidator.run(
                    turns=turns,
                    session_id=req.session_id,
                    retrieved_context=req.retrieved_context,
                    user_id=user_id,
                )
            except Exception:
                logger.exception("background consolidation failed session=%s", req.session_id)

        thread = threading.Thread(
            target=_run,
            daemon=True,
            name=f"consolidate-{req.session_id[:8]}",
        )
        thread.start()
        return MemoryConsolidateResponse(status="started")

    result = pipeline.consolidator.run(
        turns=turns,
        session_id=req.session_id,
        retrieved_context=req.retrieved_context,
        user_id=user_id,
    )
    return MemoryConsolidateResponse(
        status="skipped" if result.get("skipped_reason") else "done",
        entities_added=result.get("entities_added", 0),
        skills_added=result.get("skills_added", 0),
        facts_added=result.get("facts_added", 0),
        has_novelty=result.get("has_novelty"),
        skipped_reason=result.get("skipped_reason"),
        steps=result.get("steps", []),
    )


# ── Memory: Search ──


@router.post("/memory/search", response_model=MemorySearchResponse)
async def memory_search(req: MemorySearchRequest, pipeline=Depends(get_pipeline), user=Depends(get_optional_user)):
    """统一记忆检索：同时查询图谱和技能库。"""
    user_id = user.user_id if user else ""
    return run_memory_search(pipeline, req, user_id=user_id)


@router.post("/memory/smart-search", response_model=MemorySmartSearchResponse)
async def memory_smart_search(
    req: MemorySmartSearchRequest,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """实验性 one-shot 检索包装器：search，可选自动 synthesize。"""
    user_id = user.user_id if user else ""
    return run_memory_smart_search(pipeline, req, user_id=user_id)


# ── Memory: Synthesize ──


@router.post("/memory/synthesize", response_model=MemorySynthesizeResponse)
async def memory_synthesize(
    req: MemorySynthesizeRequest,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """对多源检索结果进行 LLM-as-Judge 评分与融合，生成 background_context。

    典型用法：衔接 POST /memory/search 的响应，将 graph_results / skill_results 传入。
    输出的 background_context 和 skill_reuse_plan 可直接传给 POST /memory/reason。
    """
    return run_memory_synthesize(pipeline, req)


# ── Memory: Append Turn ──


@router.post("/memory/append-turn", response_model=MemoryAppendTurnResponse)
async def memory_append_turn(
    req: MemoryAppendTurnRequest,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """追加单条外部宿主对话轮次，为后续 consolidate 提供 transcript bridge。"""
    user_id = user.user_id if user else ""
    return run_memory_append_turn(pipeline, req, user_id=user_id)


# ── Memory: Reason ──


@router.post("/memory/reason", response_model=MemoryReasonResponse)
async def memory_reason(
    req: MemoryReasonRequest,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """基于合成上下文对用户查询进行最终推理，生成 assistant 回答。

    当 append_to_session=True（默认）时：
    - 将用户问题追加到会话（含 token 预算检查与按需压缩），作为下一轮历史
    - 将 assistant 回答追加到会话，供后续 POST /memory/consolidate 使用

    当 append_to_session=False 时：仅读取历史，不写入会话。
    """
    user_id = user.user_id if user else ""
    wm = pipeline.wm_manager

    if req.append_to_session:
        # 追加用户消息到会话（含 token 计数及双阈值压缩）
        wm_result = wm.run(
            session_id=req.session_id,
            user_query=req.user_query,
            user_id=user_id,
        )
        compressed_history = wm_result["compressed_history"]
        wm_token_usage = wm_result["wm_token_usage"]
    else:
        # 只读历史，不写入会话
        log = wm.session_store.get_or_create(req.session_id, user_id=user_id)
        compressed_history = wm.compressor.render_context(log.turns, log.summaries)
        wm_token_usage = wm.compressor.count_tokens(compressed_history)

    result = pipeline.reasoner.run(
        user_query=req.user_query,
        compressed_history=compressed_history,
        background_context=req.background_context,
        skill_reuse_plan=req.skill_reuse_plan,
        retrieved_skills=req.retrieved_skills,
    )

    if req.append_to_session:
        wm.append_assistant_turn(
            req.session_id,
            result["final_response"],
            user_id=user_id,
        )

    return MemoryReasonResponse(
        response=result["final_response"],
        session_id=req.session_id,
        wm_token_usage=wm_token_usage,
    )


# ── Memory: Consolidate ──


@router.post("/memory/consolidate", response_model=MemoryConsolidateResponse)
async def memory_consolidate(
    req: MemoryConsolidateRequest,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """对当前会话进行记忆萃取固化，提取实体/事实写入图谱，提取技能写入技能库。

    retrieved_context 建议传入本轮 /memory/synthesize 的 background_context，
    用于新颖性判断（避免将已有记忆重复固化）。

    background=True（默认）：在后台线程中异步执行，立即返回 status="started"；
        与 Pipeline 内部行为一致，适合生产调用（固化耗时通常超过 60 秒）。
    background=False：同步等待完成后返回详细结果，适合调试/验证。
    """
    user_id = user.user_id if user else ""
    return run_memory_consolidate(pipeline, req, user_id=user_id)


# ── Memory: Graph ──


@router.get("/memory/graph", response_model=GraphResponse)
async def get_graph(pipeline=Depends(get_pipeline), user=Depends(get_optional_user)):
    user_id = user.user_id if user else ""
    sc = pipeline.search_coordinator
    try:
        data = sc.semantic_engine.export_debug(user_id=user_id)
        nodes = []
        edges = []
        for u in data.get("units", []):
            nodes.append({
                "id": u.get("unit_id", ""),
                "name": u.get("normalized_text", "")[:80],
                "label": u.get("memory_type", "unit"),
                "properties": {
                    "semantic_text": u.get("semantic_text", ""),
                    "entities": u.get("entities", []),
                    "confidence": u.get("confidence", 1.0),
                    "created_at": u.get("created_at", ""),
                },
            })
        for s in data.get("synthesized", []):
            nodes.append({
                "id": s.get("synth_id", ""),
                "name": s.get("normalized_text", "")[:80],
                "label": s.get("memory_type", "synth"),
                "properties": {
                    "semantic_text": s.get("semantic_text", ""),
                    "source_unit_ids": s.get("source_unit_ids", []),
                },
            })
            for src_id in s.get("source_unit_ids", []):
                edges.append({
                    "source": s.get("synth_id", ""),
                    "target": src_id,
                    "relation": "synthesized_from",
                })
        return GraphResponse(nodes=nodes, edges=edges)
    except Exception as exc:
        logger.exception("export_debug failed")
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}")


@router.get("/memory/graph/search", response_model=GraphResponse)
async def search_graph(
    q: str = Query(..., min_length=1, description="搜索关键词"),
    top_k: int = Query(default=10, ge=1, le=100),
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """按关键词搜索记忆条目，返回匹配结果。"""
    user_id = user.user_id if user else ""
    sc = pipeline.search_coordinator
    try:
        results = sc.semantic_engine._sqlite.fulltext_search(q, user_id=user_id, limit=top_k)
        nodes = []
        for r in results:
            nodes.append({
                "id": r.get("unit_id", ""),
                "name": r.get("normalized_text", "")[:80],
                "label": r.get("memory_type", "unit"),
                "properties": {
                    "semantic_text": r.get("semantic_text", ""),
                    "entities": r.get("entities", []),
                    "is_anchor": True,
                },
            })
        return GraphResponse(nodes=nodes, edges=[])
    except Exception:
        logger.exception("Compact graph search failed")
        raise HTTPException(status_code=500, detail="Graph search failed")


@router.delete("/memory/graph/clear", response_model=DeleteResponse)
async def clear_all_graph(
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """清空当前用户的所有语义记忆。"""
    user_id = user.user_id if user else ""
    sc = pipeline.search_coordinator
    try:
        result = sc.semantic_engine.delete_all_for_user(user_id=user_id)
        return DeleteResponse(
            message=(
                f"Compact memory cleared "
                f"(units_deleted={result.get('units_deleted', 0)}, "
                f"synth_deleted={result.get('synth_deleted', 0)})."
            )
        )
    except Exception as exc:
        logger.exception("delete_all_for_user failed")
        raise HTTPException(status_code=500, detail=f"Clear failed: {exc}")


# ── Memory: Skills ──


@router.get("/memory/skills", response_model=SkillsResponse)
async def get_skills(pipeline=Depends(get_pipeline), user=Depends(get_optional_user)):
    skill_store = pipeline.search_coordinator.skill_store
    user_id = user.user_id if user else ""
    skills = skill_store.get_all(user_id=user_id)
    return SkillsResponse(skills=skills, total=len(skills))


# NOTE: 固定路径 /memory/skills/clear 必须在参数化路径 /memory/skills/{skill_id} 前定义，
# 否则 FastAPI 会将 "clear" 当作 skill_id 参数处理。
@router.delete("/memory/skills/clear", response_model=DeleteResponse)
async def clear_all_skills(
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """清空当前用户的所有技能记忆。"""
    skill_store = pipeline.search_coordinator.skill_store
    user_id = user.user_id if user else ""
    if hasattr(skill_store, "delete_all"):
        skill_store.delete_all(user_id=user_id)
    else:
        # 降级：逐一删除
        all_skills = skill_store.get_all(user_id=user_id)
        ids = [s.get("id") or s.get("skill_id") or "" for s in all_skills]
        ids = [i for i in ids if i]
        if ids:
            skill_store.delete(ids, user_id=user_id)
    return DeleteResponse(message="All skills cleared.")


@router.delete("/memory/skills/{skill_id}", response_model=DeleteResponse)
async def delete_skill(skill_id: str, pipeline=Depends(get_pipeline), user=Depends(get_optional_user)):
    """删除指定技能条目。"""
    skill_store = pipeline.search_coordinator.skill_store
    user_id = user.user_id if user else ""
    skill_store.delete([skill_id], user_id=user_id)
    return DeleteResponse(message=f"Skill '{skill_id}' deleted.")


