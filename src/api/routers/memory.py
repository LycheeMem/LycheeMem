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


def _normalize_graph_id(value: Any) -> str:
    return str(value or "").strip()


def _build_tree_relationships(
    composites: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, list[str]]]:
    source_sets: dict[str, set[str]] = {}
    explicit_candidates: dict[str, set[str]] = {}
    parent_ids_with_explicit: set[str] = set()

    for composite in composites:
        composite_id = _normalize_graph_id(composite.get("composite_id"))
        if not composite_id:
            continue
        source_sets[composite_id] = {
            _normalize_graph_id(source_id)
            for source_id in (composite.get("source_record_ids") or [])
            if _normalize_graph_id(source_id)
        }

    for composite in composites:
        parent_id = _normalize_graph_id(composite.get("composite_id"))
        parent_sources = source_sets.get(parent_id, set())
        for raw_child_id in (composite.get("child_composite_ids") or []):
            child_id = _normalize_graph_id(raw_child_id)
            child_sources = source_sets.get(child_id, set())
            if (
                child_id
                and child_id != parent_id
                and child_sources
                and child_sources.issubset(parent_sources)
                and len(parent_sources) > len(child_sources)
            ):
                explicit_candidates.setdefault(child_id, set()).add(parent_id)
                parent_ids_with_explicit.add(parent_id)

    inferred_candidates: dict[str, set[str]] = {}
    composite_ids = list(source_sets.keys())
    for parent_id in composite_ids:
        if parent_id in parent_ids_with_explicit:
            continue
        parent_sources = source_sets[parent_id]
        if not parent_sources:
            continue

        eligible_children = [
            child_id
            for child_id in composite_ids
            if child_id != parent_id
            and source_sets[child_id]
            and source_sets[child_id].issubset(parent_sources)
            and len(parent_sources) > len(source_sets[child_id])
        ]

        for child_id in eligible_children:
            child_sources = source_sets[child_id]
            has_intermediate = any(
                mid_id not in {child_id, parent_id}
                and source_sets[mid_id]
                and child_sources.issubset(source_sets[mid_id])
                and source_sets[mid_id].issubset(parent_sources)
                and len(parent_sources) > len(source_sets[mid_id]) > len(child_sources)
                for mid_id in eligible_children
            )
            if not has_intermediate:
                inferred_candidates.setdefault(child_id, set()).add(parent_id)

    parent_candidates = {child_id: set(parents) for child_id, parents in explicit_candidates.items()}
    for child_id, parents in inferred_candidates.items():
        parent_candidates.setdefault(child_id, set()).update(parents)

    child_to_parent: dict[str, str] = {}
    for child_id, parents in parent_candidates.items():
        valid_parents = [parent_id for parent_id in parents if parent_id in source_sets]
        if not valid_parents:
            continue
        child_to_parent[child_id] = min(
            valid_parents,
            key=lambda parent_id: (len(source_sets.get(parent_id, set())) or 10**9, parent_id),
        )

    parent_to_children: dict[str, list[str]] = {}
    for child_id, parent_id in child_to_parent.items():
        parent_to_children.setdefault(parent_id, []).append(child_id)
    for child_ids in parent_to_children.values():
        child_ids.sort(key=lambda child_id: (len(source_sets.get(child_id, set())), child_id))

    return source_sets, child_to_parent, parent_to_children


def _build_semantic_tree_payload(
    data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = list(data.get("records", []))
    composites = list(data.get("composites", []))

    record_by_id = {
        _normalize_graph_id(record.get("record_id")): record
        for record in records
        if _normalize_graph_id(record.get("record_id"))
    }
    composite_by_id = {
        _normalize_graph_id(composite.get("composite_id")): composite
        for composite in composites
        if _normalize_graph_id(composite.get("composite_id"))
    }

    source_sets, child_to_parent, parent_to_children = _build_tree_relationships(composites)
    covered_record_ids = set().union(*source_sets.values()) if source_sets else set()

    node_cache: dict[str, dict[str, Any]] = {}

    def _build_record_node(record_id: str) -> dict[str, Any]:
        record = record_by_id[record_id]
        if record_id in node_cache:
            return node_cache[record_id]
        node = {
            "id": record_id,
            "name": str(record.get("normalized_text") or record.get("semantic_text") or record_id)[:120],
            "label": record.get("memory_type", "record"),
            "node_kind": "record",
            "properties": {
                "semantic_text": record.get("semantic_text", ""),
                "entities": record.get("entities", []),
                "confidence": record.get("confidence", 1.0),
                "created_at": record.get("created_at", ""),
                "source_record_count": 1,
                "child_composite_count": 0,
                "direct_record_count": 0,
            },
            "children": [],
        }
        node_cache[record_id] = node
        return node

    def _build_composite_node(composite_id: str) -> dict[str, Any]:
        composite = composite_by_id[composite_id]
        if composite_id in node_cache:
            return node_cache[composite_id]

        child_ids = parent_to_children.get(composite_id, [])
        covered_by_children = set().union(*(source_sets.get(child_id, set()) for child_id in child_ids)) if child_ids else set()
        direct_record_ids = sorted(
            record_id
            for record_id in (source_sets.get(composite_id, set()) - covered_by_children)
            if record_id in record_by_id
        )

        children = [_build_composite_node(child_id) for child_id in child_ids]
        children.extend(_build_record_node(record_id) for record_id in direct_record_ids)

        node = {
            "id": composite_id,
            "name": str(composite.get("normalized_text") or composite.get("semantic_text") or composite_id)[:120],
            "label": composite.get("memory_type", "composite"),
            "node_kind": "composite",
            "properties": {
                "semantic_text": composite.get("semantic_text", ""),
                "source_record_ids": list(sorted(source_sets.get(composite_id, set()))),
                "child_composite_ids": list(child_ids),
                "child_composite_count": len(child_ids),
                "direct_record_count": len(direct_record_ids),
                "source_record_count": len(source_sets.get(composite_id, set())),
                "confidence": composite.get("confidence", 1.0),
                "created_at": composite.get("created_at", ""),
            },
            "children": children,
        }
        node_cache[composite_id] = node
        return node

    root_composite_ids = sorted(
        composite_id
        for composite_id in composite_by_id
        if composite_id not in child_to_parent
    )
    root_record_ids = sorted(
        record_id
        for record_id in record_by_id
        if record_id not in covered_record_ids
    )

    tree_roots = [_build_composite_node(composite_id) for composite_id in root_composite_ids]
    tree_roots.extend(_build_record_node(record_id) for record_id in root_record_ids)

    tree_nodes = list(node_cache.values())
    tree_edges: list[dict[str, Any]] = []
    for parent_id, child_ids in parent_to_children.items():
        for child_id in child_ids:
            tree_edges.append({
                "source": parent_id,
                "target": child_id,
                "relation": "composed_from_composite",
            })

    all_parent_ids = sorted(set(root_composite_ids) | set(parent_to_children.keys()))
    for composite_id in all_parent_ids:
        child_ids = parent_to_children.get(composite_id, [])
        covered_by_children = set().union(*(source_sets.get(child_id, set()) for child_id in child_ids)) if child_ids else set()
        direct_record_ids = sorted(
            record_id
            for record_id in (source_sets.get(composite_id, set()) - covered_by_children)
            if record_id in record_by_id
        )
        for record_id in direct_record_ids:
            tree_edges.append({
                "source": composite_id,
                "target": record_id,
                "relation": "composed_from_record",
            })

    return tree_nodes, tree_edges, tree_roots


def _build_memory_search_context(
    pipeline,
    *,
    session_id: str | None,
    user_id: str,
) -> dict[str, Any]:
    """为显式 memory/search 请求补充可选的会话上下文。"""
    if not session_id:
        return {}

    wm = pipeline.wm_manager
    log = wm.session_store.get_or_create(session_id, user_id=user_id)
    compressed_history = wm.compressor.render_context(log.turns, log.summaries)
    active_turns = [t for t in log.turns if not t.get("deleted", False)]
    return {
        "compressed_history": compressed_history,
        "raw_recent_turns": active_turns[-6:],
        "wm_token_usage": wm.compressor.count_tokens(compressed_history),
    }


def run_memory_search(
    pipeline,
    req: MemorySearchRequest,
    *,
    user_id: str = "",
) -> MemorySearchResponse:
    """执行统一记忆检索，返回可直接供 Synthesizer 消费的 richer 结构。"""
    sc = pipeline.search_coordinator

    search_runtime = _build_memory_search_context(
        pipeline,
        session_id=req.session_id,
        user_id=user_id,
    )
    search_result = sc.run(
        req.query,
        session_id=req.session_id,
        user_id=user_id,
        top_k=req.top_k,
        include_skills=req.include_skills,
        **search_runtime,
    )

    semantic_results: list[dict[str, Any]] = []
    if req.include_graph:
        semantic_results = list(search_result.get("retrieved_graph_memories", []))

    graph_results = list(semantic_results)
    skill_results: list[dict[str, Any]] = []
    if req.include_skills:
        skill_results = list(search_result.get("retrieved_skills", []))

    graph_total = 0
    for item in semantic_results:
        provenance = item.get("provenance")
        if isinstance(provenance, list) and provenance:
            graph_total += len(provenance)
        else:
            graph_total += 1

    total = graph_total + len(skill_results)
    return MemorySearchResponse(
        query=req.query,
        graph_results=graph_results,
        semantic_results=semantic_results,
        skill_results=skill_results,
        total=total,
    )


def run_memory_synthesize(
    pipeline,
    req: MemorySynthesizeRequest,
) -> MemorySynthesizeResponse:
    """执行检索结果压缩，供 HTTP Router 与 MCP 共享。"""
    semantic_results = req.semantic_results or req.graph_results
    result = pipeline.synthesizer.run(
        user_query=req.user_query,
        retrieved_graph_memories=semantic_results,
        retrieved_skills=req.skill_results,
    )

    provenance_raw = result.get("provenance", [])
    provenance_flat: list[dict] = []
    for item in provenance_raw:
        if isinstance(item, dict) and isinstance(item.get("items"), list):
            provenance_flat.extend(item["items"])
        elif isinstance(item, dict):
            provenance_flat.append(item)

    input_count = int(result.get("input_fragment_count") or (len(semantic_results) + len(req.skill_results)))
    kept_count = int(result.get("kept_count") or len(provenance_flat))
    dropped_count = int(result.get("dropped_count") or max(0, input_count - kept_count))

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
            session_id=req.session_id,
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
            semantic_results=search_result.semantic_results,
            skill_results=search_result.skill_results,
            total=search_result.total,
            synthesized=False,
        )

    if req.mode == "raw":
        return MemorySmartSearchResponse(
            query=search_result.query,
            mode=req.mode,
            graph_results=search_result.graph_results,
            semantic_results=search_result.semantic_results,
            skill_results=search_result.skill_results,
            total=search_result.total,
            synthesized=False,
        )

    synth_result = run_memory_synthesize(
        pipeline,
        MemorySynthesizeRequest(
            user_query=req.query,
            graph_results=search_result.graph_results,
            semantic_results=search_result.semantic_results,
            skill_results=search_result.skill_results,
        ),
    )
    if req.mode == "compact":
        return MemorySmartSearchResponse(
            query=search_result.query,
            mode=req.mode,
            graph_results=[],
            semantic_results=[],
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
        semantic_results=search_result.semantic_results,
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
    """执行长期记忆固化，供 HTTP Router 与 MCP 共享。

    使用固化水位线：只处理自上次固化以来新增的 turns，
    避免整个 session 被重复固化。
    """
    import threading

    store = pipeline.wm_manager.session_store
    log = store.get_or_create(req.session_id, user_id=user_id)
    watermark = log.last_consolidated_turn_index
    raw_total = len(log.turns)
    turns = [t for t in log.turns[watermark:] if not t.get("deleted", False)]

    if not turns:
        return MemoryConsolidateResponse(
            status="skipped",
            skipped_reason="no_new_turns",
            steps=[{"name": "watermark_check", "status": "skipped",
                    "detail": f"自水位线({watermark})以来无新增 turns，跳过固化"}],
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
                # 后台线程固化成功后推进水位线
                store.set_last_consolidated_turn_index(req.session_id, raw_total)
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
    # 同步固化成功后推进水位线
    store.set_last_consolidated_turn_index(req.session_id, raw_total)
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
async def get_graph(
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
    mode: str = Query(
        default="cleaned",
        description=(
            "cleaned（默认）：输出层级记忆树，保留父/子 composite 以及直接叶子 record，"
            "用于前端树视图展示；"
            "debug：导出全部 records + composites 的底层关系，用于底层排查。"
        ),
    ),
):
    user_id = user.user_id if user else ""
    sc = pipeline.search_coordinator
    try:
        data = sc.semantic_engine.export_debug(user_id=user_id)
        if mode == "cleaned":
            nodes, edges, tree_roots = _build_semantic_tree_payload(data)
            return GraphResponse(nodes=nodes, edges=edges, tree_roots=tree_roots)

        nodes = []
        edges = []
        for u in data.get("records", []):
            rid = u.get("record_id", "")
            nodes.append({
                "id": rid,
                "name": u.get("normalized_text", "")[:80],
                "label": u.get("memory_type", "record"),
                "node_kind": "record",
                "properties": {
                    "semantic_text": u.get("semantic_text", ""),
                    "entities": u.get("entities", []),
                    "confidence": u.get("confidence", 1.0),
                    "created_at": u.get("created_at", ""),
                },
            })
        for s in data.get("composites", []):
            composite_id = s.get("composite_id", "")
            nodes.append({
                "id": composite_id,
                "name": s.get("normalized_text", "")[:80],
                "label": s.get("memory_type", "composite"),
                "node_kind": "composite",
                "properties": {
                    "semantic_text": s.get("semantic_text", ""),
                    "source_record_ids": s.get("source_record_ids", []),
                    "child_composite_ids": s.get("child_composite_ids", []),
                },
            })
            for src_id in s.get("source_record_ids", []):
                edges.append({
                    "source": composite_id,
                    "target": src_id,
                    "relation": "composed_from",
                })
        _, _, tree_roots = _build_semantic_tree_payload(data)
        return GraphResponse(nodes=nodes, edges=edges, tree_roots=tree_roots)
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
                "id": r.get("record_id", ""),
                "name": r.get("normalized_text", "")[:80],
                "label": r.get("memory_type", "record"),
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
                f"(records_deleted={result.get('records_deleted', 0)}, "
                f"composites_deleted={result.get('composites_deleted', 0)})."
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


