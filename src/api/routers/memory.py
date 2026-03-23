"""记忆端点：图谱、技能、检索。"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_optional_user, get_pipeline
from src.api.models import (
    DeleteResponse,
    FactEdgesResponse,
    GraphEdgeAddRequest,
    GraphNodeAddRequest,
    GraphResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    SkillsResponse,
)

logger = logging.getLogger("src.api")

router = APIRouter()


def _strip_node_embeddings(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从节点列表中移除 embedding 向量字段，避免 API 响应体过大。"""
    result: list[dict[str, Any]] = []
    for node in nodes:
        n = {k: v for k, v in node.items() if k != "embedding"}
        props = n.get("properties")
        if isinstance(props, dict) and "embedding" in props:
            n["properties"] = {k: v for k, v in props.items() if k != "embedding"}
        result.append(n)
    return result


def _fact_row_to_edge(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(row.get("source") or row.get("subject_entity_id") or ""),
        "target": str(row.get("target") or row.get("object_entity_id") or ""),
        "relation": str(row.get("relation") or row.get("relation_type") or ""),
        "confidence": float(row.get("confidence") or 1.0),
        "fact": str(row.get("fact") or row.get("fact_text") or ""),
        "evidence": str(row.get("evidence") or row.get("evidence_text") or ""),
        "source_session": str(row.get("source_session") or ""),
        "timestamp": str(
            row.get("timestamp")
            or row.get("t_valid_from")
            or row.get("t_created")
            or row.get("t_ref")
            or ""
        ),
        "t_valid_from": str(row.get("t_valid_from") or ""),
        "t_valid_to": str(row.get("t_valid_to") or ""),
        "t_tx_created": str(row.get("t_tx_created") or ""),
        "t_tx_expired": str(row.get("t_tx_expired") or ""),
        "episode_ids": row.get("episode_ids")
        if isinstance(row.get("episode_ids"), list)
        else [],
    }


# ── Memory: Search ──


@router.post("/memory/search", response_model=MemorySearchResponse)
async def memory_search(req: MemorySearchRequest, pipeline=Depends(get_pipeline), user=Depends(get_optional_user)):
    """统一记忆检索：同时查询图谱和技能库。"""
    sc = pipeline.search_coordinator
    user_id = user.user_id if user else ""

    graph_results: list[dict[str, Any]] = []
    if req.include_graph:
        q_emb = sc.embedder.embed_query(req.query)
        r = sc.graphiti_engine.search(
            query=req.query,
            session_id=None,
            top_k=req.top_k,
            query_embedding=q_emb,
            include_communities=True,
            user_id=user_id,
        )
        graph_results = r.provenance

    skill_results: list[dict[str, Any]] = []
    if req.include_skills:
        q_emb = sc.embedder.embed_query(req.query)
        skill_results = sc.skill_store.search(req.query, top_k=req.top_k, query_embedding=q_emb, user_id=user_id)

    total = len(graph_results) + len(skill_results)
    return MemorySearchResponse(
        query=req.query,
        graph_results=graph_results,
        skill_results=skill_results,
        total=total,
    )


# ── Memory: Graph ──


@router.get("/memory/graph", response_model=GraphResponse)
async def get_graph(pipeline=Depends(get_pipeline), user=Depends(get_optional_user)):
    user_id = user.user_id if user else ""
    store = pipeline.search_coordinator.graphiti_engine.store
    try:
        data = store.export_semantic_graph(user_id=user_id)
        return GraphResponse(
            nodes=_strip_node_embeddings(data.get("nodes", [])),
            edges=data.get("edges", []),
        )
    except Exception as exc:
        logger.exception("Graphiti export_semantic_graph failed")
        raise HTTPException(status_code=500, detail=f"Graphiti export failed: {exc}")


@router.get("/memory/graph/search", response_model=GraphResponse)
async def search_graph(
    q: str = Query(..., min_length=1, description="搜索关键词"),
    top_k: int = Query(default=10, ge=1, le=100),
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """按关键词搜索图谱节点，返回子图。"""
    user_id = user.user_id if user else ""
    store = pipeline.search_coordinator.graphiti_engine.store
    try:
        candidate_ids: list[str] = []
        seen: set[str] = set()
        for r in store.fulltext_search_entities(query=q, limit=max(50, top_k * 5), user_id=user_id):
            eid = str(r.get("entity_id") or "").strip()
            if eid and eid not in seen:
                candidate_ids.append(eid)
                seen.add(eid)

        anchor_ids = candidate_ids[:top_k]
        data = store.export_semantic_subgraph(
            entity_ids=anchor_ids,
            edge_limit=min(500, max(50, top_k * 50)),
            user_id=user_id,
        )
        nodes = _strip_node_embeddings(data.get("nodes", []))
        edges = data.get("edges", [])

        anchor_set = set(anchor_ids)
        for n in nodes:
            try:
                nid = str(n.get("id") or "").strip()
                if nid and nid in anchor_set:
                    props = n.get("properties")
                    if not isinstance(props, dict):
                        props = {}
                        n["properties"] = props
                    props["is_anchor"] = True
            except Exception:
                continue

        return GraphResponse(nodes=nodes, edges=edges)
    except Exception:
        logger.exception("Graphiti native graph search failed")
        raise HTTPException(status_code=500, detail="Graphiti graph search failed")


@router.post("/memory/graph/nodes", response_model=DeleteResponse)
async def add_graph_node(
    req: GraphNodeAddRequest,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """手动 upsert 一个实体节点到 Graphiti 知识图谱。"""
    store = pipeline.search_coordinator.graphiti_engine.store
    props = dict(req.properties or {})
    user_id_val = user.user_id if user else ""
    name = str(props.get("name") or req.id)
    store.upsert_entity(
        entity_id=req.id,
        name=name,
        summary=str(props.get("summary", "")),
        type_label=req.label,
        source_session=str(props.get("source_session", "")),
        user_id=user_id_val,
    )
    return DeleteResponse(message=f"Entity '{req.id}' upserted.")


@router.post("/memory/graph/edges", response_model=DeleteResponse)
async def add_graph_edge(
    req: GraphEdgeAddRequest,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """手动 upsert 一条 Fact 边到 Graphiti 知识图谱。"""
    store = pipeline.search_coordinator.graphiti_engine.store
    props = dict(req.properties or {})
    user_id_val = user.user_id if user else ""
    fact_id = str(props.get("fact_id") or uuid.uuid4())
    fact_text = str(props.get("fact_text") or f"{req.source} {req.relation} {req.target}")
    store.upsert_fact(
        fact_id=fact_id,
        subject_entity_id=req.source,
        object_entity_id=req.target,
        relation_type=req.relation,
        fact_text=fact_text,
        evidence_text=str(props.get("evidence_text", "")),
        confidence=float(props.get("confidence", 1.0)),
        source_session=str(props.get("source_session", "")),
        user_id=user_id_val,
    )
    return DeleteResponse(message=f"Fact '{fact_id}' upserted.")


@router.delete("/memory/graph/clear", response_model=DeleteResponse)
async def clear_all_graph(
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """清空当前用户的所有图谱记忆（Entity + Fact）。"""
    user_id = user.user_id if user else ""
    store = pipeline.search_coordinator.graphiti_engine.store
    try:
        result = store.delete_all_for_user(user_id=user_id)
        return DeleteResponse(
            message=(
                f"Graph memory cleared "
                f"(facts_deleted={result['facts_deleted']}, "
                f"episodes_deleted={result.get('episodes_deleted', 0)}, "
                f"entities_deleted={result['entities_deleted']})."
            )
        )
    except Exception as exc:
        logger.exception("Graphiti delete_all_for_user failed")
        raise HTTPException(status_code=500, detail=f"Graphiti clear failed: {exc}")


@router.delete("/memory/graph/nodes/{node_id}", response_model=DeleteResponse)
async def delete_graph_node(node_id: str, pipeline=Depends(get_pipeline), user=Depends(get_optional_user)):
    """删除图谱中的一个实体（及其所有关联 Fact）。"""
    user_id = user.user_id if user else ""
    store = pipeline.search_coordinator.graphiti_engine.store
    try:
        result = store.delete_entity(entity_id=node_id, user_id=user_id)
        return DeleteResponse(
            message=(
                f"Node '{node_id}' deleted "
                f"(entity_deleted={result['entity_deleted']}, "
                f"facts_deleted={result['facts_deleted']})."
            )
        )
    except Exception as exc:
        logger.exception("Graphiti delete_entity failed")
        raise HTTPException(status_code=500, detail=f"Graphiti delete failed: {exc}")


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


# ── Memory: Graph Time & Relation Search ──


@router.get("/memory/graph/by-relation")
async def search_graph_by_relation(
    relation: str = Query(..., min_length=1),
    top_k: int = Query(default=10, ge=1, le=100),
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """按关系类型检索图谱边。"""
    user_id = user.user_id if user is not None else ""
    store = pipeline.search_coordinator.graphiti_engine.store
    try:
        edges = store.search_facts_by_relation(relation=relation, limit=top_k, user_id=user_id)
        return {"edges": edges, "total": len(edges)}
    except Exception as exc:
        logger.exception("Graphiti search_facts_by_relation failed")
        raise HTTPException(status_code=500, detail=f"Graphiti by-relation failed: {exc}")


@router.get("/memory/graph/by-time")
async def search_graph_by_time(
    since: str | None = Query(default=None),
    until: str | None = Query(default=None),
    top_k: int = Query(default=10, ge=1, le=100),
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """按时间范围检索图谱边。"""
    user_id = user.user_id if user is not None else ""
    store = pipeline.search_coordinator.graphiti_engine.store
    try:
        edges = store.search_facts_by_time(since=since, until=until, limit=top_k, user_id=user_id)
        return {"edges": edges, "total": len(edges)}
    except Exception as exc:
        logger.exception("Graphiti search_facts_by_time failed")
        raise HTTPException(status_code=500, detail=f"Graphiti by-time failed: {exc}")


# ── Memory: Graph Facts ──


@router.get("/memory/graph/facts/active", response_model=FactEdgesResponse)
async def list_active_facts(
    subject: str = Query(..., min_length=1, description="subject_entity_id"),
    relation: str | None = Query(default=None, description="relation_type (optional)"),
    top_k: int = Query(default=200, ge=1, le=1000),
    pipeline=Depends(get_pipeline),
):
    """查看当前有效事实。"""
    store = pipeline.search_coordinator.graphiti_engine.store
    try:
        if relation and hasattr(store, "list_active_facts_for_subject_relation"):
            rows = store.list_active_facts_for_subject_relation(
                subject_entity_id=subject,
                relation_type=str(relation).strip().upper(),
                limit=top_k,
            )
        elif hasattr(store, "list_active_facts_for_subject"):
            rows = store.list_active_facts_for_subject(
                subject_entity_id=subject, limit=top_k
            )
        else:
            rows = []

        edges = [_fact_row_to_edge(dict(r)) for r in (rows or [])]
        return {"edges": edges, "total": len(edges)}
    except Exception as exc:
        logger.exception("Graphiti list_active_facts failed")
        raise HTTPException(status_code=500, detail=f"Graphiti facts/active failed: {exc}")


@router.get("/memory/graph/facts/history", response_model=FactEdgesResponse)
async def list_fact_history(
    subject: str = Query(..., min_length=1, description="subject_entity_id"),
    relation: str | None = Query(default=None, description="relation_type (optional)"),
    top_k: int = Query(default=200, ge=1, le=1000),
    pipeline=Depends(get_pipeline),
):
    """查看事实历史版本（含已失效）。"""
    store = pipeline.search_coordinator.graphiti_engine.store
    try:
        if relation and hasattr(store, "list_facts_for_subject_relation"):
            rows = store.list_facts_for_subject_relation(
                subject_entity_id=subject,
                relation_type=str(relation).strip().upper(),
                limit=top_k,
            )
        elif hasattr(store, "list_facts_for_subject"):
            rows = store.list_facts_for_subject(subject_entity_id=subject, limit=top_k)
        else:
            rows = []

        edges = [_fact_row_to_edge(dict(r)) for r in (rows or [])]
        return {"edges": edges, "total": len(edges)}
    except Exception as exc:
        logger.exception("Graphiti list_fact_history failed")
        raise HTTPException(status_code=500, detail=f"Graphiti facts/history failed: {exc}")
