"""记忆端点：图谱、技能、检索。"""

from __future__ import annotations

import logging
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
    """从节点列表中移除 embedding 向量字段，避免 API 响应体过大。

    兼容两种节点结构：
    - Graphiti 路径：顶层 ``properties`` 子字典中含 ``embedding``。
    - NetworkX / Neo4j legacy 路径：顶层直接含 ``embedding``。
    """
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
        graph_results = sc.graph_store.search(req.query, top_k=req.top_k, user_id=user_id)

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
    graphiti = getattr(getattr(pipeline, "consolidator", None), "graphiti_engine", None)
    store = getattr(graphiti, "store", None) if graphiti is not None else None
    if store is not None and hasattr(store, "export_semantic_graph"):
        try:
            data = store.export_semantic_graph(user_id=user_id)
            return GraphResponse(
                nodes=_strip_node_embeddings(data.get("nodes", [])),
                edges=data.get("edges", []),
            )
        except Exception as exc:
            logger.exception("Graphiti export_semantic_graph failed")
            raise HTTPException(status_code=500, detail=f"Graphiti export failed: {exc}")

    graph_store = pipeline.search_coordinator.graph_store
    nodes = _strip_node_embeddings(graph_store.get_all(user_id=user_id))
    if hasattr(graph_store, "get_all_edges"):
        edges = graph_store.get_all_edges(user_id=user_id)
    else:
        edges = []
    return GraphResponse(nodes=nodes, edges=edges)


@router.get("/memory/graph/search", response_model=GraphResponse)
async def search_graph(
    q: str = Query(..., min_length=1, description="搜索关键词"),
    top_k: int = Query(default=10, ge=1, le=100),
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """按关键词搜索图谱节点，返回子图。"""
    user_id = user.user_id if user else ""
    graphiti = getattr(getattr(pipeline, "consolidator", None), "graphiti_engine", None)
    store = getattr(graphiti, "store", None) if graphiti is not None else None
    if store is not None and hasattr(store, "export_semantic_subgraph"):
        try:
            # 1) candidate entity ids (strict: fulltext only; no embedding scan fallback)
            candidate_ids: list[str] = []
            seen: set[str] = set()

            try:
                ft = store.fulltext_search_entities(query=q, limit=max(50, top_k * 5), user_id=user_id)
            except Exception:
                ft = []

            for r in ft:
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

            # Mark anchors (non-breaking extra metadata)
            for n in nodes:
                try:
                    nid = str(n.get("id") or "").strip()
                    if nid and nid in set(anchor_ids):
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

    sc = pipeline.search_coordinator
    graph_store = sc.graph_store
    query_embedding = None
    try:
        query_embedding = sc.embedder.embed_query(q)
    except Exception:
        query_embedding = None
    matched_nodes = _strip_node_embeddings(
        graph_store.search(q, top_k=top_k, query_embedding=query_embedding, user_id=user_id)
    )
    node_ids = {n["node_id"] for n in matched_nodes}
    if hasattr(graph_store, "get_all_edges"):
        all_edges = graph_store.get_all_edges(user_id=user_id)
        edges = [e for e in all_edges if e["source"] in node_ids or e["target"] in node_ids]
    else:
        edges = []
    return GraphResponse(nodes=matched_nodes, edges=edges)


@router.post("/memory/graph/nodes", response_model=DeleteResponse)
async def add_graph_node(
    req: GraphNodeAddRequest,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """手动添加一个实体节点到知识图谱。"""
    graph_store = pipeline.search_coordinator.graph_store
    props = dict(req.properties or {})
    if user:
        props["user_id"] = user.user_id
    graph_store.add_node(req.id, label=req.label, properties=props)
    return DeleteResponse(message=f"Node '{req.id}' added.")


@router.post("/memory/graph/edges", response_model=DeleteResponse)
async def add_graph_edge(
    req: GraphEdgeAddRequest,
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """手动添加一条关系边到知识图谱。"""
    graph_store = pipeline.search_coordinator.graph_store
    props = dict(req.properties or {})
    if user:
        props["user_id"] = user.user_id
    graph_store.add_edge(
        req.source, req.target, relation=req.relation, properties=props
    )
    return DeleteResponse(
        message=f"Edge '{req.source}' -{req.relation}-> '{req.target}' added."
    )


@router.delete("/memory/graph/clear", response_model=DeleteResponse)
async def clear_all_graph(
    pipeline=Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    """清空当前用户的所有图谱记忆（Entity + Fact / legacy 节点）。"""
    user_id = user.user_id if user else ""

    # Graphiti 路径
    graphiti = getattr(getattr(pipeline, "consolidator", None), "graphiti_engine", None)
    store = getattr(graphiti, "store", None) if graphiti is not None else None
    if store is not None and hasattr(store, "delete_all_for_user"):
        try:
            result = store.delete_all_for_user(user_id=user_id)
            return DeleteResponse(
                message=(
                    f"Graph memory cleared "
                    f"(facts_deleted={result['facts_deleted']}, "
                    f"episodes_deleted={result.get('episodes_deleted', 0)}, "
                    f"entities_deleted={result['entities_deleted']}, "
                    f"communities_deleted={result.get('communities_deleted', 0)})."
                )
            )
        except Exception as exc:
            logger.exception("Graphiti delete_all_for_user failed")
            raise HTTPException(status_code=500, detail=f"Graphiti clear failed: {exc}")

    # Legacy 路径
    graph_store = pipeline.search_coordinator.graph_store
    if hasattr(graph_store, "delete_all"):
        graph_store.delete_all(user_id=user_id)
    else:
        nodes = graph_store.get_all(user_id=user_id)
        ids = [n.get("id") or n.get("node_id") or "" for n in nodes]
        ids = [i for i in ids if i]
        if ids:
            graph_store.delete(ids, user_id=user_id)
    return DeleteResponse(message="Graph memory cleared.")


@router.delete("/memory/graph/nodes/{node_id}", response_model=DeleteResponse)
async def delete_graph_node(node_id: str, pipeline=Depends(get_pipeline), user=Depends(get_optional_user)):
    """删除图谱中的一个节点（及其所有关联边/Fact）。"""
    user_id = user.user_id if user else ""

    # Graphiti 路径：entity_id 是内容哈希，与 legacy node_id 不同，需单独处理
    graphiti = getattr(getattr(pipeline, "consolidator", None), "graphiti_engine", None)
    store = getattr(graphiti, "store", None) if graphiti is not None else None
    if store is not None and hasattr(store, "delete_entity"):
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

    # Legacy 路径
    graph_store = pipeline.search_coordinator.graph_store
    graph_store.delete([node_id], user_id=user_id)
    return DeleteResponse(message=f"Node '{node_id}' deleted.")


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
    graphiti = getattr(getattr(pipeline, "consolidator", None), "graphiti_engine", None)
    store = getattr(graphiti, "store", None) if graphiti is not None else None
    if store is not None and hasattr(store, "search_facts_by_relation"):
        try:
            edges = store.search_facts_by_relation(relation=relation, limit=top_k, user_id=user_id)
            return {"edges": edges, "total": len(edges)}
        except Exception as exc:
            logger.exception("Graphiti search_facts_by_relation failed")
            raise HTTPException(status_code=500, detail=f"Graphiti by-relation failed: {exc}")

    graph_store = pipeline.search_coordinator.graph_store
    if hasattr(graph_store, "search_by_relation"):
        edges = graph_store.search_by_relation(relation, top_k=top_k)
    else:
        edges = []
    return {"edges": edges, "total": len(edges)}


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
    graphiti = getattr(getattr(pipeline, "consolidator", None), "graphiti_engine", None)
    store = getattr(graphiti, "store", None) if graphiti is not None else None
    if store is not None and hasattr(store, "search_facts_by_time"):
        try:
            edges = store.search_facts_by_time(since=since, until=until, limit=top_k, user_id=user_id)
            return {"edges": edges, "total": len(edges)}
        except Exception as exc:
            logger.exception("Graphiti search_facts_by_time failed")
            raise HTTPException(status_code=500, detail=f"Graphiti by-time failed: {exc}")

    graph_store = pipeline.search_coordinator.graph_store
    if hasattr(graph_store, "search_by_time"):
        edges = graph_store.search_by_time(since=since, until=until, top_k=top_k)
    else:
        edges = []
    return {"edges": edges, "total": len(edges)}


# ── Memory: Graph Facts ──


@router.get("/memory/graph/facts/active", response_model=FactEdgesResponse)
async def list_active_facts(
    subject: str = Query(..., min_length=1, description="subject_entity_id"),
    relation: str | None = Query(default=None, description="relation_type (optional)"),
    top_k: int = Query(default=200, ge=1, le=1000),
    pipeline=Depends(get_pipeline),
):
    """查看当前有效事实（Graphiti 优先）。"""
    graphiti = getattr(getattr(pipeline, "consolidator", None), "graphiti_engine", None)
    store = getattr(graphiti, "store", None) if graphiti is not None else None

    if store is not None:
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

    # Legacy fallback: best-effort filter from current graph edges (no versioning).
    graph_store = pipeline.search_coordinator.graph_store
    if hasattr(graph_store, "get_all_edges"):
        all_edges = graph_store.get_all_edges()
    else:
        all_edges = [
            {"source": u, "target": v, **d} for u, v, d in graph_store.graph.edges(data=True)
        ]

    rel_upper = str(relation).strip().upper() if relation else None
    edges = []
    for e in all_edges:
        if str(e.get("source") or "") != subject:
            continue
        if rel_upper and str(e.get("relation") or "").strip().upper() != rel_upper:
            continue
        edges.append(_fact_row_to_edge(e))
        if len(edges) >= top_k:
            break
    return {"edges": edges, "total": len(edges)}


@router.get("/memory/graph/facts/history", response_model=FactEdgesResponse)
async def list_fact_history(
    subject: str = Query(..., min_length=1, description="subject_entity_id"),
    relation: str | None = Query(default=None, description="relation_type (optional)"),
    top_k: int = Query(default=200, ge=1, le=1000),
    pipeline=Depends(get_pipeline),
):
    """查看事实历史版本（含已失效）。

    Notes:
    - Graphiti store：返回 Fact-node 的版本历史（t_valid_* + t_tx_*）。
    - Legacy triples：没有事实版本概念，返回空列表。
    """
    graphiti = getattr(getattr(pipeline, "consolidator", None), "graphiti_engine", None)
    store = getattr(graphiti, "store", None) if graphiti is not None else None

    if store is not None:
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

    return {"edges": [], "total": 0}
