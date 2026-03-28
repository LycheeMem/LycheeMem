"""Pipeline 追踪信息构建器。"""

from __future__ import annotations

from typing import Any

from src.api.models import (
    ChatResponse,
    ConsolidatorTrace,
    GraphMemoryHit,
    PipelineTrace,
    ProvenanceItem,
    ReasonerTrace,
    SearchCoordinatorTrace,
    SkillHit,
    SynthesizerTrace,
    WMManagerTrace,
)


def _build_wm_trace(result: dict[str, Any]) -> WMManagerTrace:
    compressed_history = result.get("compressed_history", [])
    raw_recent_turns = result.get("raw_recent_turns", [])
    has_summary = any(t.get("role") == "system" for t in compressed_history)
    return WMManagerTrace(
        wm_token_usage=result.get("wm_token_usage", 0),
        compressed_turn_count=len(compressed_history),
        raw_recent_turn_count=len(raw_recent_turns),
        compression_happened=has_summary,
    )


def _build_search_trace(result: dict[str, Any]) -> SearchCoordinatorTrace:
    graph_mems = result.get("retrieved_graph_memories", [])
    skills = result.get("retrieved_skills", [])
    graph_hits: list[GraphMemoryHit] = []
    for mem in graph_mems:
        anchor = mem.get("anchor", mem)
        node_id_str = str(anchor.get("node_id", ""))

        # Compact 后端：将每条 provenance 展开为独立的 GraphMemoryHit
        if node_id_str == "compact_context":
            for pv in mem.get("provenance", []):
                if not isinstance(pv, dict):
                    continue
                semantic_text = str(pv.get("semantic_text") or "").strip()
                record_id = str(pv.get("record_id") or "").strip()
                memory_type = str(pv.get("memory_type") or "")
                score = float(pv.get("score") or 0.0)
                entities = pv.get("entities") or []
                entity_summary = ", ".join(str(e) for e in entities[:3]) if entities else ""
                display_name = semantic_text[:120] if semantic_text else record_id
                graph_hits.append(
                    GraphMemoryHit(
                        node_id=record_id,
                        name=display_name,
                        label=memory_type,
                        score=score,
                        neighbor_count=len(entities),
                    )
                )
            continue

        if node_id_str == "graphiti_context":
            for pv in mem.get("provenance", []):
                if not isinstance(pv, dict):
                    continue
                fact_text = str(pv.get("fact_text") or "").strip()
                subj = str(pv.get("subject_entity_id") or "").strip()
                obj = str(pv.get("object_entity_id") or "").strip()
                rel = str(pv.get("relation_type") or "").strip()
                if fact_text:
                    display_name = fact_text
                elif subj and obj:
                    display_name = f"{subj} —{rel}→ {obj}" if rel else f"{subj} → {obj}"
                else:
                    display_name = str(pv.get("fact_id") or "")
                graph_hits.append(
                    GraphMemoryHit(
                        node_id=str(pv.get("fact_id") or ""),
                        name=display_name,
                        label=rel,
                        score=float(pv.get("rrf") or 0.0),
                        neighbor_count=int(pv.get("mentions") or 0),
                    )
                )
            continue
        subgraph = mem.get("subgraph", {})
        neighbor_count = len(subgraph.get("nodes", [])) + len(subgraph.get("edges", []))
        props = anchor.get("properties", {})
        graph_hits.append(
            GraphMemoryHit(
                node_id=str(anchor.get("node_id", anchor.get("id", ""))),
                name=str(props.get("name", anchor.get("name", ""))),
                label=str(anchor.get("label", props.get("label", ""))),
                score=float(anchor.get("score", 0.0)),
                neighbor_count=neighbor_count,
            )
        )
    skill_hits = [
        SkillHit(
            skill_id=str(sk.get("id", sk.get("skill_id", ""))),
            intent=str(sk.get("intent", "")),
            score=float(sk.get("score", 0.0)),
            reusable=bool(sk.get("reusable", False)),
        )
        for sk in skills
    ]
    return SearchCoordinatorTrace(
        graph_memories=graph_hits,
        skills=skill_hits,
        total_retrieved=len(graph_hits) + len(skill_hits),
    )


def _build_synthesizer_trace(result: dict[str, Any]) -> SynthesizerTrace:
    provenance_raw = result.get("provenance", [])
    provenance: list[ProvenanceItem] = []
    for idx, p in enumerate(provenance_raw):
        if not isinstance(p, dict):
            continue
        fact_id = str(p.get("fact_id") or "").strip()
        rrf = float(p.get("rrf") or p.get("relevance") or 0.0)
        bm25_rank_val = p.get("bm25_rank")
        bfs_rank_val = p.get("bfs_rank")
        mention_count = int(p.get("mentions") or 0)
        dist = p.get("distance") or p.get("gds_distance")
        graph_distance_val = int(dist) if dist is not None else None
        cross_enc = p.get("cross_encoder_score")
        cross_enc_val = float(cross_enc) if cross_enc is not None else None
        source_eps = p.get("source_episodes") or []
        if not isinstance(source_eps, list):
            source_eps = []
        
        # 优先使用 provenance item 中的 source（compact 为 "record"/"composite"，graphiti 为具体类型）
        # 如果 source 为 graphiti_retrieval，说明这是从 search_coordinator 传来的包装结构
        wrapper_source = str(p.get("source") or "")
        legacy_summary = str(p.get("summary") or "")
        nested_items = p.get("items")
        
        # 展开 nested items（wrapper 结构下的实际 provenance）
        if isinstance(nested_items, list):
            for sub_idx, sub in enumerate(nested_items):
                if not isinstance(sub, dict):
                    continue
                sub_fact_id = str(sub.get("fact_id") or "").strip()
                sub_source = str(sub.get("source") or "").strip()
                sub_rrf = float(sub.get("rrf") or sub.get("relevance") or 0.0)
                sub_bm25 = sub.get("bm25_rank")
                sub_bfs = sub.get("bfs_rank")
                sub_mentions = int(sub.get("mentions") or 0)
                sub_dist = sub.get("distance") or sub.get("gds_distance")
                sub_gd = int(sub_dist) if sub_dist is not None else None
                sub_ce = sub.get("cross_encoder_score")
                sub_ce_val = float(sub_ce) if sub_ce is not None else None
                sub_eps = sub.get("source_episodes") or []
                if not isinstance(sub_eps, list):
                    sub_eps = []
                
                # 如果 sub 有 source 且不为空，用 sub.source；否则用 wrapper_source；否则用默认
                final_source = sub_source or wrapper_source or "compact_semantic"
                # 对于 compact 后端，使用 semantic_text；Graphiti 用 fact_text/summary
                summary_text = str(sub.get("fact_text") or sub.get("summary") or sub.get("semantic_text") or "")
                provenance.append(
                    ProvenanceItem(
                        source=final_source,
                        index=idx * 1000 + sub_idx,
                        relevance=sub_rrf,
                        fact_id=sub_fact_id,
                        summary=summary_text,
                        rrf_score=sub_rrf,
                        bm25_rank=int(sub_bm25) if sub_bm25 is not None else None,
                        bfs_rank=int(sub_bfs) if sub_bfs is not None else None,
                        mention_count=sub_mentions,
                        graph_distance=sub_gd,
                        cross_encoder_score=sub_ce_val,
                        source_episodes=sub_eps,
                    )
                )
            continue
        
        # 非嵌套项（直接的 provenance item）
        final_source = str(p.get("source") or "compact_semantic")
        # 对于 compact 后端，优先用 semantic_text；Graphiti 用 fact_text
        summary_text = legacy_summary or str(p.get("fact_text") or p.get("semantic_text") or "")
        provenance.append(
            ProvenanceItem(
                source=final_source,
                index=int(p.get("index") or idx),
                relevance=rrf,
                fact_id=fact_id,
                summary=summary_text,
                rrf_score=rrf,
                bm25_rank=int(bm25_rank_val) if bm25_rank_val is not None else None,
                bfs_rank=int(bfs_rank_val) if bfs_rank_val is not None else None,
                mention_count=mention_count,
                graph_distance=graph_distance_val,
                cross_encoder_score=cross_enc_val,
                source_episodes=source_eps,
            )
        )
    return SynthesizerTrace(
        background_context=str(result.get("background_context", "")),
        provenance=provenance,
        skill_reuse_plan=result.get("skill_reuse_plan", []),
        kept_count=len(provenance),
    )


def _build_reasoner_trace(result: dict[str, Any]) -> ReasonerTrace:
    return ReasonerTrace(response_length=len(result.get("final_response", "")))


def _build_trace(result: dict[str, Any]) -> PipelineTrace:
    """从完整的 PipelineState dict 中提取结构化的 Pipeline 追踪信息。"""
    return PipelineTrace(
        wm_manager=_build_wm_trace(result),
        search_coordinator=_build_search_trace(result),
        synthesizer=_build_synthesizer_trace(result),
        reasoner=_build_reasoner_trace(result),
        consolidator=ConsolidatorTrace(status="pending"),
    )


def _build_chat_response(session_id: str, result: dict[str, Any]) -> ChatResponse:
    memories = len(result.get("retrieved_graph_memories", [])) + len(
        result.get("retrieved_skills", [])
    )
    return ChatResponse(
        session_id=session_id,
        response=result.get("final_response", ""),
        memories_retrieved=memories,
        wm_token_usage=result.get("wm_token_usage", 0),
        trace=_build_trace(result),
    )
