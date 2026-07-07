"""Pipeline 追踪信息构建器。"""

from __future__ import annotations

from typing import Any

from src.api.models import (
    ChatResponse,
    ConsolidatorTrace,
    GraphMemoryHit,
    PipelineTrace,
    ReasonerTrace,
    SearchCoordinatorTrace,
    SkillHit,
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

        if node_id_str == "compact_context":
            for pv in mem.get("provenance", []):
                if not isinstance(pv, dict):
                    continue
                semantic_text = str(pv.get("semantic_text") or "").strip()
                record_id = str(pv.get("record_id") or "").strip()
                memory_type = str(pv.get("memory_type") or "")
                score = float(pv.get("score") or 0.0)
                entities = pv.get("entities") or []
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


def _build_reasoner_trace(result: dict[str, Any]) -> ReasonerTrace:
    return ReasonerTrace(response_length=len(result.get("final_response", "")))


def _build_trace(result: dict[str, Any]) -> PipelineTrace:
    """从完整的 PipelineState dict 中提取结构化的 Pipeline 追踪信息。"""
    return PipelineTrace(
        wm_manager=_build_wm_trace(result),
        search_coordinator=_build_search_trace(result),
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
        turn_input_tokens=result.get("turn_input_tokens", 0),
        turn_output_tokens=result.get("turn_output_tokens", 0),
        trace=_build_trace(result),
    )
