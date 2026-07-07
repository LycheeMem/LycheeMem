"""Route-level retrieval orchestration."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from src.memory.semantic.base import SemanticSearchResult
from src.memory.semantic.debug_trace import (
    make_trace_id,
    semantic_trace_enabled,
    semantic_trace_event,
    semantic_trace_int,
    truncate_trace_text,
)
from src.memory.semantic.models import EvidenceRoute, MemoryRecord, SearchPlan
from src.memory.semantic.retrieval.strategy import RetrievalStrategy


class RouteRetrievalMixin:
    def _search_fielded_evidence(
        self,
        *,
        query: str,
        session_id: str | None,
        top_k: int,
        routes: list[EvidenceRoute],
        plan: SearchPlan,
        strategy: RetrievalStrategy,
        query_embedding: list[float] | None = None,
        reference_time: str | None = None,
        trace_id: str = "",
    ) -> SemanticSearchResult:
        """Route-aware retrieval over independent evidence needs."""
        t_total = time.perf_counter()
        timings: dict[str, float] = {}
        counts: dict[str, int] = {
            "routes": len(routes),
            "variants": 0,
            "evidence_ann_calls": 0,
            "record_ann_calls": 0,
            "turn_ann_calls": 0,
        }

        route_query_texts: list[str] = []
        for route in routes:
            route_query_texts.extend(self._route_query_variants(query, route))
        route_query_texts = self._merge_unique([], route_query_texts)
        counts["variants"] = len(route_query_texts)
        trace_on = semantic_trace_enabled()
        trace_candidate_limit = semantic_trace_int(
            "LYCHEE_SEMANTIC_TRACE_CANDIDATES",
            8,
            minimum=1,
        )
        trace_max_text = semantic_trace_int(
            "LYCHEE_SEMANTIC_TRACE_MAX_TEXT",
            800,
            minimum=0,
        )

        t0 = time.perf_counter()
        query_vector_cache = self._build_search_query_vector_cache(
            query_texts=route_query_texts,
            original_query=query,
            original_query_embedding=query_embedding,
        )
        timings["embed_queries_ms"] = (time.perf_counter() - t0) * 1000
        record_cache: dict[str, MemoryRecord | None] = {}

        route_results: list[dict[str, Any]] = []
        for route in routes:
            route_result = self._search_single_evidence_route(
                query=query,
                route=route,
                plan=plan,
                strategy=strategy,
                top_k=top_k,
                query_vector_cache=query_vector_cache,
                record_cache=record_cache,
                trace_enabled=trace_on,
                trace_candidate_limit=trace_candidate_limit,
                trace_max_text=trace_max_text,
            )
            route_results.append(route_result)
            for key, value in route_result.get("timings", {}).items():
                timings[key] = timings.get(key, 0.0) + value
            for key, value in route_result.get("counts", {}).items():
                counts[key] = counts.get(key, 0) + int(value)

        t0 = time.perf_counter()
        top, selected_candidates = self._fuse_route_results(route_results, top_k)
        top = self._limit_episode_candidates(
            top,
            selected_candidates,
            top_k,
            strategy=strategy,
        )
        timings["route_fusion_ms"] = (time.perf_counter() - t0) * 1000
        counts["selected_pool"] = len(selected_candidates)
        counts["selected_records"] = len([c for c in selected_candidates if c.source == "record"])

        t0 = time.perf_counter()
        context = self._format_context(top)
        timings["format_context_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        provenance = self._build_provenance(top)
        timings["provenance_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        if trace_on:
            context_chars = semantic_trace_int(
                "LYCHEE_SEMANTIC_TRACE_CONTEXT_CHARS",
                20000,
                minimum=0,
            )
            semantic_trace_event(
                "semantic_search",
                {
                    "trace_id": trace_id or make_trace_id("search"),
                    "query": query,
                    "session_id": session_id or "",
                    "reference_time": reference_time or "",
                    "top_k": top_k,
                    "plan": self._plan_to_dict(plan),
                    "strategy": dict(strategy.__dict__),
                    "route_query_texts": route_query_texts,
                    "counts": counts,
                    "timings_ms": {key: round(value, 1) for key, value in timings.items()},
                    "routes": [
                        result.get("debug")
                        for result in route_results
                        if isinstance(result.get("debug"), dict)
                    ],
                    "fused_pool_top": self._debug_scored_payload(
                        selected_candidates[:trace_candidate_limit],
                        max_text=trace_max_text,
                    ),
                    "final_selected": self._debug_scored_payload(
                        top,
                        max_text=trace_max_text,
                    ),
                    "final_context": truncate_trace_text(context, context_chars),
                    "provenance_count": len(provenance),
                },
            )
        return SemanticSearchResult(
            context=context,
            provenance=provenance,
            retrieval_plan=self._plan_to_dict(plan),
            mode="semantic",
        )

    def _search_single_evidence_route(
        self,
        *,
        query: str,
        route: EvidenceRoute,
        plan: SearchPlan,
        strategy: RetrievalStrategy,
        top_k: int,
        query_vector_cache: dict[str, list[float]],
        record_cache: dict[str, MemoryRecord | None],
        trace_enabled: bool = False,
        trace_candidate_limit: int = 8,
        trace_max_text: int = 800,
    ) -> dict[str, Any]:
        candidate_by_id: dict[str, dict[str, Any]] = {}
        raw_turn_by_id: dict[str, dict[str, Any]] = {}
        timings: dict[str, float] = {}
        counts: dict[str, int] = {
            "evidence_ann_calls": 0,
            "record_ann_calls": 0,
            "turn_ann_calls": 0,
        }
        route_queries = self._route_query_variants(query, route)
        evidence_limit = max(top_k * 3, 30)
        evidence_limit = max(1, int(evidence_limit * strategy.evidence_limit_multiplier))
        record_limit = max(top_k * 3, 30)
        record_limit = max(1, int(record_limit * strategy.record_limit_multiplier))
        turn_limit = max(top_k, 20)
        turn_limit = max(1, int(turn_limit * strategy.turn_limit_multiplier))
        temporal_filter = route.temporal_filter or plan.temporal_filter
        trace: dict[str, Any] | None = None
        if trace_enabled:
            trace = {
                "route_id": route.route_id,
                "evidence_goal": route.evidence_goal,
                "queries": route_queries,
                "constraints": list(route.constraints or []),
                "temporal_filter": dict(temporal_filter or {}),
                "limits": {
                    "evidence_limit": evidence_limit,
                    "record_limit": record_limit,
                    "turn_limit": turn_limit,
                },
                "query_steps": [],
            }

        t0 = time.perf_counter()
        self._recall_records_from_temporal_filter(
            candidate_by_id,
            temporal_filter=temporal_filter,
            record_cache=record_cache,
            limit=max(top_k * 10, 100),
        )
        timings["temporal_ms"] = (time.perf_counter() - t0) * 1000
        if trace is not None:
            trace["after_temporal_filter"] = {
                "record_candidates": len(candidate_by_id),
                "top": self._debug_candidate_payload(
                    sorted(
                        candidate_by_id.values(),
                        key=lambda item: self._safe_float(item.get("field_score"), 0.0),
                        reverse=True,
                    )[:trace_candidate_limit],
                    max_text=trace_max_text,
                ),
            }

        for variant in route_queries:
            variant_vector = query_vector_cache.get(self._query_vector_key(variant))
            variant_trace: dict[str, Any] | None = None
            if trace is not None:
                variant_trace = {
                    "query": variant,
                    "has_embedding": variant_vector is not None,
                }

            t0 = time.perf_counter()
            before_records = len(candidate_by_id)
            evidence_calls = self._recall_records_from_evidence_nodes_multi(
                candidate_by_id,
                query=variant,
                query_vector=variant_vector,
                limit_per_type=evidence_limit,
                record_cache=record_cache,
            )
            counts["evidence_ann_calls"] += evidence_calls
            if variant_trace is not None:
                variant_trace.update({
                    "evidence_ann_calls": evidence_calls,
                    "records_after_evidence": len(candidate_by_id),
                    "records_added_by_evidence": max(0, len(candidate_by_id) - before_records),
                })
            timings["evidence_ms"] = timings.get("evidence_ms", 0.0) + (
                time.perf_counter() - t0
            ) * 1000

            t0 = time.perf_counter()
            counts["record_ann_calls"] += 1
            before_records = len(candidate_by_id)
            record_hits = 0
            for item in self._direct_record_search(
                query=variant,
                top_k=record_limit,
                query_vector=variant_vector,
                record_cache=record_cache,
            ):
                record_hits += 1
                record_id = str(item.get("id") or "").strip()
                if not record_id:
                    continue
                retrieval_score = self._distance_to_retrieval_score(
                    self._safe_float(item.get("semantic_distance"), 1.0)
                )
                existing = candidate_by_id.get(record_id)
                if existing is None:
                    item = dict(item)
                    item["field_score"] = 0.30 + 0.30 * retrieval_score
                    item["matched_queries"] = [variant]
                    item["matched_channels"] = list(item.get("matched_channels") or ["record_ann"])
                    item["retrieval_score"] = retrieval_score
                    candidate_by_id[record_id] = item
                else:
                    existing["field_score"] = min(
                        1.0,
                        self._safe_float(existing.get("field_score"), 0.0)
                        + 0.20
                        + 0.20 * retrieval_score,
                    )
                    existing["semantic_distance"] = min(
                        self._safe_float(existing.get("semantic_distance"), 1.0),
                        self._safe_float(item.get("semantic_distance"), 1.0),
                    )
                    self._append_unique(existing, "matched_queries", variant)
                    for matched_channel in item.get("matched_channels") or ["record_ann"]:
                        self._append_unique(existing, "matched_channels", matched_channel)
            if variant_trace is not None:
                variant_trace.update({
                    "record_ann_hits": record_hits,
                    "records_after_record_ann": len(candidate_by_id),
                    "records_added_by_record_ann": max(0, len(candidate_by_id) - before_records),
                })
            timings["record_ms"] = timings.get("record_ms", 0.0) + (
                time.perf_counter() - t0
            ) * 1000

            t0 = time.perf_counter()
            counts["turn_ann_calls"] += 1
            before_turns = len(raw_turn_by_id)
            raw_turn_hits = 0
            for item in self._search_raw_turns_direct(
                query=variant,
                top_k=turn_limit,
                query_vector=variant_vector,
            ):
                raw_turn_hits += 1
                episode_id = str(item.get("id") or "").strip()
                if not episode_id:
                    continue
                retrieval_score = self._distance_to_retrieval_score(
                    self._safe_float(item.get("semantic_distance"), 1.0)
                )
                current = raw_turn_by_id.get(episode_id)
                if current is None:
                    item = dict(item)
                    item["field_score"] = self._raw_turn_field_score(
                        retrieval_score,
                        item,
                        strategy,
                    )
                    item["matched_queries"] = [variant]
                    item["matched_channels"] = ["raw_turn_direct"]
                    item["retrieval_score"] = retrieval_score
                    raw_turn_by_id[episode_id] = item
                else:
                    current["field_score"] = min(
                        1.0,
                        max(
                            self._safe_float(current.get("field_score"), 0.0),
                            self._raw_turn_field_score(retrieval_score, item, strategy),
                        )
                        + 0.08,
                    )
                    self._append_unique(current, "matched_queries", variant)
                    current["semantic_distance"] = min(
                        self._safe_float(current.get("semantic_distance"), 1.0),
                        self._safe_float(item.get("semantic_distance"), 1.0),
                    )
            if variant_trace is not None:
                variant_trace.update({
                    "raw_turn_hits": raw_turn_hits,
                    "raw_turns_after_ann": len(raw_turn_by_id),
                    "raw_turns_added_by_ann": max(0, len(raw_turn_by_id) - before_turns),
                })
                trace["query_steps"].append(variant_trace)
            timings["raw_turn_ms"] = timings.get("raw_turn_ms", 0.0) + (
                time.perf_counter() - t0
            ) * 1000

        t0 = time.perf_counter()
        combined_candidates = sorted(
            list(candidate_by_id.values()) + list(raw_turn_by_id.values()),
            key=lambda item: self._safe_float(item.get("field_score"), 0.0),
            reverse=True,
        )
        combined_candidates = self._apply_strategy_candidate_boosts(
            combined_candidates,
            strategy,
        )
        combined_candidates = self._apply_temporal_boost(combined_candidates, temporal_filter)
        timings["merge_boost_ms"] = (time.perf_counter() - t0) * 1000
        counts["record_candidates"] = len(candidate_by_id)
        counts["raw_turn_candidates"] = len(raw_turn_by_id)
        counts["combined_candidates"] = len(combined_candidates)
        if trace is not None:
            trace["retrieved_before_rerank"] = {
                "record_count": len(candidate_by_id),
                "raw_turn_count": len(raw_turn_by_id),
                "combined_count": len(combined_candidates),
                "records_top": self._debug_candidate_payload(
                    sorted(
                        candidate_by_id.values(),
                        key=lambda item: self._safe_float(item.get("field_score"), 0.0),
                        reverse=True,
                    )[:trace_candidate_limit],
                    max_text=trace_max_text,
                ),
                "raw_turns_top": self._debug_candidate_payload(
                    sorted(
                        raw_turn_by_id.values(),
                        key=lambda item: self._safe_float(item.get("field_score"), 0.0),
                        reverse=True,
                    )[:trace_candidate_limit],
                    max_text=trace_max_text,
                ),
                "combined_top": self._debug_candidate_payload(
                    combined_candidates[:trace_candidate_limit],
                    max_text=trace_max_text,
                ),
            }

        t0 = time.perf_counter()
        anchor_candidates = combined_candidates
        assistant_windows = self._build_assistant_answer_windows(
            anchor_candidates,
            strategy=strategy,
            route=route,
        )
        if assistant_windows:
            anchor_candidates = self._merge_candidate_dicts(
                anchor_candidates,
                assistant_windows,
            )
            anchor_candidates = sorted(
                anchor_candidates,
                key=lambda item: (
                    self._safe_float(item.get("field_score"), 0.0),
                    self._safe_float(item.get("retrieval_score"), 0.0),
                ),
                reverse=True,
            )
        source_windows = self._materialize_source_windows_from_anchors(
            anchor_candidates,
            strategy=strategy,
            route=route,
            top_k=top_k,
        )
        if source_windows:
            combined_candidates = self._merge_candidate_dicts(
                anchor_candidates,
                source_windows,
            )
            combined_candidates = sorted(
                combined_candidates,
                key=lambda item: (
                    self._safe_float(item.get("field_score"), 0.0),
                    self._safe_float(item.get("retrieval_score"), 0.0),
                ),
                reverse=True,
            )
        else:
            combined_candidates = anchor_candidates
        timings["source_window_ms"] = (time.perf_counter() - t0) * 1000
        counts["anchor_candidates"] = len(anchor_candidates)
        counts["assistant_answer_window_anchor_candidates"] = len(assistant_windows)
        counts["source_window_candidates"] = len(source_windows)
        if trace is not None:
            trace["source_windows_before_rerank"] = {
                "anchor_count": len(anchor_candidates),
                "assistant_window_anchor_count": len(assistant_windows),
                "source_window_count": len(source_windows),
                "fallback_to_anchors": not bool(source_windows),
                "anchors_top": self._debug_candidate_payload(
                    anchor_candidates[:trace_candidate_limit],
                    max_text=trace_max_text,
                ),
                "source_windows_top": self._debug_candidate_payload(
                    source_windows[:trace_candidate_limit],
                    max_text=trace_max_text,
                ),
            }

        t0 = time.perf_counter()
        rerank_trace: dict[str, Any] | None = {} if trace is not None else None
        if self._reranker is not None and combined_candidates:
            route_plan = self._route_plan(plan, route)
            combined_candidates = self._rerank_candidate_dicts(
                query=self._route_rerank_query(query, route),
                plan=route_plan,
                candidates=combined_candidates,
                limit=max(self._reranker_candidate_limit, top_k * 4),
                distance_key="semantic_distance",
                trace_debug=rerank_trace,
                trace_candidate_limit=trace_candidate_limit,
                trace_max_text=trace_max_text,
            )
        elif rerank_trace is not None:
            rerank_trace.update({
                "enabled": self._reranker is not None,
                "input_count": len(combined_candidates),
                "skipped_reason": "no_reranker" if self._reranker is None else "no_candidates",
            })
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        if trace is not None:
            trace["rerank"] = rerank_trace or {}

        counts["after_rerank_candidates"] = len(combined_candidates)
        if trace is not None:
            trace["after_source_window_rerank"] = self._debug_candidate_payload(
                combined_candidates[:trace_candidate_limit],
                max_text=trace_max_text,
            )

        scored = self._build_scored_candidates(combined_candidates)
        for rank, candidate in enumerate(scored, 1):
            data = dict(candidate.data)
            route_info = self._route_match_payload(route, rank)
            data["primary_route_id"] = route_info["route_id"]
            data["primary_route_goal"] = route_info["evidence_goal"]
            data["route_rank"] = rank
            data["matched_routes"] = [route_info]
            candidate.data = data
        if trace is not None:
            trace["selected_by_route"] = self._debug_scored_payload(
                scored[:trace_candidate_limit],
                max_text=trace_max_text,
            )
            trace["timings_ms"] = {key: round(value, 1) for key, value in timings.items()}
            trace["counts"] = counts
        return {
            "route": route,
            "ranked": scored,
            "timings": timings,
            "counts": counts,
            "debug": trace,
        }

    @staticmethod
    def _log_search_timing(
        *,
        query: str,
        timings: dict[str, float],
        counts: dict[str, int],
    ) -> None:
        total_ms = float(timings.get("total_ms", 0.0) or 0.0)
        if total_ms < 1000.0 and os.getenv("LYCHEE_SEMANTIC_TIMING", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        rounded = {key: round(value, 1) for key, value in timings.items()}
        logging.getLogger(__name__).warning(
            "semantic_search_timing query=%r timings_ms=%s counts=%s",
            str(query or "")[:120],
            rounded,
            counts,
        )


