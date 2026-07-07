"""Rerank, coverage selection, and candidate scoring helpers."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.memory.semantic.models import SearchPlan
from src.memory.semantic.reranker import RerankCandidate
from src.memory.semantic.retrieval.strategy import RetrievalStrategy
from src.memory.semantic.scorer import ScoredCandidate


class RetrievalSelectionMixin:
    def _rerank_candidate_dicts(
        self,
        *,
        query: str,
        plan: SearchPlan,
        candidates: list[dict[str, Any]],
        limit: int,
        distance_key: str = "semantic_distance",
        trace_debug: dict[str, Any] | None = None,
        trace_candidate_limit: int = 8,
        trace_max_text: int = 800,
    ) -> list[dict[str, Any]]:
        """Rerank generic candidate dictionaries with the configured reranker."""
        if not candidates or self._reranker is None:
            if trace_debug is not None:
                trace_debug.update({
                    "enabled": self._reranker is not None,
                    "input_count": len(candidates),
                    "skipped_reason": "no_candidates" if not candidates else "no_reranker",
                })
            return candidates[:limit]

        cap = max(limit, 1)
        to_rerank = self._select_rerank_pool(candidates, cap)
        if trace_debug is not None:
            trace_debug.update({
                "enabled": True,
                "query": query,
                "limit": limit,
                "input_count": len(candidates),
                "pool_count": len(to_rerank),
                "pool_top": self._debug_candidate_payload(
                    to_rerank[:trace_candidate_limit],
                    max_text=trace_max_text,
                ),
                "plan": self._plan_to_dict(plan),
            })
        rerank_inputs: list[RerankCandidate] = []
        for item in to_rerank:
            cid = self._candidate_id(item)
            if not cid:
                continue
            distance = self._safe_float(
                item.get(distance_key, item.get("_distance", 1.0)),
                1.0,
            )
            rerank_inputs.append(
                RerankCandidate(
                    id=cid,
                    source=str(item.get("source") or "record"),
                    text=self._candidate_text(item),
                    metadata=item,
                    retrieval_score=self._distance_to_retrieval_score(distance),
                )
            )

        try:
            results = self._reranker.rerank(
                query=query,
                retrieval_plan=self._plan_to_dict(plan),
                candidates=rerank_inputs,
            )
        except Exception as exc:
            logging.getLogger(__name__).warning("semantic reranker failed: %s", exc)
            if trace_debug is not None:
                trace_debug.update({
                    "failed": True,
                    "error": repr(exc),
                })
            self._reranker = None
            return candidates[:limit]

        score_by_id = {result.id: result for result in results}
        if trace_debug is not None:
            trace_debug["result_scores_top"] = [
                {
                    "id": result.id,
                    "rerank_score": result.rerank_score,
                    "cross_encoder_score": result.cross_encoder_score,
                }
                for result in sorted(
                    results,
                    key=lambda item: item.rerank_score,
                    reverse=True,
                )[:trace_candidate_limit]
            ]
        ranked: list[dict[str, Any]] = []
        for item in to_rerank:
            cid = self._candidate_id(item)
            if not cid:
                continue
            enriched = dict(item)
            distance = self._safe_float(
                enriched.get(distance_key, enriched.get("_distance", 1.0)), 1.0
            )
            enriched["retrieval_score"] = self._distance_to_retrieval_score(distance)
            result = score_by_id.get(cid)
            if result is not None:
                enriched["rerank_score"] = result.rerank_score
                enriched["cross_encoder_score"] = result.cross_encoder_score
            ranked.append(enriched)

        ranked.sort(
            key=lambda item: (
                self._safe_float(item.get("rerank_score"), 0.0),
                self._safe_float(item.get("retrieval_score"), 0.0),
            ),
            reverse=True,
        )
        if trace_debug is not None:
            trace_debug["ranked_top"] = self._debug_candidate_payload(
                ranked[:trace_candidate_limit],
                max_text=trace_max_text,
            )
        return ranked[:limit]

    def _select_rerank_pool(
        self,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Build a rerank pool without letting one high field-score channel dominate."""
        if limit <= 0 or not candidates:
            return []
        if len(candidates) <= limit:
            return list(candidates)

        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def add(item: dict[str, Any]) -> bool:
            cid = self._candidate_id(item)
            if not cid or cid in selected_ids:
                return False
            selected_ids.add(cid)
            selected.append(item)
            return len(selected) >= limit

        global_quota = min(len(candidates), max(limit // 2, min(40, limit)))
        for item in candidates[:global_quota]:
            if add(item):
                return selected

        buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in candidates:
            source = str(item.get("source") or "record")
            channels = item.get("matched_channels") or ["unknown"]
            if not isinstance(channels, list):
                channels = [channels]
            for channel in channels:
                key = (source, str(channel or "unknown"))
                buckets.setdefault(key, []).append(item)

        positions = {key: 0 for key in buckets}
        bucket_keys = sorted(buckets)
        while bucket_keys and len(selected) < limit:
            progressed = False
            for key in list(bucket_keys):
                items = buckets[key]
                pos = positions[key]
                while pos < len(items) and self._candidate_id(items[pos]) in selected_ids:
                    pos += 1
                positions[key] = pos
                if pos >= len(items):
                    bucket_keys.remove(key)
                    continue
                if add(items[pos]):
                    return selected
                positions[key] = pos + 1
                progressed = True
            if not progressed:
                break

        for item in candidates:
            if add(item):
                break
        return selected

    def _coverage_select_dicts(
        self,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """MMR-style aggregate selection over generic diversity signals."""
        if limit <= 0 or not candidates:
            return []
        pool = candidates[: max(limit, min(len(candidates), limit * 4))]
        signatures = [self._candidate_signature(item) for item in pool]
        selected: list[dict[str, Any]] = []
        selected_indices: list[int] = []
        remaining = list(range(len(pool)))

        while remaining and len(selected) < limit:
            best_pos = 0
            best_score = float("-inf")
            for pos, index in enumerate(remaining):
                item = pool[index]
                relevance = self._safe_float(item.get("rerank_score"), 0.0)
                if not relevance:
                    relevance = self._safe_float(item.get("retrieval_score"), 0.0)
                max_similarity = max(
                    (
                        self._signature_similarity(signatures[index], signatures[chosen_index])
                        for chosen_index in selected_indices
                    ),
                    default=0.0,
                )
                score = 0.75 * relevance - 0.25 * max_similarity
                if score > best_score:
                    best_score = score
                    best_pos = pos
            chosen = remaining.pop(best_pos)
            selected_indices.append(chosen)
            selected.append(pool[chosen])
        return selected

    def _coverage_select_scored(
        self,
        candidates: list[ScoredCandidate],
        limit: int,
    ) -> list[ScoredCandidate]:
        if limit <= 0 or not candidates:
            return []
        ordered = sorted(candidates, key=lambda c: c.final_score, reverse=True)
        pool = ordered[: max(limit, min(len(ordered), limit * 4))]
        signatures = [self._candidate_signature(item.data) for item in pool]
        selected: list[ScoredCandidate] = []
        selected_indices: list[int] = []
        remaining = list(range(len(pool)))

        while remaining and len(selected) < limit:
            best_pos = 0
            best_score = float("-inf")
            for pos, index in enumerate(remaining):
                item = pool[index]
                max_similarity = max(
                    (
                        self._signature_similarity(signatures[index], signatures[chosen_index])
                        for chosen_index in selected_indices
                    ),
                    default=0.0,
                )
                score = 0.75 * item.final_score - 0.25 * max_similarity
                if score > best_score:
                    best_score = score
                    best_pos = pos
            chosen = remaining.pop(best_pos)
            selected_indices.append(chosen)
            selected.append(pool[chosen])
        return selected

    @classmethod
    def _candidate_signature(cls, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "session": cls._as_text(item.get("source_session")),
            "entities": cls._as_string_set(item.get("entities")),
            "queries": cls._as_string_set(item.get("matched_queries")),
            "evidence": cls._evidence_node_set(item.get("matched_evidence_nodes")),
            "turns": cls._turn_set(item.get("evidence_turn_range")),
            "tokens": cls._token_set(cls._candidate_text(item)),
        }

    @classmethod
    def _signature_similarity(cls, left: dict[str, Any], right: dict[str, Any]) -> float:
        left_session = cls._as_text(left.get("session"))
        right_session = cls._as_text(right.get("session"))
        session_sim = 1.0 if left_session and left_session == right_session else 0.0
        entity_sim = cls._jaccard(left.get("entities") or set(), right.get("entities") or set())
        query_sim = cls._jaccard(left.get("queries") or set(), right.get("queries") or set())
        evidence_sim = cls._jaccard(left.get("evidence") or set(), right.get("evidence") or set())
        turn_sim = cls._jaccard(left.get("turns") or set(), right.get("turns") or set())
        text_sim = cls._jaccard(left.get("tokens") or set(), right.get("tokens") or set())
        return min(
            1.0,
            0.15 * session_sim
            + 0.20 * entity_sim
            + 0.15 * query_sim
            + 0.15 * evidence_sim
            + 0.10 * turn_sim
            + 0.25 * text_sim,
        )

    @staticmethod
    def _limit_episode_candidates(
        selected: list[ScoredCandidate],
        candidates: list[ScoredCandidate],
        limit: int,
        *,
        strategy: RetrievalStrategy,
    ) -> list[ScoredCandidate]:
        """Allow raw dialogue only when it is strong supporting evidence."""
        if limit <= 0 or not candidates:
            return selected[:limit]
        if not any(candidate.source == "episode" for candidate in selected):
            return selected[:limit]

        record_scores = [
            candidate.final_score
            for candidate in candidates
            if candidate.source != "episode"
        ]
        best_record_score = max(record_scores, default=0.0)
        if best_record_score <= 0.0:
            return selected[:limit]

        episode_route_ids = {
            str(route.get("route_id") or "")
            for candidate in selected
            if candidate.source == "episode"
            for route in (candidate.data.get("matched_routes") or [])
            if isinstance(route, dict) and str(route.get("route_id") or "").strip()
        }
        episode_budget = max(
            1,
            min(
                limit,
                max(
                    2,
                    int(round(limit * max(0.0, min(1.0, strategy.episode_budget_ratio)))),
                    len(episode_route_ids),
                ),
            ),
        )
        min_episode_score = max(
            0.35,
            best_record_score * max(0.0, min(1.0, strategy.min_episode_score_ratio)),
        )
        non_episode_route_ids = {
            str(route.get("route_id") or "")
            for candidate in selected
            if candidate.source != "episode"
            for route in (candidate.data.get("matched_routes") or [])
            if isinstance(route, dict) and str(route.get("route_id") or "").strip()
        }

        kept: list[ScoredCandidate] = []
        episode_count = 0
        for candidate in selected:
            if candidate.source != "episode":
                kept.append(candidate)
                continue
            candidate_route_ids = {
                str(route.get("route_id") or "")
                for route in (candidate.data.get("matched_routes") or [])
                if isinstance(route, dict) and str(route.get("route_id") or "").strip()
            }
            route_needs_episode = bool(candidate_route_ids - non_episode_route_ids)
            if route_needs_episode:
                kept.append(candidate)
                episode_count += 1
                continue
            if (
                episode_count < episode_budget
                and candidate.final_score >= min_episode_score
            ):
                kept.append(candidate)
                episode_count += 1

        if len(kept) >= limit:
            return sorted(kept, key=lambda c: c.final_score, reverse=True)[:limit]

        selected_ids = {candidate.id for candidate in kept}
        for candidate in sorted(candidates, key=lambda c: c.final_score, reverse=True):
            if len(kept) >= limit:
                break
            if candidate.id in selected_ids or candidate.source == "episode":
                continue
            kept.append(candidate)
            selected_ids.add(candidate.id)
        return sorted(kept, key=lambda c: c.final_score, reverse=True)[:limit]

    @classmethod
    def _candidate_similarity(cls, left: dict[str, Any], right: dict[str, Any]) -> float:
        left_session = cls._as_text(left.get("source_session"))
        right_session = cls._as_text(right.get("source_session"))
        session_sim = 1.0 if left_session and left_session == right_session else 0.0
        entity_sim = cls._jaccard(
            cls._as_string_set(left.get("entities")),
            cls._as_string_set(right.get("entities")),
        )
        query_sim = cls._jaccard(
            cls._as_string_set(left.get("matched_queries")),
            cls._as_string_set(right.get("matched_queries")),
        )
        evidence_sim = cls._jaccard(
            cls._evidence_node_set(left.get("matched_evidence_nodes")),
            cls._evidence_node_set(right.get("matched_evidence_nodes")),
        )
        turn_sim = cls._jaccard(
            cls._turn_set(left.get("evidence_turn_range")),
            cls._turn_set(right.get("evidence_turn_range")),
        )
        text_sim = cls._jaccard(
            cls._token_set(cls._candidate_text(left)),
            cls._token_set(cls._candidate_text(right)),
        )
        return min(
            1.0,
            0.15 * session_sim
            + 0.20 * entity_sim
            + 0.15 * query_sim
            + 0.15 * evidence_sim
            + 0.10 * turn_sim
            + 0.25 * text_sim,
        )

    @staticmethod
    def _candidate_id(item: dict[str, Any]) -> str:
        return str(
            item.get("id")
            or item.get("record_id")
            or item.get("episode_id")
            or ""
        ).strip()

    @staticmethod
    def _candidate_text(item: dict[str, Any]) -> str:
        return str(
            item.get("display_text")
            or item.get("semantic_text")
            or item.get("normalized_text")
            or ""
        ).strip()

    @staticmethod
    def _distance_to_retrieval_score(distance: float) -> float:
        if distance != distance:
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(distance)))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _final_score_from_rerank(
        cls,
        data: dict[str, Any],
        source: str,
        *,
        bonus: float = 0.0,
    ) -> float:
        rerank_score = cls._safe_float(
            data.get("rerank_score"),
            cls._distance_to_retrieval_score(
                cls._safe_float(data.get("semantic_distance"), 1.0)
            ),
        )
        source_weight = cls._safe_float(
            data.get("source_weight_override"),
            cls._source_weight(source),
        )
        final = (
            rerank_score * source_weight
            + cls._retrieval_signal_bonus(data)
            + bonus
        )
        return max(0.0, min(1.0, final))

    @staticmethod
    def _source_weight(source: str) -> float:
        source_key = str(source or "").strip().casefold()
        if source_key == "record":
            return 1.0
        if source_key == "episode":
            return 0.88
        return 0.95

    @classmethod
    def _retrieval_signal_bonus(cls, data: dict[str, Any]) -> float:
        matched_count = len(cls._as_string_set(data.get("matched_queries")))
        distance = cls._safe_float(
            data.get("semantic_distance"),
            cls._safe_float(data.get("_distance"), 1.0),
        )
        distance_bonus = 0.03 * cls._distance_to_retrieval_score(distance)
        query_bonus = min(0.03, 0.01 * matched_count)
        return distance_bonus + query_bonus

    @staticmethod
    def _as_text(value: Any) -> str:
        return str(value or "").strip()

    @classmethod
    def _as_string_set(cls, value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            values = [value]
        return {cls._as_text(item).casefold() for item in values if cls._as_text(item)}

    @classmethod
    def _evidence_node_set(cls, value: Any) -> set[str]:
        if value is None:
            return set()
        items = value if isinstance(value, list) else [value]
        result: set[str] = set()
        for item in items:
            if isinstance(item, dict):
                node_id = cls._as_text(item.get("node_id")).casefold()
                if node_id:
                    result.add(node_id)
            else:
                text = cls._as_text(item).casefold()
                if text:
                    result.add(text)
        return result

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[\w]+", str(text or "").casefold())
            if token
        }

    @classmethod
    def _turn_set(cls, value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            return {cls._as_text(item) for item in value if cls._as_text(item)}
        return {cls._as_text(value)} if cls._as_text(value) else set()

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, len(left | right))


