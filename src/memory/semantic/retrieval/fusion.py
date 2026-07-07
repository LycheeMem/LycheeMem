"""Route result scoring and fusion helpers."""

from __future__ import annotations

from typing import Any

from src.memory.semantic.models import EvidenceRoute
from src.memory.semantic.scorer import ScoredCandidate


class RetrievalFusionMixin:
    def _build_scored_candidates(self, candidates: list[dict[str, Any]]) -> list[ScoredCandidate]:
        scored: list[ScoredCandidate] = []
        seen_ids: set[str] = set()
        for item in candidates:
            candidate_id = self._candidate_id(item)
            if not candidate_id or candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            source = str(item.get("source") or "record")
            if self._reranker is not None and "rerank_score" in item:
                final_score = self._final_score_from_rerank(item, source, bonus=0.01)
            else:
                final_score = min(1.0, self._safe_float(item.get("field_score"), 0.0))
            scored.append(ScoredCandidate(
                id=candidate_id,
                source=source,
                final_score=final_score,
                score_breakdown={
                    "field_score": item.get("field_score", 0.0),
                    "matched_channels": item.get("matched_channels", []),
                    "rerank_score": item.get("rerank_score", ""),
                    "cross_encoder_score": item.get("cross_encoder_score", ""),
                },
                data=dict(item),
            ))
        scored.sort(key=lambda c: c.final_score, reverse=True)
        return scored

    def _fuse_route_results(
        self,
        route_results: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[ScoredCandidate], list[ScoredCandidate]]:
        """Fuse route-local rankings without erasing route coverage."""
        if top_k <= 0 or not route_results:
            return [], []

        route_count = max(1, len(route_results))
        route_quota = max(1, min(4, top_k // route_count if route_count else top_k))
        fused: dict[str, dict[str, Any]] = {}

        for result in route_results:
            route = result.get("route")
            ranked: list[ScoredCandidate] = list(result.get("ranked") or [])
            for rank, candidate in enumerate(ranked[: max(top_k * 4, route_quota * 4)], 1):
                entry = fused.setdefault(candidate.id, {
                    "candidate": candidate,
                    "best_score": 0.0,
                    "rrf": 0.0,
                    "routes": [],
                    "best_rank": rank,
                })
                if candidate.final_score > entry["best_score"]:
                    entry["candidate"] = candidate
                    entry["best_score"] = candidate.final_score
                entry["rrf"] += 1.0 / (60.0 + rank)
                entry["best_rank"] = min(int(entry["best_rank"]), rank)
                entry["routes"].append(self._route_match_payload(route, rank))

        max_rrf = max((float(entry["rrf"]) for entry in fused.values()), default=1.0)
        all_candidates: list[ScoredCandidate] = []
        for entry in fused.values():
            base = entry["candidate"]
            data = dict(base.data)
            data["matched_routes"] = self._dedupe_route_matches(entry["routes"])
            if data["matched_routes"]:
                data["primary_route_id"] = data["matched_routes"][0]["route_id"]
                data["primary_route_goal"] = data["matched_routes"][0]["evidence_goal"]
            route_fusion_score = float(entry["rrf"]) / max_rrf if max_rrf else 0.0
            final_score = max(
                0.0,
                min(1.0, 0.82 * float(entry["best_score"]) + 0.18 * route_fusion_score),
            )
            breakdown = dict(base.score_breakdown)
            breakdown["route_fusion_score"] = route_fusion_score
            breakdown["best_route_rank"] = entry["best_rank"]
            all_candidates.append(ScoredCandidate(
                id=base.id,
                source=base.source,
                final_score=final_score,
                score_breakdown=breakdown,
                data=data,
            ))

        by_id = {candidate.id: candidate for candidate in all_candidates}
        selected: list[ScoredCandidate] = []
        selected_ids: set[str] = set()

        def add(candidate: ScoredCandidate) -> None:
            if candidate.id in selected_ids or len(selected) >= top_k:
                return
            selected_ids.add(candidate.id)
            selected.append(candidate)

        for result in route_results:
            kept = 0
            for candidate in result.get("ranked") or []:
                fused_candidate = by_id.get(candidate.id)
                if fused_candidate is None or fused_candidate.id in selected_ids:
                    continue
                add(fused_candidate)
                kept += 1
                if kept >= route_quota or len(selected) >= top_k:
                    break

        for candidate in sorted(all_candidates, key=lambda c: c.final_score, reverse=True):
            if len(selected) >= top_k:
                break
            add(candidate)

        selected = self._coverage_select_scored(selected, top_k)
        return selected, sorted(all_candidates, key=lambda c: c.final_score, reverse=True)

    @staticmethod
    def _route_match_payload(route: EvidenceRoute | None, rank: int) -> dict[str, Any]:
        if route is None:
            return {
                "route_id": "default",
                "evidence_goal": "",
                "rank": int(rank),
            }
        return {
            "route_id": str(route.route_id or "default"),
            "evidence_goal": str(route.evidence_goal or ""),
            "rank": int(rank),
        }

    @staticmethod
    def _dedupe_route_matches(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in routes:
            key = str(item.get("route_id") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item)
        result.sort(key=lambda item: int(item.get("rank") or 999999))
        return result


