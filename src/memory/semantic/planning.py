"""Retrieval plan normalization helpers for CompactSemanticEngine."""

from __future__ import annotations

from typing import Any

from src.memory.semantic.models import EvidenceRoute, SearchPlan


class RetrievalPlanningMixin:
    def _dict_to_plan(self, d: dict[str, Any]) -> SearchPlan:
        """dict → SearchPlan。"""
        temporal_filter = self._clean_temporal_filter(d.get("temporal_filter"))
        return SearchPlan(
            semantic_queries=[
                str(x) for x in (d.get("semantic_queries") or []) if str(x or "").strip()
            ][: self.MAX_SEMANTIC_QUERIES],
            pragmatic_queries=[
                str(x) for x in (d.get("pragmatic_queries") or []) if str(x or "").strip()
            ][: self.MAX_SEMANTIC_QUERIES],
            mode=self._clean_plan_choice(
                d.get("mode"),
                {"answer", "action", "mixed"},
                "answer",
            ),
            temporal_filter=temporal_filter,
            depth=self._depth_for_plan(),
            question_type=self._clean_question_type(
                d.get("question_type")
                or ("aggregate" if d.get("is_aggregate_query") else "")
                or d.get("kind")
            ),
            evidence_target=str(
                d.get("evidence_target") or d.get("aggregate_target") or ""
            ),
            evidence_constraints=[
                str(x)
                for x in (
                    d.get("evidence_constraints")
                    or d.get("aggregate_constraints")
                    or []
                )
                if str(x or "").strip()
            ],
            constraints=self._clean_constraints(d.get("constraints")),
            evidence_routes=self._clean_evidence_routes(d.get("evidence_routes")),
            reasoning=str(d.get("reason") or d.get("reasoning") or ""),
        )

    @staticmethod
    def _plan_to_dict(plan: SearchPlan) -> dict[str, Any]:
        return {
            "reason": plan.reasoning,
            "mode": plan.mode,
            "question_type": plan.question_type,
            "semantic_queries": list(plan.semantic_queries),
            "pragmatic_queries": list(plan.pragmatic_queries),
            "temporal_filter": dict(plan.temporal_filter or {}),
            "evidence_target": plan.evidence_target,
            "evidence_constraints": list(plan.evidence_constraints),
            "constraints": list(plan.constraints),
            "evidence_routes": [
                {
                    "route_id": route.route_id,
                    "evidence_goal": route.evidence_goal,
                    "queries": list(route.queries),
                    "constraints": list(route.constraints),
                    "temporal_filter": dict(route.temporal_filter or {}),
                }
                for route in plan.evidence_routes
            ],
        }

    @staticmethod
    def _depth_for_plan() -> int:
        return 15

    @classmethod
    def _normalize_plan_for_query(cls, query: str, plan: SearchPlan) -> SearchPlan:
        """Apply execution settings from the LLM plan without rule-based re-planning."""
        plan.depth = cls._depth_for_plan()
        if not plan.semantic_queries:
            plan.semantic_queries = [str(query or "").strip()]
        if not plan.evidence_target:
            plan.evidence_target = str(query or "").strip()
        if not plan.evidence_routes:
            plan.evidence_routes = [
                EvidenceRoute(
                    route_id="r1",
                    evidence_goal=plan.evidence_target or str(query or "").strip(),
                    queries=cls._build_plan_query_variants(query, plan),
                    constraints=list(plan.constraints),
                    temporal_filter=plan.temporal_filter,
                )
            ]
        for index, route in enumerate(plan.evidence_routes, 1):
            if not route.route_id:
                route.route_id = f"r{index}"
            if not route.evidence_goal:
                route.evidence_goal = plan.evidence_target or str(query or "").strip()
            if not route.queries:
                route.queries = cls._build_plan_query_variants(query, plan)
            if route.temporal_filter is None and plan.temporal_filter:
                route.temporal_filter = dict(plan.temporal_filter)
        return plan

    def _build_evidence_routes(self, query: str, plan: SearchPlan) -> list[EvidenceRoute]:
        plan = self._normalize_plan_for_query(query, plan)
        return list(plan.evidence_routes or [])

    def _clean_evidence_routes(self, raw: Any) -> list[EvidenceRoute]:
        if not isinstance(raw, list):
            return []
        routes: list[EvidenceRoute] = []
        for index, item in enumerate(raw[: self.MAX_EVIDENCE_ROUTES], 1):
            if not isinstance(item, dict):
                continue
            queries = [
                str(x) for x in (item.get("queries") or []) if str(x or "").strip()
            ][: self.MAX_ROUTE_QUERIES]
            evidence_goal = str(
                item.get("evidence_goal")
                or item.get("objective")
                or item.get("goal")
                or ""
            ).strip()
            route = EvidenceRoute(
                route_id=str(item.get("route_id") or f"r{index}").strip(),
                evidence_goal=evidence_goal,
                queries=queries,
                constraints=self._clean_constraints(item.get("constraints")),
                temporal_filter=self._clean_temporal_filter(item.get("temporal_filter")),
            )
            if route.evidence_goal or route.queries:
                routes.append(route)
        return routes

    @classmethod
    def _clean_question_type(cls, value: Any) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        return cls._clean_plan_choice(
            text,
            {
                "single",
                "aggregate",
                "temporal",
                "comparison",
                "personalized_advice",
                "prior_assistant_response",
                "other",
            },
            "single",
        )

    @staticmethod
    def _clean_plan_choice(value: Any, allowed: set[str], default: str) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        return text if text in allowed else default

    @classmethod
    def _clean_temporal_filter(cls, raw: Any) -> dict[str, str] | None:
        if not isinstance(raw, dict):
            return None
        cleaned = {
            key: cls._date_key(raw.get(key))
            for key in ("since", "until")
            if cls._date_key(raw.get(key))
        }
        return cleaned or None

    @classmethod
    def _clean_constraints(cls, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        constraints: list[dict[str, Any]] = []
        for item in raw[: cls.MAX_CONSTRAINTS]:
            if isinstance(item, dict):
                value = str(item.get("value") or item.get("text") or "").strip()
                if not value:
                    continue
                constraints.append({
                    "kind": str(item.get("kind") or item.get("type") or "other").strip() or "other",
                    "value": value,
                })
            else:
                value = str(item or "").strip()
                if value:
                    constraints.append({"kind": "other", "value": value})
        return constraints

    @classmethod
    def _build_plan_query_variants(cls, query: str, plan: SearchPlan) -> list[str]:
        secondary = list(plan.semantic_queries or [])
        if plan.mode in {"action", "mixed"}:
            secondary.extend(plan.pragmatic_queries or [])
        secondary.extend([plan.evidence_target])
        secondary.extend(plan.evidence_constraints or [])
        return cls._merge_unique([query], secondary)

    @staticmethod
    def _merge_unique(primary: list[str], secondary: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for value in list(primary or []) + list(secondary or []):
            item = str(value or "").strip()
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged


