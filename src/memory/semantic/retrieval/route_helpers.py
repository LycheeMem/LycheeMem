"""Route query and rerank prompt helpers."""

from __future__ import annotations

from typing import Any

from src.memory.semantic.models import EvidenceRoute, SearchPlan


class RetrievalRouteHelperMixin:
    def _route_plan(self, plan: SearchPlan, route: EvidenceRoute) -> SearchPlan:
        return SearchPlan(
            semantic_queries=self._route_query_variants("", route),
            temporal_filter=route.temporal_filter or plan.temporal_filter,
            depth=plan.depth,
            question_type=plan.question_type,
            evidence_target=route.evidence_goal or plan.evidence_target,
            evidence_constraints=self._constraint_texts(route.constraints) or list(plan.evidence_constraints),
            constraints=list(route.constraints or plan.constraints or []),
            evidence_routes=[route],
            reasoning=plan.reasoning,
        )

    @classmethod
    def _route_rerank_query(cls, query: str, route: EvidenceRoute) -> str:
        parts = [f"Question: {str(query or '').strip()}"]
        if route.evidence_goal:
            parts.append(f"Evidence goal: {route.evidence_goal}")
        constraint_texts = cls._constraint_texts(route.constraints)
        if constraint_texts:
            parts.append("Required constraints: " + " | ".join(constraint_texts))
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _constraint_texts(constraints: list[dict[str, Any]] | None) -> list[str]:
        texts: list[str] = []
        for item in constraints or []:
            if not isinstance(item, dict):
                text = str(item or "").strip()
                if text:
                    texts.append(text)
                continue
            value = str(item.get("value") or "").strip()
            if not value:
                continue
            kind = str(item.get("kind") or "other").strip()
            prefix = f"{kind}: " if kind else ""
            texts.append(f"{prefix}{value}")
        return texts

    @classmethod
    def _route_query_variants(cls, query: str, route: EvidenceRoute) -> list[str]:
        secondary: list[str] = []
        if route.evidence_goal:
            secondary.append(route.evidence_goal)
        secondary.extend(route.queries or [])
        secondary.extend(cls._constraint_texts(route.constraints))
        if not secondary and query:
            secondary.append(query)
        return cls._merge_unique([], secondary)[: cls.MAX_ROUTE_QUERIES + 2]


