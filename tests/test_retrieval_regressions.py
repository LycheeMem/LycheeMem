from __future__ import annotations

from typing import Any

from src.memory.semantic.models import EvidenceRoute, MemoryRecord, SearchPlan
from src.memory.semantic.retrieval.candidate_utils import RetrievalCandidateUtilsMixin
from src.memory.semantic.retrieval.expansion import RetrievalExpansionMixin
from src.memory.semantic.retrieval.route_helpers import RetrievalRouteHelperMixin
from src.memory.semantic.retrieval.strategy import RetrievalStrategy
from src.memory.semantic.retrieval.strategy_policy import RetrievalStrategyPolicyMixin


class _RouteHarness(RetrievalRouteHelperMixin):
    MAX_ROUTE_QUERIES = 8

    @staticmethod
    def _merge_unique(primary: list[str], secondary: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in primary + secondary:
            text = str(value or "").strip()
            marker = text.casefold()
            if text and marker not in seen:
                seen.add(marker)
                result.append(text)
        return result


class _CandidateHarness(RetrievalCandidateUtilsMixin):
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


class _StrategyHarness(RetrievalStrategyPolicyMixin):
    pass


class _ExpansionHarness(RetrievalExpansionMixin):
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _candidate_id(item: dict[str, Any]) -> str:
        return str(item.get("id") or "")

    @staticmethod
    def _candidate_text(item: dict[str, Any]) -> str:
        return str(item.get("semantic_text") or "")

    @staticmethod
    def _anchor_score(item: dict[str, Any]) -> float:
        return float(item.get("field_score") or 0.0)

    @staticmethod
    def _merge_any_unique(primary: list[Any], secondary: list[Any]) -> list[Any]:
        result: list[Any] = []
        for value in primary + secondary:
            if value not in result:
                result.append(value)
        return result


def _record() -> MemoryRecord:
    return MemoryRecord(
        record_id="record-1",
        memory_type="fact",
        semantic_text="Melanie painted a sunrise.",
        normalized_text="fact: Melanie painted a sunrise.",
    )


def test_route_queries_keep_original_question_and_exclude_constraints() -> None:
    route = EvidenceRoute(
        evidence_goal="Find the painting event",
        queries=["Melanie sunrise painting"],
        constraints=[{"kind": "entity", "value": "Melanie"}],
    )

    variants = _RouteHarness._route_query_variants(
        "When did Melanie paint a sunrise?",
        route,
    )

    assert variants == [
        "When did Melanie paint a sunrise?",
        "Find the painting event",
        "Melanie sunrise painting",
    ]
    assert "entity: Melanie" not in variants


def test_duplicate_evidence_hits_do_not_accumulate_or_forge_distance() -> None:
    harness = _CandidateHarness()
    candidates: dict[str, dict[str, Any]] = {}

    harness._merge_record_candidate(
        candidates,
        _record(),
        score=0.34,
        matched_query="query one",
        channel="entity_ann",
        node={"node_id": "entity:melanie", "node_type": "entity"},
    )
    harness._merge_record_candidate(
        candidates,
        _record(),
        score=0.38,
        matched_query="query two",
        channel="entity_tag_ann",
        node={"node_id": "entity_tag:melanie", "node_type": "entity_tag"},
    )

    candidate = candidates["record-1"]
    assert candidate["field_score"] == 0.38
    assert candidate["semantic_distance"] == 1.0
    assert candidate["matched_queries"] == ["query one", "query two"]


def test_default_strategy_does_not_downweight_raw_turns() -> None:
    strategy = _StrategyHarness._resolve_retrieval_strategy(SearchPlan(question_type="single"))

    assert strategy.raw_turn_score_multiplier == 1.0
    assert strategy.episode_source_weight == 1.0
    assert strategy.episode_budget_ratio >= 0.40
    assert strategy.min_episode_score_ratio <= 0.72


def test_source_window_inherits_anchor_score_without_amplification() -> None:
    item = _ExpansionHarness()._make_source_window_candidate(
        session_id="session-1",
        turns=[
            {
                "turn_index": 3,
                "role": "user",
                "speaker": "Melanie",
                "content": "I painted a sunrise.",
            }
        ],
        anchor={
            "id": "record-1",
            "source": "record",
            "semantic_text": "Melanie painted a sunrise.",
            "field_score": 0.38,
            "retrieval_score": 0.35,
            "semantic_distance": 0.40,
        },
        route=EvidenceRoute(route_id="route-1", evidence_goal="find the painting date"),
        strategy=RetrievalStrategy(question_type="single"),
    )

    assert item is not None
    assert item["field_score"] == 0.38
    assert item["retrieval_score"] == 0.35
    assert item["semantic_distance"] == 0.40
