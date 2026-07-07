"""Question-type retrieval strategy policy."""

from __future__ import annotations

from src.memory.semantic.models import SearchPlan
from src.memory.semantic.retrieval.strategy import RetrievalStrategy


class RetrievalStrategyPolicyMixin:
    @staticmethod
    def _resolve_retrieval_strategy(plan: SearchPlan) -> RetrievalStrategy:
        question_type = str(plan.question_type or "single").strip().lower()

        strategy = {
            "question_type": question_type,
            "raw_turn_score_multiplier": 0.72,
            "episode_source_weight": 0.72,
            "episode_budget_ratio": 0.22,
            "min_episode_score_ratio": 0.86,
            "source_window_enabled": True,
            "source_window_size": 1,
            "source_window_anchor_limit_per_route": 24,
            "source_window_merge_gap": 1,
            "source_window_max_chars": 2800,
        }

        if question_type == "prior_assistant_response":
            strategy.update(
                evidence_limit_multiplier=0.95,
                record_limit_multiplier=0.95,
                turn_limit_multiplier=3.0,
                raw_turn_score_multiplier=1.22,
                assistant_turn_bonus=0.18,
                user_turn_bonus=0.04,
                episode_source_weight=1.0,
                episode_budget_ratio=0.62,
                min_episode_score_ratio=0.52,
                expansion_window=4,
                expansion_limit_per_route=10,
                expansion_score=0.68,
                expansion_roles=("assistant", "user"),
                source_window_size=4,
                source_window_anchor_limit_per_route=32,
                source_window_merge_gap=2,
                source_window_max_chars=7000,
            )

        if question_type == "personalized_advice":
            strategy.update(
                evidence_limit_multiplier=max(strategy.get("evidence_limit_multiplier", 1.0), 1.25),
                record_limit_multiplier=max(strategy.get("record_limit_multiplier", 1.0), 1.2),
                turn_limit_multiplier=max(strategy.get("turn_limit_multiplier", 1.0), 1.8),
                raw_turn_score_multiplier=max(strategy.get("raw_turn_score_multiplier", 0.72), 1.05),
                assistant_turn_bonus=max(strategy.get("assistant_turn_bonus", 0.0), 0.04),
                user_turn_bonus=max(strategy.get("user_turn_bonus", 0.0), 0.08),
                episode_source_weight=max(strategy.get("episode_source_weight", 0.72), 0.95),
                episode_budget_ratio=max(strategy.get("episode_budget_ratio", 0.22), 0.48),
                min_episode_score_ratio=min(strategy.get("min_episode_score_ratio", 0.75), 0.62),
                expansion_window=max(strategy.get("expansion_window", 0), 3),
                expansion_limit_per_route=max(strategy.get("expansion_limit_per_route", 0), 8),
                expansion_score=max(strategy.get("expansion_score", 0.0), 0.62),
                expansion_roles=("user", "assistant"),
                source_window_size=max(strategy.get("source_window_size", 1), 3),
                source_window_anchor_limit_per_route=max(
                    strategy.get("source_window_anchor_limit_per_route", 24), 28
                ),
                source_window_merge_gap=max(strategy.get("source_window_merge_gap", 1), 2),
                source_window_max_chars=max(strategy.get("source_window_max_chars", 2800), 5000),
            )

        if question_type == "aggregate":
            strategy.update(
                evidence_limit_multiplier=max(strategy.get("evidence_limit_multiplier", 1.0), 1.55),
                record_limit_multiplier=max(strategy.get("record_limit_multiplier", 1.0), 1.45),
                turn_limit_multiplier=max(strategy.get("turn_limit_multiplier", 1.0), 0.9),
                episode_budget_ratio=max(strategy.get("episode_budget_ratio", 1.0 / 3.0), 0.34),
                min_episode_score_ratio=min(strategy.get("min_episode_score_ratio", 0.75), 0.70),
                expansion_window=max(strategy.get("expansion_window", 0), 1),
                expansion_limit_per_route=max(strategy.get("expansion_limit_per_route", 0), 3),
                expansion_score=max(strategy.get("expansion_score", 0.0), 0.52),
                expansion_roles=strategy.get("expansion_roles", ("user", "assistant")) or ("user", "assistant"),
                source_window_size=max(strategy.get("source_window_size", 1), 1),
                source_window_anchor_limit_per_route=max(
                    strategy.get("source_window_anchor_limit_per_route", 24), 40
                ),
                source_window_merge_gap=max(strategy.get("source_window_merge_gap", 1), 1),
                source_window_max_chars=max(strategy.get("source_window_max_chars", 2800), 2600),
            )

        if question_type == "temporal":
            strategy.update(
                evidence_limit_multiplier=max(strategy.get("evidence_limit_multiplier", 1.0), 1.35),
                record_limit_multiplier=max(strategy.get("record_limit_multiplier", 1.0), 1.3),
                turn_limit_multiplier=max(strategy.get("turn_limit_multiplier", 1.0), 1.45),
                episode_source_weight=max(strategy.get("episode_source_weight", 0.72), 0.94),
                episode_budget_ratio=max(strategy.get("episode_budget_ratio", 0.22), 0.42),
                min_episode_score_ratio=min(strategy.get("min_episode_score_ratio", 0.75), 0.64),
                expansion_window=max(strategy.get("expansion_window", 0), 2),
                expansion_limit_per_route=max(strategy.get("expansion_limit_per_route", 0), 5),
                expansion_score=max(strategy.get("expansion_score", 0.0), 0.60),
                expansion_roles=("user", "assistant"),
                source_window_size=max(strategy.get("source_window_size", 1), 2),
                source_window_anchor_limit_per_route=max(
                    strategy.get("source_window_anchor_limit_per_route", 24), 28
                ),
                source_window_merge_gap=max(strategy.get("source_window_merge_gap", 1), 1),
                source_window_max_chars=max(strategy.get("source_window_max_chars", 2800), 3600),
            )

        if question_type == "comparison":
            strategy.update(
                evidence_limit_multiplier=max(strategy.get("evidence_limit_multiplier", 1.0), 1.35),
                record_limit_multiplier=max(strategy.get("record_limit_multiplier", 1.0), 1.3),
                turn_limit_multiplier=max(strategy.get("turn_limit_multiplier", 1.0), 1.3),
                episode_source_weight=max(strategy.get("episode_source_weight", 0.72), 0.92),
                episode_budget_ratio=max(strategy.get("episode_budget_ratio", 0.22), 0.38),
                min_episode_score_ratio=min(strategy.get("min_episode_score_ratio", 0.75), 0.66),
                expansion_window=max(strategy.get("expansion_window", 0), 2),
                expansion_limit_per_route=max(strategy.get("expansion_limit_per_route", 0), 5),
                expansion_score=max(strategy.get("expansion_score", 0.0), 0.58),
                expansion_roles=("user", "assistant"),
                source_window_size=max(strategy.get("source_window_size", 1), 2),
                source_window_anchor_limit_per_route=max(
                    strategy.get("source_window_anchor_limit_per_route", 24), 28
                ),
                source_window_merge_gap=max(strategy.get("source_window_merge_gap", 1), 1),
                source_window_max_chars=max(strategy.get("source_window_max_chars", 2800), 3600),
            )

        return RetrievalStrategy(**strategy)


