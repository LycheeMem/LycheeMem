"""Retrieval strategy policy object."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalStrategy:
    """Internal execution policy derived from abstract question semantics."""

    question_type: str
    evidence_limit_multiplier: float = 1.0
    record_limit_multiplier: float = 1.0
    turn_limit_multiplier: float = 1.0
    raw_turn_score_multiplier: float = 1.0
    assistant_turn_bonus: float = 0.0
    user_turn_bonus: float = 0.0
    episode_source_weight: float = 1.0
    episode_budget_ratio: float = 0.40
    min_episode_score_ratio: float = 0.72
    expansion_window: int = 0
    expansion_limit_per_route: int = 0
    expansion_score: float = 0.0
    expansion_roles: tuple[str, ...] = ()
    source_window_enabled: bool = True
    source_window_size: int = 1
    source_window_anchor_limit_per_route: int = 24
    source_window_merge_gap: int = 1
    source_window_max_chars: int = 2800

