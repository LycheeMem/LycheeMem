"""Semantic retrieval candidate types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoredCandidate:
    """评分后的候选条目。"""
    id: str              # record_id 或 episode_id
    source: str          # "record" 或 "episode"
    final_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    # 原始数据引用
    data: dict[str, Any] = field(default_factory=dict)
