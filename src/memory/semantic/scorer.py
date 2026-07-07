"""Memory Scorer（评分部分）。

对多通道召回的候选 MemoryRecord / CompositeRecord 做综合评分：

    Score = α·SemanticRelevance + β·ActionUtility + κ·SlotUtility
          + γ·TemporalFit + δ·Recency + η·EvidenceDensity − λ·TokenCost

所有系数在 0–1 范围内，用户可通过 config 调整。

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoringWeights:
    """评分权重配置。"""
    alpha: float = 0.25   # SemanticRelevance
    beta: float = 0.25    # ActionUtility
    kappa: float = 0.15   # SlotUtility
    gamma: float = 0.15   # TemporalFit
    delta: float = 0.10   # Recency
    eta: float = 0.10     # EvidenceDensity
    lam: float = 0.10     # TokenCost penalty


@dataclass
class ScoredCandidate:
    """评分后的候选条目。"""
    id: str              # record_id 或 composite_id
    source: str          # "record" 或 "composite"
    final_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
