"""Memory Scorer（评分部分）。

对多通道召回的候选 MemoryRecord / CompositeRecord 做综合评分：

    Score = α·SemanticRelevance + β·ActionUtility + γ·TemporalFit
          + δ·Recency + η·EvidenceDensity − λ·TokenCost

所有系数在 0–1 范围内，用户可通过 config 调整。

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ScoringWeights:
    """评分权重配置。"""
    alpha: float = 0.30   # SemanticRelevance
    beta: float = 0.25    # ActionUtility
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
    # 原始数据引用
    data: dict[str, Any] = field(default_factory=dict)


class MemoryScorer:
    """多维度记忆评分器。"""

    def __init__(self, weights: ScoringWeights | None = None):
        self.w = weights or ScoringWeights()

    def score_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        plan_mode: str = "answer",
        plan_tool_hints: list[str] | None = None,
        plan_required_constraints: list[str] | None = None,
        now: datetime | None = None,
    ) -> list[ScoredCandidate]:
        """对候选列表综合评分并排序。

        每条 candidate dict 需要以下字段：
        - id: str
        - source: "record" | "composite"
        - semantic_distance: float (越小越相似，0=完美匹配)
        - memory_type: str
        - tool_tags: list[str]
        - constraint_tags: list[str]
        - task_tags: list[str]
        - failure_tags: list[str]
        - affordance_tags: list[str]
        - created_at: str (ISO)
        - retrieval_count: int
        - retrieval_hit_count: int
        - action_success_count: int
        - action_fail_count: int
        - evidence_turn_range: list[int] | None
        - temporal: dict
        - semantic_text: str (用于估计 token cost)
        """
        if not candidates:
            return []

        now = now or datetime.now(timezone.utc)
        tool_hints = set(plan_tool_hints or [])
        req_constraints = set(plan_required_constraints or [])

        scored: list[ScoredCandidate] = []

        for c in candidates:
            bd = {}

            # 1. SemanticRelevance: 距离转相似度 (distance → similarity)
            dist = float(c.get("semantic_distance", 1.0))
            bd["semantic"] = max(0.0, 1.0 - dist)

            # 2. ActionUtility: tag 匹配度
            bd["action_utility"] = self._compute_action_utility(
                c, plan_mode, tool_hints, req_constraints,
            )

            # 3. TemporalFit: 时间匹配度（近期的时间引用 → 高分）
            bd["temporal_fit"] = self._compute_temporal_fit(c, now)

            # 4. Recency: 创建时间距今的衰减
            bd["recency"] = self._compute_recency(c.get("created_at", ""), now)

            # 5. EvidenceDensity: 使用统计信号
            bd["evidence_density"] = self._compute_evidence_density(c)

            # 6. TokenCost: 文本长度惩罚
            text_len = len(c.get("semantic_text", ""))
            bd["token_cost"] = min(1.0, text_len / 2000.0)

            final = (
                self.w.alpha * bd["semantic"]
                + self.w.beta * bd["action_utility"]
                + self.w.gamma * bd["temporal_fit"]
                + self.w.delta * bd["recency"]
                + self.w.eta * bd["evidence_density"]
                - self.w.lam * bd["token_cost"]
            )

            scored.append(ScoredCandidate(
                id=c.get("id", ""),
                source=c.get("source", "record"),
                final_score=final,
                score_breakdown=bd,
                data=c,
            ))

        scored.sort(key=lambda s: s.final_score, reverse=True)
        return scored

    # ──────────────────────────────────────
    # 子维度计算
    # ──────────────────────────────────────

    def _compute_action_utility(
        self,
        c: dict[str, Any],
        plan_mode: str,
        tool_hints: set[str],
        req_constraints: set[str],
    ) -> float:
        """行动实用性评分。

        对于 action/mixed 模式，tool_tags / constraint_tags 匹配度很重要。
        对于 answer 模式，action_utility 权重较低，但仍考虑约束匹配。
        """
        if plan_mode == "answer":
            # answer 模式下 action_utility 贡献较小
            return 0.2

        score = 0.0
        parts = 0

        # tool_tags 匹配
        if tool_hints:
            c_tools = set(c.get("tool_tags", []))
            overlap = len(c_tools & tool_hints)
            score += overlap / len(tool_hints) if tool_hints else 0
            parts += 1

        # constraint_tags 匹配
        if req_constraints:
            c_constraints = set(c.get("constraint_tags", []))
            overlap = len(c_constraints & req_constraints)
            score += overlap / len(req_constraints) if req_constraints else 0
            parts += 1

        # memory_type 加分
        mt = c.get("memory_type", "")
        action_types = {"procedure", "constraint", "failure_pattern", "tool_affordance"}
        if mt in action_types:
            score += 0.5
            parts += 1

        # failure_tags 有值时额外加分（失败模式在 action 模式下有价值）
        if c.get("failure_tags"):
            score += 0.3
            parts += 1

        # affordance_tags 加分
        if c.get("affordance_tags"):
            score += 0.2
            parts += 1

        return min(1.0, score / max(1, parts))

    @staticmethod
    def _compute_temporal_fit(c: dict[str, Any], now: datetime) -> float:
        """时间匹配度。有时间标注且在合理范围内的 → 高分。"""
        temporal = c.get("temporal", {})
        if not temporal:
            return 0.3  # 无时间信息，给中等分

        # 如果有 t_valid_to 且已过期 → 低分
        t_valid_to = temporal.get("t_valid_to", "")
        if t_valid_to:
            try:
                valid_to = datetime.fromisoformat(t_valid_to.replace("Z", "+00:00"))
                if valid_to < now:
                    return 0.1
            except (ValueError, TypeError):
                pass

        return 0.7  # 有时间标注且未过期

    @staticmethod
    def _compute_recency(created_at: str, now: datetime) -> float:
        """创建时间衰减：指数衰减，半衰期 30 天。"""
        if not created_at:
            return 0.3

        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            days_ago = (now - created).total_seconds() / 86400.0
            # 指数衰减，半衰期 30 天
            return math.exp(-0.693 * days_ago / 30.0)
        except (ValueError, TypeError):
            return 0.3

    @staticmethod
    def _compute_evidence_density(c: dict[str, Any]) -> float:
        """使用统计信号：命中率高 + 行动成功多 → 高分。"""
        retrieval_count = max(1, int(c.get("retrieval_count", 0)))
        hit_count = int(c.get("retrieval_hit_count", 0))
        success_count = int(c.get("action_success_count", 0))
        fail_count = int(c.get("action_fail_count", 0))

        if retrieval_count <= 1:
            # 新条目，给中等分
            return 0.5

        hit_rate = hit_count / retrieval_count
        total_actions = success_count + fail_count
        success_rate = success_count / total_actions if total_actions > 0 else 0.5

        return 0.5 * hit_rate + 0.5 * success_rate
