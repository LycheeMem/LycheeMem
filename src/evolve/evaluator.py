"""Prompt 健康诊断器：从历史信号生成 per-prompt 健康报告。

职责：
- 聚合 prompt_metrics 表中的时序数据
- 分析 prompt_failure_cases 中的失败模式
- 对每个 prompt 输出综合健康评分 + 瓶颈诊断
- 排名出"最值得优化的 prompt"供 Optimizer 消费
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.evolve.prompt_store import PromptStore

logger = logging.getLogger("src.evolve.evaluator")


@dataclass
class PromptHealthReport:
    """单个 prompt 的健康诊断报告。"""
    prompt_name: str
    version: int
    health_score: float  # 0.0 ~ 1.0, higher = healthier
    sample_count: int = 0
    metrics_summary: dict[str, float] = field(default_factory=dict)
    failure_count: int = 0
    failure_breakdown: dict[str, int] = field(default_factory=dict)
    diagnosis: str = ""
    optimization_priority: float = 0.0  # higher = more urgent
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


# ── 每个 prompt 的关键指标定义 ──

_PROMPT_KEY_METRICS: dict[str, list[str]] = {
    "compact_encoding": ["records_added"],
    "novelty_check": [],
    "retrieval_planning": ["required_supplementary"],
    "retrieval_adequacy_check": ["adequacy_pass"],
    "retrieval_additional_queries": [],
    "composite_filter": [],
    "search_coordinator": [],
    "synthesis": ["kept_ratio"],
    "reasoning": [],
    "consolidation": ["skills_added"],
    "wm_compression": [],
    "synthesis_execute": [],
    "feedback_classification": [],
}

# 每个指标的理想方向和基准
_METRIC_BENCHMARKS: dict[str, dict[str, Any]] = {
    "kept_ratio": {"direction": "higher", "baseline": 0.5, "weight": 1.5},
    "adequacy_pass": {"direction": "higher", "baseline": 0.6, "weight": 2.0},
    "required_supplementary": {"direction": "lower", "baseline": 0.4, "weight": 1.5},
    "records_added": {"direction": "higher", "baseline": 1.0, "weight": 1.0},
    "skills_added": {"direction": "higher", "baseline": 0.5, "weight": 0.5},
}


class PromptEvaluator:
    """从 PromptStore 中聚合信号并生成健康报告。"""

    def __init__(self, store: PromptStore, min_samples: int = 10):
        self._store = store
        self._min_samples = min_samples

    def evaluate_prompt(self, prompt_name: str, version: int | None = None) -> PromptHealthReport:
        """对单个 prompt 进行健康评估。"""
        if version is None:
            active = self._store.get_active_version(prompt_name)
            version = active.version if active else 0

        key_metrics = _PROMPT_KEY_METRICS.get(prompt_name, [])
        metrics_summary: dict[str, float] = {}
        total_samples = 0
        scores: list[float] = []

        for metric_name in key_metrics:
            snapshots = self._store.get_metrics(prompt_name, version=version, metric_name=metric_name, limit=200)
            if not snapshots:
                continue

            # 不同指标可能来自同一批请求，使用 max 而非 sum 避免重复累计样本数。
            total_samples = max(total_samples, len(snapshots))

            values = [s.metric_value for s in snapshots]
            avg = sum(values) / len(values)

            metrics_summary[metric_name] = round(avg, 4)

            benchmark = _METRIC_BENCHMARKS.get(metric_name)
            if benchmark:
                if benchmark["direction"] == "higher":
                    normalized = min(1.0, avg / max(benchmark["baseline"], 0.01))
                else:
                    normalized = max(0.0, 1.0 - avg / max(benchmark["baseline"], 0.01))
                scores.append(normalized * benchmark["weight"])

        failure_count = self._store.count_failures(prompt_name, version=version)
        failures = self._store.get_failures(prompt_name, version=version, limit=100)
        failure_breakdown: dict[str, int] = {}
        for f in failures:
            failure_breakdown[f.case_type] = failure_breakdown.get(f.case_type, 0) + 1

        if scores:
            total_weight = sum(
                _METRIC_BENCHMARKS.get(m, {}).get("weight", 1.0)
                for m in key_metrics
                if m in metrics_summary
            )
            health_score = sum(scores) / max(total_weight, 0.01)
        else:
            health_score = 0.5  # 无数据时默认中等

        failure_penalty = min(0.3, failure_count * 0.01)
        health_score = max(0.0, health_score - failure_penalty)

        diagnosis = self._diagnose(prompt_name, metrics_summary, failure_breakdown, health_score)

        optimization_priority = self._calc_priority(
            health_score, total_samples, failure_count, prompt_name
        )

        return PromptHealthReport(
            prompt_name=prompt_name,
            version=version,
            health_score=round(health_score, 4),
            sample_count=total_samples,
            metrics_summary=metrics_summary,
            failure_count=failure_count,
            failure_breakdown=failure_breakdown,
            diagnosis=diagnosis,
            optimization_priority=round(optimization_priority, 4),
        )

    def evaluate_all(self) -> list[PromptHealthReport]:
        """对所有已注册 prompt 进行健康评估，按优化优先级排序。"""
        from src.evolve.prompt_registry import get_registry, IMMUTABLE_PROMPTS

        registry = get_registry()
        if registry is None:
            return []

        reports = []
        for name in registry.list_registered():
            if name in IMMUTABLE_PROMPTS:
                continue
            report = self.evaluate_prompt(name)
            reports.append(report)

        reports.sort(key=lambda r: r.optimization_priority, reverse=True)
        return reports

    def get_top_optimization_targets(self, n: int = 3) -> list[PromptHealthReport]:
        """获取最值得优化的 top-N prompt。"""
        reports = self.evaluate_all()
        return [
            r for r in reports[:n]
            if r.sample_count >= self._min_samples
        ]

    def _diagnose(
        self,
        prompt_name: str,
        metrics: dict[str, float],
        failures: dict[str, int],
        health_score: float,
    ) -> str:
        """根据指标和失败模式生成文字诊断。"""
        parts: list[str] = []

        if health_score >= 0.8:
            parts.append("Health: GOOD")
        elif health_score >= 0.5:
            parts.append("Health: MODERATE")
        else:
            parts.append("Health: POOR")

        kept_ratio = metrics.get("kept_ratio")
        if kept_ratio is not None and kept_ratio < 0.3:
            parts.append(f"Synthesis kept ratio very low ({kept_ratio:.0%}), may be over-filtering")

        adequacy = metrics.get("adequacy_pass")
        if adequacy is not None and adequacy < 0.5:
            parts.append(f"Retrieval adequacy pass rate low ({adequacy:.0%})")

        supp = metrics.get("required_supplementary")
        if supp is not None and supp > 0.5:
            parts.append(f"Frequent supplementary retrieval needed ({supp:.0%})")

        top_failures = sorted(failures.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_failures:
            parts.append("Top failure types: " + ", ".join(f"{t}({c})" for t, c in top_failures))

        return "; ".join(parts)

    @staticmethod
    def _calc_priority(
        health_score: float,
        sample_count: int,
        failure_count: int,
        prompt_name: str,
    ) -> float:
        """计算优化优先级。

        考虑因素：
        1. 健康评分越低 → 优先级越高
        2. 样本量足够 → 优先级越高（有足够数据支撑）
        3. 失败案例多 → 优先级越高
        4. 高频调用路径 → 优先级加权
        """
        _HIGH_IMPACT_PROMPTS = {
            "reasoning", "synthesis", "compact_encoding",
            "retrieval_planning", "search_coordinator",
        }

        priority = (1.0 - health_score) * 5.0

        if sample_count >= 20:
            priority += 1.0
        elif sample_count >= 10:
            priority += 0.5

        priority += min(2.0, failure_count * 0.05)

        if prompt_name in _HIGH_IMPACT_PROMPTS:
            priority *= 1.3

        return priority
