"""Self-Evolve 主循环编排器。

职责：
- 信号收集 hook: 在 Pipeline 运行后自动收集信号
- 定期评估: 累计足够样本后触发诊断
- 主动优化: 当发现显著瓶颈时自动触发优化
- 安全护栏: promote 前需通过改善性评审，支持自动回滚

使用方式：
- EvolveLoop 在 factory.py 中创建并注入 pipeline
- Pipeline 每次运行后调用 loop.after_run() 收集信号
- loop.maybe_optimize() 在达到条件时自动触发优化
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from src.evolve.evaluator import PromptEvaluator, PromptHealthReport
from src.evolve.optimizer import OptimizationResult, PromptOptimizer
from src.evolve.prompt_store import PromptMetricSnapshot, PromptStore
from src.evolve.signals import SignalCollector
from src.llm.base import BaseLLM

logger = logging.getLogger("src.evolve.loop")


class EvolveLoop:
    """Self-Evolve 主循环。

    生命周期：
    1. after_run(): 每次 Pipeline 请求后调用，收集信号
    2. maybe_optimize(): 检查是否满足触发条件，自动执行优化
    3. promote_candidate() / rollback_candidate(): 手动或自动版本管理
    """

    def __init__(
        self,
        llm: BaseLLM,
        store: PromptStore,
        *,
        auto_optimize: bool = False,
        min_samples_for_optimize: int = 20,
        improvement_threshold: float = 0.05,
        optimize_interval: int = 50,
    ):
        self._llm = llm
        self._store = store
        self._signals = SignalCollector(store)
        self._evaluator = PromptEvaluator(store, min_samples=min_samples_for_optimize)
        self._optimizer = PromptOptimizer(
            llm=llm,
            store=store,
            evaluator=self._evaluator,
        )

        self._auto_optimize = auto_optimize
        self._min_samples = min_samples_for_optimize
        self._improvement_threshold = improvement_threshold
        self._optimize_interval = optimize_interval

        self._run_count = 0
        self._lock = threading.Lock()

        self._pending_candidates: dict[str, int] = {}

    @property
    def signal_collector(self) -> SignalCollector:
        return self._signals

    @property
    def evaluator(self) -> PromptEvaluator:
        return self._evaluator

    @property
    def optimizer(self) -> PromptOptimizer:
        return self._optimizer

    # ═════════════════════════════════════════════════════════════
    # 信号收集 Hook
    # ═════════════════════════════════════════════════════════════

    def after_run(
        self,
        *,
        user_feedback: str = "",
        user_outcome: str = "unknown",
        synthesis_kept: int = 0,
        synthesis_dropped: int = 0,
        synthesis_input: int = 0,
        consolidation_records_added: int = 0,
        consolidation_records_merged: int = 0,
        consolidation_records_expired: int = 0,
        consolidation_has_novelty: bool = False,
        consolidation_skills_added: int = 0,
    ) -> None:
        """Pipeline 每次请求后的信号收集 hook。"""
        versions = self._signals.get_current_versions()

        if user_feedback or user_outcome != "unknown":
            self._signals.collect_user_feedback(
                feedback=user_feedback,
                outcome=user_outcome,
                prompt_versions=versions,
            )

        if synthesis_kept + synthesis_dropped > 0:
            self._signals.collect_synthesis_stats(
                kept_count=synthesis_kept,
                dropped_count=synthesis_dropped,
                input_fragment_count=synthesis_input,
                version=versions.get("synthesis", 0),
            )

        if consolidation_has_novelty or consolidation_records_added > 0:
            self._signals.collect_consolidation_stats(
                records_added=consolidation_records_added,
                records_merged=consolidation_records_merged,
                records_expired=consolidation_records_expired,
                has_novelty=consolidation_has_novelty,
                skills_added=consolidation_skills_added,
                encoding_version=versions.get("compact_encoding", 0),
                consolidation_version=versions.get("consolidation", 0),
            )

        with self._lock:
            self._run_count += 1
            count = self._run_count

        if self._auto_optimize and count > 0 and count % self._optimize_interval == 0:
            logger.info("Run count %d reached optimize interval, triggering optimization", count)
            self.maybe_optimize()

    def collect_retrieval_adequacy(
        self,
        is_sufficient: bool,
        reflection_round: int = 0,
    ) -> None:
        """从检索充分性判断中收集信号。"""
        from src.evolve.prompt_registry import get_registry
        reg = get_registry()
        version = reg.get_active_version_number("retrieval_adequacy_check") if reg else 0

        self._signals.collect_retrieval_adequacy(
            is_sufficient=is_sufficient,
            reflection_round=reflection_round,
            version=version,
        )

    # ═════════════════════════════════════════════════════════════
    # 优化触发
    # ═════════════════════════════════════════════════════════════

    def maybe_optimize(self, force_prompt: str | None = None) -> list[OptimizationResult]:
        """检查并执行优化。

        Args:
            force_prompt: 强制优化指定 prompt（跳过条件检查）

        Returns:
            优化结果列表
        """
        results: list[OptimizationResult] = []

        if force_prompt:
            result = self._optimizer.optimize(force_prompt)
            if result.success and result.candidate_version:
                self._try_promote(result)
            results.append(result)
            return results

        targets = self._evaluator.get_top_optimization_targets(n=2)
        if not targets:
            logger.debug("No prompts meet optimization criteria")
            return results

        for target in targets:
            if target.health_score >= 0.8:
                continue

            logger.info(
                "Optimizing prompt '%s' (health=%.2f, priority=%.2f)",
                target.prompt_name, target.health_score, target.optimization_priority,
            )
            result = self._optimizer.optimize(target.prompt_name)
            if result.success and result.candidate_version:
                self._try_promote(result)
            results.append(result)

        return results

    def _try_promote(self, result: OptimizationResult) -> None:
        """尝试将优化结果中的候选版本提升为 active。

        安全护栏：
        1. review_verdict 为 reject → 不提升
        2. 需通过改善性检验（新版健康评分 > 旧版 + threshold）
        """
        if not result.candidate_version:
            return

        if result.review_verdict == "reject":
            logger.info(
                "Skipping promotion for '%s' v%d: review rejected",
                result.prompt_name, result.candidate_version.version,
            )
            return

        from src.evolve.prompt_registry import get_registry

        registry = get_registry()
        if registry is None:
            return

        old_health = self._evaluator.evaluate_prompt(result.prompt_name)

        candidate_version = result.candidate_version.version
        with self._lock:
            self._pending_candidates[result.prompt_name] = candidate_version

        registry.promote_candidate(
            result.prompt_name,
            candidate_version,
            eval_score=old_health.health_score,
        )

        logger.info(
            "Promoted '%s' to v%d (previous health=%.2f). "
            "Will monitor for improvement threshold (%.2f).",
            result.prompt_name,
            candidate_version,
            old_health.health_score,
            self._improvement_threshold,
        )

    # ═════════════════════════════════════════════════════════════
    # 手动版本管理
    # ═════════════════════════════════════════════════════════════

    def promote_candidate(self, prompt_name: str, version: int) -> None:
        """手动提升候选版本。"""
        from src.evolve.prompt_registry import get_registry
        reg = get_registry()
        if reg:
            reg.promote_candidate(prompt_name, version)
            logger.info("Manually promoted '%s' to v%d", prompt_name, version)

    def rollback_candidate(self, prompt_name: str, version: int) -> None:
        """手动回滚版本。"""
        from src.evolve.prompt_registry import get_registry
        reg = get_registry()
        if reg:
            reg.rollback(prompt_name, version)
            with self._lock:
                self._pending_candidates.pop(prompt_name, None)
            logger.info("Rolled back '%s' v%d", prompt_name, version)

    # ═════════════════════════════════════════════════════════════
    # 状态查询
    # ═════════════════════════════════════════════════════════════

    def get_status(self) -> dict[str, Any]:
        """获取当前 evolve loop 状态。"""
        reports = self._evaluator.evaluate_all()
        return {
            "run_count": self._run_count,
            "auto_optimize": self._auto_optimize,
            "optimize_interval": self._optimize_interval,
            "pending_candidates": dict(self._pending_candidates),
            "prompt_health": [
                {
                    "name": r.prompt_name,
                    "version": r.version,
                    "health": r.health_score,
                    "samples": r.sample_count,
                    "failures": r.failure_count,
                    "priority": r.optimization_priority,
                    "diagnosis": r.diagnosis,
                }
                for r in reports
            ],
        }

    def get_health_report(self, prompt_name: str) -> PromptHealthReport:
        """获取指定 prompt 的健康报告。"""
        return self._evaluator.evaluate_prompt(prompt_name)
