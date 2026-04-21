"""Self-Evolve 主循环编排器。

职责：
- 信号收集 hook: 在 Pipeline 运行后自动收集信号
- 定期评估: 累计足够样本后触发诊断
- 主动优化: 当发现显著瓶颈时自动触发优化
- 安全护栏: promote 前需通过改善性评审，支持自动回滚

使用方式：
- EvolveLoop 在 factory.py 中创建并注入 pipeline
- Pipeline 在信号产生时调用 loop.after_run() 收集信号
- Pipeline 每次用户请求完成后调用 loop.record_request() 递增运行计数
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
    1. after_run(): 在信号产生时调用，写入指标
    2. record_request(): 每次用户请求完成后调用，递增计数并检查自动优化
    3. maybe_optimize(): 检查是否满足触发条件，自动执行优化
    4. promote_candidate() / rollback_candidate(): 手动或自动版本管理
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

        # name → {version, pre_health, promoted_at_run, candidate_samples_at_promote}
        self._pending_candidates: dict[str, dict[str, Any]] = {}
        self._probation_check_interval = max(10, min_samples_for_optimize // 2)

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
        prompt_versions_used: dict[str, int] | None = None,
        synthesis_kept: int = 0,
        synthesis_dropped: int = 0,
        synthesis_input: int = 0,
        consolidation_records_added: int = 0,
        consolidation_records_merged: int = 0,
        consolidation_records_expired: int = 0,
        consolidation_has_novelty: bool = False,
        consolidation_skills_added: int = 0,
    ) -> None:
        """写入一次由 Pipeline 产生的信号。"""
        versions = prompt_versions_used or self._signals.get_current_versions()

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

    def record_request(self) -> None:
        """记录一次完整用户请求，并按请求频率检查 probation 和自动优化。"""
        with self._lock:
            self._run_count += 1
            count = self._run_count

        if count > 0 and count % self._probation_check_interval == 0:
            try:
                self._check_probation()
            except Exception:
                logger.warning("Probation check failed", exc_info=True)

        if self._auto_optimize and count > 0 and count % self._optimize_interval == 0:
            logger.info("Run count %d reached optimize interval, triggering optimization", count)
            self.maybe_optimize()

    def collect_retrieval_adequacy(
        self,
        is_sufficient: bool,
        reflection_round: int = 0,
        prompt_versions_used: dict[str, int] | None = None,
    ) -> None:
        """从检索充分性判断中收集信号。"""
        versions = prompt_versions_used or self._signals.get_current_versions()
        try:
            adequacy_version = int(versions.get("retrieval_adequacy_check", 0) or 0)
        except Exception:
            adequacy_version = 0
        try:
            planning_version = int(versions.get("retrieval_planning", 0) or 0)
        except Exception:
            planning_version = 0

        self._signals.collect_retrieval_adequacy(
            is_sufficient=is_sufficient,
            reflection_round=reflection_round,
            adequacy_version=adequacy_version,
            planning_version=planning_version,
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
        """将候选版本提升为 active 并进入 probation（试用期）。

        安全护栏：
        1. review_verdict 为 reject → 不提升
        2. 提升后进入 probation，累积 _min_samples 后自动比较改善性
        3. 改善不足 improvement_threshold → 自动回滚
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

        pre_health = self._evaluator.evaluate_prompt(result.prompt_name)
        candidate_version = result.candidate_version.version

        registry.promote_candidate(
            result.prompt_name,
            candidate_version,
            eval_score=pre_health.health_score,
        )

        # 提升后立刻查询新版本的初始样本数（通常为 0）
        # probation 检查时用"新版本当前样本数 - 此值"来判断积累了多少
        new_version_initial_samples = self._evaluator.evaluate_prompt(
            result.prompt_name, version=candidate_version
        ).sample_count

        with self._lock:
            self._pending_candidates[result.prompt_name] = {
                "version": candidate_version,
                "pre_health": pre_health.health_score,
                "promoted_at_run": self._run_count,
                "candidate_samples_at_promote": new_version_initial_samples,
            }

        logger.info(
            "Promoted '%s' to v%d (pre-health=%.2f), entering probation. "
            "Will auto-rollback if improvement < %.2f after %d samples.",
            result.prompt_name, candidate_version,
            pre_health.health_score, self._improvement_threshold,
            self._min_samples,
        )

    def _check_probation(self) -> None:
        """检查所有处于 probation 的候选版本是否达到改善门槛。

        条件满足时保留，不满足时自动回滚。
        """
        with self._lock:
            pending = dict(self._pending_candidates)

        if not pending:
            return

        for prompt_name, info in pending.items():
            version = info["version"]
            pre_health = info["pre_health"]
            candidate_samples_at_promote = info["candidate_samples_at_promote"]

            current_report = self._evaluator.evaluate_prompt(prompt_name, version=version)
            new_samples = current_report.sample_count - candidate_samples_at_promote
            if new_samples < self._min_samples:
                continue

            improvement = current_report.health_score - pre_health
            if improvement >= self._improvement_threshold:
                logger.info(
                    "Probation PASSED for '%s' v%d: "
                    "health %.2f → %.2f (improvement=%.3f >= threshold=%.3f)",
                    prompt_name, version,
                    pre_health, current_report.health_score,
                    improvement, self._improvement_threshold,
                )
                with self._lock:
                    self._pending_candidates.pop(prompt_name, None)
            else:
                logger.warning(
                    "Probation FAILED for '%s' v%d: "
                    "health %.2f → %.2f (improvement=%.3f < threshold=%.3f). Rolling back.",
                    prompt_name, version,
                    pre_health, current_report.health_score,
                    improvement, self._improvement_threshold,
                )
                self.rollback_candidate(prompt_name, version)

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
            "pending_candidates": {
                k: {
                    "version": v["version"],
                    "pre_health": v["pre_health"],
                    "runs_since_promote": self._run_count - v["promoted_at_run"],
                }
                for k, v in self._pending_candidates.items()
            },
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
