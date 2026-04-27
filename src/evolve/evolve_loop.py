"""Self-Evolve 主循环编排器。

职责：
- 信号收集 hook: 在 Pipeline 运行后自动收集信号
- 定期评估: 累计足够样本后触发诊断
- 主动优化: 当发现显著瓶颈时自动触发优化
- 安全护栏: promote 前需通过改善性评审，支持自动回滚

使用方式：
- EvolveLoop 在 factory.py 中创建并注入 pipeline
- Pipeline 在信号产生时调用 loop.after_run() 收集信号
- 各核心 API 完成后调用 loop.record_api_call() 记录 prompt 使用
- 手动接口仍可通过 loop.maybe_optimize() 直接触发优化
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from src.evolve.evaluator import PromptEvaluator, PromptHealthReport
from src.evolve.optimizer import OptimizationResult, PromptOptimizer
from src.evolve.prompt_store import EvolveEvent, PromptMetricSnapshot, PromptStore
from src.evolve.signals import SignalCollector
from src.llm.base import BaseLLM

logger = logging.getLogger("src.evolve.loop")


class EvolveLoop:
    """Self-Evolve 主循环。

    生命周期：
    1. after_run(): 在信号产生时调用，写入指标
    2. record_api_call(): 每次核心 API 完成后调用，按参与 prompt 计数
    3. maybe_optimize(): 手动或调试场景下触发优化
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
        self._api_call_counts: dict[str, int] = {}
        self._prompt_usage_counts: dict[str, dict[int, int]] = {}
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

    def record_api_call(
        self,
        *,
        api_name: str,
        prompt_versions_used: dict[str, int] | None = None,
    ) -> None:
        """记录一次核心 API 调用，并按参与 prompt 的使用次数触发自动优化。"""
        normalized_api_name = str(api_name or "").strip() or "unknown"
        versions: dict[str, int] = {}
        for raw_name, raw_version in (prompt_versions_used or {}).items():
            name = str(raw_name or "").strip()
            if not name:
                continue
            try:
                versions[name] = int(raw_version or 0)
            except Exception:
                versions[name] = 0

        pending_prompt_checks: list[tuple[str, int, int]] = []
        with self._lock:
            self._run_count += 1
            count = self._run_count
            self._api_call_counts[normalized_api_name] = self._api_call_counts.get(normalized_api_name, 0) + 1

            for prompt_name, version in versions.items():
                version_counts = self._prompt_usage_counts.setdefault(prompt_name, {})
                version_counts[version] = version_counts.get(version, 0) + 1
                usage_count = version_counts[version]
                if self._auto_optimize and usage_count > 0 and usage_count % self._optimize_interval == 0:
                    pending_prompt_checks.append((prompt_name, version, usage_count))

        if count > 0 and count % self._probation_check_interval == 0:
            try:
                self._check_probation()
            except Exception:
                logger.warning("Probation check failed", exc_info=True)

        for prompt_name, version, usage_count in pending_prompt_checks:
            try:
                self._maybe_auto_optimize_prompt(
                    prompt_name,
                    expected_version=version,
                    usage_count=usage_count,
                )
            except Exception:
                logger.warning(
                    "Auto prompt optimization failed prompt=%s version=%s usage_count=%s",
                    prompt_name,
                    version,
                    usage_count,
                    exc_info=True,
                )

    def record_request(
        self,
        prompt_versions_used: dict[str, int] | None = None,
        *,
        api_name: str = "chat",
    ) -> None:
        """兼容旧调用方的别名，语义等同于 record_api_call()."""
        self.record_api_call(
            api_name=api_name,
            prompt_versions_used=prompt_versions_used,
        )

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

    def _maybe_auto_optimize_prompt(
        self,
        prompt_name: str,
        *,
        expected_version: int,
        usage_count: int,
    ) -> OptimizationResult | None:
        """当某个 prompt 的 API 使用次数达到阈值时，尝试仅优化该 prompt。"""
        from src.evolve.prompt_registry import get_registry

        registry = get_registry()
        if registry is None or registry.is_immutable(prompt_name):
            return None

        active_version = registry.get_active_version_number(prompt_name)
        if active_version != expected_version:
            logger.debug(
                "Skipping auto optimize for '%s': expected v%d but active is v%d",
                prompt_name,
                expected_version,
                active_version,
            )
            return None

        # 以轨迹数量作为准入门槛：积累足够的真实行为样本后，LLM 自行判断是否需要改写。
        # 不依赖 health_score，避免"没有 key_metrics 的 prompt 永远被排除"的问题。
        trace_count = self._store.count_traces(prompt_name)
        if trace_count < self._min_samples:
            logger.debug(
                "Skipping auto optimize for '%s' v%d: trace_count=%d < min_samples=%d",
                prompt_name,
                expected_version,
                trace_count,
                self._min_samples,
            )
            return None

        logger.info(
            "Prompt '%s' v%d usage count reached %d (traces=%d), triggering auto optimization",
            prompt_name,
            expected_version,
            usage_count,
            trace_count,
        )
        result = self._optimizer.optimize(prompt_name)
        if result.success and result.candidate_version:
            self._try_promote(result)
        return result

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
        2. 提升后进入 probation，累积 _min_samples 条新轨迹后由 LLM 对比新旧版本行为
        3. LLM 判断 rollback → 自动回滚
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

        original_version = result.original_version
        candidate_version = result.candidate_version.version
        change_summary = result.candidate_version.reason or ""

        registry.promote_candidate(
            result.prompt_name,
            candidate_version,
        )

        # 记录候选版本上线时已有的轨迹数（通常为 0）
        # probation 检查时用"当前轨迹数 - 此值"来判断新轨迹是否已累积到门槛
        candidate_traces_at_promote = self._store.count_traces(
            result.prompt_name, version=candidate_version
        )

        with self._lock:
            self._pending_candidates[result.prompt_name] = {
                "version": candidate_version,
                "original_version": original_version,
                "change_summary": change_summary,
                "promoted_at_run": self._run_count,
                "candidate_traces_at_promote": candidate_traces_at_promote,
            }
        try:
            self._store.record_event(EvolveEvent(
                event_type="promote_probation",
                prompt_name=result.prompt_name,
                from_version=original_version,
                to_version=candidate_version,
                summary=(
                    f"提升候选进入 probation：v{candidate_version} "
                    f"(original=v{original_version}, min_samples={self._min_samples})"
                ),
                payload={
                    "original_version": original_version,
                    "candidate_version": candidate_version,
                    "change_summary": change_summary,
                    "candidate_traces_at_promote": candidate_traces_at_promote,
                    "min_samples": self._min_samples,
                },
            ))
        except Exception:
            pass

        logger.info(
            "Promoted '%s' to v%d, entering probation. "
            "Will LLM-evaluate after %d new traces.",
            result.prompt_name, candidate_version, self._min_samples,
        )
        try:
            self._store.record_event(EvolveEvent(
                event_type="promote_probation",
                prompt_name=result.prompt_name,
                from_version=original_version,
                to_version=candidate_version,
                summary=(
                    f"提升候选进入 probation：v{candidate_version} "
                    f"(original=v{original_version}, min_samples={self._min_samples})"
                ),
                payload={
                    "original_version": original_version,
                    "candidate_version": candidate_version,
                    "change_summary": change_summary,
                    "candidate_traces_at_promote": candidate_traces_at_promote,
                    "min_samples": self._min_samples,
                },
            ))
        except Exception:
            pass

        logger.info(
            "Promoted '%s' to v%d, entering probation. "
            "Will LLM-evaluate after %d new traces.",
            result.prompt_name, candidate_version, self._min_samples,
        )

    def _check_probation(self) -> None:
        """LLM 对比新旧版本轨迹，判断候选版本是否保留。

        走流：
        1. 候选版本上线后累积足 min_samples 条轨迹
        2. 调用 optimizer.evaluate_probation() 进行 LLM 轨迹对比
        3. verdict=keep → 保留；verdict=rollback → 回滚至原始版本
        """
        with self._lock:
            pending = dict(self._pending_candidates)

        if not pending:
            return

        for prompt_name, info in pending.items():
            candidate_version = info["version"]
            original_version = info["original_version"]
            change_summary = info.get("change_summary", "")
            candidate_traces_at_promote = info["candidate_traces_at_promote"]

            # 检查候选版本是否已累积足够新轨迹
            current_trace_count = self._store.count_traces(
                prompt_name, version=candidate_version
            )
            new_traces = current_trace_count - candidate_traces_at_promote
            if new_traces < self._min_samples:
                continue

            # LLM 对比轨迹
            try:
                review = self._optimizer.evaluate_probation(
                    prompt_name=prompt_name,
                    original_version=original_version,
                    candidate_version=candidate_version,
                    change_summary=change_summary,
                )
            except Exception:
                logger.warning(
                    "Probation LLM review failed for '%s' v%d, skipping this check.",
                    prompt_name, candidate_version, exc_info=True,
                )
                continue

            verdict = review.get("verdict", "keep")
            confidence = review.get("confidence", 0.0)
            reasoning = review.get("reasoning", "")

            if verdict == "rollback":
                logger.warning(
                    "Probation ROLLBACK for '%s' v%d (confidence=%.2f): %s",
                    prompt_name, candidate_version, confidence, reasoning[:200],
                )
                try:
                    self._store.record_event(EvolveEvent(
                        event_type="probation_fail",
                        prompt_name=prompt_name,
                        from_version=candidate_version,
                        to_version=original_version,
                        summary=(
                            f"probation LLM判定回滚：v{candidate_version} → v{original_version} "
                            f"(confidence={confidence:.2f}, new_traces={new_traces})"
                        ),
                        payload={
                            "review": review,
                            "new_traces": new_traces,
                            "original_version": original_version,
                            "candidate_version": candidate_version,
                        },
                    ))
                except Exception:
                    pass
                self.rollback_candidate(prompt_name, candidate_version)
            else:
                logger.info(
                    "Probation PASSED for '%s' v%d (confidence=%.2f): %s",
                    prompt_name, candidate_version, confidence, reasoning[:200],
                )
                with self._lock:
                    self._pending_candidates.pop(prompt_name, None)
                try:
                    self._store.record_event(EvolveEvent(
                        event_type="probation_pass",
                        prompt_name=prompt_name,
                        from_version=original_version,
                        to_version=candidate_version,
                        summary=(
                            f"probation LLM判定保留：v{candidate_version} "
                            f"(confidence={confidence:.2f}, new_traces={new_traces})"
                        ),
                        payload={
                            "review": review,
                            "new_traces": new_traces,
                            "original_version": original_version,
                            "candidate_version": candidate_version,
                        },
                    ))
                except Exception:
                    pass

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
            try:
                self._store.record_event(EvolveEvent(
                    event_type="rollback",
                    prompt_name=prompt_name,
                    from_version=None,
                    to_version=version,
                    summary=f"回滚版本：{prompt_name} v{version}",
                    payload={},
                ))
            except Exception:
                pass

    # ═════════════════════════════════════════════════════════════
    # 状态查询
    # ═════════════════════════════════════════════════════════════

    def get_status(self) -> dict[str, Any]:
        """获取当前 evolve loop 状态。"""
        reports = self._evaluator.evaluate_all()
        return {
            "run_count": self._run_count,
            "api_call_counts": dict(self._api_call_counts),
            "prompt_usage_counts": {
                name: {str(version): count for version, count in sorted(version_counts.items())}
                for name, version_counts in self._prompt_usage_counts.items()
            },
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
