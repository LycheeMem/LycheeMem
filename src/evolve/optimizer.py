"""Prompt 优化器：LLM 驱动的 prompt 改写与评审。

流程：
1. 收集目标 prompt 的健康报告 + 失败案例
2. 调用 LLM 诊断根因
3. 调用 LLM 生成候选修改版
4. （可选）调用 LLM 做候选评审
5. 返回候选版本供 EvolveLoop 决策
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.evolve.evaluator import PromptEvaluator, PromptHealthReport
from src.evolve.prompt_store import EvolveEvent, PromptFailureCase, PromptStore, PromptVersion
from src.evolve.prompts import (
    CANDIDATE_REVIEW_SYSTEM,
    FAILURE_DIAGNOSIS_SYSTEM,
    PROMPT_REWRITE_SYSTEM,
)
from src.llm.base import BaseLLM, set_llm_call_source

logger = logging.getLogger("src.evolve.optimizer")


@dataclass
class OptimizationResult:
    """一次优化尝试的完整结果。"""
    prompt_name: str
    original_version: int
    candidate_version: PromptVersion | None = None
    diagnosis: dict[str, Any] = field(default_factory=dict)
    rewrite_analysis: dict[str, Any] = field(default_factory=dict)
    review_verdict: str = ""  # approve / reject / neutral
    review_detail: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    reason: str = ""


class PromptOptimizer:
    """LLM 驱动的 Prompt 优化器。"""

    def __init__(
        self,
        llm: BaseLLM,
        store: PromptStore,
        evaluator: PromptEvaluator,
        *,
        max_failure_cases: int = 10,
        enable_review: bool = True,
    ):
        self._llm = llm
        self._store = store
        self._evaluator = evaluator
        self._max_failure_cases = max_failure_cases
        self._enable_review = enable_review

    def optimize(self, prompt_name: str) -> OptimizationResult:
        """对指定 prompt 执行一次完整的优化尝试。

        Steps:
        1. 评估当前健康状态
        2. 诊断失败根因
        3. 生成候选版本
        4. （可选）评审候选
        5. 返回结果
        """
        from src.evolve.prompt_registry import get_registry, IMMUTABLE_PROMPTS

        if prompt_name in IMMUTABLE_PROMPTS:
            return OptimizationResult(
                prompt_name=prompt_name,
                original_version=0,
                reason=f"Prompt '{prompt_name}' is immutable",
            )

        registry = get_registry()
        if registry is None:
            return OptimizationResult(
                prompt_name=prompt_name,
                original_version=0,
                reason="Registry not initialized",
            )

        health = self._evaluator.evaluate_prompt(prompt_name)
        current_text = registry.get(prompt_name)
        current_version = registry.get_active_version_number(prompt_name)

        result = OptimizationResult(
            prompt_name=prompt_name,
            original_version=current_version,
        )

        # 1. 诊断
        failures = self._store.get_failures(
            prompt_name, version=current_version, limit=self._max_failure_cases
        )
        diagnosis = self._diagnose(prompt_name, current_text, health, failures)
        result.diagnosis = diagnosis
        try:
            self._store.record_event(EvolveEvent(
                event_type="optimize_diagnosis",
                prompt_name=prompt_name,
                from_version=current_version,
                to_version=None,
                summary=f"诊断完成：health={health.health_score:.3f}, samples={health.sample_count}, failures={health.failure_count}",
                payload={
                    "health": health.__dict__,
                    "diagnosis": diagnosis,
                    "failures_count": len(failures),
                    "active_prompt": current_text,
                },
            ))
        except Exception:
            # 可观测性失败不应影响主流程
            pass

        feasibility = diagnosis.get("optimization_feasibility", "low")
        if feasibility == "low" and health.health_score >= 0.8:
            result.reason = "Prompt is healthy, no optimization needed"
            return result

        # 2. 改写
        rewrite = self._rewrite(prompt_name, current_text, health, failures)
        result.rewrite_analysis = rewrite

        revised_prompt = rewrite.get("revised_prompt", "")
        if not revised_prompt or revised_prompt.strip() == current_text.strip():
            result.reason = "No meaningful changes proposed"
            return result

        changes = rewrite.get("changes", [])
        if not changes:
            result.reason = "Optimizer found no specific changes to make"
            return result

        # 3. 创建候选版本
        change_summary = "; ".join(c.get("reason", "") for c in changes[:3])
        candidate = registry.create_candidate(
            name=prompt_name,
            prompt_text=revised_prompt,
            reason=f"Auto-optimize: {change_summary}"[:200],
        )
        result.candidate_version = candidate
        try:
            self._store.record_event(EvolveEvent(
                event_type="candidate_created",
                prompt_name=prompt_name,
                from_version=current_version,
                to_version=candidate.version,
                summary=f"生成候选 v{candidate.version}（changes={len(changes)}）",
                payload={
                    "changes": changes,
                    "reason": candidate.reason,
                    # 历史记录中保存完整 prompt，便于审计（不再截断）
                    "original_prompt": current_text,
                    "candidate_prompt": revised_prompt,
                    "original_prompt_excerpt": current_text,
                    "candidate_prompt_excerpt": revised_prompt,
                },
            ))
        except Exception:
            pass

        # 4. 评审（可选）
        if self._enable_review:
            review = self._review(current_text, revised_prompt, changes, failures)
            result.review_detail = review
            result.review_verdict = review.get("verdict", "neutral")
            try:
                self._store.record_event(EvolveEvent(
                    event_type="candidate_review",
                    prompt_name=prompt_name,
                    from_version=current_version,
                    to_version=candidate.version,
                    summary=f"评审：{result.review_verdict or 'neutral'}",
                    payload={
                        "verdict": result.review_verdict,
                        "detail": review,
                        "original_prompt": current_text,
                        "candidate_prompt": revised_prompt,
                    },
                ))
            except Exception:
                pass

            if result.review_verdict == "reject":
                result.reason = f"Review rejected: {review.get('reasoning', '')}"
                return result

        result.success = True
        result.reason = f"Candidate v{candidate.version} created"
        logger.info(
            "Optimization for '%s': candidate v%d created (changes: %d)",
            prompt_name, candidate.version, len(changes),
        )
        return result

    # ── LLM 调用 ──

    def _diagnose(
        self,
        prompt_name: str,
        current_text: str,
        health: PromptHealthReport,
        failures: list[PromptFailureCase],
    ) -> dict[str, Any]:
        """调用 LLM 诊断失败根因。"""
        failure_text = self._format_failures(failures)
        metrics_text = json.dumps(health.metrics_summary, indent=2, ensure_ascii=False)

        user_content = (
            f"<PROMPT_NAME>\n{prompt_name}\n</PROMPT_NAME>\n\n"
            f"<CURRENT_PROMPT>\n{current_text[:3000]}\n</CURRENT_PROMPT>\n\n"
            f"<FAILURE_CASES>\n{failure_text}\n</FAILURE_CASES>\n\n"
            f"<METRICS>\n{metrics_text}\n</METRICS>"
        )

        set_llm_call_source("evolve_diagnosis")
        response = self._llm.generate([
            {"role": "system", "content": FAILURE_DIAGNOSIS_SYSTEM},
            {"role": "user", "content": user_content},
        ])

        return self._safe_parse(response)

    def _rewrite(
        self,
        prompt_name: str,
        current_text: str,
        health: PromptHealthReport,
        failures: list[PromptFailureCase],
    ) -> dict[str, Any]:
        """调用 LLM 生成候选版本。"""
        failure_text = self._format_failures(failures)
        health_text = (
            f"Health Score: {health.health_score:.2f}\n"
            f"Sample Count: {health.sample_count}\n"
            f"Diagnosis: {health.diagnosis}\n"
            f"Metrics: {json.dumps(health.metrics_summary, indent=2, ensure_ascii=False)}\n"
            f"Failure Count: {health.failure_count}\n"
            f"Failure Breakdown: {json.dumps(health.failure_breakdown, indent=2, ensure_ascii=False)}"
        )

        user_content = (
            f"<PROMPT_NAME>\n{prompt_name}\n</PROMPT_NAME>\n\n"
            f"<CURRENT_PROMPT>\n{current_text}\n</CURRENT_PROMPT>\n\n"
            f"<HEALTH_REPORT>\n{health_text}\n</HEALTH_REPORT>\n\n"
            f"<FAILURE_CASES>\n{failure_text}\n</FAILURE_CASES>"
        )

        set_llm_call_source("evolve_rewrite")
        response = self._llm.generate([
            {"role": "system", "content": PROMPT_REWRITE_SYSTEM},
            {"role": "user", "content": user_content},
        ])

        return self._safe_parse(response)

    def _review(
        self,
        original_text: str,
        candidate_text: str,
        changes: list[dict],
        failures: list[PromptFailureCase],
    ) -> dict[str, Any]:
        """调用 LLM 对候选版本进行评审。"""
        change_log = json.dumps(changes, indent=2, ensure_ascii=False)
        test_cases = self._format_failures(failures[:5])

        user_content = (
            f"<ORIGINAL_PROMPT>\n{original_text[:2000]}\n</ORIGINAL_PROMPT>\n\n"
            f"<CANDIDATE_PROMPT>\n{candidate_text[:2000]}\n</CANDIDATE_PROMPT>\n\n"
            f"<CHANGE_LOG>\n{change_log}\n</CHANGE_LOG>\n\n"
            f"<TEST_CASES>\n{test_cases}\n</TEST_CASES>"
        )

        set_llm_call_source("evolve_review")
        response = self._llm.generate([
            {"role": "system", "content": CANDIDATE_REVIEW_SYSTEM},
            {"role": "user", "content": user_content},
        ])

        return self._safe_parse(response)

    # ── 工具方法 ──

    @staticmethod
    def _format_failures(failures: list[PromptFailureCase]) -> str:
        if not failures:
            return "(no failure cases)"
        lines = []
        for i, f in enumerate(failures, 1):
            lines.append(f"Case {i} [{f.case_type}]:")
            if f.input_summary:
                lines.append(f"  Input: {f.input_summary[:300]}")
            if f.expected:
                lines.append(f"  Expected: {f.expected[:200]}")
            if f.actual:
                lines.append(f"  Actual: {f.actual[:200]}")
            if f.diagnosis:
                lines.append(f"  Diagnosis: {f.diagnosis[:200]}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _safe_parse(response: str) -> dict[str, Any]:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {"raw_response": response[:2000]}
