"""信号收集器：从系统各组件聚合反馈信号。

信号源：
1. UsageLog（用户反馈 + 行动结果）
2. SynthesizerAgent 输出（kept/dropped 比例）
3. ConsolidationResult（records_added/merged/expired）
4. LLM Token 统计（成本维度）
5. 检索充分性判断（adequacy_check 通过率）

信号汇总到 PromptStore 的 prompt_metrics + prompt_failure_cases 中。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.evolve.prompt_store import PromptFailureCase, PromptMetricSnapshot, PromptStore

logger = logging.getLogger("src.evolve.signals")


# ── 信号事件 ──

@dataclass
class EvolveSignal:
    """一个原子信号事件。"""
    prompt_name: str
    signal_type: str  # user_feedback / synthesis_stats / consolidation_stats / retrieval_adequacy / ...
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ── prompt_name ↔ llm_call_source 映射 ──

_SOURCE_TO_PROMPT: dict[str, str] = {
    "compact_encoding": "compact_encoding",
    "novelty_check": "novelty_check",
    "composite_filter": "composite_filter",
    "adequacy_check": "retrieval_adequacy_check",
    "additional_queries": "retrieval_additional_queries",
    "feedback_classification": "feedback_classification",
    "retrieval_planning": "retrieval_planning",
    "query_analysis_and_hyde": "search_coordinator",
    "synthesis_scoring": "synthesis",
    "reasoning": "reasoning",
    "wm_compression": "wm_compression",
    "skill_extraction": "consolidation",
}


def source_to_prompt_name(call_source: str) -> str:
    """将 set_llm_call_source 标签转换为 prompt name。"""
    return _SOURCE_TO_PROMPT.get(call_source, call_source)


class SignalCollector:
    """从 Pipeline 运行结果中提取信号并写入 PromptStore。

    设计为无状态：每次调用 collect_* 方法即完成一轮信号写入。
    """

    def __init__(self, store: PromptStore):
        self._store = store

    def collect_user_feedback(
        self,
        feedback: str,
        outcome: str,
        *,
        prompt_versions: dict[str, int] | None = None,
    ) -> None:
        """收集用户反馈信号。

        feedback: positive / negative / correction / ""
        outcome: success / fail / unknown
        prompt_versions: 当前各 prompt 的 active 版本号
        """
        if outcome == "unknown":
            return

        versions = prompt_versions or {}

        relevant_prompts = [
            "reasoning", "synthesis", "search_coordinator",
            "retrieval_planning", "compact_encoding",
        ]

        for pname in relevant_prompts:
            ver = versions.get(pname, 0)
            self._store.record_metric(PromptMetricSnapshot(
                prompt_name=pname,
                version=ver,
                metric_name="user_feedback_positive",
                metric_value=1.0 if outcome == "success" else 0.0,
                sample_count=1,
            ))

        if outcome == "fail" and feedback in ("negative", "correction"):
            for pname in ["reasoning", "synthesis"]:
                ver = versions.get(pname, 0)
                self._store.record_failure(PromptFailureCase(
                    prompt_name=pname,
                    version=ver,
                    case_type="user_negative_feedback",
                    diagnosis=f"feedback={feedback}, outcome={outcome}",
                ))

    def collect_synthesis_stats(
        self,
        kept_count: int,
        dropped_count: int,
        input_fragment_count: int,
        *,
        version: int = 0,
    ) -> None:
        """收集 SynthesizerAgent 的 kept/dropped 统计。"""
        total = kept_count + dropped_count
        if total == 0:
            return

        kept_ratio = kept_count / total
        self._store.record_metric(PromptMetricSnapshot(
            prompt_name="synthesis",
            version=version,
            metric_name="kept_ratio",
            metric_value=kept_ratio,
            sample_count=1,
            detail=json.dumps({
                "kept": kept_count,
                "dropped": dropped_count,
                "input": input_fragment_count,
                "fragments_total": total,
            }),
        ))

    def collect_consolidation_stats(
        self,
        records_added: int,
        records_merged: int,
        records_expired: int,
        has_novelty: bool,
        skills_added: int,
        *,
        encoding_version: int = 0,
        consolidation_version: int = 0,
    ) -> None:
        """收集固化阶段统计。"""
        self._store.record_metric(PromptMetricSnapshot(
            prompt_name="compact_encoding",
            version=encoding_version,
            metric_name="records_added",
            metric_value=float(records_added),
            sample_count=1,
            detail=json.dumps({
                "merged": records_merged,
                "expired": records_expired,
                "has_novelty": has_novelty,
            }),
        ))

        if skills_added > 0:
            self._store.record_metric(PromptMetricSnapshot(
                prompt_name="consolidation",
                version=consolidation_version,
                metric_name="skills_added",
                metric_value=float(skills_added),
                sample_count=1,
            ))

    def collect_retrieval_adequacy(
        self,
        is_sufficient: bool,
        reflection_round: int,
        *,
        adequacy_version: int = 0,
        planning_version: int = 0,
    ) -> None:
        """收集检索充分性判断结果。"""
        self._store.record_metric(PromptMetricSnapshot(
            prompt_name="retrieval_adequacy_check",
            version=adequacy_version,
            metric_name="adequacy_pass",
            metric_value=1.0 if is_sufficient else 0.0,
            sample_count=1,
            detail=json.dumps({"reflection_round": reflection_round}),
        ))

        if not is_sufficient:
            self._store.record_metric(PromptMetricSnapshot(
                prompt_name="retrieval_planning",
                version=planning_version,
                metric_name="required_supplementary",
                metric_value=1.0,
                sample_count=1,
            ))

    def collect_retrieval_miss(
        self,
        query: str,
        plan_summary: str,
        *,
        planning_version: int = 0,
    ) -> None:
        """记录检索未命中的失败案例。"""
        self._store.record_failure(PromptFailureCase(
            prompt_name="retrieval_planning",
            version=planning_version,
            case_type="retrieval_miss",
            input_summary=query[:500],
            actual=plan_summary[:500],
            diagnosis="Retrieval plan did not yield sufficient results",
        ))

    def collect_session_continuation(
        self,
        *,
        prompt_versions: dict[str, int] | None = None,
    ) -> None:
        """记录一次会话延续信号（隐式正面反馈）。

        当用户在同一 session 中发出后续消息但没有负面反馈时，
        视为对上一轮回答的隐式认可。权重低于显式反馈。
        """
        versions = prompt_versions or {}
        implicit_positive_score = 0.6

        for pname in ["reasoning", "synthesis", "search_coordinator", "retrieval_planning"]:
            ver = versions.get(pname, 0)
            self._store.record_metric(PromptMetricSnapshot(
                prompt_name=pname,
                version=ver,
                metric_name="user_feedback_positive",
                metric_value=implicit_positive_score,
                sample_count=1,
                detail=json.dumps({
                    "source": "session_continuation",
                    "score": implicit_positive_score,
                }),
            ))

    def collect_encoding_loss(
        self,
        input_turns: str,
        output_records_count: int,
        *,
        version: int = 0,
    ) -> None:
        """记录编码可能遗漏的案例（输出为空但输入有实质内容）。"""
        if output_records_count == 0:
            self._store.record_failure(PromptFailureCase(
                prompt_name="compact_encoding",
                version=version,
                case_type="encoding_empty_output",
                input_summary=input_turns[:500],
                actual="0 records extracted",
                diagnosis="Encoder returned empty results for non-trivial input",
            ))

    def get_current_versions(self) -> dict[str, int]:
        """获取所有 prompt 的当前 active 版本号。"""
        from src.evolve.prompt_registry import get_registry

        reg = get_registry()
        if reg is None:
            return {}

        versions = {}
        for name in reg.list_registered():
            versions[name] = reg.get_active_version_number(name)
        return versions
