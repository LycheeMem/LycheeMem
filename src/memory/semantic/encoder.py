"""Compact Semantic Encoder（模块一）。

流水线：对话 → 原子抽取 → 指代消解 → action metadata 标注 → MemoryUnit 列表

对应 idea 论文的 Module 1: Compact Semantic Encoding。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from src.llm.base import BaseLLM
from src.memory.semantic.models import MemoryUnit, VALID_MEMORY_TYPES
from src.memory.semantic.prompts import (
    COMPACT_ENCODING_SYSTEM,
    DECONTEXTUALIZE_SYSTEM,
    ACTION_METADATA_SYSTEM,
)


class CompactEncoder:
    """紧凑语义编码器：将对话轮次编码为 MemoryUnit 列表。"""

    def __init__(self, llm: BaseLLM):
        self._llm = llm

    def encode_conversation(
        self,
        current_turns: list[dict[str, Any]],
        *,
        previous_turns: list[dict[str, Any]] | None = None,
        session_id: str = "",
        user_id: str = "",
    ) -> list[MemoryUnit]:
        """完整编码流水线：抽取 → 去上下文化 → 元数据标注。

        Args:
            current_turns: 需要处理的当前对话轮次
            previous_turns: 最近的上文轮次（供指代消解参考，可选）
            session_id: 会话 ID
            user_id: 用户 ID

        Returns:
            编码完成的 MemoryUnit 列表
        """
        # Stage 1: 原子抽取
        raw_units = self._extract_raw_units(current_turns, previous_turns or [])
        if not raw_units:
            return []

        context_text = self._format_turns(current_turns, previous_turns or [])
        now_iso = datetime.now(timezone.utc).isoformat()
        results: list[MemoryUnit] = []

        for raw in raw_units:
            semantic_text = raw.get("semantic_text", "")
            if not semantic_text.strip():
                continue

            # Stage 2: 指代消解（去上下文化）
            decontextualized = self._decontextualize(semantic_text, context_text)

            # Stage 3: Action metadata 标注
            memory_type = raw.get("memory_type", "fact")
            if memory_type not in VALID_MEMORY_TYPES:
                memory_type = "fact"
            metadata = self._annotate_action_metadata(decontextualized, memory_type)

            normalized_text = metadata.get("normalized_text", decontextualized)
            unit_id = self._make_unit_id(normalized_text)

            unit = MemoryUnit(
                unit_id=unit_id,
                memory_type=memory_type,
                semantic_text=decontextualized,
                normalized_text=normalized_text,
                entities=raw.get("entities", []),
                temporal=raw.get("temporal", {}),
                task_tags=metadata.get("task_tags", []),
                tool_tags=metadata.get("tool_tags", []),
                constraint_tags=metadata.get("constraint_tags", []),
                failure_tags=metadata.get("failure_tags", []),
                affordance_tags=metadata.get("affordance_tags", []),
                confidence=1.0,
                evidence_turn_range=raw.get("evidence_turns", []),
                source_session=session_id,
                user_id=user_id,
                created_at=now_iso,
                updated_at=now_iso,
            )
            results.append(unit)

        return results

    # ──────────────────────────────────────
    # Stage 1: 原子事实抽取
    # ──────────────────────────────────────

    def _extract_raw_units(
        self,
        current_turns: list[dict[str, Any]],
        previous_turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """LLM 驱动的原子事实抽取。"""
        prev_text = self._format_section(previous_turns) if previous_turns else "（无上文）"
        curr_text = self._format_section(current_turns)

        user_content = (
            f"<PREVIOUS_TURNS>\n{prev_text}\n</PREVIOUS_TURNS>\n\n"
            f"<CURRENT_TURNS>\n{curr_text}\n</CURRENT_TURNS>"
        )

        response = self._llm.generate([
            {"role": "system", "content": COMPACT_ENCODING_SYSTEM},
            {"role": "user", "content": user_content},
        ])

        try:
            parsed = self._parse_json(response)
            units = parsed.get("units", [])
            if isinstance(units, list):
                return units
        except (ValueError, json.JSONDecodeError):
            pass
        return []

    # ──────────────────────────────────────
    # Stage 2: 指代消解
    # ──────────────────────────────────────

    def _decontextualize(self, original_text: str, context: str) -> str:
        """通过 LLM 消解指代，使文本自洽。"""
        user_content = (
            f"<ORIGINAL_TEXT>\n{original_text}\n</ORIGINAL_TEXT>\n\n"
            f"<CONTEXT>\n{context}\n</CONTEXT>"
        )

        response = self._llm.generate([
            {"role": "system", "content": DECONTEXTUALIZE_SYSTEM},
            {"role": "user", "content": user_content},
        ])

        try:
            parsed = self._parse_json(response)
            result = parsed.get("decontextualized_text", "")
            if result.strip():
                return result
        except (ValueError, json.JSONDecodeError):
            pass
        # 兜底：返回原文
        return original_text

    # ──────────────────────────────────────
    # Stage 3: Action Metadata 标注
    # ──────────────────────────────────────

    def _annotate_action_metadata(
        self, semantic_text: str, memory_type: str,
    ) -> dict[str, Any]:
        """LLM 标注 normalized_text + tags。"""
        user_content = (
            f"<SEMANTIC_TEXT>\n{semantic_text}\n</SEMANTIC_TEXT>\n\n"
            f"<MEMORY_TYPE>\n{memory_type}\n</MEMORY_TYPE>"
        )

        response = self._llm.generate([
            {"role": "system", "content": ACTION_METADATA_SYSTEM},
            {"role": "user", "content": user_content},
        ])

        try:
            parsed = self._parse_json(response)
            return {
                "normalized_text": parsed.get("normalized_text", semantic_text),
                "task_tags": parsed.get("task_tags", []),
                "tool_tags": parsed.get("tool_tags", []),
                "constraint_tags": parsed.get("constraint_tags", []),
                "failure_tags": parsed.get("failure_tags", []),
                "affordance_tags": parsed.get("affordance_tags", []),
            }
        except (ValueError, json.JSONDecodeError):
            return {"normalized_text": semantic_text}

    # ──────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────

    @staticmethod
    def _make_unit_id(normalized_text: str) -> str:
        """SHA256(normalized_text) 作为幂等 ID。"""
        return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_turns(
        current_turns: list[dict[str, Any]],
        previous_turns: list[dict[str, Any]],
    ) -> str:
        parts = []
        if previous_turns:
            parts.append("[上文]")
            for t in previous_turns:
                parts.append(f"{t.get('role', 'unknown')}: {t.get('content', '')}")
        parts.append("[当前]")
        for t in current_turns:
            parts.append(f"{t.get('role', 'unknown')}: {t.get('content', '')}")
        return "\n".join(parts)

    @staticmethod
    def _format_section(turns: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in turns
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """从 LLM 输出中提取 JSON。"""
        text = text.strip()
        # 去除可能的代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
