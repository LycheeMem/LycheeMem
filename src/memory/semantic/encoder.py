"""Compact Semantic Encoder。

流水线：对话 → 单次 LLM 调用（类型化提取 + 指代消解 + action metadata 标注）→ MemoryRecord 列表。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from src.llm.base import BaseLLM
from src.memory.semantic.models import MemoryRecord, VALID_MEMORY_TYPES
from src.memory.semantic.prompts import COMPACT_ENCODING_SYSTEM


class CompactSemanticEncoder:
    """Compact Semantic Encoder：将对话轮次编码为 MemoryRecord 列表。"""

    def __init__(self, llm: BaseLLM):
        self._llm = llm

    def encode_conversation(
        self,
        current_turns: list[dict[str, Any]],
        *,
        previous_turns: list[dict[str, Any]] | None = None,
        session_id: str = "",
        turn_index_offset: int = 0,
    ) -> list[MemoryRecord]:
        """单次 LLM 调用完成抽取 + 指代消解 + action metadata 标注。

        Args:
            current_turns: 需要处理的当前对话轮次
            previous_turns: 最近的上文轮次（供指代消解参考，可选）
            session_id: 会话 ID
            turn_index_offset: current_turns 在完整 session 中的起始 turn 索引

        Returns:
            编码完成的 MemoryRecord 列表
        """
        raw_records = self._encode_records(current_turns, previous_turns or [])
        if not raw_records:
            return []

        now_iso = datetime.now(timezone.utc).isoformat()
        results: list[MemoryRecord] = []

        for raw in raw_records:
            semantic_text = raw.get("semantic_text", "")
            if not semantic_text.strip():
                continue

            memory_type = raw.get("memory_type", "fact")
            if memory_type not in VALID_MEMORY_TYPES:
                memory_type = "fact"

            # normalized_text 由 LLM 直接给出；fallback 到 semantic_text
            normalized_text = raw.get("normalized_text", "") or semantic_text

            raw_src = raw.get("source_role", "")
            source_role = raw_src if raw_src in ("user", "assistant", "both") else ""

            record_id = self._make_record_id(semantic_text)

            record = MemoryRecord(
                record_id=record_id,
                memory_type=memory_type,
                semantic_text=semantic_text,
                normalized_text=normalized_text,
                entities=raw.get("entities", []),
                temporal=raw.get("temporal", {}),
                task_tags=raw.get("task_tags", []),
                tool_tags=raw.get("tool_tags", []),
                constraint_tags=raw.get("constraint_tags", []),
                failure_tags=raw.get("failure_tags", []),
                affordance_tags=raw.get("affordance_tags", []),
                confidence=1.0,
                evidence_turn_range=self._normalize_evidence_turns(
                    raw.get("evidence_turns", []),
                    turn_index_offset=turn_index_offset,
                ),
                source_session=session_id,
                source_role=source_role,
                created_at=now_iso,
                updated_at=now_iso,
            )
            results.append(record)

        return results

    # ──────────────────────────────────────
    # 单次 LLM 编码（抽取 + 指代消解 + metadata）
    # ──────────────────────────────────────

    def _encode_records(
        self,
        current_turns: list[dict[str, Any]],
        previous_turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """单次 LLM 调用：输出包含全部字段的 record 列表。"""
        prev_text = self._format_section(previous_turns) if previous_turns else "(no previous turns)"
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
            records = parsed.get("records", [])
            if isinstance(records, list):
                return records
        except (ValueError, json.JSONDecodeError):
            pass
        return []

    # ──────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────

    @staticmethod
    def _make_record_id(semantic_text: str) -> str:
        """SHA256(semantic_text) 作为记录 ID。

        使用 semantic_text（保留完整语义、不裁剪时态信息）而非 normalized_text，
        避免时间不同的同类事件因 normalized_text 相同而产生 ID 碰撞。
        """
        return hashlib.sha256(semantic_text.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_section(turns: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in turns
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """从 LLM 输出中提取 JSON。"""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    @staticmethod
    def _normalize_evidence_turns(
        evidence_turns: Any,
        *,
        turn_index_offset: int = 0,
    ) -> list[int]:
        if not isinstance(evidence_turns, list):
            return []

        normalized: list[int] = []
        seen: set[int] = set()
        base_offset = max(0, int(turn_index_offset or 0))
        for raw in evidence_turns:
            try:
                absolute_index = base_offset + int(raw)
            except (TypeError, ValueError):
                continue
            if absolute_index < 0 or absolute_index in seen:
                continue
            seen.add(absolute_index)
            normalized.append(absolute_index)
        normalized.sort()
        return normalized
