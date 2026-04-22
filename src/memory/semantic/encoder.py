"""Compact Semantic Encoder。

流水线：对话 → 单次 LLM 调用（类型化提取 + 指代消解 + action metadata 标注）→ MemoryRecord 列表。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from src.llm.base import BaseLLM, set_llm_call_source
from src.memory.semantic.models import MemoryRecord, VALID_MEMORY_TYPES
from src.memory.semantic.prompts import COMPACT_ENCODING_SYSTEM
from src.evolve.prompt_registry import get_prompt


class CompactSemanticEncoder:
    """Compact Semantic Encoder：将对话轮次编码为 MemoryRecord 列表。"""

    def __init__(self, llm: BaseLLM):
        self._llm = llm

    def encode_conversation(
        self,
        current_turns: list[dict[str, Any]],
        *,
        previous_turns: list[dict[str, Any]] | None = None,
        ingest_scope: str = "user_only",
        session_id: str = "",
        turn_index_offset: int = 0,
        session_date: str | None = None,
    ) -> list[MemoryRecord]:
        """单次 LLM 调用完成抽取 + 指代消解 + metadata 标注。

        Args:
            current_turns: 需要处理的当前对话轮次
            previous_turns: 最近的上文轮次（供指代消解参考，可选）
            ingest_scope: 抽取范围；`user_only` 只固化用户内容，
                `user_and_assistant` 允许固化满足条件的 assistant 内容
            session_id: 会话 ID
            turn_index_offset: current_turns 在完整 session 中的起始 turn 索引
            session_date: 对话发生的日期（自由文本，如 "May 8, 2023"）。
                提供后将作为显式时间锚点注入 system prompt，
                指导 LLM 将相对时间表达（yesterday/last week/next month）
                转换为绝对日期。

        Returns:
            编码完成的 MemoryRecord 列表
        """
        normalized_scope = self._normalize_ingest_scope(ingest_scope)
        raw_records = self._encode_records(
            current_turns,
            previous_turns or [],
            ingest_scope=normalized_scope,
            session_date=session_date,
        )
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

            # normalized_text 由 Python 规则生成，不依赖 LLM 输出
            normalized_text = f"{memory_type}: {semantic_text}"

            raw_src = raw.get("source_role", "")
            raw_evidence_turns = raw.get("evidence_turns", [])
            source_role = self._resolve_source_role(
                raw_src,
                raw_evidence_turns,
                current_turns,
            )
            if normalized_scope == "user_only":
                if source_role in ("assistant", "both"):
                    continue
                if not source_role:
                    source_role = "user"

            record_id = self._make_record_id(semantic_text)

            record = MemoryRecord(
                record_id=record_id,
                memory_type=memory_type,
                semantic_text=semantic_text,
                normalized_text=normalized_text,
                entities=raw.get("entities", []),
                temporal=raw.get("temporal", {}),
                tags=raw.get("tags", []),
                confidence=1.0,
                evidence_turn_range=self._normalize_evidence_turns(
                    raw_evidence_turns,
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
        session_date: str | None = None,
        ingest_scope: str = "user_only",
    ) -> list[dict[str, Any]]:
        """单次 LLM 调用：输出包含全部字段的 record 列表。"""
        prev_text = self._format_section(previous_turns) if previous_turns else "(no previous turns)"
        curr_text = self._format_section(current_turns)

        # session_date 注入 user message 头部，system prompt 保持不变。
        # 调用方在 Python 侧可靠地提取日期后传入，encoder 不自行解析格式。
        if session_date:
            date_header = (
                f"<SESSION_DATE>{session_date}</SESSION_DATE>\n"
                f"Use the SESSION_DATE above as \"today\" to convert all relative time "
                f"references (yesterday, last week, next month, etc.) to absolute dates.\n\n"
            )
        else:
            date_header = ""

        user_content = (
            f"{date_header}"
            f"<INGEST_SCOPE>{self._normalize_ingest_scope(ingest_scope)}</INGEST_SCOPE>\n\n"
            f"<PREVIOUS_TURNS>\n{prev_text}\n</PREVIOUS_TURNS>\n\n"
            f"<CURRENT_TURNS>\n{curr_text}\n</CURRENT_TURNS>"
        )

        set_llm_call_source("compact_encoding")
        response = self._llm.generate([
            {"role": "system", "content": get_prompt("compact_encoding", COMPACT_ENCODING_SYSTEM)},
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
    def _normalize_ingest_scope(value: Any) -> str:
        scope = str(value or "").strip().lower()
        return "user_and_assistant" if scope == "user_and_assistant" else "user_only"

    @classmethod
    def _resolve_source_role(
        cls,
        raw_source_role: Any,
        evidence_turns: Any,
        current_turns: list[dict[str, Any]],
    ) -> str:
        roles: set[str] = set()
        if isinstance(evidence_turns, list):
            for raw_turn in evidence_turns:
                try:
                    turn_index = int(raw_turn)
                except (TypeError, ValueError):
                    continue
                if 0 <= turn_index < len(current_turns):
                    role = str(current_turns[turn_index].get("role", "")).strip().lower()
                    if role in ("user", "assistant"):
                        roles.add(role)

        if roles == {"user"}:
            return "user"
        if roles == {"assistant"}:
            return "assistant"
        if roles == {"user", "assistant"}:
            return "both"

        fallback = str(raw_source_role or "").strip().lower()
        if fallback in ("user", "assistant", "both"):
            return fallback
        return ""

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
