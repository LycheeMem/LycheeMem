"""Pragmatic Memory Synthesizer（模块二）。

在 ingest 完成后，对同一用户的 MemoryUnit 做在线合成：
1. 检测新写入 units 与已有 units 之间是否可合成
2. LLM 判断合成可行性 + 分组
3. LLM 执行合成，生成 SynthesizedUnit
4. 写入 sqlite + vector

对应 idea 论文的 Module 2: Pragmatic Memory Synthesis。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from src.llm.base import BaseLLM
from src.memory.semantic.models import (
    MemoryUnit,
    SynthesizedUnit,
    VALID_SYNTH_TYPES,
)
from src.memory.semantic.prompts import (
    SYNTHESIS_JUDGE_SYSTEM,
    SYNTHESIS_EXECUTE_SYSTEM,
)
from src.memory.semantic.sqlite_store import SQLiteSemanticStore
from src.memory.semantic.vector_index import LanceVectorIndex


class PragmaticSynthesizer:
    """实用记忆合成器：在写入新 units 后自动检测并执行合成。"""

    def __init__(
        self,
        llm: BaseLLM,
        sqlite_store: SQLiteSemanticStore,
        vector_index: LanceVectorIndex,
        *,
        similarity_threshold: float = 0.75,
        min_units_for_synthesis: int = 2,
        max_units_per_group: int = 8,
    ):
        self._llm = llm
        self._sqlite = sqlite_store
        self._vector = vector_index
        self._similarity_threshold = similarity_threshold
        self._min_units = min_units_for_synthesis
        self._max_units_per_group = max_units_per_group

    def synthesize_on_ingest(
        self,
        new_units: list[MemoryUnit],
        *,
        user_id: str = "",
    ) -> list[SynthesizedUnit]:
        """在新 units 写入后触发合成流程。

        1. 对每个新 unit，通过 FTS + 向量检索找到相关的已有 units
        2. 将新 unit + 相关 units 组合为候选集
        3. LLM 判断是否可合成 + 分组
        4. 对每组执行合成
        5. 写入存储

        Returns:
            生成的 SynthesizedUnit 列表
        """
        if not new_units:
            return []

        # 收集所有候选 units（新写入的 + 与其相关的旧 units）
        candidate_ids: set[str] = {u.unit_id for u in new_units}
        candidate_map: dict[str, MemoryUnit] = {u.unit_id: u for u in new_units}

        for unit in new_units:
            # FTS 检索相关旧条目
            fts_results = self._sqlite.find_similar_by_normalized_text(
                unit.normalized_text,
                user_id=user_id,
                limit=5,
            )
            for r in fts_results:
                if r.unit_id not in candidate_ids:
                    candidate_ids.add(r.unit_id)
                    candidate_map[r.unit_id] = r

        candidates = list(candidate_map.values())

        if len(candidates) < self._min_units:
            return []

        # LLM 判断合成可行性
        groups = self._judge_synthesis(candidates)
        if not groups:
            return []

        # 执行合成
        now_iso = datetime.now(timezone.utc).isoformat()
        synthesized: list[SynthesizedUnit] = []

        for group in groups:
            source_ids = group.get("source_unit_ids", [])
            source_units = [candidate_map[uid] for uid in source_ids if uid in candidate_map]

            if len(source_units) < self._min_units:
                continue
            if len(source_units) > self._max_units_per_group:
                source_units = source_units[: self._max_units_per_group]

            reason = group.get("synthesis_reason", "")
            suggested_type = group.get("suggested_type", "synthesized_pattern")
            if suggested_type not in VALID_SYNTH_TYPES:
                suggested_type = "synthesized_pattern"

            synth_result = self._execute_synthesis(source_units, reason, suggested_type)
            if not synth_result:
                continue

            synth_id = self._make_synth_id(
                [u.unit_id for u in source_units], synth_result.get("semantic_text", "")
            )

            su = SynthesizedUnit(
                synth_id=synth_id,
                memory_type=suggested_type,
                semantic_text=synth_result.get("semantic_text", ""),
                normalized_text=synth_result.get("normalized_text", ""),
                source_unit_ids=[u.unit_id for u in source_units],
                synthesis_reason=reason,
                entities=synth_result.get("entities", []),
                temporal=synth_result.get("temporal", {}),
                task_tags=synth_result.get("task_tags", []),
                tool_tags=synth_result.get("tool_tags", []),
                constraint_tags=synth_result.get("constraint_tags", []),
                failure_tags=synth_result.get("failure_tags", []),
                affordance_tags=synth_result.get("affordance_tags", []),
                confidence=synth_result.get("confidence", 0.9),
                user_id=user_id,
                created_at=now_iso,
                updated_at=now_iso,
            )

            # 写入 SQLite
            self._sqlite.upsert_synthesized(su)
            # 写入向量索引
            self._vector.upsert_synthesized(
                synth_id=su.synth_id,
                user_id=su.user_id,
                memory_type=su.memory_type,
                semantic_text=su.semantic_text,
                normalized_text=su.normalized_text,
            )

            synthesized.append(su)

        return synthesized

    def _judge_synthesis(
        self, candidates: list[MemoryUnit],
    ) -> list[dict[str, Any]]:
        """LLM 判断候选集是否可合成 + 分组。"""
        units_json = json.dumps(
            [
                {
                    "unit_id": u.unit_id,
                    "memory_type": u.memory_type,
                    "semantic_text": u.semantic_text,
                    "normalized_text": u.normalized_text,
                    "entities": u.entities,
                    "task_tags": u.task_tags,
                    "tool_tags": u.tool_tags,
                }
                for u in candidates
            ],
            ensure_ascii=False,
            indent=2,
        )

        user_content = f"<UNITS>\n{units_json}\n</UNITS>"

        response = self._llm.generate([
            {"role": "system", "content": SYNTHESIS_JUDGE_SYSTEM},
            {"role": "user", "content": user_content},
        ])

        try:
            parsed = self._parse_json(response)
            if not parsed.get("should_synthesize", False):
                return []
            return parsed.get("groups", [])
        except (ValueError, json.JSONDecodeError):
            return []

    def _execute_synthesis(
        self,
        source_units: list[MemoryUnit],
        reason: str,
        suggested_type: str,
    ) -> dict[str, Any] | None:
        """LLM 执行合成，返回合成结果。"""
        units_json = json.dumps(
            [
                {
                    "unit_id": u.unit_id,
                    "memory_type": u.memory_type,
                    "semantic_text": u.semantic_text,
                    "normalized_text": u.normalized_text,
                    "entities": u.entities,
                    "temporal": u.temporal,
                    "task_tags": u.task_tags,
                    "tool_tags": u.tool_tags,
                    "constraint_tags": u.constraint_tags,
                    "failure_tags": u.failure_tags,
                    "affordance_tags": u.affordance_tags,
                    "confidence": u.confidence,
                }
                for u in source_units
            ],
            ensure_ascii=False,
            indent=2,
        )

        user_content = (
            f"<SOURCE_UNITS>\n{units_json}\n</SOURCE_UNITS>\n\n"
            f"<SYNTHESIS_REASON>\n{reason}\n</SYNTHESIS_REASON>\n\n"
            f"<SUGGESTED_TYPE>\n{suggested_type}\n</SUGGESTED_TYPE>"
        )

        response = self._llm.generate([
            {"role": "system", "content": SYNTHESIS_EXECUTE_SYSTEM},
            {"role": "user", "content": user_content},
        ])

        try:
            return self._parse_json(response)
        except (ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _make_synth_id(source_ids: list[str], semantic_text: str) -> str:
        raw = "|".join(sorted(source_ids)) + "|" + semantic_text
        return "synth_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
