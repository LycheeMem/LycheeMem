"""Record Fusion Engine。

在 ingest 完成后，对同一用户的 MemoryRecord 执行在线聚合：
1. 检测新写入 records 与已有 records 之间是否可融合
2. LLM 判断融合可行性 + 分组
3. LLM 执行融合，生成 CompositeRecord
4. 写入 sqlite + vector
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from src.llm.base import BaseLLM
from src.memory.semantic.models import (
    MemoryRecord,
    CompositeRecord,
    VALID_SYNTH_TYPES,
)
from src.memory.semantic.prompts import (
    SYNTHESIS_JUDGE_SYSTEM,
    get_synthesis_execute_system,
)
from src.memory.semantic.sqlite_store import SQLiteSemanticStore
from src.memory.semantic.vector_index import LanceVectorIndex


class RecordFusionEngine:
    """记录融合引擎：在写入新 records 后自动检测并执行聚合。"""

    def __init__(
        self,
        llm: BaseLLM,
        sqlite_store: SQLiteSemanticStore,
        vector_index: LanceVectorIndex,
        *,
        similarity_threshold: float = 0.75,
        min_records_for_synthesis: int = 2,
        max_records_per_group: int = 8,
    ):
        self._llm = llm
        self._sqlite = sqlite_store
        self._vector = vector_index
        self._similarity_threshold = similarity_threshold
        self._min_records = min_records_for_synthesis
        self._max_records_per_group = max_records_per_group

    def synthesize_on_ingest(
        self,
        new_records: list[MemoryRecord],
        *,
        user_id: str = "",
    ) -> list[CompositeRecord]:
        """在新 records 写入后触发融合流程。

        1. 对每个新 record，通过 FTS + 向量检索找到相关的已有 records
        2. 将新 record + 相关 records 组合为候选集
        3. LLM 判断是否可融合 + 分组
        4. 对每组执行融合
        5. 写入存储

        Returns:
            生成的 CompositeRecord 列表
        """
        if not new_records:
            return []

        # 收集所有候选 records（新写入的 + 与其相关的旧 records）
        candidate_ids: set[str] = {r.record_id for r in new_records}
        candidate_map: dict[str, MemoryRecord] = {r.record_id: r for r in new_records}

        for record in new_records:
            # FTS 检索相关旧条目
            fts_results = self._sqlite.find_similar_by_normalized_text(
                record.normalized_text,
                user_id=user_id,
                limit=5,
            )
            for r in fts_results:
                if r.record_id not in candidate_ids:
                    candidate_ids.add(r.record_id)
                    candidate_map[r.record_id] = r

        candidates = list(candidate_map.values())

        if len(candidates) < self._min_records:
            return []

        # LLM 判断合成可行性
        groups = self._judge_synthesis(candidates)
        if not groups:
            return []

        # 执行合成
        now_iso = datetime.now(timezone.utc).isoformat()
        synthesized: list[CompositeRecord] = []

        for group in groups:
            source_ids = group.get("source_record_ids", [])
            source_records = [candidate_map[uid] for uid in source_ids if uid in candidate_map]

            if len(source_records) < self._min_records:
                continue
            if len(source_records) > self._max_records_per_group:
                source_records = source_records[: self._max_records_per_group]

            reason = group.get("synthesis_reason", "")
            suggested_type = group.get("suggested_type", "composite_pattern")
            if suggested_type not in VALID_SYNTH_TYPES:
                suggested_type = "composite_pattern"

            synth_result = self._execute_synthesis(source_records, reason, suggested_type)
            if not synth_result:
                continue

            composite_id = self._make_composite_id(
                [u.record_id for u in source_records], synth_result.get("semantic_text", "")
            )

            composite = CompositeRecord(
                composite_id=composite_id,
                memory_type=suggested_type,
                semantic_text=synth_result.get("semantic_text", ""),
                normalized_text=synth_result.get("normalized_text", ""),
                source_record_ids=[u.record_id for u in source_records],
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
            self._sqlite.upsert_synthesized(composite)
            # 写入向量索引
            self._vector.upsert_synthesized(
                composite_id=composite.composite_id,
                user_id=composite.user_id,
                memory_type=composite.memory_type,
                semantic_text=composite.semantic_text,
                normalized_text=composite.normalized_text,
            )

            synthesized.append(composite)

        return synthesized

    def _judge_synthesis(
        self, candidates: list[MemoryRecord],
    ) -> list[dict[str, Any]]:
        """LLM 判断候选集是否可融合 + 分组。"""
        records_json = json.dumps(
            [
                {
                    "record_id": u.record_id,
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

        user_content = f"<RECORDS>\n{records_json}\n</RECORDS>"

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
        source_records: list[MemoryRecord],
        reason: str,
        suggested_type: str,
    ) -> dict[str, Any] | None:
        """LLM 执行融合，返回聚合结果。"""
        records_json = json.dumps(
            [
                {
                    "record_id": u.record_id,
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
                for u in source_records
            ],
            ensure_ascii=False,
            indent=2,
        )

        user_content = (
            f"<SOURCE_RECORDS>\n{records_json}\n</SOURCE_RECORDS>\n\n"
            f"<SYNTHESIS_REASON>\n{reason}\n</SYNTHESIS_REASON>\n\n"
            f"<SUGGESTED_TYPE>\n{suggested_type}\n</SUGGESTED_TYPE>"
        )

        response = self._llm.generate([
            {"role": "system", "content": get_synthesis_execute_system()},
            {"role": "user", "content": user_content},
        ])

        try:
            return self._parse_json(response)
        except (ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _make_composite_id(source_ids: list[str], semantic_text: str) -> str:
        raw = "|".join(sorted(source_ids)) + "|" + semantic_text
        return "comp_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
