"""Record Fusion Engine。

在 ingest 完成后，对同一用户的记忆执行在线聚合：
1. 检测新写入 records 与已有 records 之间是否可融合
2. LLM 判断融合可行性 + 分组（同一轮输出互不重叠 group）
3. LLM 执行融合，生成 CompositeRecord
4. 对新 record / 新 composite 与已有 root composite 再做多轮层级融合
5. 持久化直接 child composite 关系，形成层级树
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
    VALID_MEMORY_TYPES,
    VALID_SYNTH_TYPES,
)
from src.memory.semantic.prompts import (
    SYNTHESIS_JUDGE_SYSTEM,
    SYNTHESIS_EXECUTE_SYSTEM,
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
        max_hierarchy_rounds: int = 2,
    ):
        self._llm = llm
        self._sqlite = sqlite_store
        self._vector = vector_index
        self._similarity_threshold = similarity_threshold
        self._min_records = min_records_for_synthesis
        self._max_records_per_group = max_records_per_group
        self._max_hierarchy_rounds = max(0, int(max_hierarchy_rounds))
        self._last_run_stats = self._empty_run_stats()

    @staticmethod
    def _empty_run_stats() -> dict[str, Any]:
        return {
            "expired_record_ids": [],
            "updated_record_ids": [],
            "invalidated_composite_ids": [],
        }

    def get_last_run_stats(self) -> dict[str, Any]:
        return {
            "expired_record_ids": list(self._last_run_stats.get("expired_record_ids", [])),
            "updated_record_ids": list(self._last_run_stats.get("updated_record_ids", [])),
            "invalidated_composite_ids": list(self._last_run_stats.get("invalidated_composite_ids", [])),
        }

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
        self._last_run_stats = self._empty_run_stats()
        if not new_records:
            return []

        existing_composites = self._sqlite.list_synthesized(user_id=user_id)
        covered_record_ids, _ = self._build_existing_hierarchy_maps(existing_composites)

        active_new_records, updated_existing_records = self._resolve_conflicts_on_ingest(
            new_records,
            user_id=user_id,
        )
        frontier_records_map = {
            record.record_id: record
            for record in [*active_new_records, *updated_existing_records]
            if str(record.record_id or "").strip() and not record.expired
        }
        if not frontier_records_map:
            return []

        if self._last_run_stats.get("invalidated_composite_ids"):
            existing_composites = self._sqlite.list_synthesized(user_id=user_id)
            covered_record_ids, _ = self._build_existing_hierarchy_maps(existing_composites)

        frontier_records = list(frontier_records_map.values())

        synthesized = self._synthesize_records_on_ingest(
            frontier_records,
            user_id=user_id,
            covered_record_ids=covered_record_ids,
        )
        if self._max_hierarchy_rounds <= 0:
            return synthesized

        all_synthesized = list(synthesized)
        covered_by_first_layer = {
            source_id
            for composite in synthesized
            for source_id in composite.source_record_ids
        }
        frontier: list[MemoryRecord | CompositeRecord] = list(synthesized)
        frontier.extend(
            record
            for record in frontier_records
            if record.record_id not in covered_by_first_layer
        )
        seen_ids = {item.composite_id for item in synthesized}

        for _ in range(self._max_hierarchy_rounds):
            next_level = self._synthesize_hierarchy_on_ingest(frontier, user_id=user_id)
            next_level = [item for item in next_level if item.composite_id not in seen_ids]
            if not next_level:
                break

            all_synthesized.extend(next_level)
            frontier = list(next_level)
            seen_ids.update(item.composite_id for item in next_level)

        return all_synthesized

    def _synthesize_records_on_ingest(
        self,
        new_records: list[MemoryRecord],
        *,
        user_id: str = "",
        covered_record_ids: set[str] | None = None,
    ) -> list[CompositeRecord]:
        """第一层融合：new/existing records -> composite。"""
        new_records = [record for record in new_records if not record.expired]
        if not new_records:
            return []

        covered_record_ids = {
            str(record_id or "").strip()
            for record_id in (covered_record_ids or set())
            if str(record_id or "").strip()
        }

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
                if r.record_id in covered_record_ids:
                    continue
                if r.record_id not in candidate_ids:
                    candidate_ids.add(r.record_id)
                    candidate_map[r.record_id] = r

        candidates = list(candidate_map.values())

        if len(candidates) < self._min_records:
            return []

        # LLM 判断合成可行性
        decision = self._judge_synthesis(
            candidates,
            new_record_ids={record.record_id for record in new_records},
        )
        groups = self._normalize_disjoint_groups(
            decision.get("groups", []),
            allowed_ids=candidate_map.keys(),
        )
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

            flattened_source_ids = self._flatten_source_record_ids(source_records)
            if self._is_source_group_already_covered(flattened_source_ids):
                continue

            synth_result = self._execute_synthesis(source_records, reason, suggested_type)
            if not synth_result:
                continue

            composite_id = self._make_composite_id(
                flattened_source_ids,
                synth_result.get("semantic_text", ""),
            )

            composite = CompositeRecord(
                composite_id=composite_id,
                memory_type=suggested_type,
                semantic_text=synth_result.get("semantic_text", ""),
                normalized_text=synth_result.get("normalized_text", ""),
                source_record_ids=flattened_source_ids,
                child_composite_ids=[],
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

            self._persist_composite(composite)
            synthesized.append(composite)

        return synthesized

    def _synthesize_hierarchy_on_ingest(
        self,
        new_items: list[MemoryRecord | CompositeRecord],
        *,
        user_id: str = "",
    ) -> list[CompositeRecord]:
        """层级融合：new record/composite + related root composites -> parent composite。"""
        if not new_items:
            return []

        existing_composites = self._sqlite.list_synthesized(user_id=user_id)
        _, child_to_parent = self._build_existing_hierarchy_maps(existing_composites)

        candidate_map: dict[str, MemoryRecord | CompositeRecord] = {}
        for item in new_items:
            item_id = self._item_id(item)
            if not item_id:
                continue
            if isinstance(item, CompositeRecord) and item_id in child_to_parent:
                continue
            candidate_map[item_id] = item

        if len(candidate_map) < self._min_records:
            return []

        for item in list(candidate_map.values()):
            for related in self._find_related_root_composites(
                item,
                user_id=user_id,
                child_to_parent=child_to_parent,
            ):
                item_id = self._item_id(item)
                if related.composite_id == item_id:
                    continue
                if self._has_coverage_relation(
                    self._flatten_source_record_ids([item]),
                    related.source_record_ids,
                ):
                    continue
                candidate_map.setdefault(related.composite_id, related)

        candidates = list(candidate_map.values())
        if len(candidates) < self._min_records:
            return []

        decision = self._judge_synthesis(candidates)
        groups = self._normalize_disjoint_groups(
            decision.get("groups", []),
            allowed_ids=candidate_map.keys(),
        )
        if not groups:
            return []

        now_iso = datetime.now(timezone.utc).isoformat()
        synthesized: list[CompositeRecord] = []

        for group in groups:
            source_ids = [
                str(uid or "").strip()
                for uid in group.get("source_record_ids", [])
                if str(uid or "").strip()
            ]
            source_items = [candidate_map[uid] for uid in source_ids if uid in candidate_map]

            if len(source_items) < self._min_records:
                continue
            if len(source_items) > self._max_records_per_group:
                source_items = source_items[: self._max_records_per_group]

            flattened_source_ids = self._flatten_source_record_ids(source_items)
            if len(flattened_source_ids) < self._min_records:
                continue

            reason = group.get("synthesis_reason", "")
            suggested_type = group.get("suggested_type", "composite_pattern")
            if suggested_type not in VALID_SYNTH_TYPES:
                suggested_type = "composite_pattern"

            if self._is_source_group_already_covered(flattened_source_ids):
                continue

            synth_result = self._execute_synthesis(source_items, reason, suggested_type)
            if not synth_result:
                continue

            composite_id = self._make_composite_id(
                flattened_source_ids,
                synth_result.get("semantic_text", ""),
            )
            if composite_id in candidate_map:
                continue

            child_composite_ids = sorted(
                self._item_id(item)
                for item in source_items
                if isinstance(item, CompositeRecord)
            )

            composite = CompositeRecord(
                composite_id=composite_id,
                memory_type=suggested_type,
                semantic_text=synth_result.get("semantic_text", ""),
                normalized_text=synth_result.get("normalized_text", ""),
                source_record_ids=flattened_source_ids,
                child_composite_ids=child_composite_ids,
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

            self._persist_composite(composite)
            synthesized.append(composite)

        return synthesized

    def _resolve_conflicts_on_ingest(
        self,
        new_records: list[MemoryRecord],
        *,
        user_id: str = "",
    ) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
        active_new_records = [record for record in new_records if not record.expired]
        if not active_new_records:
            return [], []

        new_record_ids = {
            str(record.record_id or "").strip()
            for record in active_new_records
            if str(record.record_id or "").strip()
        }
        candidate_map: dict[str, MemoryRecord] = {
            record.record_id: record
            for record in active_new_records
            if str(record.record_id or "").strip()
        }
        existing_record_ids: set[str] = set()

        for record in active_new_records:
            for related in self._find_related_existing_records(
                record,
                user_id=user_id,
                exclude_ids=new_record_ids,
            ):
                if related.expired:
                    continue
                candidate_map.setdefault(related.record_id, related)
                existing_record_ids.add(related.record_id)

        if not existing_record_ids:
            return active_new_records, []

        decision = self._judge_synthesis(
            list(candidate_map.values()),
            new_record_ids=new_record_ids,
        )
        conflicts = self._normalize_conflicts(
            decision.get("conflicts", []),
            allowed_ids=candidate_map.keys(),
            new_ids=new_record_ids,
            existing_ids=existing_record_ids,
        )
        if not conflicts:
            return active_new_records, []

        now_iso = datetime.now(timezone.utc).isoformat()
        updated_records: list[MemoryRecord] = []
        expired_record_ids: set[str] = set()
        invalidated_composite_ids: set[str] = set()

        for conflict in conflicts:
            anchor_id = conflict["anchor_record_id"]
            incoming_ids = conflict["incoming_record_ids"]
            anchor_record = candidate_map.get(anchor_id)
            incoming_records = [candidate_map[record_id] for record_id in incoming_ids if record_id in candidate_map]
            if anchor_record is None or not incoming_records:
                continue

            resolved = self._execute_synthesis(
                [anchor_record, *incoming_records],
                conflict.get("conflict_reason", ""),
                anchor_record.memory_type,
                resolution_mode="conflict_update",
                target_record_id=anchor_id,
                new_record_ids=new_record_ids,
            )
            if not resolved:
                continue

            resolved_memory_type = str(resolved.get("resolved_memory_type") or anchor_record.memory_type).strip()
            if resolved_memory_type not in VALID_MEMORY_TYPES:
                resolved_memory_type = anchor_record.memory_type

            merged_source_role = self._merge_source_roles(
                [anchor_record, *incoming_records],
                preferred=getattr(incoming_records[-1], "source_role", "") if incoming_records else anchor_record.source_role,
            )
            merged_session = next(
                (
                    record.source_session
                    for record in reversed(incoming_records)
                    if str(record.source_session or "").strip()
                ),
                anchor_record.source_session,
            )
            merged_evidence_turns = sorted({
                *[int(turn) for turn in (anchor_record.evidence_turn_range or []) if isinstance(turn, int)],
                *[
                    int(turn)
                    for record in incoming_records
                    for turn in (record.evidence_turn_range or [])
                    if isinstance(turn, int)
                ],
            })

            updated_record = MemoryRecord(
                record_id=anchor_record.record_id,
                memory_type=resolved_memory_type,
                semantic_text=str(resolved.get("semantic_text") or anchor_record.semantic_text).strip(),
                normalized_text=str(resolved.get("normalized_text") or anchor_record.normalized_text).strip(),
                entities=[str(v) for v in (resolved.get("entities") or anchor_record.entities or []) if str(v or "").strip()],
                temporal=resolved.get("temporal") or dict(anchor_record.temporal),
                task_tags=[str(v) for v in (resolved.get("task_tags") or anchor_record.task_tags or []) if str(v or "").strip()],
                tool_tags=[str(v) for v in (resolved.get("tool_tags") or anchor_record.tool_tags or []) if str(v or "").strip()],
                constraint_tags=[str(v) for v in (resolved.get("constraint_tags") or anchor_record.constraint_tags or []) if str(v or "").strip()],
                failure_tags=[str(v) for v in (resolved.get("failure_tags") or anchor_record.failure_tags or []) if str(v or "").strip()],
                affordance_tags=[str(v) for v in (resolved.get("affordance_tags") or anchor_record.affordance_tags or []) if str(v or "").strip()],
                confidence=float(resolved.get("confidence") or anchor_record.confidence or 1.0),
                evidence_turn_range=merged_evidence_turns,
                source_session=merged_session,
                source_role=merged_source_role,
                user_id=anchor_record.user_id,
                created_at=anchor_record.created_at,
                updated_at=now_iso,
                retrieval_count=anchor_record.retrieval_count,
                retrieval_hit_count=anchor_record.retrieval_hit_count,
                action_success_count=anchor_record.action_success_count,
                action_fail_count=anchor_record.action_fail_count,
                last_retrieved_at=anchor_record.last_retrieved_at,
                expired=False,
                expired_at="",
                expired_reason="",
            )

            self._sqlite.upsert_record(updated_record)
            self._vector.upsert(
                record_id=updated_record.record_id,
                user_id=updated_record.user_id,
                memory_type=updated_record.memory_type,
                semantic_text=updated_record.semantic_text,
                normalized_text=updated_record.normalized_text,
            )
            updated_records.append(updated_record)

            for incoming_record in incoming_records:
                self._sqlite.expire_record(
                    incoming_record.record_id,
                    expired_at=now_iso,
                    expired_reason=f"conflict_update:{anchor_record.record_id}",
                )
                self._vector.mark_expired(incoming_record.record_id)
                expired_record_ids.add(incoming_record.record_id)

            affected_source_ids = [anchor_record.record_id, *incoming_ids]
            stale_composites = self._sqlite.get_synthesized_by_source(affected_source_ids)
            stale_composite_ids = sorted({
                composite.composite_id
                for composite in stale_composites
                if str(composite.composite_id or "").strip()
            })
            if stale_composite_ids:
                self._sqlite.delete_synthesized_many(stale_composite_ids, user_id=user_id)
                for composite_id in stale_composite_ids:
                    self._vector.delete_synthesized(composite_id)
                invalidated_composite_ids.update(stale_composite_ids)

        surviving_new_records = [
            record
            for record in active_new_records
            if record.record_id not in expired_record_ids
        ]

        self._last_run_stats = {
            "expired_record_ids": sorted(expired_record_ids),
            "updated_record_ids": sorted({record.record_id for record in updated_records}),
            "invalidated_composite_ids": sorted(invalidated_composite_ids),
        }

        return surviving_new_records, updated_records

    def _find_related_existing_records(
        self,
        record: MemoryRecord,
        *,
        user_id: str = "",
        exclude_ids: set[str] | None = None,
    ) -> list[MemoryRecord]:
        exclude_ids = {
            str(record_id or "").strip()
            for record_id in (exclude_ids or set())
            if str(record_id or "").strip()
        }
        related: dict[str, MemoryRecord] = {}
        queries: list[str] = []

        for text in [
            record.normalized_text,
            record.semantic_text,
            " ".join(record.entities),
            " ".join([
                *record.task_tags,
                *record.tool_tags,
                *record.constraint_tags,
                *record.failure_tags,
                *record.affordance_tags,
            ]),
        ]:
            cleaned = str(text or "").strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)

        for query_text in queries[:4]:
            for result in self._sqlite.fulltext_search(
                query_text,
                user_id=user_id,
                limit=6,
            ):
                record_id = str(result.get("record_id") or "").strip()
                if not record_id or record_id == record.record_id or record_id in exclude_ids:
                    continue
                fetched = self._sqlite.get_record(record_id)
                if fetched is not None and not fetched.expired:
                    related[record_id] = fetched

            for column in ("vector", "normalized_vector"):
                for result in self._vector.search(
                    query_text,
                    user_id=user_id,
                    column=column,
                    limit=6,
                ):
                    record_id = str(result.get("record_id") or "").strip()
                    if not record_id or record_id == record.record_id or record_id in exclude_ids:
                        continue
                    fetched = self._sqlite.get_record(record_id)
                    if fetched is not None and not fetched.expired:
                        related[record_id] = fetched

        return list(related.values())

    def _persist_composite(self, composite: CompositeRecord) -> None:
        self._sqlite.upsert_synthesized(composite)
        self._vector.upsert_synthesized(
            composite_id=composite.composite_id,
            user_id=composite.user_id,
            memory_type=composite.memory_type,
            semantic_text=composite.semantic_text,
            normalized_text=composite.normalized_text,
        )

    def _is_source_group_already_covered(self, source_record_ids: list[str]) -> bool:
        source_ids_set = {
            str(uid or "").strip()
            for uid in source_record_ids
            if str(uid or "").strip()
        }
        if len(source_ids_set) < self._min_records:
            return True

        existing_composites = self._sqlite.get_synthesized_by_source(list(source_ids_set))
        for existing in existing_composites:
            existing_set = {
                str(uid or "").strip()
                for uid in existing.source_record_ids
                if str(uid or "").strip()
            }
            if not existing_set:
                continue
            if source_ids_set.issubset(existing_set):
                return True

            overlap = len(source_ids_set & existing_set)
            union = len(source_ids_set | existing_set)
            jaccard = overlap / union if union > 0 else 0.0
            if jaccard >= 0.7:
                return True

        return False

    def _find_related_root_composites(
        self,
        item: MemoryRecord | CompositeRecord,
        *,
        user_id: str = "",
        child_to_parent: dict[str, str] | None = None,
    ) -> list[CompositeRecord]:
        related: dict[str, CompositeRecord] = {}
        child_to_parent = child_to_parent or {}

        queries: list[str] = []
        for text in [
            item.normalized_text,
            item.semantic_text,
            " ".join(item.entities),
            " ".join([*item.task_tags, *item.tool_tags, *item.constraint_tags]),
        ]:
            cleaned = str(text or "").strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)

        for query_text in queries[:4]:
            fts_results = self._sqlite.fulltext_search_synthesized(
                query_text,
                user_id=user_id,
                limit=6,
            )
            for result in fts_results:
                composite_id = str(result.get("composite_id", "")).strip()
                if not composite_id or composite_id == self._item_id(item):
                    continue
                if composite_id in child_to_parent:
                    continue
                fetched = self._sqlite.get_synthesized(composite_id)
                if fetched is not None:
                    related[composite_id] = fetched

            for column in ("vector", "normalized_vector"):
                vector_results = self._vector.search_synthesized(
                    query_text,
                    user_id=user_id,
                    column=column,
                    limit=6,
                )
                for result in vector_results:
                    composite_id = str(result.get("composite_id", "")).strip()
                    if not composite_id or composite_id == self._item_id(item):
                        continue
                    if composite_id in child_to_parent:
                        continue
                    fetched = self._sqlite.get_synthesized(composite_id)
                    if fetched is not None:
                        related[composite_id] = fetched

        return list(related.values())

    def _build_existing_hierarchy_maps(
        self,
        composites: list[CompositeRecord],
    ) -> tuple[set[str], dict[str, str]]:
        covered_record_ids: set[str] = set()
        source_sets: dict[str, set[str]] = {}
        explicit_candidates: dict[str, set[str]] = {}

        for composite in composites:
            composite_id = self._item_id(composite)
            if not composite_id:
                continue
            source_set = {
                str(source_id or "").strip()
                for source_id in composite.source_record_ids
                if str(source_id or "").strip()
            }
            source_sets[composite_id] = source_set
            covered_record_ids.update(source_set)

        for composite in composites:
            parent_id = self._item_id(composite)
            parent_sources = source_sets.get(parent_id, set())
            for child_id in composite.child_composite_ids:
                normalized_child = str(child_id or "").strip()
                child_sources = source_sets.get(normalized_child, set())
                if (
                    normalized_child
                    and normalized_child != parent_id
                    and child_sources
                    and child_sources.issubset(parent_sources)
                    and len(parent_sources) > len(child_sources)
                ):
                    explicit_candidates.setdefault(normalized_child, set()).add(parent_id)

        inferred_candidates: dict[str, set[str]] = {}
        composite_ids = list(source_sets.keys())
        for parent_id in composite_ids:
            if any(parent_id in parents for parents in explicit_candidates.values()):
                continue
            parent_sources = source_sets[parent_id]
            if len(parent_sources) < self._min_records:
                continue

            eligible_children = [
                child_id
                for child_id in composite_ids
                if child_id != parent_id
                and source_sets[child_id]
                and source_sets[child_id].issubset(parent_sources)
                and len(parent_sources) > len(source_sets[child_id])
            ]

            for child_id in eligible_children:
                child_sources = source_sets[child_id]
                has_intermediate = any(
                    mid_id not in {child_id, parent_id}
                    and source_sets[mid_id]
                    and child_sources.issubset(source_sets[mid_id])
                    and source_sets[mid_id].issubset(parent_sources)
                    and len(parent_sources) > len(source_sets[mid_id]) > len(child_sources)
                    for mid_id in eligible_children
                )
                if not has_intermediate:
                    inferred_candidates.setdefault(child_id, set()).add(parent_id)

        parent_candidates = explicit_candidates or inferred_candidates
        if explicit_candidates:
            for child_id, parents in inferred_candidates.items():
                if child_id not in parent_candidates:
                    parent_candidates[child_id] = set(parents)

        child_to_parent: dict[str, str] = {}
        for child_id, parents in parent_candidates.items():
            valid_parents = [parent_id for parent_id in parents if parent_id in source_sets]
            if not valid_parents:
                continue
            best_parent = min(
                valid_parents,
                key=lambda parent_id: (len(source_sets.get(parent_id, set())) or 10**9, parent_id),
            )
            child_to_parent[child_id] = best_parent

        return covered_record_ids, child_to_parent

    def _normalize_disjoint_groups(
        self,
        groups: list[dict[str, Any]],
        *,
        allowed_ids: Any,
        blocked_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        allowed = {
            str(item_id or "").strip()
            for item_id in allowed_ids
            if str(item_id or "").strip()
        }
        blocked_ids = {
            str(item_id or "").strip()
            for item_id in (blocked_ids or set())
            if str(item_id or "").strip()
        }
        normalized_groups: list[dict[str, Any]] = []
        for group in groups or []:
            raw_ids = group.get("source_record_ids", []) or []
            source_ids: list[str] = []
            seen: set[str] = set()
            for raw_id in raw_ids:
                normalized = str(raw_id or "").strip()
                if (
                    not normalized
                    or normalized not in allowed
                    or normalized in seen
                    or normalized in blocked_ids
                ):
                    continue
                seen.add(normalized)
                source_ids.append(normalized)

            if len(source_ids) < self._min_records:
                continue

            normalized_groups.append({
                "source_record_ids": source_ids,
                "synthesis_reason": str(group.get("synthesis_reason", "") or "").strip(),
                "suggested_type": str(group.get("suggested_type", "composite_pattern") or "composite_pattern").strip(),
            })

        normalized_groups.sort(
            key=lambda group: (
                -len(group["source_record_ids"]),
                group.get("suggested_type", ""),
                group.get("source_record_ids", []),
            )
        )

        used_ids: set[str] = set()
        disjoint_groups: list[dict[str, Any]] = []
        for group in normalized_groups:
            source_ids = group.get("source_record_ids", [])
            if any(source_id in used_ids for source_id in source_ids):
                continue
            disjoint_groups.append(group)
            used_ids.update(source_ids)

        return disjoint_groups

    def _normalize_conflicts(
        self,
        conflicts: list[dict[str, Any]],
        *,
        allowed_ids: Any,
        new_ids: set[str],
        existing_ids: set[str],
    ) -> list[dict[str, Any]]:
        allowed = {
            str(item_id or "").strip()
            for item_id in allowed_ids
            if str(item_id or "").strip()
        }
        new_ids = {
            str(item_id or "").strip()
            for item_id in new_ids
            if str(item_id or "").strip()
        }
        existing_ids = {
            str(item_id or "").strip()
            for item_id in existing_ids
            if str(item_id or "").strip()
        }

        merged_by_anchor: dict[str, dict[str, Any]] = {}
        for conflict in conflicts or []:
            anchor_id = str(conflict.get("anchor_record_id") or "").strip()
            if not anchor_id or anchor_id not in allowed or anchor_id not in existing_ids:
                continue

            incoming_ids: list[str] = []
            seen_incoming: set[str] = set()
            for raw_id in conflict.get("incoming_record_ids", []) or []:
                incoming_id = str(raw_id or "").strip()
                if (
                    not incoming_id
                    or incoming_id not in allowed
                    or incoming_id not in new_ids
                    or incoming_id == anchor_id
                    or incoming_id in seen_incoming
                ):
                    continue
                seen_incoming.add(incoming_id)
                incoming_ids.append(incoming_id)

            if not incoming_ids:
                continue

            entry = merged_by_anchor.setdefault(
                anchor_id,
                {
                    "anchor_record_id": anchor_id,
                    "incoming_record_ids": [],
                    "conflict_reason": "",
                    "resolution_mode": "update_existing",
                },
            )
            entry["incoming_record_ids"] = self._merge_unique_str(
                entry["incoming_record_ids"],
                incoming_ids,
            )
            if not entry["conflict_reason"]:
                entry["conflict_reason"] = str(conflict.get("conflict_reason") or "").strip()

        normalized = sorted(
            merged_by_anchor.values(),
            key=lambda item: (-len(item["incoming_record_ids"]), item["anchor_record_id"]),
        )

        disjoint_conflicts: list[dict[str, Any]] = []
        used_anchors: set[str] = set()
        used_incoming: set[str] = set()
        for conflict in normalized:
            anchor_id = conflict["anchor_record_id"]
            incoming_ids = conflict["incoming_record_ids"]
            if anchor_id in used_anchors or any(record_id in used_incoming for record_id in incoming_ids):
                continue
            disjoint_conflicts.append(conflict)
            used_anchors.add(anchor_id)
            used_incoming.update(incoming_ids)

        return disjoint_conflicts

    @staticmethod
    def _merge_unique_str(primary: list[str], secondary: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for value in list(primary or []) + list(secondary or []):
            item = str(value or "").strip()
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    @staticmethod
    def _merge_source_roles(
        records: list[MemoryRecord],
        *,
        preferred: str = "",
    ) -> str:
        preferred = str(preferred or "").strip()
        if preferred:
            return preferred

        roles = {
            str(record.source_role or "").strip()
            for record in records
            if str(record.source_role or "").strip()
        }
        if not roles:
            return ""
        if len(roles) == 1:
            return next(iter(roles))
        return "both"

    @staticmethod
    def _item_id(item: MemoryRecord | CompositeRecord) -> str:
        if isinstance(item, CompositeRecord):
            return str(item.composite_id or "").strip()
        return str(item.record_id or "").strip()

    @staticmethod
    def _flatten_source_record_ids(
        source_items: list[MemoryRecord | CompositeRecord],
    ) -> list[str]:
        source_ids: set[str] = set()
        for item in source_items:
            if isinstance(item, CompositeRecord):
                raw_ids = item.source_record_ids
            else:
                raw_ids = [item.record_id]

            for raw_id in raw_ids:
                normalized = str(raw_id or "").strip()
                if normalized:
                    source_ids.add(normalized)

        return sorted(source_ids)

    @staticmethod
    def _has_coverage_relation(left_ids: list[str], right_ids: list[str]) -> bool:
        left_set = {str(uid or "").strip() for uid in left_ids if str(uid or "").strip()}
        right_set = {str(uid or "").strip() for uid in right_ids if str(uid or "").strip()}
        if not left_set or not right_set:
            return False
        return left_set.issubset(right_set) or right_set.issubset(left_set)

    def _judge_synthesis(
        self,
        candidates: list[MemoryRecord | CompositeRecord],
        *,
        new_record_ids: set[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """LLM 判断候选集是否可融合或做冲突更新。"""
        records_json = json.dumps(
            [
                self._serialize_candidate_for_judge(item, new_record_ids=new_record_ids)
                for item in candidates
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
            return {
                "groups": parsed.get("groups", []) or [],
                "conflicts": parsed.get("conflicts", []) or [],
            }
        except (ValueError, json.JSONDecodeError):
            return {"groups": [], "conflicts": []}

    def _execute_synthesis(
        self,
        source_records: list[MemoryRecord | CompositeRecord],
        reason: str,
        suggested_type: str,
        *,
        resolution_mode: str = "synthesize",
        target_record_id: str = "",
        new_record_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """LLM 执行融合，返回聚合结果。"""
        records_json = json.dumps(
            [
                self._serialize_candidate_for_execute(item, new_record_ids=new_record_ids)
                for item in source_records
            ],
            ensure_ascii=False,
            indent=2,
        )

        user_content = (
            f"<SOURCE_RECORDS>\n{records_json}\n</SOURCE_RECORDS>\n\n"
            f"<SYNTHESIS_REASON>\n{reason}\n</SYNTHESIS_REASON>\n\n"
            f"<SUGGESTED_TYPE>\n{suggested_type}\n</SUGGESTED_TYPE>\n\n"
            f"<RESOLUTION_MODE>\n{resolution_mode}\n</RESOLUTION_MODE>\n\n"
            f"<TARGET_RECORD_ID>\n{target_record_id}\n</TARGET_RECORD_ID>"
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
    def _make_composite_id(source_ids: list[str], semantic_text: str) -> str:
        raw = "|".join(sorted(source_ids)) + "|" + semantic_text
        return "comp_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _serialize_candidate_for_judge(
        item: MemoryRecord | CompositeRecord,
        *,
        new_record_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        new_record_ids = {
            str(record_id or "").strip()
            for record_id in (new_record_ids or set())
            if str(record_id or "").strip()
        }
        if isinstance(item, CompositeRecord):
            item_id = item.composite_id
            source_record_ids = list(item.source_record_ids)
            item_kind = "composite"
        else:
            item_id = item.record_id
            source_record_ids = [item.record_id]
            item_kind = "record"

        return {
            "record_id": item_id,
            "item_kind": item_kind,
            "ingest_status": "new" if item_id in new_record_ids else "existing",
            "memory_type": item.memory_type,
            "semantic_text": item.semantic_text,
            "normalized_text": item.normalized_text,
            "entities": item.entities,
            "temporal": item.temporal,
            "task_tags": item.task_tags,
            "tool_tags": item.tool_tags,
            "constraint_tags": item.constraint_tags,
            "failure_tags": item.failure_tags,
            "affordance_tags": item.affordance_tags,
            "source_record_ids": source_record_ids,
            "child_composite_ids": list(item.child_composite_ids) if isinstance(item, CompositeRecord) else [],
        }

    @staticmethod
    def _serialize_candidate_for_execute(
        item: MemoryRecord | CompositeRecord,
        *,
        new_record_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        new_record_ids = {
            str(record_id or "").strip()
            for record_id in (new_record_ids or set())
            if str(record_id or "").strip()
        }
        if isinstance(item, CompositeRecord):
            item_id = item.composite_id
            source_record_ids = list(item.source_record_ids)
            item_kind = "composite"
        else:
            item_id = item.record_id
            source_record_ids = [item.record_id]
            item_kind = "record"

        return {
            "record_id": item_id,
            "item_kind": item_kind,
            "ingest_status": "new" if item_id in new_record_ids else "existing",
            "memory_type": item.memory_type,
            "semantic_text": item.semantic_text,
            "normalized_text": item.normalized_text,
            "entities": item.entities,
            "temporal": item.temporal,
            "task_tags": item.task_tags,
            "tool_tags": item.tool_tags,
            "constraint_tags": item.constraint_tags,
            "failure_tags": item.failure_tags,
            "affordance_tags": item.affordance_tags,
            "confidence": item.confidence,
            "source_record_ids": source_record_ids,
            "child_composite_ids": list(item.child_composite_ids) if isinstance(item, CompositeRecord) else [],
        }

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
