"""Record Fusion Engine — Embedding Similarity Based.

记忆融合流程（纯 embedding 数学，无 LLM 调用）：
1. 去重：new record vs existing records，距离 < dedup 阈值 → 过期旧记录
2. 聚类：距离 < synthesis 阈值 → Union-Find 连通分量 → cluster
3. CompositeRecord：代表记录文本 + 合并 metadata
4. 层级聚合（可选）：composite 之间再做一轮相似度聚类
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from src.memory.semantic.models import (
    MemoryRecord,
    CompositeRecord,
    VALID_SYNTH_TYPES,
)
from src.memory.semantic.sqlite_store import SQLiteSemanticStore
from src.memory.semantic.vector_index import LanceVectorIndex


class RecordFusionEngine:
    """记录融合引擎：embedding 余弦相似度聚类 + 去重，不使用 LLM。"""

    def __init__(
        self,
        sqlite_store: SQLiteSemanticStore,
        vector_index: LanceVectorIndex,
        *,
        synthesis_similarity: float = 0.75,
        dedup_threshold: float = 0.85,
        min_records_for_synthesis: int = 2,
        max_records_per_group: int = 8,
        max_hierarchy_rounds: int = 2,
    ):
        self._sqlite = sqlite_store
        self._vector = vector_index

        # 将余弦相似度阈值转换为 L2 距离阈值
        # 对归一化向量: L2_distance = 2 * (1 - cosine_similarity)
        self._synthesis_max_dist = 2.0 * (1.0 - synthesis_similarity)
        self._dedup_max_dist = 2.0 * (1.0 - dedup_threshold)
        self._min_records = min_records_for_synthesis
        self._max_per_group = max_records_per_group
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

    # ──────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────

    def synthesize_on_ingest(
        self,
        new_records: list[MemoryRecord],
    ) -> list[CompositeRecord]:
        """在新 records 写入后触发融合流程。

        1. 去重：检测近似重复，过期旧记录
        2. 聚类：embedding 距离 < synthesis 阈值 → cluster
        3. 每个 cluster → CompositeRecord（代表记录文本，合并 metadata）
        4. 层级聚合（可选）：composite 之间再做一轮

        Returns:
            生成的 CompositeRecord 列表
        """
        self._last_run_stats = self._empty_run_stats()
        if not new_records:
            return []

        # Step 1: 去重
        surviving_records, expired_ids, invalidated_cids = self._dedup_on_ingest(new_records)
        self._last_run_stats["expired_record_ids"] = sorted(expired_ids)
        self._last_run_stats["invalidated_composite_ids"] = sorted(invalidated_cids)

        if not surviving_records:
            return []

        # Step 2: 找到已有 composite 覆盖的 record_ids
        existing_composites = self._sqlite.list_synthesized()
        covered_record_ids: set[str] = set()
        for comp in existing_composites:
            for rid in comp.source_record_ids:
                rid_s = str(rid or "").strip()
                if rid_s:
                    covered_record_ids.add(rid_s)

        # Step 3: 聚类 + 构建 CompositeRecord
        composites = self._cluster_and_build(
            surviving_records,
            covered_record_ids=covered_record_ids,
        )

        if self._max_hierarchy_rounds <= 0 or not composites:
            return composites

        # Step 4: 层级聚合
        all_composites = list(composites)
        covered_by_first: set[str] = set()
        for c in composites:
            covered_by_first.update(c.source_record_ids)

        frontier: list[MemoryRecord | CompositeRecord] = list(composites)
        frontier.extend(
            r for r in surviving_records
            if r.record_id not in covered_by_first
        )
        seen_ids = {c.composite_id for c in composites}

        for _ in range(self._max_hierarchy_rounds):
            next_level = self._hierarchy_round(frontier, seen_ids)
            if not next_level:
                break
            all_composites.extend(next_level)
            frontier = list(next_level)
            seen_ids.update(c.composite_id for c in next_level)

        return all_composites

    # ──────────────────────────────────────
    # Step 1: 去重
    # ──────────────────────────────────────

    def _dedup_on_ingest(
        self,
        new_records: list[MemoryRecord],
    ) -> tuple[list[MemoryRecord], set[str], set[str]]:
        """对每个新 record，检测与已有记录的近似重复。

        判定条件：embedding 距离 < dedup 阈值 AND 同一 memory_type。
        处理：过期旧记录（保留新记录，因为新记录包含最新信息）。

        Returns:
            (surviving_new_records, expired_record_ids, invalidated_composite_ids)
        """
        expired_ids: set[str] = set()
        invalidated_cids: set[str] = set()
        new_ids = {
            str(r.record_id or "").strip()
            for r in new_records
            if str(r.record_id or "").strip()
        }
        now_iso = datetime.now(timezone.utc).isoformat()

        valid_records = [r for r in new_records if not r.expired and r.record_id]

        # 并行查询：每条 record 独立调用 vector.search（只读，并发安全）
        def _search_dedup(rec: MemoryRecord) -> tuple[str, list[dict]]:
            results = self._vector.search(
                rec.semantic_text, column="vector", limit=10, include_expired=False,
            )
            return rec.record_id, results

        dedup_search_map: dict[str, list[dict]] = {}
        if valid_records:
            with ThreadPoolExecutor(max_workers=min(8, len(valid_records))) as _pool:
                for _rid, _results in _pool.map(_search_dedup, valid_records):
                    dedup_search_map[_rid] = _results

        for record in valid_records:
            results = dedup_search_map.get(record.record_id, [])
            for hit in results:
                hit_id = str(hit.get("record_id", "")).strip()
                dist = float(hit.get("_distance", 999.0))

                if not hit_id or hit_id == record.record_id or hit_id in new_ids:
                    continue
                if hit_id in expired_ids:
                    continue
                if dist > self._dedup_max_dist:
                    continue
                if hit.get("memory_type", "") != record.memory_type:
                    continue

                # 近似重复：过期旧记录
                self._sqlite.expire_record(
                    hit_id,
                    expired_at=now_iso,
                    expired_reason=f"dedup:{record.record_id}",
                )
                self._vector.mark_expired(hit_id)
                expired_ids.add(hit_id)

                # 使包含该旧记录的 composite 失效
                stale = self._sqlite.get_synthesized_by_source([hit_id])
                for comp in stale:
                    cid = str(comp.composite_id or "").strip()
                    if cid and cid not in invalidated_cids:
                        self._sqlite.delete_synthesized(cid)
                        self._vector.delete_synthesized(cid)
                        invalidated_cids.add(cid)

        surviving = [
            r for r in new_records
            if not r.expired and r.record_id not in expired_ids
        ]
        return surviving, expired_ids, invalidated_cids

    # ──────────────────────────────────────
    # Step 2-3: 聚类 + 构建 CompositeRecord
    # ──────────────────────────────────────

    def _cluster_and_build(
        self,
        new_records: list[MemoryRecord],
        *,
        covered_record_ids: set[str] | None = None,
    ) -> list[CompositeRecord]:
        """对新记录做 embedding 聚类，构建 CompositeRecord。"""
        covered = covered_record_ids or set()

        if len(new_records) < self._min_records:
            return []

        # 收集候选：新记录 + 与其相似的已有记录
        candidate_map: dict[str, MemoryRecord] = {
            r.record_id: r for r in new_records
            if str(r.record_id or "").strip()
        }
        new_ids = set(candidate_map.keys())

        # 并行 ANN 搜索，建立相似度边
        def _search_cluster(rec: MemoryRecord) -> tuple[str, list[dict]]:
            return rec.record_id, self._vector.search(
                rec.semantic_text, column="vector", limit=15, include_expired=False,
            )

        cluster_search_map: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(new_records))) as _pool:
            for _rid, _results in _pool.map(_search_cluster, new_records):
                cluster_search_map[_rid] = _results

        edges: list[tuple[str, str]] = []
        for record in new_records:
            for hit in cluster_search_map.get(record.record_id, []):
                hit_id = str(hit.get("record_id", "")).strip()
                dist = float(hit.get("_distance", 999.0))

                if not hit_id or hit_id == record.record_id:
                    continue
                if dist > self._synthesis_max_dist:
                    continue
                if hit_id in covered:
                    continue

                # 添加边
                edges.append((record.record_id, hit_id))

                # 如果是已有记录，加入候选
                if hit_id not in candidate_map:
                    fetched = self._sqlite.get_record(hit_id)
                    if fetched and not fetched.expired:
                        candidate_map[hit_id] = fetched

        if not edges:
            return []

        # Union-Find 构建连通分量
        clusters = self._union_find_clusters(edges, candidate_map.keys())

        # 过滤：每个 cluster 至少 min_records 个 record，且至少含一个新记录
        valid_clusters: list[list[str]] = []
        for cluster_ids in clusters:
            if len(cluster_ids) < self._min_records:
                continue
            if not any(rid in new_ids for rid in cluster_ids):
                continue
            valid_clusters.append(cluster_ids)

        if not valid_clusters:
            return []

        # 构建 CompositeRecord
        now_iso = datetime.now(timezone.utc).isoformat()
        composites: list[CompositeRecord] = []

        for cluster_ids in valid_clusters:
            cluster_ids = cluster_ids[: self._max_per_group]
            cluster_records = [
                candidate_map[rid] for rid in cluster_ids
                if rid in candidate_map
            ]
            if len(cluster_records) < self._min_records:
                continue

            flat_ids = sorted({r.record_id for r in cluster_records if r.record_id})
            if self._is_source_group_already_covered(flat_ids):
                continue

            composite = self._build_composite(
                cluster_records,
                flat_ids=flat_ids,
                child_composite_ids=[],
                now_iso=now_iso,
            )
            if composite:
                self._persist_composite(composite)
                composites.append(composite)

        return composites

    # ──────────────────────────────────────
    # Step 4: 层级聚合
    # ──────────────────────────────────────

    def _hierarchy_round(
        self,
        items: list[MemoryRecord | CompositeRecord],
        seen_ids: set[str],
    ) -> list[CompositeRecord]:
        """一轮层级聚合：items + 相关 root composites → parent composites。"""
        if not items:
            return []

        existing_composites = self._sqlite.list_synthesized()
        child_to_parent = self._build_child_to_parent(existing_composites)

        candidate_map: dict[str, MemoryRecord | CompositeRecord] = {}
        for item in items:
            item_id = self._item_id(item)
            if not item_id or item_id in child_to_parent:
                continue
            candidate_map[item_id] = item

        # 搜索相关的 root composites
        edges: list[tuple[str, str]] = []
        for item in list(candidate_map.values()):
            item_id = self._item_id(item)
            results = self._vector.search_synthesized(
                item.semantic_text,
                column="vector",
                limit=10,
            )
            for hit in results:
                cid = str(hit.get("composite_id", "")).strip()
                dist = float(hit.get("_distance", 999.0))

                if not cid or cid == item_id or cid in seen_ids:
                    continue
                if cid in child_to_parent:
                    continue
                if dist > self._synthesis_max_dist:
                    continue

                # 检查是否有包含关系
                if cid not in candidate_map:
                    fetched = self._sqlite.get_synthesized(cid)
                    if fetched:
                        if self._has_coverage_relation(
                            self._flatten_source_ids([item]),
                            fetched.source_record_ids,
                        ):
                            continue
                        candidate_map[cid] = fetched

                edges.append((item_id, cid))

        if not edges or len(candidate_map) < self._min_records:
            return []

        clusters = self._union_find_clusters(edges, candidate_map.keys())

        now_iso = datetime.now(timezone.utc).isoformat()
        composites: list[CompositeRecord] = []

        for cluster_ids in clusters:
            if len(cluster_ids) < self._min_records:
                continue

            cluster_ids = cluster_ids[: self._max_per_group]
            cluster_items = [
                candidate_map[uid] for uid in cluster_ids
                if uid in candidate_map
            ]
            if len(cluster_items) < self._min_records:
                continue

            flat_ids = self._flatten_source_ids(cluster_items)
            if len(flat_ids) < self._min_records:
                continue
            if self._is_source_group_already_covered(flat_ids):
                continue

            child_cids = sorted(
                self._item_id(item)
                for item in cluster_items
                if isinstance(item, CompositeRecord)
            )

            composite_id = self._make_composite_id(
                flat_ids,
                max(cluster_items, key=lambda x: (x.confidence, x.updated_at)).semantic_text,
            )
            if composite_id in seen_ids:
                continue

            composite = self._build_composite(
                cluster_items,
                flat_ids=flat_ids,
                child_composite_ids=child_cids,
                now_iso=now_iso,
            )
            if composite:
                self._persist_composite(composite)
                composites.append(composite)

        return composites

    # ──────────────────────────────────────
    # CompositeRecord 构建
    # ──────────────────────────────────────

    def _build_composite(
        self,
        items: list[MemoryRecord | CompositeRecord],
        *,
        flat_ids: list[str],
        child_composite_ids: list[str],
        now_iso: str,
    ) -> CompositeRecord | None:
        """从一组 items 构建 CompositeRecord。

        - semantic_text: 选择 confidence 最高 / updated_at 最新的代表记录
        - metadata: 合并所有 items 的 entities、tags、temporal
        """
        if len(items) < self._min_records:
            return None

        # 选择代表记录
        representative = max(items, key=lambda x: (x.confidence, x.updated_at))

        # 合并 metadata
        all_entities = self._merge_unique(
            [e for item in items for e in (item.entities or [])]
        )
        all_tags = self._merge_unique(
            [t for item in items for t in (item.tags or [])]
        )
        merged_temporal = self._merge_temporal(
            [item.temporal for item in items if item.temporal]
        )
        avg_confidence = sum(item.confidence for item in items) / len(items)

        # 推断 composite memory_type
        type_counts: dict[str, int] = {}
        for item in items:
            type_counts[item.memory_type] = type_counts.get(item.memory_type, 0) + 1
        dominant_type = max(type_counts, key=lambda k: type_counts[k])
        synth_type = self._infer_synth_type(dominant_type)

        composite_id = self._make_composite_id(flat_ids, representative.semantic_text)

        return CompositeRecord(
            composite_id=composite_id,
            memory_type=synth_type,
            semantic_text=representative.semantic_text,
            normalized_text=f"{synth_type}: {representative.semantic_text}",
            source_record_ids=flat_ids,
            child_composite_ids=child_composite_ids,
            entities=all_entities,
            temporal=merged_temporal,
            tags=all_tags,
            confidence=round(avg_confidence, 4),
            created_at=now_iso,
            updated_at=now_iso,
        )

    # ──────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────

    @staticmethod
    def _union_find_clusters(
        edges: list[tuple[str, str]],
        all_ids: Any,
    ) -> list[list[str]]:
        """Union-Find 算法，根据边集构建连通分量。"""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for uid in all_ids:
            parent.setdefault(str(uid), str(uid))

        for a, b in edges:
            a_s, b_s = str(a).strip(), str(b).strip()
            if a_s and b_s:
                parent.setdefault(a_s, a_s)
                parent.setdefault(b_s, b_s)
                union(a_s, b_s)

        groups: dict[str, list[str]] = defaultdict(list)
        for uid in parent:
            groups[find(uid)].append(uid)

        return [sorted(ids) for ids in groups.values() if len(ids) >= 2]

    @staticmethod
    def _merge_unique(items: list[str]) -> list[str]:
        """去重合并字符串列表（大小写不敏感去重）。"""
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            s = str(item or "").strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                result.append(s)
        return result

    @staticmethod
    def _merge_temporal(temporals: list[dict[str, Any]]) -> dict[str, str]:
        """合并多个 temporal 字典：t_valid_from 取最早，t_valid_to 取最晚。"""
        merged: dict[str, str] = {"t_ref": "", "t_valid_from": "", "t_valid_to": ""}
        for t in temporals:
            if not isinstance(t, dict):
                continue
            for k in ("t_ref", "t_valid_from", "t_valid_to"):
                v = str(t.get(k, "") or "").strip()
                if v:
                    if not merged[k]:
                        merged[k] = v
                    elif k == "t_valid_from":
                        merged[k] = min(merged[k], v)
                    elif k == "t_valid_to":
                        merged[k] = max(merged[k], v)
                    # t_ref: keep first non-empty
        return merged

    @staticmethod
    def _infer_synth_type(dominant_memory_type: str) -> str:
        """根据主导 memory_type 推断合适的 composite 类型。"""
        mapping = {
            "preference": "composite_preference",
            "constraint": "composite_constraint",
            "failure_pattern": "composite_pattern",
            "procedure": "composite_pattern",
        }
        result = mapping.get(dominant_memory_type, "usage_pattern")
        if result not in VALID_SYNTH_TYPES:
            result = "usage_pattern"
        return result

    def _persist_composite(self, composite: CompositeRecord) -> None:
        self._sqlite.upsert_synthesized(composite)
        self._vector.upsert_synthesized(
            composite_id=composite.composite_id,
            memory_type=composite.memory_type,
            semantic_text=composite.semantic_text,
            normalized_text=composite.normalized_text,
        )

    def _is_source_group_already_covered(self, source_record_ids: list[str]) -> bool:
        source_set = {s for s in source_record_ids if s}
        if len(source_set) < self._min_records:
            return True

        existing = self._sqlite.get_synthesized_by_source(list(source_set))
        for comp in existing:
            existing_set = {
                str(uid or "").strip()
                for uid in comp.source_record_ids
                if str(uid or "").strip()
            }
            if not existing_set:
                continue
            if source_set.issubset(existing_set):
                return True
            overlap = len(source_set & existing_set)
            union_size = len(source_set | existing_set)
            if union_size > 0 and overlap / union_size >= 0.7:
                return True
        return False

    @staticmethod
    def _item_id(item: MemoryRecord | CompositeRecord) -> str:
        if isinstance(item, CompositeRecord):
            return str(item.composite_id or "").strip()
        return str(item.record_id or "").strip()

    @staticmethod
    def _flatten_source_ids(
        items: list[MemoryRecord | CompositeRecord],
    ) -> list[str]:
        ids: set[str] = set()
        for item in items:
            if isinstance(item, CompositeRecord):
                for rid in item.source_record_ids:
                    s = str(rid or "").strip()
                    if s:
                        ids.add(s)
            else:
                s = str(item.record_id or "").strip()
                if s:
                    ids.add(s)
        return sorted(ids)

    @staticmethod
    def _has_coverage_relation(left: list[str], right: list[str]) -> bool:
        ls = {s for s in left if s}
        rs = {s for s in right if s}
        if not ls or not rs:
            return False
        return ls.issubset(rs) or rs.issubset(ls)

    def _build_child_to_parent(
        self, composites: list[CompositeRecord],
    ) -> dict[str, str]:
        """构建 child composite → parent composite 映射。"""
        source_sets: dict[str, set[str]] = {}
        for comp in composites:
            cid = str(comp.composite_id or "").strip()
            if cid:
                source_sets[cid] = {
                    str(s or "").strip()
                    for s in comp.source_record_ids
                    if str(s or "").strip()
                }

        child_to_parent: dict[str, str] = {}
        for comp in composites:
            parent_id = str(comp.composite_id or "").strip()
            parent_sources = source_sets.get(parent_id, set())
            for child_cid in comp.child_composite_ids:
                child_cid_s = str(child_cid or "").strip()
                child_sources = source_sets.get(child_cid_s, set())
                if (
                    child_cid_s
                    and child_cid_s != parent_id
                    and child_sources
                    and child_sources.issubset(parent_sources)
                    and len(parent_sources) > len(child_sources)
                ):
                    child_to_parent[child_cid_s] = parent_id

        return child_to_parent

    @staticmethod
    def _make_composite_id(source_ids: list[str], semantic_text: str) -> str:
        raw = "|".join(sorted(source_ids)) + "|" + semantic_text
        return "comp_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
