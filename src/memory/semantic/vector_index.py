"""LanceDB 向量索引。

职责：
- memory_records 表：MemoryRecord 的 semantic / normalized embedding
- evidence_nodes 表：entity/tag/entity_tag/time/event-frame 等结构化证据索引的 embedding
- episode_turns 表：原始对话 turn 的 embedding
- 提供 ANN 向量检索

依赖 lancedb (已列入 pyproject.toml) + pyarrow。
"""

from __future__ import annotations

import os
import threading
from typing import Any

import lancedb
import pyarrow as pa

from src.embedder.base import BaseEmbedder


def _make_record_schema(dim: int) -> pa.Schema:
    """构建 memory_records 表 schema。dim > 0 时使用 FixedSizeList（推荐）。"""
    vec_type = pa.list_(pa.float32(), dim) if dim > 0 else pa.list_(pa.float32())
    return pa.schema([
        pa.field("record_id", pa.utf8()),
        pa.field("memory_type", pa.utf8()),
        pa.field("vector", vec_type),
        pa.field("normalized_vector", vec_type),
        pa.field("expired", pa.bool_()),
    ])


def _make_evidence_node_schema(dim: int) -> pa.Schema:
    """构建 evidence_nodes 表 schema。"""
    vec_type = pa.list_(pa.float32(), dim) if dim > 0 else pa.list_(pa.float32())
    return pa.schema([
        pa.field("node_id", pa.utf8()),
        pa.field("node_type", pa.utf8()),
        pa.field("key", pa.utf8()),
        pa.field("vector", vec_type),
    ])


def _make_episode_turn_schema(dim: int) -> pa.Schema:
    """构建 episode_turns 表 schema：原始对话 turn 的向量索引。"""
    vec_type = pa.list_(pa.float32(), dim) if dim > 0 else pa.list_(pa.float32())
    return pa.schema([
        pa.field("episode_id", pa.utf8()),
        pa.field("session_id", pa.utf8()),
        pa.field("turn_index", pa.int32()),
        pa.field("role", pa.utf8()),
        pa.field("content", pa.utf8()),
        pa.field("vector", vec_type),
        pa.field("created_at", pa.utf8()),
    ])


class LanceVectorIndex:
    """基于 LanceDB 的向量索引，ANN 检索 MemoryRecord / EvidenceNode / 原始 Turn。"""

    MEMORY_TABLE = "memory_records"
    EVIDENCE_TABLE = "evidence_nodes"
    EPISODE_TABLE = "episode_turns"

    def __init__(
        self,
        db_path: str = "lychee_compact_vector",
        embedder: BaseEmbedder | None = None,
        embedding_dim: int = 0,
    ):
        self._db_path = db_path
        self._embedder = embedder
        self._embedding_dim = embedding_dim  # 0 = 自动检测 / 使用变长 list
        self._local = threading.local()
        self._meta_lock = threading.Lock()
        self._table_nonempty_cache: dict[str, bool] = {}
        self._vector_dim_cache: dict[tuple[str, str], int] = {}
        self._exact_max_rows = max(0, self._env_int("LYCHEE_VECTOR_EXACT_MAX_ROWS", 20_000))
        self._exact_rows_cache: dict[str, list[dict[str, Any]]] = {}
        self._exact_matrix_cache: dict[tuple[str, str], Any] = {}
        self._search_semaphore = threading.BoundedSemaphore(
            max(
                1,
                self._env_int(
                    "LYCHEE_VECTOR_SEARCH_CONCURRENCY",
                    min(16, max(1, os.cpu_count() or 1)),
                ),
            )
        )
        os.makedirs(db_path, exist_ok=True)
        self._db = lancedb.connect(db_path)
        self._ensure_tables()

    def set_embedder(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    # ──────────────────────────────────────
    # 表初始化
    # ──────────────────────────────────────

    def _ensure_tables(self) -> None:
        """创建或验证 LanceDB 表。

        - 若表不存在：按 embedding_dim 构建 schema 并创建。
        - 若表已存在但 vector 列是变长 list（旧 schema）且 embedding_dim 已知：
          删除旧表并用正确 FixedSizeList schema 重建。
        - 若 embedding_dim=0：建变长 list schema（底存，仅用于无配置场景）。
        """
        target_mr = _make_record_schema(self._embedding_dim)
        target_ev = _make_evidence_node_schema(self._embedding_dim)
        target_ep = _make_episode_turn_schema(self._embedding_dim)
        existing = set(self._db.table_names())

        for table_name, schema in [
            (self.MEMORY_TABLE, target_mr),
            (self.EVIDENCE_TABLE, target_ev),
            (self.EPISODE_TABLE, target_ep),
        ]:
            if table_name not in existing:
                self._db.create_table(table_name, schema=schema)
            elif self._embedding_dim > 0:
                # 检查现有 vector 列类型是否为 FixedSizeList
                try:
                    tbl = self._db.open_table(table_name)
                    existing_schema = tbl.schema
                    vec_field = existing_schema.field("vector")
                    if not pa.types.is_fixed_size_list(vec_field.type):
                        # 旧 variable-length schema — 重建
                        self._db.drop_table(table_name)
                        self._db.create_table(table_name, schema=schema)
                except Exception:
                    pass  # 无法检查则保持现有状态


    # ──────────────────────────────────────
    # MemoryRecord 向量写入
    # ──────────────────────────────────────

    def upsert(
        self,
        record_id: str,
        memory_type: str,
        semantic_text: str,
        normalized_text: str,
        *,
        expired: bool = False,
        semantic_vector: list[float] | None = None,
        normalized_vector: list[float] | None = None,
    ) -> None:
        """写入 / 更新一条 MemoryRecord 的向量。
        
        如果未提供 vector，则使用 embedder 实时计算。
        """
        if semantic_vector is None or normalized_vector is None:
            if self._embedder is None:
                raise RuntimeError("LanceVectorIndex: embedder 未设置且未提供向量")
            vecs = self._embedder.embed([semantic_text, normalized_text])
            semantic_vector = semantic_vector or vecs[0]
            normalized_vector = normalized_vector or vecs[1]

        table = self._db.open_table(self.MEMORY_TABLE)
        # 先删除旧记录（如存在），再插入新记录
        try:
            table.delete(f"record_id = '{self._escape_sql(record_id)}'")
        except Exception:
            pass
        table.add([{
            "record_id": record_id,
            "memory_type": memory_type,
            "vector": semantic_vector,
            "normalized_vector": normalized_vector,
            "expired": expired,
        }])
        self._mark_table_nonempty(self.MEMORY_TABLE)
        self._invalidate_exact_cache(self.MEMORY_TABLE)

    def upsert_batch(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        """批量写入 MemoryRecord 向量。
        
        每条 record 需要: record_id, memory_type, semantic_text, normalized_text。
        可选: semantic_vector, normalized_vector, expired。
        """
        if not records:
            return

        # 计算缺失的向量
        texts_to_embed: list[str] = []
        embed_indices: list[tuple[int, str]] = []  # (record_idx, "semantic"|"normalized")
        for i, r in enumerate(records):
            if "semantic_vector" not in r or r["semantic_vector"] is None:
                embed_indices.append((i, "semantic"))
                texts_to_embed.append(r["semantic_text"])
            if "normalized_vector" not in r or r["normalized_vector"] is None:
                embed_indices.append((i, "normalized"))
                texts_to_embed.append(r["normalized_text"])

        if texts_to_embed:
            if self._embedder is None:
                raise RuntimeError("LanceVectorIndex: embedder 未设置且未提供向量")
            embedded = self._embedder.embed(texts_to_embed)
            for (idx, kind), vec in zip(embed_indices, embedded):
                if kind == "semantic":
                    records[idx]["semantic_vector"] = vec
                else:
                    records[idx]["normalized_vector"] = vec

        table = self._db.open_table(self.MEMORY_TABLE)
        # 批量删除旧记录
        ids = [r["record_id"] for r in records]
        for rid in ids:
            try:
                table.delete(f"record_id = '{self._escape_sql(rid)}'")
            except Exception:
                pass

        rows = [
            {
                "record_id": r["record_id"],
                "memory_type": r["memory_type"],
                "vector": r["semantic_vector"],
                "normalized_vector": r["normalized_vector"],
                "expired": r.get("expired", False),
            }
            for r in records
        ]
        table.add(rows)
        self._mark_table_nonempty(self.MEMORY_TABLE)
        self._invalidate_exact_cache(self.MEMORY_TABLE)

    def search(
        self,
        query_text: str,
        *,
        column: str = "vector",
        limit: int = 20,
        memory_types: list[str] | None = None,
        include_expired: bool = False,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """向量 ANN 检索 MemoryRecord。

        column: "vector"（语义检索） 或 "normalized_vector"（归一化/实用检索）
        """
        if query_vector is None:
            if self._embedder is None:
                raise RuntimeError("LanceVectorIndex: embedder 未设置且未提供查询向量")
            query_vector = self._embedder.embed_query(query_text)

        table = self._open_table(self.MEMORY_TABLE)
        if table is None or not self._table_has_rows(self.MEMORY_TABLE, table):
            return []
        query_vector = self._normalize_query_vector(self.MEMORY_TABLE, table, column, query_vector)
        if query_vector is None:
            return []

        try:
            search_limit = self._search_limit_with_filter_headroom(
                limit,
                has_filter=(not include_expired or bool(memory_types)),
            )
            results = self._exact_vector_search(
                table_name=self.MEMORY_TABLE,
                table=table,
                vector_column=column,
                query_vector=query_vector,
                limit=search_limit,
            )
            if results is None:
                with self._search_semaphore:
                    results = (
                        table.search(query_vector, vector_column_name=column)
                        .limit(search_limit)
                        .to_list()
                    )
        except Exception:
            # 表为空或列不存在等，返回空
            return []

        allowed_types = {str(t) for t in (memory_types or [])}
        filtered: list[dict[str, Any]] = []
        for row in results:
            if not include_expired and bool(row.get("expired", False)):
                continue
            if allowed_types and str(row.get("memory_type", "")) not in allowed_types:
                continue
            filtered.append(row)
            if len(filtered) >= limit:
                break

        return [
            {
                "record_id": r.get("record_id", ""),
                "memory_type": r.get("memory_type", ""),
                "_distance": r.get("_distance", 999.0),
            }
            for r in filtered
        ]

    # ──────────────────────────────────────
    # EvidenceNode 向量写入 / 检索
    # ──────────────────────────────────────

    def upsert_evidence_nodes_batch(self, nodes: list[dict[str, Any]]) -> None:
        """批量写入 entity/tag/entity_tag/time/event-frame 证据索引向量。

        每条 node 需要: node_id, node_type, key, search_text。
        可选: vector。
        """
        if not nodes:
            return

        texts_to_embed: list[str] = []
        embed_indices: list[int] = []
        for i, node in enumerate(nodes):
            if "vector" not in node or node["vector"] is None:
                embed_indices.append(i)
                texts_to_embed.append(str(node.get("search_text") or node.get("key") or ""))

        if texts_to_embed:
            if self._embedder is None:
                raise RuntimeError("LanceVectorIndex: embedder 未设置且未提供向量")
            embedded = self._embedder.embed(texts_to_embed)
            for idx, vec in zip(embed_indices, embedded):
                nodes[idx]["vector"] = vec

        table = self._db.open_table(self.EVIDENCE_TABLE)
        for node in nodes:
            node_id = self._escape_sql(str(node.get("node_id", "")))
            if not node_id:
                continue
            try:
                table.delete(f"node_id = '{node_id}'")
            except Exception:
                pass

        rows = [
            {
                "node_id": str(node.get("node_id", "")),
                "node_type": str(node.get("node_type", "")),
                "key": str(node.get("key", "")),
                "vector": node["vector"],
            }
            for node in nodes
            if str(node.get("node_id", "")).strip() and node.get("vector") is not None
        ]
        if rows:
            table.add(rows)
            self._mark_table_nonempty(self.EVIDENCE_TABLE)
            self._invalidate_exact_cache(self.EVIDENCE_TABLE)

    def search_evidence_nodes(
        self,
        query_text: str,
        *,
        node_types: list[str] | None = None,
        limit: int = 20,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """对 evidence_nodes 做 ANN 检索。"""
        if query_vector is None:
            if self._embedder is None:
                raise RuntimeError("LanceVectorIndex: embedder 未设置且未提供查询向量")
            query_vector = self._embedder.embed_query(query_text)

        table = self._open_table(self.EVIDENCE_TABLE)
        if table is None or not self._table_has_rows(self.EVIDENCE_TABLE, table):
            return []
        query_vector = self._normalize_query_vector(self.EVIDENCE_TABLE, table, "vector", query_vector)
        if query_vector is None:
            return []

        try:
            search_limit = self._search_limit_with_filter_headroom(
                limit,
                has_filter=bool(node_types),
            )
            results = self._exact_vector_search(
                table_name=self.EVIDENCE_TABLE,
                table=table,
                vector_column="vector",
                query_vector=query_vector,
                limit=search_limit,
            )
            if results is None:
                with self._search_semaphore:
                    results = (
                        table.search(query_vector, vector_column_name="vector")
                        .limit(search_limit)
                        .to_list()
                    )
        except Exception:
            return []

        allowed_types = {str(t) for t in (node_types or [])}
        filtered: list[dict[str, Any]] = []
        for row in results:
            if allowed_types and str(row.get("node_type", "")) not in allowed_types:
                continue
            filtered.append(row)
            if len(filtered) >= limit:
                break

        return [
            {
                "node_id": r.get("node_id", ""),
                "node_type": r.get("node_type", ""),
                "key": r.get("key", ""),
                "_distance": r.get("_distance", 999.0),
            }
            for r in filtered
        ]

    # ──────────────────────────────────────
    # 批量删除
    # ──────────────────────────────────────

    # ──────────────────────────────────────
    # Episode Turn 向量写入 / 检索
    # ──────────────────────────────────────

    def upsert_turns_batch(self, turns: list[dict[str, Any]]) -> None:
        """批量写入原始对话 turn 的向量。

        每条 turn 需要: episode_id, session_id, turn_index, role, content
        可选: vector (float list), created_at
        """
        if not turns:
            return

        texts_to_embed: list[str] = []
        embed_indices: list[int] = []
        for i, t in enumerate(turns):
            if "vector" not in t or t["vector"] is None:
                embed_indices.append(i)
                texts_to_embed.append(str(t.get("content", "")))

        if texts_to_embed:
            if self._embedder is None:
                raise RuntimeError("LanceVectorIndex: embedder 未设置且未提供向量")
            embedded = self._embedder.embed(texts_to_embed)
            for idx, vec in zip(embed_indices, embedded):
                turns[idx]["vector"] = vec

        table = self._db.open_table(self.EPISODE_TABLE)
        for t in turns:
            ep_id = self._escape_sql(str(t.get("episode_id", "")))
            try:
                table.delete(f"episode_id = '{ep_id}'")
            except Exception:
                pass

        rows = [
            {
                "episode_id": str(t.get("episode_id", "")),
                "session_id": str(t.get("session_id", "")),
                "turn_index": int(t.get("turn_index", 0)),
                "role": str(t.get("role", "")),
                "content": str(t.get("content", "")),
                "vector": t["vector"],
                "created_at": str(t.get("created_at", "")),
            }
            for t in turns
        ]
        table.add(rows)
        if rows:
            self._mark_table_nonempty(self.EPISODE_TABLE)
            self._invalidate_exact_cache(self.EPISODE_TABLE)

    def search_turns(
        self,
        query_text: str,
        *,
        limit: int = 20,
        query_vector: list[float] | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """对原始对话 turn 做 ANN 向量检索。"""
        if query_vector is None:
            if self._embedder is None:
                raise RuntimeError("LanceVectorIndex: embedder 未设置且未提供查询向量")
            query_vector = self._embedder.embed_query(query_text)

        table = self._open_table(self.EPISODE_TABLE)
        if table is None or not self._table_has_rows(self.EPISODE_TABLE, table):
            return []
        query_vector = self._normalize_query_vector(self.EPISODE_TABLE, table, "vector", query_vector)
        if query_vector is None:
            return []

        try:
            search_limit = self._search_limit_with_filter_headroom(
                limit,
                has_filter=bool(session_id),
            )
            results = self._exact_vector_search(
                table_name=self.EPISODE_TABLE,
                table=table,
                vector_column="vector",
                query_vector=query_vector,
                limit=search_limit,
            )
            if results is None:
                with self._search_semaphore:
                    results = (
                        table.search(query_vector, vector_column_name="vector")
                        .limit(search_limit)
                        .to_list()
                    )
        except Exception:
            return []

        target_session = str(session_id or "")
        filtered: list[dict[str, Any]] = []
        for row in results:
            if target_session and str(row.get("session_id", "")) != target_session:
                continue
            filtered.append(row)
            if len(filtered) >= limit:
                break

        return [
            {
                "episode_id": r.get("episode_id", ""),
                "session_id": r.get("session_id", ""),
                "turn_index": r.get("turn_index", 0),
                "role": r.get("role", ""),
                "content": r.get("content", ""),
                "created_at": r.get("created_at", ""),
                "_distance": r.get("_distance", 999.0),
            }
            for r in filtered
        ]

    def get_all_episode_ids(self) -> set[str]:
        """返回 episode_turns 表中已索引的全部 episode_id 集合（用于增量补全）。"""
        try:
            table = self._open_table(self.EPISODE_TABLE)
            if table is None or not self._table_has_rows(self.EPISODE_TABLE, table):
                return set()
            with self._search_semaphore:
                results = table.search().select(["episode_id"]).limit(10_000_000).to_list()
            return {str(r.get("episode_id", "")) for r in results if r.get("episode_id")}
        except Exception:
            return set()

    # ──────────────────────────────────────
    # 批量删除
    # ──────────────────────────────────────

    def delete_all(self) -> None:
        """删除全部向量数据。"""
        for tname in [self.MEMORY_TABLE, self.EVIDENCE_TABLE, self.EPISODE_TABLE]:
            try:
                table = self._open_table(tname)
                if table is None:
                    continue
                table.delete("1 = 1")
                self._mark_table_empty(tname)
                self._invalidate_exact_cache(tname)
            except Exception:
                pass

    # ──────────────────────────────────────
    # 工具
    # ──────────────────────────────────────

    @staticmethod
    def _escape_sql(value: str) -> str:
        """防止 SQL 注入（LanceDB filter 为字符串拼接）。"""
        return value.replace("'", "''").replace("\\", "\\\\")

    @staticmethod
    def _search_limit_with_filter_headroom(limit: int, *, has_filter: bool) -> int:
        base = max(1, int(limit or 1))
        if not has_filter:
            return base
        return max(base * 5, min(base + 80, 200))

    def _open_table(self, table_name: str) -> Any | None:
        """Safely open a LanceDB table.

        Some LanceDB failures are raised before the search call, especially when
        a table directory is missing or half-written. Keep retrieval robust by
        treating those cases as empty indexes.
        """
        table_cache: dict[str, Any] | None = getattr(self._local, "table_cache", None)
        if table_cache is None:
            table_cache = {}
            self._local.table_cache = table_cache
        table = table_cache.get(table_name)
        if table is not None:
            return table
        try:
            table = self._db.open_table(table_name)
            table_cache[table_name] = table
            return table
        except Exception:
            return None

    def _table_has_rows(self, table_name: str, table: Any) -> bool:
        with self._meta_lock:
            cached = self._table_nonempty_cache.get(table_name)
        if cached is not None:
            return cached
        try:
            has_rows = int(table.count_rows()) > 0
        except Exception:
            has_rows = True
        with self._meta_lock:
            self._table_nonempty_cache[table_name] = has_rows
        return has_rows

    def _normalize_query_vector(
        self,
        table_name: str,
        table: Any,
        vector_column: str,
        query_vector: Any,
    ) -> list[float] | None:
        """Coerce and validate query vectors before entering LanceDB native search."""
        if query_vector is None:
            return None
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()
        if not isinstance(query_vector, list):
            try:
                vector = [float(x) for x in query_vector]
            except (TypeError, ValueError):
                return None
        else:
            vector = query_vector

        try:
            vector_len = len(vector)
        except TypeError:
            return None

        expected_dim = self._table_vector_dim(table_name, table, vector_column) or self._embedding_dim
        if expected_dim > 0 and vector_len != expected_dim:
            return None
        return vector

    def _table_vector_dim(self, table_name: str, table: Any, vector_column: str) -> int:
        cache_key = (table_name, vector_column)
        with self._meta_lock:
            cached = self._vector_dim_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            vec_type = table.schema.field(vector_column).type
            if pa.types.is_fixed_size_list(vec_type):
                dim = int(vec_type.list_size)
                with self._meta_lock:
                    self._vector_dim_cache[cache_key] = dim
                return dim
        except Exception:
            return 0
        return 0

    def _mark_table_nonempty(self, table_name: str) -> None:
        with self._meta_lock:
            self._table_nonempty_cache[table_name] = True

    def _mark_table_empty(self, table_name: str) -> None:
        with self._meta_lock:
            self._table_nonempty_cache[table_name] = False

    def _exact_vector_search(
        self,
        *,
        table_name: str,
        table: Any,
        vector_column: str,
        query_vector: list[float],
        limit: int,
    ) -> list[dict[str, Any]] | None:
        rows = self._exact_rows(table_name, table)
        if rows is None:
            return None
        if not rows:
            return []
        cache_key = (table_name, vector_column)
        try:
            import numpy as np

            with self._meta_lock:
                matrix = self._exact_matrix_cache.get(cache_key)
            if matrix is None:
                matrix = np.asarray([row.get(vector_column) or [] for row in rows], dtype="float32")
                with self._meta_lock:
                    self._exact_matrix_cache[cache_key] = matrix
            query = np.asarray(query_vector, dtype="float32")
            if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.shape[0]:
                return None
            distances = np.sum((matrix - query) ** 2, axis=1)
            take = min(max(1, int(limit or 1)), len(rows))
            if take < len(rows):
                order = np.argpartition(distances, take - 1)[:take]
                order = order[np.argsort(distances[order])]
            else:
                order = np.argsort(distances)
            return [
                {**rows[int(index)], "_distance": float(distances[int(index)])}
                for index in order[:take]
            ]
        except Exception:
            return self._exact_vector_search_python(
                rows=rows,
                vector_column=vector_column,
                query_vector=query_vector,
                limit=limit,
            )

    def _exact_vector_search_python(
        self,
        *,
        rows: list[dict[str, Any]],
        vector_column: str,
        query_vector: list[float],
        limit: int,
    ) -> list[dict[str, Any]] | None:
        try:
            query = [float(x) for x in query_vector]
            scored: list[tuple[float, dict[str, Any]]] = []
            for row in rows:
                vector = row.get(vector_column) or []
                if len(vector) != len(query):
                    return None
                distance = 0.0
                for left, right in zip(vector, query):
                    diff = float(left) - right
                    distance += diff * diff
                scored.append((distance, row))
            scored.sort(key=lambda item: item[0])
            take = min(max(1, int(limit or 1)), len(scored))
            return [
                {**row, "_distance": float(distance)}
                for distance, row in scored[:take]
            ]
        except Exception:
            return None

    def _exact_rows(self, table_name: str, table: Any) -> list[dict[str, Any]] | None:
        if self._exact_max_rows <= 0:
            return None
        with self._meta_lock:
            cached = self._exact_rows_cache.get(table_name)
        if cached is not None:
            return cached
        try:
            row_count = int(table.count_rows())
            if row_count > self._exact_max_rows:
                return None
            rows = table.to_arrow().to_pylist()
        except Exception:
            return None
        with self._meta_lock:
            self._exact_rows_cache[table_name] = rows
        return rows

    def _invalidate_exact_cache(self, table_name: str) -> None:
        with self._meta_lock:
            self._exact_rows_cache.pop(table_name, None)
            for key in list(self._exact_matrix_cache):
                if key[0] == table_name:
                    self._exact_matrix_cache.pop(key, None)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, "") or default)
        except (TypeError, ValueError):
            return default
