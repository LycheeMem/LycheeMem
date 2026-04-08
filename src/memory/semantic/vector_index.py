"""LanceDB 向量索引。

职责：
- memory_records 表：MemoryRecord 的 semantic / normalized embedding
- composite_records 表：CompositeRecord 的 embedding
- 提供 ANN 向量检索（semantic_query / pragmatic_query 两种向量）

依赖 lancedb (已列入 pyproject.toml) + pyarrow。
"""

from __future__ import annotations

import os
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


def _make_composite_schema(dim: int) -> pa.Schema:
    """构建 composite_records 表 schema。"""
    vec_type = pa.list_(pa.float32(), dim) if dim > 0 else pa.list_(pa.float32())
    return pa.schema([
        pa.field("composite_id", pa.utf8()),
        pa.field("memory_type", pa.utf8()),
        pa.field("vector", vec_type),
        pa.field("normalized_vector", vec_type),
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
    """基于 LanceDB 的向量索引，ANN 检索 MemoryRecord / CompositeRecord / 原始 Turn。"""

    MEMORY_TABLE = "memory_records"
    SYNTH_TABLE = "composite_records"
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
        target_cr = _make_composite_schema(self._embedding_dim)
        target_ep = _make_episode_turn_schema(self._embedding_dim)
        existing = set(self._db.table_names())

        for table_name, schema in [
            (self.MEMORY_TABLE, target_mr),
            (self.SYNTH_TABLE, target_cr),
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

        table = self._db.open_table(self.MEMORY_TABLE)

        # 构造过滤条件
        filters: list[str] = []
        if not include_expired:
            filters.append("expired = false")
        if memory_types:
            type_strs = ", ".join(f"'{self._escape_sql(t)}'" for t in memory_types)
            filters.append(f"memory_type IN ({type_strs})")

        where = " AND ".join(filters) if filters else None

        try:
            q = table.search(query_vector, vector_column_name=column).limit(limit)
            if where:
                q = q.where(where)
            results = q.to_list()
        except Exception:
            # 表为空或列不存在等，返回空
            return []

        return [
            {
                "record_id": r.get("record_id", ""),
                "memory_type": r.get("memory_type", ""),
                "_distance": r.get("_distance", 999.0),
            }
            for r in results
        ]

    # ──────────────────────────────────────
    # CompositeRecord 向量写入 / 检索
    # ──────────────────────────────────────

    def upsert_synthesized(
        self,
        composite_id: str,
        memory_type: str,
        semantic_text: str,
        normalized_text: str,
        *,
        semantic_vector: list[float] | None = None,
        normalized_vector: list[float] | None = None,
    ) -> None:
        if semantic_vector is None or normalized_vector is None:
            if self._embedder is None:
                raise RuntimeError("LanceVectorIndex: embedder 未设置且未提供向量")
            vecs = self._embedder.embed([semantic_text, normalized_text])
            semantic_vector = semantic_vector or vecs[0]
            normalized_vector = normalized_vector or vecs[1]

        table = self._db.open_table(self.SYNTH_TABLE)
        try:
            table.delete(f"composite_id = '{self._escape_sql(composite_id)}'")
        except Exception:
            pass
        table.add([{
            "composite_id": composite_id,
            "memory_type": memory_type,
            "vector": semantic_vector,
            "normalized_vector": normalized_vector,
        }])

    def search_synthesized(
        self,
        query_text: str,
        *,
        column: str = "vector",
        limit: int = 10,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if query_vector is None:
            if self._embedder is None:
                raise RuntimeError("LanceVectorIndex: embedder 未设置且未提供查询向量")
            query_vector = self._embedder.embed_query(query_text)

        table = self._db.open_table(self.SYNTH_TABLE)

        filters: list[str] = []
        where = " AND ".join(filters) if filters else None

        try:
            q = table.search(query_vector, vector_column_name=column).limit(limit)
            if where:
                q = q.where(where)
            results = q.to_list()
        except Exception:
            return []

        return [
            {
                "composite_id": r.get("composite_id", ""),
                "memory_type": r.get("memory_type", ""),
                "_distance": r.get("_distance", 999.0),
            }
            for r in results
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
            # 分批嵌入，避免超出 API 单批限制（部分 provider 上限 10）
            _EMBED_CHUNK = 8
            all_embedded: list[list[float]] = []
            for _i in range(0, len(texts_to_embed), _EMBED_CHUNK):
                all_embedded.extend(self._embedder.embed(texts_to_embed[_i : _i + _EMBED_CHUNK]))
            for idx, vec in zip(embed_indices, all_embedded):
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

        table = self._db.open_table(self.EPISODE_TABLE)
        filters: list[str] = []
        if session_id:
            filters.append(f"session_id = '{self._escape_sql(session_id)}'")
        where = " AND ".join(filters) if filters else None

        try:
            q = table.search(query_vector, vector_column_name="vector").limit(limit)
            if where:
                q = q.where(where)
            results = q.to_list()
        except Exception:
            return []

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
            for r in results
        ]

    def get_all_episode_ids(self) -> set[str]:
        """返回 episode_turns 表中已索引的全部 episode_id 集合（用于增量补全）。"""
        try:
            table = self._db.open_table(self.EPISODE_TABLE)
            results = table.search().select(["episode_id"]).limit(10_000_000).to_list()
            return {str(r.get("episode_id", "")) for r in results if r.get("episode_id")}
        except Exception:
            return set()

    # ──────────────────────────────────────
    # 批量删除
    # ──────────────────────────────────────

    def delete_all(self) -> None:
        """删除全部向量数据。"""
        for tname in [self.MEMORY_TABLE, self.SYNTH_TABLE, self.EPISODE_TABLE]:
            try:
                table = self._db.open_table(tname)
                table.delete("1 = 1")
            except Exception:
                pass

    def delete_record(self, record_id: str) -> None:
        try:
            table = self._db.open_table(self.MEMORY_TABLE)
            table.delete(f"record_id = '{self._escape_sql(record_id)}'")
        except Exception:
            pass

    def delete_synthesized(self, composite_id: str) -> None:
        try:
            table = self._db.open_table(self.SYNTH_TABLE)
            table.delete(f"composite_id = '{self._escape_sql(composite_id)}'")
        except Exception:
            pass

    def mark_expired(self, record_id: str) -> None:
        """标记过期（向量层面：删除后以 expired=True 重新插入开销大，直接删除即可，
        具体过期条目不参与 ANN 召回，靠 sqlite 侧查询）。"""
        self.delete_record(record_id)

    # ──────────────────────────────────────
    # 工具
    # ──────────────────────────────────────

    @staticmethod
    def _escape_sql(value: str) -> str:
        """防止 SQL 注入（LanceDB filter 为字符串拼接）。"""
        return value.replace("'", "''").replace("\\", "\\\\")
