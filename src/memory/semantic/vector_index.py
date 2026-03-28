"""LanceDB 向量索引。

职责：
- memory_units 表：MemoryUnit 的 semantic / normalized embedding
- synthesized_units 表：SynthesizedUnit 的 embedding
- 提供 ANN 向量检索（semantic_query / pragmatic_query 两种向量）

依赖 lancedb (已列入 pyproject.toml) + pyarrow。
"""

from __future__ import annotations

import os
from typing import Any

import lancedb
import pyarrow as pa

from src.embedder.base import BaseEmbedder


def _make_mu_schema(dim: int) -> pa.Schema:
    """构建 memory_units 表 schema。dim > 0 时使用 FixedSizeList（推荐）。"""
    vec_type = pa.list_(pa.float32(), dim) if dim > 0 else pa.list_(pa.float32())
    return pa.schema([
        pa.field("unit_id", pa.utf8()),
        pa.field("user_id", pa.utf8()),
        pa.field("memory_type", pa.utf8()),
        pa.field("vector", vec_type),
        pa.field("normalized_vector", vec_type),
        pa.field("expired", pa.bool_()),
    ])


def _make_su_schema(dim: int) -> pa.Schema:
    """构建 synthesized_units 表 schema。"""
    vec_type = pa.list_(pa.float32(), dim) if dim > 0 else pa.list_(pa.float32())
    return pa.schema([
        pa.field("synth_id", pa.utf8()),
        pa.field("user_id", pa.utf8()),
        pa.field("memory_type", pa.utf8()),
        pa.field("vector", vec_type),
        pa.field("normalized_vector", vec_type),
    ])


class LanceVectorIndex:
    """基于 LanceDB 的向量索引，ANN 检索 MemoryUnit / SynthesizedUnit。"""

    MEMORY_TABLE = "memory_units"
    SYNTH_TABLE = "synthesized_units"

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
          删除旧表并用正确 FixedSizeList schema 重建（旧数据已丢失不影响，
          因为 0 向量根本无法检索）。
        - 若 embedding_dim=0：建变长 list schema（兜底，仅用于无配置场景）。
        """
        target_mu = _make_mu_schema(self._embedding_dim)
        target_su = _make_su_schema(self._embedding_dim)
        existing = set(self._db.table_names())

        for table_name, schema in [
            (self.MEMORY_TABLE, target_mu),
            (self.SYNTH_TABLE, target_su),
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
    # MemoryUnit 向量写入
    # ──────────────────────────────────────

    def upsert(
        self,
        unit_id: str,
        user_id: str,
        memory_type: str,
        semantic_text: str,
        normalized_text: str,
        *,
        expired: bool = False,
        semantic_vector: list[float] | None = None,
        normalized_vector: list[float] | None = None,
    ) -> None:
        """写入 / 更新一条 MemoryUnit 的向量。
        
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
            table.delete(f"unit_id = '{self._escape_sql(unit_id)}'")
        except Exception:
            pass
        table.add([{
            "unit_id": unit_id,
            "user_id": user_id,
            "memory_type": memory_type,
            "vector": semantic_vector,
            "normalized_vector": normalized_vector,
            "expired": expired,
        }])

    def upsert_batch(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        """批量写入 MemoryUnit 向量。
        
        每条 record 需要: unit_id, user_id, memory_type, semantic_text, normalized_text。
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
        ids = [r["unit_id"] for r in records]
        for uid in ids:
            try:
                table.delete(f"unit_id = '{self._escape_sql(uid)}'")
            except Exception:
                pass

        rows = [
            {
                "unit_id": r["unit_id"],
                "user_id": r["user_id"],
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
        user_id: str = "",
        column: str = "vector",
        limit: int = 20,
        memory_types: list[str] | None = None,
        include_expired: bool = False,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """向量 ANN 检索 MemoryUnit。

        column: "vector"（语义检索） 或 "normalized_vector"（归一化/实用检索）
        """
        if query_vector is None:
            if self._embedder is None:
                raise RuntimeError("LanceVectorIndex: embedder 未设置且未提供查询向量")
            query_vector = self._embedder.embed_query(query_text)

        table = self._db.open_table(self.MEMORY_TABLE)

        # 构造过滤条件
        filters: list[str] = []
        if user_id:
            filters.append(f"user_id = '{self._escape_sql(user_id)}'")
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
                "unit_id": r.get("unit_id", ""),
                "user_id": r.get("user_id", ""),
                "memory_type": r.get("memory_type", ""),
                "_distance": r.get("_distance", 999.0),
            }
            for r in results
        ]

    # ──────────────────────────────────────
    # SynthesizedUnit 向量写入 / 检索
    # ──────────────────────────────────────

    def upsert_synthesized(
        self,
        synth_id: str,
        user_id: str,
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
            table.delete(f"synth_id = '{self._escape_sql(synth_id)}'")
        except Exception:
            pass
        table.add([{
            "synth_id": synth_id,
            "user_id": user_id,
            "memory_type": memory_type,
            "vector": semantic_vector,
            "normalized_vector": normalized_vector,
        }])

    def search_synthesized(
        self,
        query_text: str,
        *,
        user_id: str = "",
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
        if user_id:
            filters.append(f"user_id = '{self._escape_sql(user_id)}'")
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
                "synth_id": r.get("synth_id", ""),
                "user_id": r.get("user_id", ""),
                "memory_type": r.get("memory_type", ""),
                "_distance": r.get("_distance", 999.0),
            }
            for r in results
        ]

    # ──────────────────────────────────────
    # 批量删除
    # ──────────────────────────────────────

    def delete_all_for_user(self, user_id: str) -> None:
        """删除指定用户的全部向量。"""
        escaped = self._escape_sql(user_id)
        for tname in [self.MEMORY_TABLE, self.SYNTH_TABLE]:
            try:
                table = self._db.open_table(tname)
                table.delete(f"user_id = '{escaped}'")
            except Exception:
                pass

    def delete_unit(self, unit_id: str) -> None:
        try:
            table = self._db.open_table(self.MEMORY_TABLE)
            table.delete(f"unit_id = '{self._escape_sql(unit_id)}'")
        except Exception:
            pass

    def mark_expired(self, unit_id: str) -> None:
        """标记过期（向量层面：删除后以 expired=True 重新插入开销大，直接删除即可，
        具体过期条目不参与 ANN 召回，靠 sqlite 侧查询）。"""
        self.delete_unit(unit_id)

    # ──────────────────────────────────────
    # 工具
    # ──────────────────────────────────────

    @staticmethod
    def _escape_sql(value: str) -> str:
        """防止 SQL 注入（LanceDB filter 为字符串拼接）。"""
        return value.replace("'", "''").replace("\\", "\\\\")
