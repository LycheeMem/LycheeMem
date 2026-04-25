"""SQLite + FTS5 + LanceDB 技能库存储（程序记忆持久化）。

职责：
- skills 表：SkillEntry 主存储（SQLite WAL 模式）
- skills_fts：FTS5 全文索引（覆盖 intent + doc_markdown）
- LanceDB skills 表：intent 向量索引（cosine ANN 检索）

接口与 FileSkillStore 完全兼容（BaseMemoryStore API + record_usage）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

import lancedb
import pyarrow as pa

from src.embedder.base import BaseEmbedder
from src.memory.base import BaseMemoryStore


# ──────────────────────────────────────
# 工具函数
# ──────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(obj: Any) -> str:
    if obj is None:
        return "{}"
    return json.dumps(obj, ensure_ascii=False)


def _json_loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _escape_sql(s: str) -> str:
    """转义 SQL 字符串中的单引号（用于 LanceDB 过滤字符串拼接）。"""
    return s.replace("'", "''")


def _make_skill_vector_schema(dim: int) -> pa.Schema:
    """构建 LanceDB skills 表 schema。dim > 0 时使用 FixedSizeList（推荐）。"""
    vec_type = pa.list_(pa.float32(), dim) if dim > 0 else pa.list_(pa.float32())
    return pa.schema([
        pa.field("skill_id", pa.utf8()),
        pa.field("vector", vec_type),
    ])


# ──────────────────────────────────────
# 主类
# ──────────────────────────────────────

class SQLiteSkillStore(BaseMemoryStore):
    """基于 SQLite + FTS5 + LanceDB 的技能库存储。

    - SQLite：结构化字段 + FTS5 全文检索
    - LanceDB：intent 向量 ANN 检索（cosine metric）

    Args:
        db_path: SQLite 数据库文件路径。
        vector_db_path: LanceDB 目录路径。
        embedder: 用于在未提供 query_embedding 时自动计算查询向量。
        embedding_dim: 向量维度（0 = 变长，不推荐）。
    """

    SKILL_TABLE = "skills"

    def __init__(
        self,
        db_path: str = "data/skill_store.db",
        vector_db_path: str = "data/skill_vector",
        embedder: BaseEmbedder | None = None,
        embedding_dim: int = 0,
    ):
        self._db_path = db_path
        self._vector_db_path = vector_db_path
        self._embedder = embedder
        self._embedding_dim = embedding_dim
        self._lock = threading.Lock()
        self._local = threading.local()

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(vector_db_path, exist_ok=True)

        self._init_schema()

        self._ldb = lancedb.connect(vector_db_path)
        self._ensure_vector_table()

    @property
    def _conn(self) -> sqlite3.Connection:
        """返回当前线程专属的 SQLite 连接（按需创建）。"""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

        self._ldb = lancedb.connect(vector_db_path)
        self._ensure_vector_table()

    def set_embedder(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    # ──────────────────────────────────────
    # Schema 初始化
    # ──────────────────────────────────────

    def _init_schema(self) -> None:
        """创建 SQLite 表 + FTS5 索引（幂等）。"""
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id      TEXT PRIMARY KEY,
                    intent        TEXT NOT NULL,
                    doc_markdown  TEXT NOT NULL,
                    conditions    TEXT NOT NULL DEFAULT '',
                    metadata      TEXT NOT NULL DEFAULT '{}',
                    success_count INTEGER NOT NULL DEFAULT 0,
                    last_used     TEXT NOT NULL DEFAULT '',
                    created_at    TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_skills_created ON skills(created_at);
            """)

            # FTS5 虚拟表（分开创建，兼容 IF NOT EXISTS）
            try:
                self._conn.execute("""
                    CREATE VIRTUAL TABLE skills_fts USING fts5(
                        skill_id     UNINDEXED,
                        intent,
                        doc_markdown,
                        content=skills,
                        content_rowid=rowid,
                        tokenize='unicode61'
                    )
                """)
            except sqlite3.OperationalError:
                pass  # 已存在

            self._conn.commit()

    def _ensure_vector_table(self) -> None:
        """创建或验证 LanceDB 表 schema（幂等）。"""
        schema = _make_skill_vector_schema(self._embedding_dim)
        existing = set(self._ldb.table_names())

        if self.SKILL_TABLE not in existing:
            self._ldb.create_table(self.SKILL_TABLE, schema=schema)
        elif self._embedding_dim > 0:
            # 若现有表是变长 list schema，重建为 FixedSizeList
            try:
                tbl = self._ldb.open_table(self.SKILL_TABLE)
                vec_field = tbl.schema.field("vector")
                if not pa.types.is_fixed_size_list(vec_field.type):
                    self._ldb.drop_table(self.SKILL_TABLE)
                    self._ldb.create_table(self.SKILL_TABLE, schema=schema)
            except Exception:
                pass

    # ──────────────────────────────────────
    # BaseMemoryStore API
    # ──────────────────────────────────────

    def add(self, items: list[dict[str, Any]]) -> None:
        """写入一批技能条目（幂等 upsert，以 skill_id 为主键）。

        每条 item 需要 intent + doc_markdown；可选 id、embedding、conditions、metadata
        等字段。若未提供 embedding 且 embedder 已配置，则自动对 intent 编码。
        """
        if not items:
            return
        now = _now_iso()
        for item in items:
            skill_id = item.get("id", str(uuid.uuid4()))
            intent = item["intent"]
            doc_markdown = item["doc_markdown"]
            conditions = item.get("conditions", "")
            metadata = item.get("metadata", {})
            success_count = item.get("success_count", 0)
            last_used = item.get("last_used", "")
            created_at = item.get("created_at", now)

            # ── SQLite upsert ──
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO skills
                        (skill_id, intent, doc_markdown, conditions, metadata,
                         success_count, last_used, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(skill_id) DO UPDATE SET
                        intent        = excluded.intent,
                        doc_markdown  = excluded.doc_markdown,
                        conditions    = excluded.conditions,
                        metadata      = excluded.metadata,
                        last_used     = excluded.last_used
                    """,
                    (skill_id, intent, doc_markdown, conditions,
                     _json_dumps(metadata), success_count, last_used, created_at),
                )
                # FTS5 同步
                self._conn.execute(
                    "INSERT OR REPLACE INTO skills_fts"
                    "  (rowid, skill_id, intent, doc_markdown)"
                    "  SELECT rowid, skill_id, intent, doc_markdown"
                    "  FROM skills WHERE skill_id = ?",
                    (skill_id,),
                )
                self._conn.commit()

            # ── LanceDB upsert ──
            vec: list[float] | None = item.get("embedding")
            if vec is None and self._embedder is not None:
                vec = self._embedder.embed_query(intent)
            if vec is not None:
                self._upsert_vector(skill_id, vec)

    def search(
        self,
        query: str,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """向量 ANN 检索技能库（cosine 相似度）。

        仅使用向量通道：调用方（SearchCoordinator）已通过 HyDE 生成了最优查询向量，
        与 FileSkillStore 接口保持一致。

        Returns:
            列表，每条含 id / intent / doc_markdown / score / metadata /
            success_count / conditions。score ∈ [0, 1]，值越大越相关。
        """
        if query_embedding is None:
            if self._embedder is None:
                return []
            query_embedding = self._embedder.embed_query(query)

        # ── LanceDB 向量检索（cosine metric）──
        try:
            tbl = self._ldb.open_table(self.SKILL_TABLE)
            q = (
                tbl.search(query_embedding, vector_column_name="vector")
                .metric("cosine")
                .limit(top_k * 3)  # 多取一些，过滤后再截断
            )
            hits = q.to_list()
        except Exception:
            return []

        if not hits:
            return []

        # ── 按 skill_id 批量查 SQLite ──
        skill_ids = [h["skill_id"] for h in hits]
        # cosine distance ∈ [0, 1]：0 = 完全相同，1 = 完全不同
        # cosine similarity = 1 - distance
        dist_map = {h["skill_id"]: float(h.get("_distance", 1.0)) for h in hits}

        placeholders = ",".join("?" for _ in skill_ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM skills WHERE skill_id IN ({placeholders})",
                skill_ids,
            ).fetchall()

        if not rows:
            return []

        results: list[dict[str, Any]] = []
        for row in rows:
            sid = row["skill_id"]
            dist = dist_map.get(sid, 1.0)
            sim = max(0.0, 1.0 - dist)  # cosine distance → cosine similarity
            results.append({
                "id": sid,
                "intent": row["intent"],
                "doc_markdown": row["doc_markdown"],
                "score": sim,
                "metadata": _json_loads_dict(row["metadata"]),
                "success_count": row["success_count"],
                "conditions": row["conditions"],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete(self, ids: list[str]) -> None:
        """删除指定 ID 的技能。"""
        if not ids:
            return

        with self._lock:
            for skill_id in ids:
                self._conn.execute(
                    "DELETE FROM skills WHERE skill_id = ?",
                    (skill_id,),
                )
            self._conn.commit()

        try:
            tbl = self._ldb.open_table(self.SKILL_TABLE)
            for skill_id in ids:
                tbl.delete(f"skill_id = '{_escape_sql(skill_id)}'")
        except Exception:
            pass

    def delete_all(self) -> None:
        """清空技能库。"""
        with self._lock:
            self._conn.execute("DELETE FROM skills")
            self._conn.commit()

        try:
            # 重建空表（LanceDB 不支持 TRUNCATE）
            self._ldb.drop_table(self.SKILL_TABLE)
            schema = _make_skill_vector_schema(self._embedding_dim)
            self._ldb.create_table(self.SKILL_TABLE, schema=schema)
        except Exception:
            pass

    def get_all(self) -> list[dict[str, Any]]:
        """返回全部技能条目（不含向量）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM skills ORDER BY created_at DESC"
            ).fetchall()

        return [
            {
                "id": row["skill_id"],
                "intent": row["intent"],
                "doc_markdown": row["doc_markdown"],
                "metadata": _json_loads_dict(row["metadata"]),
                "success_count": row["success_count"],
                "last_used": row["last_used"],
                "conditions": row["conditions"],
            }
            for row in rows
        ]

    # ──────────────────────────────────────
    # 额外 API（与 FileSkillStore 兼容）
    # ──────────────────────────────────────

    def record_usage(self, skill_id: str) -> None:
        """递增 success_count 并更新 last_used 时间戳。"""
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                "UPDATE skills SET success_count = success_count + 1, last_used = ? "
                "WHERE skill_id = ?",
                (now, skill_id),
            )
            self._conn.commit()

    def fulltext_search(
        self,
        query: str,
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """FTS5 全文检索（按 intent / doc_markdown 关键词匹配）。

        作为向量检索的补充，可用于关键词精确命中场景。
        """
        safe_q = query.replace('"', '""')
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT s.*, fts.rank
                    FROM skills_fts fts
                    JOIN skills s ON s.skill_id = fts.skill_id
                    WHERE skills_fts MATCH ?
                    ORDER BY fts.rank
                    LIMIT ?
                    """,
                    (safe_q, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []

        result = []
        for row in rows:
            result.append({
                "id": row["skill_id"],
                "intent": row["intent"],
                "doc_markdown": row["doc_markdown"],
                "score": 1.0,
                "metadata": _json_loads_dict(row["metadata"]),
                "success_count": row["success_count"],
                "conditions": row["conditions"],
            })
        return result

    # ──────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────

    def _upsert_vector(self, skill_id: str, vector: list[float]) -> None:
        """在 LanceDB 中写入或更新一条技能向量。"""
        try:
            tbl = self._ldb.open_table(self.SKILL_TABLE)
            tbl.delete(f"skill_id = '{_escape_sql(skill_id)}'")
            tbl.add([{"skill_id": skill_id, "vector": vector}])
        except Exception:
            pass
