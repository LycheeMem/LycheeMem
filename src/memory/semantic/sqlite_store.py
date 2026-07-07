"""SQLite + FTS5 结构化存储。

职责：
- memory_records 表：MemoryRecord 主存储
- evidence_nodes 表：由 entity/tag/time/session-turn 等结构化字段形成的无损证据索引
- memory_records_fts：FTS5 全文索引（覆盖 normalized_text + entities + tags）
- evidence_nodes_fts：FTS5 全文索引

所有写操作线程安全（sqlite3 + threading.Lock）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any

from src.memory.semantic.models import MemoryRecord
from src.utils.time_utils import normalize_date_key


def _json_dumps(obj: Any) -> str:
    """JSON 序列化，空值返回空数组字符串。"""
    if obj is None:
        return "[]"
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(s: str | None) -> Any:
    if not s:
        return []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []


def _json_loads_dict(s: str | None) -> dict:
    if not s:
        return {}
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _merge_unique_values(*values: Any) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = [value]
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
    return result


def _merge_temporal_dict(existing: Any, incoming: Any) -> dict[str, Any]:
    left = existing if isinstance(existing, dict) else {}
    right = incoming if isinstance(incoming, dict) else {}
    merged: dict[str, Any] = dict(left)
    for key, value in right.items():
        if value in ("", None, [], {}):
            continue
        if key in {"start", "t_valid_from"}:
            current = str(merged.get(key) or "")
            incoming_text = normalize_date_key(value) or str(value)
            current = normalize_date_key(current) or current
            merged[key] = min(current, incoming_text) if current else incoming_text
        elif key in {"end", "t_valid_to"}:
            current = str(merged.get(key) or "")
            incoming_text = normalize_date_key(value) or str(value)
            current = normalize_date_key(current) or current
            merged[key] = max(current, incoming_text) if current else incoming_text
        else:
            merged[key] = value
    return merged


class SQLiteSemanticStore:
    """基于 SQLite + FTS5 的 Compact Semantic Memory 结构化存储。"""

    def __init__(self, db_path: str = "lychee_compact_memory.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._local = threading.local()
        self._evidence_type_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # init_schema() 通过 self._conn property 在当前线程建立首条连接
        self.init_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        """返回当前线程专属的 SQLite 连接（按需创建）。

        每个线程持有独立连接，彻底消除跨线程共享连接导致的
        sqlite3.InterfaceError: bad parameter or other API misuse。
        WAL 模式下多读一写不互相阻塞；写操作仍由 self._lock 序列化。
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
        return conn

    def init_schema(self) -> None:
        """创建表 + FTS5 索引（幂等）。"""
        with self._lock:
            c = self._conn
            if self._db_path != ":memory:":
                c.execute("PRAGMA journal_mode=WAL")
            c.executescript("""
                CREATE TABLE IF NOT EXISTS memory_records (
                    record_id        TEXT PRIMARY KEY,
                    memory_type      TEXT NOT NULL,
                    semantic_text    TEXT NOT NULL,
                    normalized_text  TEXT NOT NULL,
                    entities         TEXT NOT NULL DEFAULT '[]',
                    temporal         TEXT NOT NULL DEFAULT '{}',
                    tags             TEXT NOT NULL DEFAULT '[]',
                    confidence       REAL NOT NULL DEFAULT 1.0,
                    evidence_turn_range TEXT NOT NULL DEFAULT '[]',
                    source_session   TEXT NOT NULL DEFAULT '',
                    source_role      TEXT NOT NULL DEFAULT '',
                    created_at       TEXT NOT NULL DEFAULT '',
                    updated_at       TEXT NOT NULL DEFAULT '',
                    expired          INTEGER NOT NULL DEFAULT 0,
                    expired_at       TEXT NOT NULL DEFAULT '',
                    expired_reason   TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS evidence_nodes (
                    node_id          TEXT PRIMARY KEY,
                    node_type        TEXT NOT NULL,
                    key              TEXT NOT NULL,
                    label            TEXT NOT NULL DEFAULT '',
                    search_text      TEXT NOT NULL DEFAULT '',
                    record_ids       TEXT NOT NULL DEFAULT '[]',
                    session_ids      TEXT NOT NULL DEFAULT '[]',
                    entities         TEXT NOT NULL DEFAULT '[]',
                    tags             TEXT NOT NULL DEFAULT '[]',
                    temporal         TEXT NOT NULL DEFAULT '{}',
                    evidence_turn_ranges TEXT NOT NULL DEFAULT '[]',
                    metadata         TEXT NOT NULL DEFAULT '{}',
                    created_at       TEXT NOT NULL DEFAULT '',
                    updated_at       TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_mr_memory_type ON memory_records(memory_type);
                CREATE INDEX IF NOT EXISTS idx_mr_expired ON memory_records(expired);
                CREATE INDEX IF NOT EXISTS idx_mr_created_at ON memory_records(created_at);
                CREATE INDEX IF NOT EXISTS idx_en_type ON evidence_nodes(node_type);
                CREATE INDEX IF NOT EXISTS idx_en_key ON evidence_nodes(key);
            """)

            # FTS5 虚拟表
            try:
                c.execute("""
                    CREATE VIRTUAL TABLE memory_records_fts USING fts5(
                        record_id UNINDEXED,
                        normalized_text,
                        temporal,
                        entities,
                        tags,
                        content=memory_records,
                        content_rowid=rowid,
                        tokenize='unicode61'
                    )
                """)
            except sqlite3.OperationalError:
                pass  # 已存在

            try:
                c.execute("""
                    CREATE VIRTUAL TABLE evidence_nodes_fts USING fts5(
                        node_id UNINDEXED,
                        node_type UNINDEXED,
                        key,
                        label,
                        search_text,
                        entities,
                        tags,
                        content=evidence_nodes,
                        content_rowid=rowid,
                        tokenize='unicode61'
                    )
                """)
            except sqlite3.OperationalError:
                pass

            c.commit()

    # ──────────────────────────────────────
    # MemoryRecord CRUD
    # ──────────────────────────────────────

    def upsert_record(self, record: MemoryRecord) -> None:
        """写入或更新一个 MemoryRecord（幂等）。"""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_records (
                    record_id, memory_type, semantic_text, normalized_text,
                    entities, temporal, tags,
                    confidence, evidence_turn_range, source_session, source_role,
                    created_at, updated_at,
                    expired, expired_at, expired_reason
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?
                )
                ON CONFLICT(record_id) DO UPDATE SET
                    memory_type = excluded.memory_type,
                    semantic_text = excluded.semantic_text,
                    normalized_text = excluded.normalized_text,
                    entities = excluded.entities,
                    temporal = excluded.temporal,
                    tags = excluded.tags,
                    confidence = excluded.confidence,
                    evidence_turn_range = excluded.evidence_turn_range,
                    source_role = excluded.source_role,
                    updated_at = excluded.updated_at,
                    expired = excluded.expired,
                    expired_at = excluded.expired_at,
                    expired_reason = excluded.expired_reason
                """,
                (
                    record.record_id, record.memory_type,
                    record.semantic_text, record.normalized_text,
                    _json_dumps(record.entities), _json_dumps(record.temporal),
                    _json_dumps(record.tags),
                    record.confidence, _json_dumps(record.evidence_turn_range),
                    record.source_session, record.source_role,
                    record.created_at, record.updated_at,
                    int(record.expired), record.expired_at, record.expired_reason,
                ),
            )
            # 同步 FTS5
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_records_fts(rowid, record_id, "
                "normalized_text, temporal, entities, tags) "
                "SELECT rowid, record_id, normalized_text, temporal, entities, tags "
                "FROM memory_records WHERE record_id = ?",
                (record.record_id,),
            )
            self._conn.commit()

    def get_record(self, record_id: str) -> MemoryRecord | None:
        """按 ID 获取单个 MemoryRecord。"""
        row = self._conn.execute(
            "SELECT * FROM memory_records WHERE record_id = ?", (record_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_records_by_ids(self, record_ids: list[str]) -> dict[str, MemoryRecord]:
        """Batch fetch MemoryRecord rows by ID."""
        cleaned: list[str] = []
        seen: set[str] = set()
        for record_id in record_ids:
            rid = str(record_id or "").strip()
            if not rid or rid in seen:
                continue
            seen.add(rid)
            cleaned.append(rid)
        if not cleaned:
            return {}

        found: dict[str, MemoryRecord] = {}
        for start in range(0, len(cleaned), 500):
            chunk = cleaned[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT * FROM memory_records WHERE record_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                record = self._row_to_record(row)
                found[record.record_id] = record
        return found

    def delete_all(self) -> dict[str, int]:
        """清空全部数据。"""
        with self._lock:
            c1 = self._conn.execute(
                "SELECT COUNT(*) FROM memory_records"
            ).fetchone()[0]
            c2 = self._conn.execute(
                "SELECT COUNT(*) FROM evidence_nodes"
            ).fetchone()[0]
            self._conn.execute("DELETE FROM memory_records")
            self._conn.execute("DELETE FROM evidence_nodes")
            for fts_table in ("memory_records_fts", "evidence_nodes_fts"):
                try:
                    self._conn.execute(f"DELETE FROM {fts_table}")
                except sqlite3.OperationalError:
                    pass
            self._conn.commit()
            self._clear_evidence_type_cache()
        return {"records_deleted": c1, "evidence_nodes_deleted": c2}

    # ──────────────────────────────────────
    # FTS5 全文检索
    # ──────────────────────────────────────

    def fulltext_search(
        self,
        query: str,
        *,
        limit: int = 30,
        memory_types: list[str] | None = None,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        """BM25 全文召回 MemoryRecord。"""
        fts_query = self._escape_fts_query(query)
        if not fts_query:
            return []

        sql = (
            "SELECT m.*, bm25(memory_records_fts) AS fts_score "
            "FROM memory_records_fts f "
            "JOIN memory_records m ON f.rowid = m.rowid "
            "WHERE memory_records_fts MATCH ? "
        )
        params: list[Any] = [fts_query]

        if not include_expired:
            sql += "AND m.expired = 0 "
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            sql += f"AND m.memory_type IN ({placeholders}) "
            params.extend(memory_types)

        sql += "ORDER BY fts_score LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ──────────────────────────────────────
    # EvidenceNode CRUD / 检索
    # ──────────────────────────────────────

    def upsert_evidence_nodes(self, nodes: list[dict[str, Any]]) -> None:
        """写入由 entity/tag/time/event-frame 构成的无损证据索引节点。"""
        if not nodes:
            return
        now_nodes: list[dict[str, Any]] = []
        with self._lock:
            for node in nodes:
                node_id = str(node.get("node_id") or "").strip()
                node_type = str(node.get("node_type") or "").strip()
                key = str(node.get("key") or "").strip()
                if not node_id or not node_type or not key:
                    continue

                existing_row = self._conn.execute(
                    "SELECT * FROM evidence_nodes WHERE node_id = ?", (node_id,)
                ).fetchone()
                if existing_row is not None:
                    existing = self._row_to_dict(existing_row)
                else:
                    existing = {}

                existing_text = str(existing.get("search_text") or "").strip()
                incoming_text = str(node.get("search_text") or "").strip()
                if existing_text and incoming_text and incoming_text not in existing_text:
                    search_text = (existing_text + "\n" + incoming_text).strip()
                else:
                    search_text = incoming_text or existing_text
                if len(search_text) > 8000:
                    search_text = search_text[-8000:].lstrip()

                created_at = str(
                    existing.get("created_at")
                    or node.get("created_at")
                    or ""
                )
                merged = {
                    "node_id": node_id,
                    "node_type": node_type,
                    "key": key,
                    "label": str(node.get("label") or existing.get("label") or key),
                    "search_text": search_text,
                    "record_ids": _merge_unique_values(
                        existing.get("record_ids"), node.get("record_ids")
                    ),
                    "session_ids": _merge_unique_values(
                        existing.get("session_ids"), node.get("session_ids")
                    ),
                    "entities": _merge_unique_values(
                        existing.get("entities"), node.get("entities")
                    ),
                    "tags": _merge_unique_values(existing.get("tags"), node.get("tags")),
                    "temporal": _merge_temporal_dict(
                        existing.get("temporal"), node.get("temporal")
                    ),
                    "evidence_turn_ranges": _merge_unique_values(
                        existing.get("evidence_turn_ranges"),
                        node.get("evidence_turn_ranges"),
                    ),
                    "metadata": {
                        **(existing.get("metadata") if isinstance(existing.get("metadata"), dict) else {}),
                        **(node.get("metadata") if isinstance(node.get("metadata"), dict) else {}),
                    },
                    "created_at": created_at,
                    "updated_at": str(node.get("updated_at") or ""),
                }
                now_nodes.append(merged)

                self._conn.execute(
                    """
                    INSERT INTO evidence_nodes (
                        node_id, node_type, key, label, search_text,
                        record_ids, session_ids, entities, tags, temporal,
                        evidence_turn_ranges, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        node_type = excluded.node_type,
                        key = excluded.key,
                        label = excluded.label,
                        search_text = excluded.search_text,
                        record_ids = excluded.record_ids,
                        session_ids = excluded.session_ids,
                        entities = excluded.entities,
                        tags = excluded.tags,
                        temporal = excluded.temporal,
                        evidence_turn_ranges = excluded.evidence_turn_ranges,
                        metadata = excluded.metadata,
                        updated_at = excluded.updated_at
                    """,
                    (
                        merged["node_id"], merged["node_type"], merged["key"],
                        merged["label"], merged["search_text"],
                        _json_dumps(merged["record_ids"]),
                        _json_dumps(merged["session_ids"]),
                        _json_dumps(merged["entities"]),
                        _json_dumps(merged["tags"]),
                        _json_dumps(merged["temporal"]),
                        _json_dumps(merged["evidence_turn_ranges"]),
                        _json_dumps(merged["metadata"]),
                        merged["created_at"], merged["updated_at"],
                    ),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO evidence_nodes_fts(rowid, node_id, "
                    "node_type, key, label, search_text, entities, tags) "
                    "SELECT rowid, node_id, node_type, key, label, search_text, entities, tags "
                    "FROM evidence_nodes WHERE node_id = ?",
                    (merged["node_id"],),
                )
            self._conn.commit()
            self._clear_evidence_type_cache()

    def get_evidence_node(self, node_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM evidence_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def get_evidence_nodes_by_ids(self, node_ids: list[str]) -> list[dict[str, Any]]:
        cleaned = [str(node_id or "").strip() for node_id in node_ids if str(node_id or "").strip()]
        if not cleaned:
            return []
        placeholders = ",".join("?" for _ in cleaned)
        rows = self._conn.execute(
            f"SELECT * FROM evidence_nodes WHERE node_id IN ({placeholders})",
            cleaned,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_evidence_nodes_by_type(
        self,
        node_type: str,
        *,
        limit: int = 100_000,
    ) -> list[dict[str, Any]]:
        cleaned_type = str(node_type or "").strip()
        if not cleaned_type:
            return []
        cache_key = (cleaned_type, max(1, int(limit or 1)))
        with self._cache_lock:
            cached = self._evidence_type_cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        rows = self._conn.execute(
            "SELECT * FROM evidence_nodes WHERE node_type = ? ORDER BY key LIMIT ?",
            cache_key,
        ).fetchall()
        result = [self._row_to_dict(r) for r in rows]
        with self._cache_lock:
            self._evidence_type_cache[cache_key] = result
        return [dict(item) for item in result]

    def fulltext_search_evidence_nodes(
        self,
        query: str,
        *,
        node_types: list[str] | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        fts_query = self._escape_fts_query(query)
        if not fts_query:
            return []
        sql = (
            "SELECT n.*, bm25(evidence_nodes_fts) AS fts_score "
            "FROM evidence_nodes_fts f "
            "JOIN evidence_nodes n ON f.rowid = n.rowid "
            "WHERE evidence_nodes_fts MATCH ? "
        )
        params: list[Any] = [fts_query]
        if node_types:
            placeholders = ",".join("?" for _ in node_types)
            sql += f"AND n.node_type IN ({placeholders}) "
            params.extend(node_types)
        sql += "ORDER BY fts_score LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ──────────────────────────────────────
    # 调试导出
    # ──────────────────────────────────────

    def export_all(self) -> dict[str, Any]:
        """导出全部数据用于调试/前端。"""
        records = [self._row_to_dict(r) for r in self._conn.execute("SELECT * FROM memory_records").fetchall()]
        evidence_nodes = [self._row_to_dict(r) for r in self._conn.execute("SELECT * FROM evidence_nodes").fetchall()]
        return {
            "records": records,
            "evidence_nodes": evidence_nodes,
            "total_records": len(records),
            "total_evidence_nodes": len(evidence_nodes),
        }

    def count_records(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM memory_records WHERE expired = 0"
        ).fetchone()[0]

    def count_evidence_nodes(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM evidence_nodes"
        ).fetchone()[0]

    def count_composites(self) -> int:
        """Compatibility alias for older API status code."""
        return self.count_evidence_nodes()

    # ──────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────

    def _clear_evidence_type_cache(self) -> None:
        with self._cache_lock:
            self._evidence_type_cache.clear()

    @staticmethod
    def _escape_fts_query(query: str) -> str:
        """将查询文本转为 FTS5 OR 查询，同时添加前缀通配符提升召回率。

        例：'machine learning cat' → '"machine" OR "machine"* OR "learning" OR "learning"* OR "cat" OR "cat"*'
        使用 OR 而非隐式 AND，避免多词查询因缺少某个词而返回 0 结果。
        前缀通配符 * 支持前缀匹配，对中文实体词有一定帮助。
        """
        if not query or not query.strip():
            return ""
        # 按空格分词；忽略空 token
        tokens = [t.strip() for t in query.strip().split() if t.strip()]
        if not tokens:
            return ""
        # 每个 token 产生两项：精确匹配 + 前缀通配；用 OR 连接所有项
        parts: list[str] = []
        for t in tokens:
            # 去掉 token 中 FTS5 保留特殊字符，只保留字母/数字/中文
            safe = t.replace('"', '').replace("'", "").replace("*", "")
            if not safe:
                continue
            parts.append(f'"{safe}"')       # 精确 token 匹配
            parts.append(f'"{safe}"*')      # 前缀通配符
        if not parts:
            return ""
        return " OR ".join(parts)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in (
            "entities", "temporal", "tags",
            "evidence_turn_range",
            "record_ids", "session_ids", "evidence_turn_ranges", "metadata",
        ):
            if key in d and isinstance(d[key], str):
                d[key] = _json_loads(d[key]) if key != "temporal" else _json_loads_dict(d[key])
        if "expired" in d:
            d["expired"] = bool(d["expired"])
        return d

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        d = dict(row)
        return MemoryRecord(
            record_id=d["record_id"],
            memory_type=d["memory_type"],
            semantic_text=d["semantic_text"],
            normalized_text=d["normalized_text"],
            entities=_json_loads(d["entities"]),
            temporal=_json_loads_dict(d["temporal"]),
            tags=_json_loads(d["tags"]),
            confidence=float(d["confidence"] or 0.0),
            evidence_turn_range=_json_loads(d["evidence_turn_range"]),
            source_session=d["source_session"],
            source_role=d.get("source_role", ""),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            expired=bool(d["expired"]),
            expired_at=d["expired_at"],
            expired_reason=d["expired_reason"],
        )
