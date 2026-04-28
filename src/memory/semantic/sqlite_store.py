"""SQLite + FTS5 结构化存储。

职责：
- memory_records 表：MemoryRecord 主存储
- composite_records 表：CompositeRecord 聚合条目
- memory_records_fts：FTS5 全文索引（覆盖 normalized_text + entities + tags）
- composite_records_fts：FTS5 全文索引
- usage_logs 表：检索使用记录

所有写操作线程安全（sqlite3 + threading.Lock）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any

from src.memory.semantic.models import MemoryRecord, CompositeRecord, UsageLog


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


class SQLiteSemanticStore:
    """基于 SQLite + FTS5 的 Compact Semantic Memory 结构化存储。"""

    def __init__(self, db_path: str = "lychee_compact_memory.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.init_schema()

    def init_schema(self) -> None:
        """创建表 + FTS5 索引（幂等）。"""
        with self._lock:
            c = self._conn
            c.executescript("""
                CREATE TABLE IF NOT EXISTS memory_records (
                    record_id        TEXT PRIMARY KEY,
                    memory_type      TEXT NOT NULL,
                    semantic_text    TEXT NOT NULL,
                    normalized_text  TEXT NOT NULL,
                    entities         TEXT NOT NULL DEFAULT '[]',
                    temporal         TEXT NOT NULL DEFAULT '{}',
                    task_tags        TEXT NOT NULL DEFAULT '[]',
                    tool_tags        TEXT NOT NULL DEFAULT '[]',
                    constraint_tags  TEXT NOT NULL DEFAULT '[]',
                    failure_tags     TEXT NOT NULL DEFAULT '[]',
                    affordance_tags  TEXT NOT NULL DEFAULT '[]',
                    confidence       REAL NOT NULL DEFAULT 1.0,
                    evidence_turn_range TEXT NOT NULL DEFAULT '[]',
                    source_session   TEXT NOT NULL DEFAULT '',
                    source_role      TEXT NOT NULL DEFAULT '',
                    created_at       TEXT NOT NULL DEFAULT '',
                    updated_at       TEXT NOT NULL DEFAULT '',
                    retrieval_count       INTEGER NOT NULL DEFAULT 0,
                    retrieval_hit_count   INTEGER NOT NULL DEFAULT 0,
                    action_success_count  INTEGER NOT NULL DEFAULT 0,
                    action_fail_count     INTEGER NOT NULL DEFAULT 0,
                    last_retrieved_at     TEXT NOT NULL DEFAULT '',
                    expired          INTEGER NOT NULL DEFAULT 0,
                    expired_at       TEXT NOT NULL DEFAULT '',
                    expired_reason   TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS composite_records (
                    composite_id     TEXT PRIMARY KEY,
                    memory_type      TEXT NOT NULL,
                    semantic_text    TEXT NOT NULL,
                    normalized_text  TEXT NOT NULL,
                    source_record_ids TEXT NOT NULL DEFAULT '[]',
                    child_composite_ids TEXT NOT NULL DEFAULT '[]',
                    synthesis_reason TEXT NOT NULL DEFAULT '',
                    entities         TEXT NOT NULL DEFAULT '[]',
                    temporal         TEXT NOT NULL DEFAULT '{}',
                    task_tags        TEXT NOT NULL DEFAULT '[]',
                    tool_tags        TEXT NOT NULL DEFAULT '[]',
                    constraint_tags  TEXT NOT NULL DEFAULT '[]',
                    failure_tags     TEXT NOT NULL DEFAULT '[]',
                    affordance_tags  TEXT NOT NULL DEFAULT '[]',
                    confidence       REAL NOT NULL DEFAULT 1.0,
                    created_at       TEXT NOT NULL DEFAULT '',
                    updated_at       TEXT NOT NULL DEFAULT '',
                    retrieval_count       INTEGER NOT NULL DEFAULT 0,
                    retrieval_hit_count   INTEGER NOT NULL DEFAULT 0,
                    action_success_count  INTEGER NOT NULL DEFAULT 0,
                    action_fail_count     INTEGER NOT NULL DEFAULT 0,
                    last_retrieved_at     TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS usage_logs (
                    log_id           TEXT PRIMARY KEY,
                    session_id       TEXT NOT NULL,
                    timestamp        TEXT NOT NULL,
                    query            TEXT NOT NULL,
                    retrieval_plan   TEXT NOT NULL DEFAULT '{}',
                    action_state     TEXT NOT NULL DEFAULT '{}',
                    retrieved_record_ids TEXT NOT NULL DEFAULT '[]',
                    kept_record_ids  TEXT NOT NULL DEFAULT '[]',
                    final_response_excerpt TEXT NOT NULL DEFAULT '',
                    user_feedback    TEXT NOT NULL DEFAULT '',
                    action_outcome   TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_mr_memory_type ON memory_records(memory_type);
                CREATE INDEX IF NOT EXISTS idx_mr_expired ON memory_records(expired);
                CREATE INDEX IF NOT EXISTS idx_mr_created_at ON memory_records(created_at);
                CREATE INDEX IF NOT EXISTS idx_ul_session ON usage_logs(session_id);
            """)

            # 兼容旧库：若 source_role 列不存在则追加（ALTER TABLE 幂等）
            try:
                c.execute("ALTER TABLE memory_records ADD COLUMN source_role TEXT NOT NULL DEFAULT ''")
                c.commit()
            except sqlite3.OperationalError:
                pass  # 列已存在

            try:
                c.execute("ALTER TABLE usage_logs ADD COLUMN action_state TEXT NOT NULL DEFAULT '{}' ")
                c.commit()
            except sqlite3.OperationalError:
                pass  # 列已存在

            try:
                c.execute("ALTER TABLE composite_records ADD COLUMN child_composite_ids TEXT NOT NULL DEFAULT '[]'")
                c.commit()
            except sqlite3.OperationalError:
                pass  # 列已存在

            # FTS5 虚拟表（分开创建避免 IF NOT EXISTS 不兼容问题）
            try:
                c.execute("""
                    CREATE VIRTUAL TABLE memory_records_fts USING fts5(
                        record_id UNINDEXED,
                        normalized_text,
                        temporal,
                        entities,
                        task_tags,
                        tool_tags,
                        constraint_tags,
                        failure_tags,
                        affordance_tags,
                        content=memory_records,
                        content_rowid=rowid,
                        tokenize='unicode61'
                    )
                """)
            except sqlite3.OperationalError:
                pass  # 已存在

            # 迁移：旧 FTS5 缺少 temporal 列时重建（content table 模式; 数据保留在 memory_records）
            try:
                fts_cols = [
                    row[2]
                    for row in c.execute("PRAGMA table_info(memory_records_fts)").fetchall()
                ]
                if fts_cols and "temporal" not in fts_cols:
                    c.execute("DROP TABLE IF EXISTS memory_records_fts")
                    c.execute("""
                        CREATE VIRTUAL TABLE memory_records_fts USING fts5(
                            record_id UNINDEXED,
                            normalized_text,
                            temporal,
                            entities,
                            task_tags,
                            tool_tags,
                            constraint_tags,
                            failure_tags,
                            affordance_tags,
                            content=memory_records,
                            content_rowid=rowid,
                            tokenize='unicode61'
                        )
                    """)
                    c.execute("INSERT INTO memory_records_fts(memory_records_fts) VALUES('rebuild')")
                    c.commit()
            except Exception:
                pass  # 迁移失败不阻断启动

            try:
                c.execute("""
                    CREATE VIRTUAL TABLE composite_records_fts USING fts5(
                        composite_id UNINDEXED,
                        normalized_text,
                        entities,
                        task_tags,
                        tool_tags,
                        constraint_tags,
                        content=composite_records,
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
                    entities, temporal, task_tags, tool_tags,
                    constraint_tags, failure_tags, affordance_tags,
                    confidence, evidence_turn_range, source_session, source_role,
                    created_at, updated_at,
                    retrieval_count, retrieval_hit_count,
                    action_success_count, action_fail_count,
                    last_retrieved_at,
                    expired, expired_at, expired_reason
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?,
                    ?, ?, ?
                )
                ON CONFLICT(record_id) DO UPDATE SET
                    memory_type = excluded.memory_type,
                    semantic_text = excluded.semantic_text,
                    normalized_text = excluded.normalized_text,
                    entities = excluded.entities,
                    temporal = excluded.temporal,
                    task_tags = excluded.task_tags,
                    tool_tags = excluded.tool_tags,
                    constraint_tags = excluded.constraint_tags,
                    failure_tags = excluded.failure_tags,
                    affordance_tags = excluded.affordance_tags,
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
                    _json_dumps(record.task_tags), _json_dumps(record.tool_tags),
                    _json_dumps(record.constraint_tags), _json_dumps(record.failure_tags),
                    _json_dumps(record.affordance_tags),
                    record.confidence, _json_dumps(record.evidence_turn_range),
                    record.source_session, record.source_role,
                    record.created_at, record.updated_at,
                    record.retrieval_count, record.retrieval_hit_count,
                    record.action_success_count, record.action_fail_count,
                    record.last_retrieved_at,
                    int(record.expired), record.expired_at, record.expired_reason,
                ),
            )
            # 同步 FTS5
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_records_fts(rowid, record_id, "
                "normalized_text, temporal, entities, task_tags, tool_tags, "
                "constraint_tags, failure_tags, affordance_tags) "
                "SELECT rowid, record_id, normalized_text, temporal, entities, task_tags, "
                "tool_tags, constraint_tags, failure_tags, affordance_tags "
                "FROM memory_records WHERE record_id = ?",
                (record.record_id,),
            )
            self._conn.commit()

    def upsert_records(self, records: list[MemoryRecord]) -> None:
        """批量写入 MemoryRecord。"""
        for u in records:
            self.upsert_record(u)

    def get_record(self, record_id: str) -> MemoryRecord | None:
        """按 ID 获取单个 MemoryRecord。"""
        row = self._conn.execute(
            "SELECT * FROM memory_records WHERE record_id = ?", (record_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_records_by_ids(self, record_ids: list[str]) -> list[MemoryRecord]:
        """按 ID 列表批量获取。"""
        if not record_ids:
            return []
        placeholders = ",".join("?" for _ in record_ids)
        rows = self._conn.execute(
            f"SELECT * FROM memory_records WHERE record_id IN ({placeholders})",
            record_ids,
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def delete_record(self, record_id: str) -> None:
        """硬删除一个 MemoryRecord。"""
        with self._lock:
            self._conn.execute(
                "DELETE FROM memory_records WHERE record_id = ?", (record_id,)
            )
            self._conn.commit()

    def expire_record(
        self, record_id: str, *, expired_at: str = "", expired_reason: str = "",
    ) -> None:
        """软删除（标记过期）。"""
        with self._lock:
            self._conn.execute(
                "UPDATE memory_records SET expired = 1, expired_at = ?, expired_reason = ? "
                "WHERE record_id = ?",
                (expired_at, expired_reason, record_id),
            )
            self._conn.commit()

    def delete_all(self) -> dict[str, int]:
        """清空全部数据。"""
        with self._lock:
            c1 = self._conn.execute(
                "SELECT COUNT(*) FROM memory_records"
            ).fetchone()[0]
            c2 = self._conn.execute(
                "SELECT COUNT(*) FROM composite_records"
            ).fetchone()[0]
            self._conn.execute("DELETE FROM memory_records")
            self._conn.execute("DELETE FROM composite_records")
            self._conn.execute("DELETE FROM usage_logs")
            self._conn.commit()
        return {"records_deleted": c1, "composites_deleted": c2}

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

    def fulltext_search_synthesized(
        self,
        query: str,
        *,
        limit: int = 10,
        memory_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 全文召回复合记录。"""
        fts_query = self._escape_fts_query(query)
        if not fts_query:
            return []

        sql = (
            "SELECT s.*, bm25(composite_records_fts) AS fts_score "
            "FROM composite_records_fts f "
            "JOIN composite_records s ON f.rowid = s.rowid "
            "WHERE composite_records_fts MATCH ? "
        )
        params: list[Any] = [fts_query]

        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            sql += f"AND s.memory_type IN ({placeholders}) "
            params.extend(memory_types)

        sql += "ORDER BY fts_score LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ──────────────────────────────────────
    # 时间范围查询
    # ──────────────────────────────────────

    def search_by_time(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int = 30,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        """按事件有效时间范围查询 MemoryRecord。

        优先匹配 temporal.t_valid_from / t_valid_to（事件真实发生时间窗口）；
        同时对 created_at（写入时刻）做回退匹配，取两者的 UNION，保证不遗漏。
        """
        params: list[Any] = []
        filter_clauses: list[str] = []
        if not include_expired:
            filter_clauses.append("expired = 0")

        # 事件时间子查询：JSON_EXTRACT 读取 temporal 字段中的时间值
        # t_valid_from / t_valid_to 均存储为 ISO 字符串，可直接做字符串比较
        event_time_clauses: list[str] = []
        event_params: list[Any] = []
        write_time_clauses: list[str] = []
        write_params: list[Any] = []

        if since:
            # 事件结束时间 >= since（事件在查询区间内仍有效）
            # 若 t_valid_to 为空则退化为按 t_valid_from >= since
            event_time_clauses.append(
                "(JSON_EXTRACT(temporal, '$.t_valid_to') >= ? OR "
                "(JSON_EXTRACT(temporal, '$.t_valid_to') = '' AND "
                " JSON_EXTRACT(temporal, '$.t_valid_from') >= ?))"
            )
            event_params.extend([since, since])
            write_time_clauses.append("created_at >= ?")
            write_params.append(since)
        if until:
            # 事件开始时间 <= until（事件在查询区间内尚未结束）
            event_time_clauses.append(
                "(JSON_EXTRACT(temporal, '$.t_valid_from') <= ? OR "
                "(JSON_EXTRACT(temporal, '$.t_valid_from') = '' AND "
                " JSON_EXTRACT(temporal, '$.t_valid_to') <= ?))"
            )
            event_params.extend([until, until])
            write_time_clauses.append("created_at <= ?")
            write_params.append(until)

        if event_time_clauses and write_time_clauses:
            # UNION：事件时间命中 OR 写入时间命中
            event_where_parts = [*filter_clauses, *event_time_clauses]
            write_where_parts = [*filter_clauses, *write_time_clauses]
            event_where = " AND ".join(event_where_parts)
            write_where = " AND ".join(write_where_parts)
            sql = (
                f"SELECT * FROM memory_records WHERE {event_where} "
                f"UNION "
                f"SELECT * FROM memory_records WHERE {write_where} "
                f"ORDER BY created_at DESC LIMIT ?"
            )
            params.extend(event_params)
            params.extend(write_params)
            params.append(limit)
        else:
            # 没有时间约束，退化为全量查询
            where_sql = " AND ".join(filter_clauses) if filter_clauses else "1=1"
            sql = f"SELECT * FROM memory_records WHERE {where_sql} ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ──────────────────────────────────────
    # 按 tag 筛选
    # ──────────────────────────────────────

    def search_by_tags(
        self,
        *,
        tool_tags: list[str] | None = None,
        constraint_tags: list[str] | None = None,
        task_tags: list[str] | None = None,
        failure_tags: list[str] | None = None,
        affordance_tags: list[str] | None = None,
        memory_types: list[str] | None = None,
        limit: int = 30,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        """按 tag 字段做 LIKE 匹配查询。"""
        sql = "SELECT * FROM memory_records WHERE 1=1 "
        params: list[Any] = []

        if not include_expired:
            sql += "AND expired = 0 "
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            sql += f"AND memory_type IN ({placeholders}) "
            params.extend(memory_types)

        for tags, col in [
            (tool_tags, "tool_tags"),
            (constraint_tags, "constraint_tags"),
            (task_tags, "task_tags"),
            (failure_tags, "failure_tags"),
            (affordance_tags, "affordance_tags"),
        ]:
            if tags:
                conditions = []
                for tag in tags:
                    conditions.append(f"{col} LIKE ?")
                    params.append(f"%{tag}%")
                sql += "AND (" + " OR ".join(conditions) + ") "

        sql += "ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search_by_slot_hints(
        self,
        *,
        slot_terms: list[str],
        memory_types: list[str] | None = None,
        limit: int = 30,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        """根据缺失 slot 关键字，在文本与 tags 上做定向召回。"""
        cleaned_terms = [str(term or "").strip() for term in (slot_terms or []) if str(term or "").strip()]
        if not cleaned_terms:
            return []

        sql = "SELECT * FROM memory_records WHERE 1=1 "
        params: list[Any] = []

        if not include_expired:
            sql += "AND expired = 0 "
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            sql += f"AND memory_type IN ({placeholders}) "
            params.extend(memory_types)

        group_conditions: list[str] = []
        for term in cleaned_terms:
            group_conditions.extend([
                "normalized_text LIKE ?",
                "semantic_text LIKE ?",
                "entities LIKE ?",
                "constraint_tags LIKE ?",
                "tool_tags LIKE ?",
                "affordance_tags LIKE ?",
            ])
            like_term = f"%{term}%"
            params.extend([like_term] * 6)

        sql += "AND (" + " OR ".join(group_conditions) + ") "
        sql += "ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ──────────────────────────────────────
    # CompositeRecord CRUD
    # ──────────────────────────────────────

    def upsert_synthesized(self, composite: CompositeRecord) -> None:
        """写入或更新 CompositeRecord。"""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO composite_records (
                    composite_id, memory_type, semantic_text, normalized_text,
                    source_record_ids, child_composite_ids, synthesis_reason,
                    entities, temporal, task_tags, tool_tags,
                    constraint_tags, failure_tags, affordance_tags,
                    confidence, created_at, updated_at,
                    retrieval_count, retrieval_hit_count,
                    action_success_count, action_fail_count, last_retrieved_at
                ) VALUES (?, ?, ?, ?,  ?, ?, ?,  ?, ?, ?, ?,  ?, ?, ?,  ?, ?, ?,  ?, ?,  ?, ?, ?)
                ON CONFLICT(composite_id) DO UPDATE SET
                    memory_type = excluded.memory_type,
                    semantic_text = excluded.semantic_text,
                    normalized_text = excluded.normalized_text,
                    source_record_ids = excluded.source_record_ids,
                    child_composite_ids = excluded.child_composite_ids,
                    synthesis_reason = excluded.synthesis_reason,
                    entities = excluded.entities,
                    temporal = excluded.temporal,
                    task_tags = excluded.task_tags,
                    tool_tags = excluded.tool_tags,
                    constraint_tags = excluded.constraint_tags,
                    failure_tags = excluded.failure_tags,
                    affordance_tags = excluded.affordance_tags,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (
                    composite.composite_id, composite.memory_type,
                    composite.semantic_text, composite.normalized_text,
                    _json_dumps(composite.source_record_ids),
                    _json_dumps(composite.child_composite_ids),
                    composite.synthesis_reason,
                    _json_dumps(composite.entities), _json_dumps(composite.temporal),
                    _json_dumps(composite.task_tags), _json_dumps(composite.tool_tags),
                    _json_dumps(composite.constraint_tags), _json_dumps(composite.failure_tags),
                    _json_dumps(composite.affordance_tags),
                    composite.confidence, composite.created_at, composite.updated_at,
                    composite.retrieval_count, composite.retrieval_hit_count,
                    composite.action_success_count, composite.action_fail_count,
                    composite.last_retrieved_at,
                ),
            )
            # 同步 FTS5
            self._conn.execute(
                "INSERT OR REPLACE INTO composite_records_fts(rowid, composite_id, "
                "normalized_text, entities, task_tags, tool_tags, constraint_tags) "
                "SELECT rowid, composite_id, normalized_text, entities, task_tags, "
                "tool_tags, constraint_tags "
                "FROM composite_records WHERE composite_id = ?",
                (composite.composite_id,),
            )
            self._conn.commit()

    def get_synthesized_by_source(
        self, source_record_ids: list[str],
    ) -> list[CompositeRecord]:
        """查找包含指定 source_record_id 的合成条目。"""
        if not source_record_ids:
            return []
        rows = self._conn.execute(
            "SELECT * FROM composite_records"
        ).fetchall()
        results = []
        for r in rows:
            stored_ids = _json_loads(r["source_record_ids"])
            if any(uid in stored_ids for uid in source_record_ids):
                results.append(self._row_to_synth(r))
        return results

    def get_synthesized(self, composite_id: str) -> CompositeRecord | None:
        """按 composite_id 获取单个 CompositeRecord。"""
        row = self._conn.execute(
            "SELECT * FROM composite_records WHERE composite_id = ?", (composite_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_synth(row)

    def delete_synthesized(self, composite_id: str) -> None:
        """硬删除单个 CompositeRecord。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT rowid FROM composite_records WHERE composite_id = ?",
                (composite_id,),
            ).fetchone()

            if row is None:
                return

            rowid = row["rowid"]
            self._conn.execute(
                "DELETE FROM composite_records_fts WHERE rowid = ?",
                (rowid,),
            )
            self._conn.execute(
                "DELETE FROM composite_records WHERE composite_id = ?",
                (composite_id,),
            )
            self._conn.commit()

    def delete_synthesized_many(self, composite_ids: list[str]) -> None:
        """批量硬删除 CompositeRecord。"""
        for composite_id in composite_ids:
            if not str(composite_id or "").strip():
                continue
            self.delete_synthesized(str(composite_id).strip())

    def list_synthesized(self) -> list[CompositeRecord]:
        """获取全部 CompositeRecord。"""
        rows = self._conn.execute("SELECT * FROM composite_records").fetchall()
        return [self._row_to_synth(r) for r in rows]

    # ──────────────────────────────────────
    # 重复检测
    # ──────────────────────────────────────

    def find_similar_by_normalized_text(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """FTS5 模糊匹配，用于写入时去重。"""
        fts_query = self._escape_fts_query(normalized_text)
        if not fts_query:
            return []

        sql = (
            "SELECT m.* FROM memory_records_fts f "
            "JOIN memory_records m ON f.rowid = m.rowid "
            "WHERE memory_records_fts MATCH ? AND m.expired = 0 "
        )
        params: list[Any] = [fts_query]
        sql += "LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ──────────────────────────────────────
    # Usage Logs
    # ──────────────────────────────────────

    def insert_usage_log(self, log: UsageLog) -> None:
        """插入一条使用日志。"""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO usage_logs "
                "(log_id, session_id, timestamp, query, "
                "retrieval_plan, action_state, retrieved_record_ids, kept_record_ids, "
                "final_response_excerpt, user_feedback, action_outcome) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    log.log_id, log.session_id, log.timestamp,
                    log.query, _json_dumps(log.retrieval_plan),
                    _json_dumps(log.action_state),
                    _json_dumps(log.retrieved_record_ids),
                    _json_dumps(log.kept_record_ids),
                    log.final_response_excerpt,
                    log.user_feedback, log.action_outcome,
                ),
            )
            self._conn.commit()

    def get_usage_log(self, log_id: str) -> UsageLog | None:
        row = self._conn.execute(
            "SELECT * FROM usage_logs WHERE log_id = ?",
            (log_id,),
        ).fetchone()
        if row is None:
            return None
        return UsageLog(
            log_id=row["log_id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            query=row["query"],
            retrieval_plan=_json_loads_dict(row["retrieval_plan"]),
            action_state=_json_loads_dict(row["action_state"]),
            retrieved_record_ids=_json_loads(row["retrieved_record_ids"]),
            kept_record_ids=_json_loads(row["kept_record_ids"]),
            final_response_excerpt=row["final_response_excerpt"],
            user_feedback=row["user_feedback"],
            action_outcome=row["action_outcome"],
        )

    def update_usage_log(
        self,
        log_id: str,
        *,
        final_response_excerpt: str | None = None,
        user_feedback: str | None = None,
        action_outcome: str | None = None,
    ) -> None:
        fields: list[str] = []
        params: list[Any] = []

        if final_response_excerpt is not None:
            fields.append("final_response_excerpt = ?")
            params.append(final_response_excerpt)
        if user_feedback is not None:
            fields.append("user_feedback = ?")
            params.append(user_feedback)
        if action_outcome is not None:
            fields.append("action_outcome = ?")
            params.append(action_outcome)

        if not fields:
            return

        params.append(log_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE usage_logs SET {', '.join(fields)} WHERE log_id = ?",
                params,
            )
            self._conn.commit()

    def get_recent_usage_logs(
        self, *, session_id: str, limit: int = 20,
    ) -> list[UsageLog]:
        """获取最近的使用日志。"""
        rows = self._conn.execute(
            "SELECT * FROM usage_logs WHERE session_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [
            UsageLog(
                log_id=r["log_id"],
                session_id=r["session_id"],
                timestamp=r["timestamp"],
                query=r["query"],
                retrieval_plan=_json_loads_dict(r["retrieval_plan"]),
                action_state=_json_loads_dict(r["action_state"]),
                retrieved_record_ids=_json_loads(r["retrieved_record_ids"]),
                kept_record_ids=_json_loads(r["kept_record_ids"]),
                final_response_excerpt=r["final_response_excerpt"],
                user_feedback=r["user_feedback"],
                action_outcome=r["action_outcome"],
            )
            for r in rows
        ]

    # ──────────────────────────────────────
    # 使用统计批量更新
    # ──────────────────────────────────────

    def increment_retrieval_count(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        with self._lock:
            placeholders = ",".join("?" for _ in record_ids)
            self._conn.execute(
                f"UPDATE memory_records SET retrieval_count = retrieval_count + 1 "
                f"WHERE record_id IN ({placeholders})",
                record_ids,
            )
            self._conn.execute(
                f"UPDATE composite_records SET retrieval_count = retrieval_count + 1 "
                f"WHERE composite_id IN ({placeholders})",
                record_ids,
            )
            self._conn.commit()

    def increment_hit_count(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        with self._lock:
            placeholders = ",".join("?" for _ in record_ids)
            self._conn.execute(
                f"UPDATE memory_records SET retrieval_hit_count = retrieval_hit_count + 1 "
                f"WHERE record_id IN ({placeholders})",
                record_ids,
            )
            self._conn.execute(
                f"UPDATE composite_records SET retrieval_hit_count = retrieval_hit_count + 1 "
                f"WHERE composite_id IN ({placeholders})",
                record_ids,
            )
            self._conn.commit()

    def increment_action_success(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        with self._lock:
            placeholders = ",".join("?" for _ in record_ids)
            self._conn.execute(
                f"UPDATE memory_records SET action_success_count = action_success_count + 1 "
                f"WHERE record_id IN ({placeholders})",
                record_ids,
            )
            self._conn.execute(
                f"UPDATE composite_records SET action_success_count = action_success_count + 1 "
                f"WHERE composite_id IN ({placeholders})",
                record_ids,
            )
            self._conn.commit()

    def increment_action_fail(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        with self._lock:
            placeholders = ",".join("?" for _ in record_ids)
            self._conn.execute(
                f"UPDATE memory_records SET action_fail_count = action_fail_count + 1 "
                f"WHERE record_id IN ({placeholders})",
                record_ids,
            )
            self._conn.execute(
                f"UPDATE composite_records SET action_fail_count = action_fail_count + 1 "
                f"WHERE composite_id IN ({placeholders})",
                record_ids,
            )
            self._conn.commit()

    # ──────────────────────────────────────
    # 调试导出
    # ──────────────────────────────────────

    def export_all(self) -> dict[str, Any]:
        """导出全部数据用于调试/前端。"""
        records = [self._row_to_dict(r) for r in self._conn.execute("SELECT * FROM memory_records").fetchall()]
        synths = [self._row_to_dict(r) for r in self._conn.execute("SELECT * FROM composite_records").fetchall()]
        return {
            "records": records,
            "composites": synths,
            "total_records": len(records),
            "total_composites": len(synths),
        }

    def count_records(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM memory_records WHERE expired = 0"
        ).fetchone()[0]

    def count_composites(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM composite_records"
        ).fetchone()[0]

    # ──────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────

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
            "entities", "temporal", "task_tags", "tool_tags",
            "constraint_tags", "failure_tags", "affordance_tags",
            "evidence_turn_range", "source_record_ids", "child_composite_ids",
            "retrieval_plan", "retrieved_record_ids", "kept_record_ids",
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
            task_tags=_json_loads(d["task_tags"]),
            tool_tags=_json_loads(d["tool_tags"]),
            constraint_tags=_json_loads(d["constraint_tags"]),
            failure_tags=_json_loads(d["failure_tags"]),
            affordance_tags=_json_loads(d["affordance_tags"]),
            confidence=float(d["confidence"] or 0.0),
            evidence_turn_range=_json_loads(d["evidence_turn_range"]),
            source_session=d["source_session"],
            source_role=d.get("source_role", ""),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            retrieval_count=int(d["retrieval_count"] or 0),
            retrieval_hit_count=int(d["retrieval_hit_count"] or 0),
            action_success_count=int(d["action_success_count"] or 0),
            action_fail_count=int(d["action_fail_count"] or 0),
            last_retrieved_at=d["last_retrieved_at"],
            expired=bool(d["expired"]),
            expired_at=d["expired_at"],
            expired_reason=d["expired_reason"],
        )

    @staticmethod
    def _row_to_synth(row: sqlite3.Row) -> CompositeRecord:
        return CompositeRecord(
            composite_id=row["composite_id"],
            memory_type=row["memory_type"],
            semantic_text=row["semantic_text"],
            normalized_text=row["normalized_text"],
            source_record_ids=_json_loads(row["source_record_ids"]),
            child_composite_ids=_json_loads(row["child_composite_ids"]),
            synthesis_reason=row["synthesis_reason"],
            entities=_json_loads(row["entities"]),
            temporal=_json_loads_dict(row["temporal"]),
            task_tags=_json_loads(row["task_tags"]),
            tool_tags=_json_loads(row["tool_tags"]),
            constraint_tags=_json_loads(row["constraint_tags"]),
            failure_tags=_json_loads(row["failure_tags"]),
            affordance_tags=_json_loads(row["affordance_tags"]),
            confidence=float(row["confidence"] or 0.0),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            retrieval_count=int(row["retrieval_count"] or 0),
            retrieval_hit_count=int(row["retrieval_hit_count"] or 0),
            action_success_count=int(row["action_success_count"] or 0),
            action_fail_count=int(row["action_fail_count"] or 0),
            last_retrieved_at=row["last_retrieved_at"],
        )
