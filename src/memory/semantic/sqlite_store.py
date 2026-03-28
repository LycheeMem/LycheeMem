"""SQLite + FTS5 结构化存储。

职责：
- memory_units 表：MemoryUnit 主存储
- synthesized_units 表：SynthesizedUnit 合成条目
- memory_units_fts：FTS5 全文索引（覆盖 normalized_text + entities + tags）
- synthesized_units_fts：FTS5 全文索引
- usage_logs 表：检索使用记录

所有写操作线程安全（sqlite3 + threading.Lock）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any

from src.memory.semantic.models import MemoryUnit, SynthesizedUnit, UsageLog


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
                CREATE TABLE IF NOT EXISTS memory_units (
                    unit_id          TEXT PRIMARY KEY,
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
                    user_id          TEXT NOT NULL DEFAULT '',
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

                CREATE TABLE IF NOT EXISTS synthesized_units (
                    synth_id         TEXT PRIMARY KEY,
                    memory_type      TEXT NOT NULL,
                    semantic_text    TEXT NOT NULL,
                    normalized_text  TEXT NOT NULL,
                    source_unit_ids  TEXT NOT NULL DEFAULT '[]',
                    synthesis_reason TEXT NOT NULL DEFAULT '',
                    entities         TEXT NOT NULL DEFAULT '[]',
                    temporal         TEXT NOT NULL DEFAULT '{}',
                    task_tags        TEXT NOT NULL DEFAULT '[]',
                    tool_tags        TEXT NOT NULL DEFAULT '[]',
                    constraint_tags  TEXT NOT NULL DEFAULT '[]',
                    failure_tags     TEXT NOT NULL DEFAULT '[]',
                    affordance_tags  TEXT NOT NULL DEFAULT '[]',
                    confidence       REAL NOT NULL DEFAULT 1.0,
                    user_id          TEXT NOT NULL DEFAULT '',
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
                    user_id          TEXT NOT NULL DEFAULT '',
                    timestamp        TEXT NOT NULL,
                    query            TEXT NOT NULL,
                    retrieval_plan   TEXT NOT NULL DEFAULT '{}',
                    retrieved_unit_ids TEXT NOT NULL DEFAULT '[]',
                    kept_unit_ids    TEXT NOT NULL DEFAULT '[]',
                    final_response_excerpt TEXT NOT NULL DEFAULT '',
                    user_feedback    TEXT NOT NULL DEFAULT '',
                    action_outcome   TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_mu_user_id ON memory_units(user_id);
                CREATE INDEX IF NOT EXISTS idx_mu_memory_type ON memory_units(memory_type);
                CREATE INDEX IF NOT EXISTS idx_mu_expired ON memory_units(expired);
                CREATE INDEX IF NOT EXISTS idx_mu_created_at ON memory_units(created_at);
                CREATE INDEX IF NOT EXISTS idx_su_user_id ON synthesized_units(user_id);
                CREATE INDEX IF NOT EXISTS idx_ul_session ON usage_logs(session_id);
            """)

            # FTS5 虚拟表（分开创建避免 IF NOT EXISTS 不兼容问题）
            try:
                c.execute("""
                    CREATE VIRTUAL TABLE memory_units_fts USING fts5(
                        unit_id UNINDEXED,
                        normalized_text,
                        entities,
                        task_tags,
                        tool_tags,
                        constraint_tags,
                        failure_tags,
                        affordance_tags,
                        content=memory_units,
                        content_rowid=rowid,
                        tokenize='unicode61'
                    )
                """)
            except sqlite3.OperationalError:
                pass  # 已存在

            try:
                c.execute("""
                    CREATE VIRTUAL TABLE synthesized_units_fts USING fts5(
                        synth_id UNINDEXED,
                        normalized_text,
                        entities,
                        task_tags,
                        tool_tags,
                        constraint_tags,
                        content=synthesized_units,
                        content_rowid=rowid,
                        tokenize='unicode61'
                    )
                """)
            except sqlite3.OperationalError:
                pass

            c.commit()

    # ──────────────────────────────────────
    # Memory Unit CRUD
    # ──────────────────────────────────────

    def upsert_unit(self, unit: MemoryUnit) -> None:
        """写入或更新一个 MemoryUnit（幂等）。"""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_units (
                    unit_id, memory_type, semantic_text, normalized_text,
                    entities, temporal, task_tags, tool_tags,
                    constraint_tags, failure_tags, affordance_tags,
                    confidence, evidence_turn_range, source_session, user_id,
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
                ON CONFLICT(unit_id) DO UPDATE SET
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
                    updated_at = excluded.updated_at,
                    expired = excluded.expired,
                    expired_at = excluded.expired_at,
                    expired_reason = excluded.expired_reason
                """,
                (
                    unit.unit_id, unit.memory_type,
                    unit.semantic_text, unit.normalized_text,
                    _json_dumps(unit.entities), _json_dumps(unit.temporal),
                    _json_dumps(unit.task_tags), _json_dumps(unit.tool_tags),
                    _json_dumps(unit.constraint_tags), _json_dumps(unit.failure_tags),
                    _json_dumps(unit.affordance_tags),
                    unit.confidence, _json_dumps(unit.evidence_turn_range),
                    unit.source_session, unit.user_id,
                    unit.created_at, unit.updated_at,
                    unit.retrieval_count, unit.retrieval_hit_count,
                    unit.action_success_count, unit.action_fail_count,
                    unit.last_retrieved_at,
                    int(unit.expired), unit.expired_at, unit.expired_reason,
                ),
            )
            # 同步 FTS5
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_units_fts(rowid, unit_id, "
                "normalized_text, entities, task_tags, tool_tags, "
                "constraint_tags, failure_tags, affordance_tags) "
                "SELECT rowid, unit_id, normalized_text, entities, task_tags, "
                "tool_tags, constraint_tags, failure_tags, affordance_tags "
                "FROM memory_units WHERE unit_id = ?",
                (unit.unit_id,),
            )
            self._conn.commit()

    def upsert_units(self, units: list[MemoryUnit]) -> None:
        """批量写入 MemoryUnit。"""
        for u in units:
            self.upsert_unit(u)

    def get_unit(self, unit_id: str) -> MemoryUnit | None:
        """按 ID 获取单个 MemoryUnit。"""
        row = self._conn.execute(
            "SELECT * FROM memory_units WHERE unit_id = ?", (unit_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_unit(row)

    def get_units_by_ids(self, unit_ids: list[str]) -> list[MemoryUnit]:
        """按 ID 列表批量获取。"""
        if not unit_ids:
            return []
        placeholders = ",".join("?" for _ in unit_ids)
        rows = self._conn.execute(
            f"SELECT * FROM memory_units WHERE unit_id IN ({placeholders})",
            unit_ids,
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def delete_unit(self, unit_id: str, *, user_id: str = "") -> None:
        """硬删除一个 MemoryUnit。"""
        with self._lock:
            if user_id:
                self._conn.execute(
                    "DELETE FROM memory_units WHERE unit_id = ? AND user_id = ?",
                    (unit_id, user_id),
                )
            else:
                self._conn.execute(
                    "DELETE FROM memory_units WHERE unit_id = ?", (unit_id,)
                )
            self._conn.commit()

    def expire_unit(
        self, unit_id: str, *, expired_at: str = "", expired_reason: str = "",
    ) -> None:
        """软删除（标记过期）。"""
        with self._lock:
            self._conn.execute(
                "UPDATE memory_units SET expired = 1, expired_at = ?, expired_reason = ? "
                "WHERE unit_id = ?",
                (expired_at, expired_reason, unit_id),
            )
            self._conn.commit()

    def delete_all_for_user(self, user_id: str) -> dict[str, int]:
        """清空指定用户的所有数据。"""
        with self._lock:
            c1 = self._conn.execute(
                "SELECT COUNT(*) FROM memory_units WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            c2 = self._conn.execute(
                "SELECT COUNT(*) FROM synthesized_units WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            self._conn.execute(
                "DELETE FROM memory_units WHERE user_id = ?", (user_id,)
            )
            self._conn.execute(
                "DELETE FROM synthesized_units WHERE user_id = ?", (user_id,)
            )
            self._conn.execute(
                "DELETE FROM usage_logs WHERE user_id = ?", (user_id,)
            )
            self._conn.commit()
        return {"units_deleted": c1, "synth_deleted": c2}

    # ──────────────────────────────────────
    # FTS5 全文检索
    # ──────────────────────────────────────

    def fulltext_search(
        self,
        query: str,
        *,
        limit: int = 30,
        user_id: str = "",
        memory_types: list[str] | None = None,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        """BM25 全文召回 MemoryUnit。"""
        fts_query = self._escape_fts_query(query)
        if not fts_query:
            return []

        sql = (
            "SELECT m.*, bm25(memory_units_fts) AS fts_score "
            "FROM memory_units_fts f "
            "JOIN memory_units m ON f.rowid = m.rowid "
            "WHERE memory_units_fts MATCH ? "
        )
        params: list[Any] = [fts_query]

        if user_id:
            sql += "AND m.user_id = ? "
            params.append(user_id)
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
        user_id: str = "",
    ) -> list[dict[str, Any]]:
        """BM25 全文召回 SynthesizedUnit。"""
        fts_query = self._escape_fts_query(query)
        if not fts_query:
            return []

        sql = (
            "SELECT s.*, bm25(synthesized_units_fts) AS fts_score "
            "FROM synthesized_units_fts f "
            "JOIN synthesized_units s ON f.rowid = s.rowid "
            "WHERE synthesized_units_fts MATCH ? "
        )
        params: list[Any] = [fts_query]

        if user_id:
            sql += "AND s.user_id = ? "
            params.append(user_id)

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
        user_id: str = "",
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        """按创建时间范围查询 MemoryUnit。"""
        sql = "SELECT * FROM memory_units WHERE 1=1 "
        params: list[Any] = []

        if user_id:
            sql += "AND user_id = ? "
            params.append(user_id)
        if not include_expired:
            sql += "AND expired = 0 "
        if since:
            sql += "AND created_at >= ? "
            params.append(since)
        if until:
            sql += "AND created_at <= ? "
            params.append(until)

        sql += "ORDER BY created_at DESC LIMIT ?"
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
        limit: int = 30,
        user_id: str = "",
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        """按 tag 字段做 LIKE 匹配查询。"""
        sql = "SELECT * FROM memory_units WHERE 1=1 "
        params: list[Any] = []

        if user_id:
            sql += "AND user_id = ? "
            params.append(user_id)
        if not include_expired:
            sql += "AND expired = 0 "

        for tags, col in [
            (tool_tags, "tool_tags"),
            (constraint_tags, "constraint_tags"),
            (task_tags, "task_tags"),
            (failure_tags, "failure_tags"),
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

    # ──────────────────────────────────────
    # Synthesized Unit CRUD
    # ──────────────────────────────────────

    def upsert_synthesized(self, unit: SynthesizedUnit) -> None:
        """写入或更新 SynthesizedUnit。"""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO synthesized_units (
                    synth_id, memory_type, semantic_text, normalized_text,
                    source_unit_ids, synthesis_reason,
                    entities, temporal, task_tags, tool_tags,
                    constraint_tags, failure_tags, affordance_tags,
                    confidence, user_id, created_at, updated_at,
                    retrieval_count, retrieval_hit_count,
                    action_success_count, action_fail_count, last_retrieved_at
                ) VALUES (?, ?, ?, ?,  ?, ?,  ?, ?, ?, ?,  ?, ?, ?,  ?, ?, ?, ?,  ?, ?,  ?, ?, ?)
                ON CONFLICT(synth_id) DO UPDATE SET
                    memory_type = excluded.memory_type,
                    semantic_text = excluded.semantic_text,
                    normalized_text = excluded.normalized_text,
                    source_unit_ids = excluded.source_unit_ids,
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
                    unit.synth_id, unit.memory_type,
                    unit.semantic_text, unit.normalized_text,
                    _json_dumps(unit.source_unit_ids), unit.synthesis_reason,
                    _json_dumps(unit.entities), _json_dumps(unit.temporal),
                    _json_dumps(unit.task_tags), _json_dumps(unit.tool_tags),
                    _json_dumps(unit.constraint_tags), _json_dumps(unit.failure_tags),
                    _json_dumps(unit.affordance_tags),
                    unit.confidence, unit.user_id, unit.created_at, unit.updated_at,
                    unit.retrieval_count, unit.retrieval_hit_count,
                    unit.action_success_count, unit.action_fail_count,
                    unit.last_retrieved_at,
                ),
            )
            # 同步 FTS5
            self._conn.execute(
                "INSERT OR REPLACE INTO synthesized_units_fts(rowid, synth_id, "
                "normalized_text, entities, task_tags, tool_tags, constraint_tags) "
                "SELECT rowid, synth_id, normalized_text, entities, task_tags, "
                "tool_tags, constraint_tags "
                "FROM synthesized_units WHERE synth_id = ?",
                (unit.synth_id,),
            )
            self._conn.commit()

    def get_synthesized_by_source(
        self, source_unit_ids: list[str],
    ) -> list[SynthesizedUnit]:
        """查找包含指定 source_unit_id 的合成条目。"""
        if not source_unit_ids:
            return []
        rows = self._conn.execute(
            "SELECT * FROM synthesized_units"
        ).fetchall()
        results = []
        for r in rows:
            stored_ids = _json_loads(r["source_unit_ids"])
            if any(uid in stored_ids for uid in source_unit_ids):
                results.append(self._row_to_synth(r))
        return results

    def get_synthesized(self, synth_id: str) -> SynthesizedUnit | None:
        """按 synth_id 获取单个 SynthesizedUnit。"""
        row = self._conn.execute(
            "SELECT * FROM synthesized_units WHERE synth_id = ?", (synth_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_synth(row)

    # ──────────────────────────────────────
    # 重复检测
    # ──────────────────────────────────────

    def find_similar_by_normalized_text(
        self,
        normalized_text: str,
        *,
        user_id: str = "",
        limit: int = 5,
    ) -> list[MemoryUnit]:
        """FTS5 模糊匹配，用于写入时去重。"""
        fts_query = self._escape_fts_query(normalized_text)
        if not fts_query:
            return []

        sql = (
            "SELECT m.* FROM memory_units_fts f "
            "JOIN memory_units m ON f.rowid = m.rowid "
            "WHERE memory_units_fts MATCH ? AND m.expired = 0 "
        )
        params: list[Any] = [fts_query]
        if user_id:
            sql += "AND m.user_id = ? "
            params.append(user_id)
        sql += "LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_unit(r) for r in rows]

    # ──────────────────────────────────────
    # Usage Logs
    # ──────────────────────────────────────

    def insert_usage_log(self, log: UsageLog) -> None:
        """插入一条使用日志。"""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO usage_logs "
                "(log_id, session_id, user_id, timestamp, query, "
                "retrieval_plan, retrieved_unit_ids, kept_unit_ids, "
                "final_response_excerpt, user_feedback, action_outcome) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    log.log_id, log.session_id, log.user_id, log.timestamp,
                    log.query, _json_dumps(log.retrieval_plan),
                    _json_dumps(log.retrieved_unit_ids),
                    _json_dumps(log.kept_unit_ids),
                    log.final_response_excerpt,
                    log.user_feedback, log.action_outcome,
                ),
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
                user_id=r["user_id"],
                timestamp=r["timestamp"],
                query=r["query"],
                retrieval_plan=_json_loads_dict(r["retrieval_plan"]),
                retrieved_unit_ids=_json_loads(r["retrieved_unit_ids"]),
                kept_unit_ids=_json_loads(r["kept_unit_ids"]),
                final_response_excerpt=r["final_response_excerpt"],
                user_feedback=r["user_feedback"],
                action_outcome=r["action_outcome"],
            )
            for r in rows
        ]

    # ──────────────────────────────────────
    # 使用统计批量更新
    # ──────────────────────────────────────

    def increment_retrieval_count(self, unit_ids: list[str]) -> None:
        if not unit_ids:
            return
        with self._lock:
            placeholders = ",".join("?" for _ in unit_ids)
            self._conn.execute(
                f"UPDATE memory_units SET retrieval_count = retrieval_count + 1 "
                f"WHERE unit_id IN ({placeholders})",
                unit_ids,
            )
            self._conn.execute(
                f"UPDATE synthesized_units SET retrieval_count = retrieval_count + 1 "
                f"WHERE synth_id IN ({placeholders})",
                unit_ids,
            )
            self._conn.commit()

    def increment_hit_count(self, unit_ids: list[str]) -> None:
        if not unit_ids:
            return
        with self._lock:
            placeholders = ",".join("?" for _ in unit_ids)
            self._conn.execute(
                f"UPDATE memory_units SET retrieval_hit_count = retrieval_hit_count + 1 "
                f"WHERE unit_id IN ({placeholders})",
                unit_ids,
            )
            self._conn.execute(
                f"UPDATE synthesized_units SET retrieval_hit_count = retrieval_hit_count + 1 "
                f"WHERE synth_id IN ({placeholders})",
                unit_ids,
            )
            self._conn.commit()

    def increment_action_success(self, unit_ids: list[str]) -> None:
        if not unit_ids:
            return
        with self._lock:
            placeholders = ",".join("?" for _ in unit_ids)
            self._conn.execute(
                f"UPDATE memory_units SET action_success_count = action_success_count + 1 "
                f"WHERE unit_id IN ({placeholders})",
                unit_ids,
            )
            self._conn.commit()

    def increment_action_fail(self, unit_ids: list[str]) -> None:
        if not unit_ids:
            return
        with self._lock:
            placeholders = ",".join("?" for _ in unit_ids)
            self._conn.execute(
                f"UPDATE memory_units SET action_fail_count = action_fail_count + 1 "
                f"WHERE unit_id IN ({placeholders})",
                unit_ids,
            )
            self._conn.commit()

    # ──────────────────────────────────────
    # 调试导出
    # ──────────────────────────────────────

    def export_all(self, *, user_id: str = "") -> dict[str, Any]:
        """导出全部数据用于调试/前端。"""
        sql_mu = "SELECT * FROM memory_units"
        sql_su = "SELECT * FROM synthesized_units"
        params: list[Any] = []
        if user_id:
            sql_mu += " WHERE user_id = ?"
            sql_su += " WHERE user_id = ?"
            params = [user_id]

        units = [self._row_to_dict(r) for r in self._conn.execute(sql_mu, params).fetchall()]
        synths = [self._row_to_dict(r) for r in self._conn.execute(sql_su, params).fetchall()]

        return {
            "units": units,
            "synthesized": synths,
            "total_units": len(units),
            "total_synthesized": len(synths),
        }

    def count_units(self, *, user_id: str = "") -> int:
        if user_id:
            return self._conn.execute(
                "SELECT COUNT(*) FROM memory_units WHERE user_id = ? AND expired = 0",
                (user_id,),
            ).fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM memory_units WHERE expired = 0"
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
            "evidence_turn_range", "source_unit_ids",
            "retrieval_plan", "retrieved_unit_ids", "kept_unit_ids",
        ):
            if key in d and isinstance(d[key], str):
                d[key] = _json_loads(d[key]) if key != "temporal" else _json_loads_dict(d[key])
        if "expired" in d:
            d["expired"] = bool(d["expired"])
        return d

    @staticmethod
    def _row_to_unit(row: sqlite3.Row) -> MemoryUnit:
        return MemoryUnit(
            unit_id=row["unit_id"],
            memory_type=row["memory_type"],
            semantic_text=row["semantic_text"],
            normalized_text=row["normalized_text"],
            entities=_json_loads(row["entities"]),
            temporal=_json_loads_dict(row["temporal"]),
            task_tags=_json_loads(row["task_tags"]),
            tool_tags=_json_loads(row["tool_tags"]),
            constraint_tags=_json_loads(row["constraint_tags"]),
            failure_tags=_json_loads(row["failure_tags"]),
            affordance_tags=_json_loads(row["affordance_tags"]),
            confidence=float(row["confidence"]),
            evidence_turn_range=_json_loads(row["evidence_turn_range"]),
            source_session=row["source_session"],
            user_id=row["user_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            retrieval_count=int(row["retrieval_count"]),
            retrieval_hit_count=int(row["retrieval_hit_count"]),
            action_success_count=int(row["action_success_count"]),
            action_fail_count=int(row["action_fail_count"]),
            last_retrieved_at=row["last_retrieved_at"],
            expired=bool(row["expired"]),
            expired_at=row["expired_at"],
            expired_reason=row["expired_reason"],
        )

    @staticmethod
    def _row_to_synth(row: sqlite3.Row) -> SynthesizedUnit:
        return SynthesizedUnit(
            synth_id=row["synth_id"],
            memory_type=row["memory_type"],
            semantic_text=row["semantic_text"],
            normalized_text=row["normalized_text"],
            source_unit_ids=_json_loads(row["source_unit_ids"]),
            synthesis_reason=row["synthesis_reason"],
            entities=_json_loads(row["entities"]),
            temporal=_json_loads_dict(row["temporal"]),
            task_tags=_json_loads(row["task_tags"]),
            tool_tags=_json_loads(row["tool_tags"]),
            constraint_tags=_json_loads(row["constraint_tags"]),
            failure_tags=_json_loads(row["failure_tags"]),
            affordance_tags=_json_loads(row["affordance_tags"]),
            confidence=float(row["confidence"]),
            user_id=row["user_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            retrieval_count=int(row["retrieval_count"]),
            retrieval_hit_count=int(row["retrieval_hit_count"]),
            action_success_count=int(row["action_success_count"]),
            action_fail_count=int(row["action_fail_count"]),
            last_retrieved_at=row["last_retrieved_at"],
        )
