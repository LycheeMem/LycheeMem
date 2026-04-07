"""
SQLite 会话存储。

持久化版本的会话日志存储，兼容 InMemorySessionStore 接口。
后续可迁移至 MySQL（只需改连接字符串和极少 SQL 语法）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from src.memory.working.session_store import SessionLog


class SQLiteSessionStore:
    """基于 SQLite 的持久化会话存储。

    与 InMemorySessionStore 完全相同的公有接口。
    使用 WAL 模式提高并发性能。
    """

    def __init__(self, db_path: str = "sessions.db"):
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """每个线程独立连接（SQLite 线程安全要求）。"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                boundary_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);

            CREATE TABLE IF NOT EXISTS session_meta (
                session_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                last_consolidated_turn_index INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # 当字段已存在时 ALTER TABLE 会抛异常，捕获就进行迁移
        for alter_sql in [
            "ALTER TABLE turns ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE turns ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE summaries ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE session_meta ADD COLUMN last_consolidated_turn_index INTEGER NOT NULL DEFAULT 0",
        ]:
            try:
                conn.execute(alter_sql)
                conn.commit()
            except Exception:
                pass  # 列已存在

    def get_or_create(self, session_id: str) -> SessionLog:
        """获取完整会话日志，不存在则返回空日志。"""
        conn = self._get_conn()
        turns = [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
                "deleted": bool(row["deleted"]),
                "token_count": row["token_count"] if "token_count" in row.keys() else 0,
            }
            for row in conn.execute(
                "SELECT role, content, created_at, deleted, token_count FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
        ]
        summaries = [
            {"boundary_index": row["boundary_index"], "content": row["content"], "token_count": row["token_count"] if "token_count" in row.keys() else 0}
            for row in conn.execute(
                "SELECT boundary_index, content, token_count FROM summaries WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
        ]
        # 从 session_meta 读取固化水位线
        meta_row = conn.execute(
            "SELECT last_consolidated_turn_index FROM session_meta WHERE session_id = ?", (session_id,)
        ).fetchone()
        last_consolidated_idx = meta_row["last_consolidated_turn_index"] if meta_row and "last_consolidated_turn_index" in meta_row.keys() else 0
        log = SessionLog(session_id=session_id)
        log.turns = turns
        log.summaries = summaries
        log.last_consolidated_turn_index = last_consolidated_idx
        return log

    def append_turn(self, session_id: str, role: str, content: str, token_count: int = 0) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO turns (session_id, role, content, token_count) VALUES (?, ?, ?, ?)",
            (session_id, role, content, token_count),
        )
        # upsert session_meta 的 updated_at
        conn.execute(
            "INSERT INTO session_meta (session_id) VALUES (?) ON CONFLICT(session_id) DO UPDATE SET updated_at=CURRENT_TIMESTAMP",
            (session_id,),
        )
        conn.commit()

    def get_turns(self, session_id: str) -> list[dict[str, str]]:
        conn = self._get_conn()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
                "deleted": bool(row["deleted"]),
                "token_count": row["token_count"] if "token_count" in row.keys() else 0,
            }
            for row in conn.execute(
                "SELECT role, content, created_at, deleted, token_count FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
        ]

    def get_turn_window(
        self,
        session_id: str,
        start_index: int,
        end_index: int,
        *,
        window: int = 0,
    ) -> list[dict[str, Any]]:
        """按绝对 turn 索引回溯原始对话窗口。"""
        log = self.get_or_create(session_id)
        if not log.turns:
            return []

        left = max(0, min(int(start_index), int(end_index)) - max(0, int(window or 0)))
        right = min(
            len(log.turns) - 1,
            max(int(start_index), int(end_index)) + max(0, int(window or 0)),
        )
        result: list[dict[str, Any]] = []
        for turn_index in range(left, right + 1):
            turn = dict(log.turns[turn_index])
            turn["turn_index"] = turn_index
            result.append(turn)
        return result

    def add_summary(self, session_id: str, boundary_index: int, summary_text: str, token_count: int = 0) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO summaries (session_id, boundary_index, content, token_count) VALUES (?, ?, ?, ?)",
            (session_id, boundary_index, summary_text, token_count),
        )
        conn.commit()

    def mark_turns_deleted(self, session_id: str, boundary_index: int) -> None:
        """将 boundary_index 之前的 turns 软删除标记（幂等，重复标记无副作用）。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id FROM turns WHERE session_id = ? ORDER BY id LIMIT ?",
            (session_id, boundary_index),
        ).fetchall()
        ids = [row["id"] for row in rows]
        if ids:
            conn.execute(
                f"UPDATE turns SET deleted=1 WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            conn.commit()

    def set_last_consolidated_turn_index(self, session_id: str, raw_turn_count: int) -> None:
        """更新固化水位线（raw_turn_count = 本次固化时 turns 列表的总长度）。仅允许单调递增。"""
        conn = self._get_conn()
        # 先确保 session_meta 行存在（若不存在则插入）
        conn.execute(
            "INSERT INTO session_meta (session_id) VALUES (?) ON CONFLICT(session_id) DO NOTHING",
            (session_id,),
        )
        # 仅当新值大于旧值时才更新（单调递增保证）
        conn.execute(
            "UPDATE session_meta SET last_consolidated_turn_index = ? "
            "WHERE session_id = ? AND last_consolidated_turn_index < ?",
            (raw_turn_count, session_id, raw_turn_count),
        )
        conn.commit()

    def delete_session(self, session_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM session_meta WHERE session_id = ?", (session_id,))
        conn.commit()

    def update_session_meta(
        self, session_id: str, topic: str | None = None, tags: list[str] | None = None
    ) -> None:
        """更新会话元数据。"""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO session_meta (session_id) VALUES (?) ON CONFLICT(session_id) DO NOTHING",
            (session_id,),
        )
        if topic is not None:
            conn.execute("UPDATE session_meta SET topic=? WHERE session_id=?", (topic, session_id))
        if tags is not None:
            conn.execute(
                "UPDATE session_meta SET tags=? WHERE session_id=?", (json.dumps(tags), session_id)
            )
        conn.commit()

    def list_sessions(self, offset: int = 0, limit: int = 50) -> list[dict]:
        """返回会话的摘要列表，按最新活动倒序，支持分页。"""
        conn = self._get_conn()
        rows = conn.execute(
                """
                SELECT
                    t.session_id,
                    SUM(CASE WHEN t.deleted=0 THEN 1 ELSE 0 END) AS turn_count,
                    (SELECT t2.content FROM turns t2
                     WHERE t2.session_id = t.session_id AND t2.role = 'user' AND t2.deleted=0
                     ORDER BY t2.id DESC LIMIT 1) AS last_message,
                    (SELECT t2.content FROM turns t2
                     WHERE t2.session_id = t.session_id AND t2.role = 'user' AND t2.deleted=0
                     ORDER BY t2.id ASC LIMIT 1) AS first_message,
                    MAX(t.created_at) AS updated_at,
                    COALESCE(m.topic, '') AS topic,
                    COALESCE(m.tags, '[]') AS tags,
                    COALESCE(m.created_at, MIN(t.created_at)) AS session_created_at
                FROM turns t
                LEFT JOIN session_meta m ON m.session_id = t.session_id
                GROUP BY t.session_id
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """,
                (limit, offset),
            ).fetchall()
        result = []
        for row in rows:
            try:
                tags = json.loads(row["tags"])
            except (json.JSONDecodeError, TypeError):
                tags = []
            result.append(
                {
                    "session_id": row["session_id"],
                    "turn_count": row["turn_count"],
                    "last_message": (row["last_message"] or "")[:120],
                    "title": row["topic"] if row["topic"] else (row["first_message"] or "")[:40],
                    "topic": row["topic"],
                    "tags": tags,
                    "created_at": row["session_created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return result

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
