"""
SQLite 会话存储。

持久化版本的会话日志存储，兼容 InMemorySessionStore 接口。
后续可迁移至 MySQL（只需改连接字符串和极少 SQL 语法）。
"""

from __future__ import annotations

import json
import sqlite3
import threading

from a_frame.memory.working.session_store import SessionLog


class SQLiteSessionStore:
    """基于 SQLite 的持久化会话存储。

    与 InMemorySessionStore 完全相同的公有接口。
    使用 WAL 模式提高并发性能。
    """

    def __init__(self, db_path: str = "a_frame_sessions.db"):
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                boundary_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);

            CREATE TABLE IF NOT EXISTS session_meta (
                session_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()

    def get_or_create(self, session_id: str) -> SessionLog:
        """获取完整会话日志，不存在则返回空日志。"""
        conn = self._get_conn()
        turns = [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in conn.execute(
                "SELECT role, content, created_at FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
        ]
        summaries = [
            {"boundary_index": row["boundary_index"], "content": row["content"]}
            for row in conn.execute(
                "SELECT boundary_index, content FROM summaries WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
        ]
        log = SessionLog(session_id=session_id)
        log.turns = turns
        log.summaries = summaries
        return log

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO turns (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
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
            }
            for row in conn.execute(
                "SELECT role, content, created_at FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
        ]

    def add_summary(self, session_id: str, boundary_index: int, summary_text: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO summaries (session_id, boundary_index, content) VALUES (?, ?, ?)",
            (session_id, boundary_index, summary_text),
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
        """返回所有会话的摘要列表，按最新活动倒序，支持分页。"""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT
                t.session_id,
                COUNT(*) AS turn_count,
                (SELECT t2.content FROM turns t2
                 WHERE t2.session_id = t.session_id AND t2.role = 'user'
                 ORDER BY t2.id DESC LIMIT 1) AS last_message,
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
