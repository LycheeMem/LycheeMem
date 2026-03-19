"""
用户存储。

SQLite 持久化存储用户信息，使用 bcrypt 对密码做安全哈希。
"""

from __future__ import annotations

import datetime
import sqlite3
import threading
import uuid
from dataclasses import dataclass

import bcrypt


@dataclass
class User:
    """用户实体。"""

    user_id: str
    username: str
    password_hash: str
    display_name: str
    created_at: str


class UserStore:
    """基于 SQLite 的用户存储（线程安全）。"""

    def __init__(self, db_path: str = "lychee_memos_users.db"):
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username);
        """)
        conn.commit()

    @staticmethod
    def _hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def _verify_password(password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

    def create_user(self, username: str, password: str, display_name: str = "") -> User:
        """创建新用户。用户名已存在时抛出 ValueError。"""
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT user_id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            raise ValueError(f"用户名 '{username}' 已存在")

        user_id = uuid.uuid4().hex
        password_hash = self._hash_password(password)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO users (user_id, username, password_hash, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, password_hash, display_name or username, now),
        )
        conn.commit()
        return User(
            user_id=user_id,
            username=username,
            password_hash=password_hash,
            display_name=display_name or username,
            created_at=now,
        )

    def authenticate(self, username: str, password: str) -> User | None:
        """验证用户凭据，成功返回 User，失败返回 None。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT user_id, username, password_hash, display_name, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return None
        if not self._verify_password(password, row["password_hash"]):
            return None
        return User(
            user_id=row["user_id"],
            username=row["username"],
            password_hash=row["password_hash"],
            display_name=row["display_name"],
            created_at=row["created_at"],
        )

    def get_user_by_id(self, user_id: str) -> User | None:
        """按 user_id 查找用户。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT user_id, username, password_hash, display_name, created_at FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return User(
            user_id=row["user_id"],
            username=row["username"],
            password_hash=row["password_hash"],
            display_name=row["display_name"],
            created_at=row["created_at"],
        )
