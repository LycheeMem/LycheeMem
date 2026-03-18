"""
会话日志存储。

开发阶段：内存字典
生产阶段：Redis
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclass
class SessionLog:
    """单个会话的完整对话日志。"""

    session_id: str
    turns: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    # summaries 结构：[{"boundary_index": int, "content": str}]
    topic: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


class InMemorySessionStore:
    """内存版会话存储，开发用。"""

    def __init__(self):
        self._store: dict[str, SessionLog] = {}

    def get_or_create(self, session_id: str) -> SessionLog:
        if session_id not in self._store:
            self._store[session_id] = SessionLog(session_id=session_id)
        return self._store[session_id]

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        log = self.get_or_create(session_id)
        log.turns.append({"role": role, "content": content, "created_at": _now_iso()})
        log.updated_at = _now_iso()

    def get_turns(self, session_id: str) -> list[dict[str, Any]]:
        return self.get_or_create(session_id).turns

    def add_summary(self, session_id: str, boundary_index: int, summary_text: str) -> None:
        log = self.get_or_create(session_id)
        log.summaries.append({"boundary_index": boundary_index, "content": summary_text})

    def mark_turns_deleted(self, session_id: str, boundary_index: int) -> None:
        """将 boundary_index 之前的 turns 软删除标记（保留数据，后端忽略，前端可渲染）。"""
        log = self.get_or_create(session_id)
        for i in range(min(boundary_index, len(log.turns))):
            log.turns[i]["deleted"] = True

    def delete_session(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def update_session_meta(
        self, session_id: str, topic: str | None = None, tags: list[str] | None = None
    ) -> None:
        """更新会话元数据。"""
        log = self.get_or_create(session_id)
        if topic is not None:
            log.topic = topic
        if tags is not None:
            log.tags = tags

    def list_sessions(self, offset: int = 0, limit: int = 50) -> list[dict]:
        """返回所有会话的摘要列表，按最新活动倒序，支持分页。"""
        result = []
        for session_id, log in self._store.items():
            active_turns = [t for t in log.turns if not t.get("deleted", False)]
            last_user = next(
                (t["content"] for t in reversed(active_turns) if t["role"] == "user"), ""
            )
            result.append(
                {
                    "session_id": session_id,
                    "turn_count": len(active_turns),
                    "last_message": last_user[:120],
                    "topic": log.topic,
                    "tags": log.tags,
                    "created_at": log.created_at,
                    "updated_at": log.updated_at,
                }
            )
        result.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
        return result[offset : offset + limit]
