"""
会话日志存储。

开发阶段：内存字典
生产阶段：Redis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionLog:
    """单个会话的完整对话日志。"""
    session_id: str
    turns: list[dict[str, str]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    # summaries 结构：[{"boundary_index": int, "content": str}]


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
        log.turns.append({"role": role, "content": content})

    def get_turns(self, session_id: str) -> list[dict[str, str]]:
        return self.get_or_create(session_id).turns

    def add_summary(self, session_id: str, boundary_index: int, summary_text: str) -> None:
        log = self.get_or_create(session_id)
        log.summaries.append({"boundary_index": boundary_index, "content": summary_text})

    def delete_session(self, session_id: str) -> None:
        self._store.pop(session_id, None)
