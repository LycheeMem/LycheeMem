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
    # summaries 结构：[{"boundary_index": int, "content": str, "token_count": int}]
    topic: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    # 固化水位线：记录上次成功固化时 turns 列表的长度（包含已软删除的 turn）。
    # 下次固化只需处理 turns[last_consolidated_turn_index:] 的新增部分，
    # 从而彻底消除跨轮重复固化问题。
    last_consolidated_turn_index: int = 0


class InMemorySessionStore:
    """内存版会话存储，开发用。"""

    def __init__(self):
        self._store: dict[str, SessionLog] = {}

    def get_or_create(self, session_id: str) -> SessionLog:
        if session_id not in self._store:
            self._store[session_id] = SessionLog(session_id=session_id)
        return self._store[session_id]

    def append_turn(self, session_id: str, role: str, content: str, token_count: int = 0) -> None:
        log = self.get_or_create(session_id)
        log.turns.append({"role": role, "content": content, "token_count": token_count, "created_at": _now_iso()})
        log.updated_at = _now_iso()

    def get_turns(self, session_id: str) -> list[dict[str, Any]]:
        return self.get_or_create(session_id).turns

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
        log = self.get_or_create(session_id)
        log.summaries.append({"boundary_index": boundary_index, "content": summary_text, "token_count": token_count})

    def mark_turns_deleted(self, session_id: str, boundary_index: int) -> None:
        """将 boundary_index 之前的 turns 软删除标记（保留数据，后端忽略，前端可渲染）。"""
        log = self.get_or_create(session_id)
        for i in range(min(boundary_index, len(log.turns))):
            log.turns[i]["deleted"] = True

    def set_last_consolidated_turn_index(self, session_id: str, raw_turn_count: int) -> None:
        """更新固化水位线（raw_turn_count = 本次固化时 turns 列表的总长度）。"""
        log = self.get_or_create(session_id)
        if raw_turn_count > log.last_consolidated_turn_index:
            log.last_consolidated_turn_index = raw_turn_count

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
        """返回会话的摘要列表，按最新活动倒序，支持分页。"""
        result = []
        for session_id, log in self._store.items():
            active_turns = [t for t in log.turns if not t.get("deleted", False)]
            first_user = next(
                (t["content"] for t in active_turns if t["role"] == "user"), ""
            )
            last_user = next(
                (t["content"] for t in reversed(active_turns) if t["role"] == "user"), ""
            )
            result.append(
                {
                    "session_id": session_id,
                    "turn_count": len(active_turns),
                    "last_message": last_user[:120],
                    "title": log.topic if log.topic else first_user[:40],
                    "topic": log.topic,
                    "tags": log.tags,
                    "created_at": log.created_at,
                    "updated_at": log.updated_at,
                }
            )
        result.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
        return result[offset : offset + limit]
