"""程序记忆 / 技能库。

Key = 任务意图 embedding
Value = 技能说明文档（Markdown）

开发阶段：内存字典 + numpy cosine similarity
生产阶段：向量数据库（可替换实现）
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from a_frame.memory.base import BaseMemoryStore


@dataclass
class SkillEntry:
    """技能条目。"""

    id: str
    intent: str  # 任务意图描述
    embedding: list[float]  # 意图的向量表示
    doc_markdown: str  # 技能说明（Markdown）
    metadata: dict[str, Any] = field(default_factory=dict)
    success_count: int = 0  # 成功使用次数
    last_used: str | None = None  # 最后使用时间 (ISO 格式)
    conditions: str = ""  # 适用条件描述


class InMemorySkillStore(BaseMemoryStore):
    """内存版技能库，开发用。"""

    def __init__(self):
        self._skills: dict[str, SkillEntry] = {}

    def add(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            skill = SkillEntry(
                id=item.get("id", str(uuid.uuid4())),
                intent=item["intent"],
                embedding=item["embedding"],
                doc_markdown=item["doc_markdown"],
                metadata=item.get("metadata", {}),
                success_count=item.get("success_count", 0),
                last_used=item.get("last_used"),
                conditions=item.get("conditions", ""),
            )
            self._skills[skill.id] = skill

    def record_usage(self, skill_id: str) -> None:
        """记录技能被成功使用一次。"""
        import datetime

        if skill_id in self._skills:
            self._skills[skill_id].success_count += 1
            self._skills[skill_id].last_used = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()

    def search(
        self, query: str, top_k: int = 5, query_embedding: list[float] | None = None
    ) -> list[dict[str, Any]]:
        """向量相似度检索。需要传入 query_embedding。"""
        if query_embedding is None or not self._skills:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        scored = []
        for skill in self._skills.values():
            s_vec = np.array(skill.embedding, dtype=np.float32)
            # cosine similarity
            cos_sim = float(
                np.dot(q_vec, s_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(s_vec) + 1e-9)
            )
            scored.append((cos_sim, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": s.id,
                "intent": s.intent,
                "doc_markdown": s.doc_markdown,
                "score": score,
                "metadata": s.metadata,
                "success_count": s.success_count,
                "conditions": s.conditions,
            }
            for score, s in scored[:top_k]
        ]

    def delete(self, ids: list[str]) -> None:
        for skill_id in ids:
            self._skills.pop(skill_id, None)

    def get_all(self) -> list[dict[str, Any]]:
        return [
            {
                "id": s.id,
                "intent": s.intent,
                "doc_markdown": s.doc_markdown,
                "metadata": s.metadata,
                "success_count": s.success_count,
                "last_used": s.last_used,
                "conditions": s.conditions,
            }
            for s in self._skills.values()
        ]
