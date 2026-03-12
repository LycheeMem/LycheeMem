"""
程序记忆 / 技能库。

Key = 任务意图 embedding
Value = 成功的工具调用序列（JSON）

开发阶段：内存字典 + numpy cosine similarity
生产阶段：Qdrant
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
    intent: str                    # 任务意图描述
    embedding: list[float]         # 意图的向量表示
    tool_chain: list[dict[str, Any]]  # 工具调用序列
    metadata: dict[str, Any] = field(default_factory=dict)


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
                tool_chain=item["tool_chain"],
                metadata=item.get("metadata", {}),
            )
            self._skills[skill.id] = skill

    def search(self, query: str, top_k: int = 5, query_embedding: list[float] | None = None) -> list[dict[str, Any]]:
        """向量相似度检索。需要传入 query_embedding。"""
        if query_embedding is None or not self._skills:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        scored = []
        for skill in self._skills.values():
            s_vec = np.array(skill.embedding, dtype=np.float32)
            # cosine similarity
            cos_sim = float(np.dot(q_vec, s_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(s_vec) + 1e-9))
            scored.append((cos_sim, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": s.id,
                "intent": s.intent,
                "tool_chain": s.tool_chain,
                "score": score,
                "metadata": s.metadata,
            }
            for score, s in scored[:top_k]
        ]

    def delete(self, ids: list[str]) -> None:
        for skill_id in ids:
            self._skills.pop(skill_id, None)

    def get_all(self) -> list[dict[str, Any]]:
        return [
            {"id": s.id, "intent": s.intent, "tool_chain": s.tool_chain, "metadata": s.metadata}
            for s in self._skills.values()
        ]
