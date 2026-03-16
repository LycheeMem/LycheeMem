"""本地文件技能库（轻量级向量存储）。

目标：提供“零依赖/单文件落盘”的技能库实现。
- 存储格式：一个 JSON 文件，包含技能条目列表（含 embedding/doc_markdown/metadata 等）。
- 检索方式：numpy 余弦相似度（适用于中小规模技能库）。

说明：不兼容旧的 tool_chain JSON 数据格式。
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict
from typing import Any

import numpy as np

from a_frame.memory.base import BaseMemoryStore
from a_frame.memory.procedural.skill_store import SkillEntry


class FileSkillStore(BaseMemoryStore):
    """基于单 JSON 文件的技能库。"""

    def __init__(self, file_path: str = "a_frame_skills.json"):
        self._file_path = file_path
        self._lock = threading.Lock()
        self._skills: dict[str, SkillEntry] = {}
        self._load()

    # ──────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._file_path):
            self._skills = {}
            return

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                if not raw:
                    self._skills = {}
                    return
                data = json.loads(raw)
        except Exception:
            self._skills = {}
            return

        if not isinstance(data, list):
            self._skills = {}
            return

        skills: dict[str, SkillEntry] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                if "doc_markdown" not in item:
                    # 不兼容旧数据：缺少 doc_markdown 的条目直接忽略
                    continue
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
                skills[skill.id] = skill
            except Exception:
                continue
        self._skills = skills

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._file_path) or ".", exist_ok=True)
        tmp_path = self._file_path + ".tmp"
        payload = [asdict(s) for s in self._skills.values()]
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, self._file_path)

    # ──────────────────────────────────────
    # BaseMemoryStore API
    # ──────────────────────────────────────

    def add(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        with self._lock:
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
            self._save()

    def search(
        self,
        query: str,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if query_embedding is None:
            return []

        with self._lock:
            if not self._skills:
                return []
            q_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = float(np.linalg.norm(q_vec) + 1e-9)

            scored: list[tuple[float, SkillEntry]] = []
            for skill in self._skills.values():
                s_vec = np.array(skill.embedding, dtype=np.float32)
                s_norm = float(np.linalg.norm(s_vec) + 1e-9)
                cos_sim = float(np.dot(q_vec, s_vec) / (q_norm * s_norm))
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
        if not ids:
            return
        with self._lock:
            for skill_id in ids:
                self._skills.pop(skill_id, None)
            self._save()

    def get_all(self) -> list[dict[str, Any]]:
        with self._lock:
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

    # ──────────────────────────────────────
    # Extra API
    # ──────────────────────────────────────

    def record_usage(self, skill_id: str) -> None:
        import datetime

        with self._lock:
            if skill_id not in self._skills:
                return
            self._skills[skill_id].success_count += 1
            self._skills[skill_id].last_used = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._save()
