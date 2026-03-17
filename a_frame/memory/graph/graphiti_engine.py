"""Graphiti(论文) 风格图谱引擎：Search/Rerank/Constructor 的对外门面。

PR1 目标：提供可注入的引擎骨架与兼容导出接口；真正的 ingest/search 能力
在后续 PR 分步实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import datetime
import hashlib

from a_frame.memory.graph.graphiti_neo4j_store import GraphitiNeo4jStore


@dataclass(slots=True)
class GraphitiSearchResult:
    context: str
    provenance: list[dict[str, Any]]


class GraphitiEngine:
    """Graphiti 引擎（面向论文的 f(α)=χ(ρ(φ(α)))）。"""

    def __init__(self, store: GraphitiNeo4jStore):
        self.store = store

    @staticmethod
    def _default_episode_id(*, session_id: str, turn_index: int, role: str, content: str) -> str:
        raw = f"{session_id}|{turn_index}|{role}|{content}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def ingest_episode(
        self,
        *,
        session_id: str,
        turn_index: int,
        role: str,
        content: str,
        t_ref: str | None = None,
        episode_id: str | None = None,
    ) -> str:
        """写入一个 Episode（幂等）。

        PR2 仅做“原始事件落盘 + 可追溯 id”，不触发实体/事实解析。
        """

        if t_ref is None:
            t_ref = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if episode_id is None:
            episode_id = self._default_episode_id(
                session_id=session_id,
                turn_index=turn_index,
                role=role,
                content=content,
            )

        return self.store.upsert_episode(
            episode_id=episode_id,
            session_id=session_id,
            role=role,
            content=content,
            turn_index=turn_index,
            t_ref=t_ref,
        )

    def search(
        self, *, query: str, session_id: str | None = None, top_k: int = 10
    ) -> GraphitiSearchResult:
        """论文式检索：Search→Rerank→Constructor。

        PR1 仅提供签名；在 PR5 才实现。
        """
        raise NotImplementedError

    def export_semantic_graph(self) -> dict[str, list[dict[str, Any]]]:
        """为 API/web-demo 提供兼容的语义图视图导出。"""
        return self.store.export_semantic_graph()
