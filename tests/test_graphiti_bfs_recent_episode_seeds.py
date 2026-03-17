"""Paper parity: BFS seeded by recent episodes.

Graphiti paper (3.1 Search) notes that BFS over the KG can accept nodes as parameters and is
particularly valuable when using recent episodes as seeds, to incorporate recently mentioned
entities/relationships into retrieved context.

This test locks the behavior:
- When session_id is provided, GraphitiEngine.search() should seed BFS with entity ids derived
  from recent episodes (store.list_recent_entity_ids_from_episodes).
- Even if query-based channels return nothing, BFS recall can still surface context.
"""

from __future__ import annotations

from typing import Any

import pytest

from a_frame.memory.graph.graphiti_engine import GraphitiEngine


class FakeStore:
    def __init__(self) -> None:
        self.called_recent = 0

    # Query channels return empty
    def fulltext_search_facts(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return []

    def fulltext_search_entities(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return []

    # Strict GraphitiEngine also requires mentions counting when session_id is provided.
    def count_mentions_in_session(
        self,
        *,
        session_id: str,
        entity_ids: list[str] | None = None,
        fact_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        return {"entities": {}, "facts": {}}

    # Paper: recent-episode derived entity seeds
    def list_recent_entity_ids_from_episodes(
        self, *, session_id: str, episode_limit: int = 4, entity_limit: int = 20
    ) -> list[str]:
        self.called_recent += 1
        return ["E1"]

    def export_semantic_subgraph(self, *, entity_ids: list[str], edge_limit: int = 200) -> dict[str, Any]:
        # If BFS is seeded with E1, we expose an edge/fact that can be recalled.
        if "E1" not in entity_ids:
            return {"nodes": [], "edges": []}
        return {
            "nodes": [
                {"id": "E1", "name": "Alice", "properties": {"summary": "person"}},
                {"id": "E2", "name": "Acme", "properties": {"summary": "company"}},
            ],
            "edges": [
                {
                    "fact_id": "F1",
                    "source": "E1",
                    "target": "E2",
                    "relation": "WORKS_FOR",
                    "fact": "Alice works for Acme",
                    "evidence": "I joined Acme",
                    "t_valid_from": "2026-01-01T00:00:00+00:00",
                    "t_valid_to": "",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            ],
        }

    def get_entities_by_ids(self, *, entity_ids: list[str]) -> list[dict[str, Any]]:
        # Not needed for this test.
        return []


def test_graphiti_bfs_seeds_include_recent_episodes_entities() -> None:
    engine = GraphitiEngine(store=FakeStore(), strict=False)

    result = engine.search(
        query="unrelated query",
        session_id="s1",
        top_k=5,
        query_embedding=None,
        include_communities=False,
    )

    assert "Alice works for Acme" in result.context
    assert engine.store.called_recent == 1


def test_graphiti_strict_requires_recent_episode_seed_capability() -> None:
    class NoRecentStore:
        # Provide minimum surface area required by GraphitiEngine.search(strict=True)
        # while intentionally omitting list_recent_entity_ids_from_episodes.

        def fulltext_search_facts(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
            return []

        def fulltext_search_entities(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
            return []

        def count_mentions_in_session(
            self,
            *,
            session_id: str,
            entity_ids: list[str] | None = None,
            fact_ids: list[str] | None = None,
        ) -> dict[str, dict[str, int]]:
            return {"entities": {}, "facts": {}}

        def export_semantic_subgraph(
            self, *, entity_ids: list[str], edge_limit: int = 200
        ) -> dict[str, Any]:
            return {"nodes": [], "edges": []}

        def get_entities_by_ids(self, *, entity_ids: list[str]) -> list[dict[str, Any]]:
            return []

    engine = GraphitiEngine(store=NoRecentStore(), strict=True)

    with pytest.raises(RuntimeError, match="list_recent_entity_ids_from_episodes"):
        engine.search(
            query="q",
            session_id="s1",
            top_k=5,
            query_embedding=None,
            include_communities=False,
        )
