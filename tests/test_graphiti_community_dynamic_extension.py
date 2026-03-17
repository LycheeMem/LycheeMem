from __future__ import annotations

from a_frame.memory.graph.graphiti_engine import GraphitiEngine


def test_graphiti_engine_community_dynamic_extension_fills_pool() -> None:
    class FakeStore:
        def fulltext_search_facts(self, *, query: str, limit: int = 10):
            return []

        def fulltext_search_entities(self, *, query: str, limit: int = 10):
            return []

        def export_semantic_subgraph(self, *, entity_ids, edge_limit: int = 600):
            return {"nodes": [], "edges": []}

        def get_entities_by_ids(self, *, entity_ids):
            return []

        def fulltext_search_communities(self, *, query: str, limit: int = 10):
            return [
                {
                    "community_id": "c1",
                    "name": "Base",
                    "summary": "base summary",
                    "score": 3.0,
                }
            ]

        def expand_communities_via_entities(self, *, community_ids: list[str], limit: int = 5):
            assert community_ids == ["c1"]
            assert limit >= 1
            return [
                {
                    "community_id": "c2",
                    "name": "Expanded",
                    "summary": "expanded summary",
                    "score": 1.0,
                }
            ]

    engine = GraphitiEngine(store=FakeStore())
    r = engine.search(query="q", top_k=1, query_embedding=None, include_communities=True)

    assert "<COMMUNITIES>" in r.context
    assert "Base" in r.context
    assert "Expanded" in r.context
