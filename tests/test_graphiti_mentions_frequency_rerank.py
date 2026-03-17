from __future__ import annotations

from a_frame.memory.graph.graphiti_engine import GraphitiEngine


def test_graphiti_engine_mentions_frequency_can_flip_order() -> None:
    class FakeStore:
        def fulltext_search_facts(self, *, query: str, limit: int = 10):
            # BM25 rank prefers f1 first.
            return [
                {
                    "fact_id": "f1",
                    "source": "e1",
                    "target": "e2",
                    "relation": "RELATED",
                    "fact": "Fact 1",
                    "evidence": "",
                    "timestamp": "2026-01-02T00:00:00+00:00",
                    "t_valid_from": "2026-01-02T00:00:00+00:00",
                    "t_valid_to": "",
                },
                {
                    "fact_id": "f2",
                    "source": "e3",
                    "target": "e4",
                    "relation": "RELATED",
                    "fact": "Fact 2",
                    "evidence": "",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "t_valid_from": "2026-01-01T00:00:00+00:00",
                    "t_valid_to": "",
                },
            ]

        def fulltext_search_entities(self, *, query: str, limit: int = 10):
            return []

        def export_semantic_subgraph(self, *, entity_ids, edge_limit: int = 600):
            return {"nodes": [], "edges": []}

        def get_entities_by_ids(self, *, entity_ids):
            return []

        def count_mentions_in_session(self, *, session_id: str, entity_ids, fact_ids):
            assert session_id == "s1"
            # Make f2 heavily mentioned in the session.
            return {
                "entities": {},
                "facts": {
                    "f1": 0,
                    "f2": 10,
                },
            }

    engine = GraphitiEngine(store=FakeStore(), strict=False)
    r = engine.search(query="q", session_id="s1", top_k=2, query_embedding=None)

    assert len(r.provenance) == 2
    assert r.provenance[0]["fact_id"] == "f2"
    assert r.provenance[0]["mentions"] == 10
    assert r.provenance[1]["fact_id"] == "f1"
