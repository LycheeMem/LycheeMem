"""PR5: GraphitiEngine.search (Search→Rerank→Constructor) minimal tests.

These tests avoid real Neo4j by using a FakeStore that mimics the store API.
"""

from a_frame.memory.graph.graphiti_engine import GraphitiEngine


def test_graphiti_engine_search_constructs_context_and_provenance():
    class FakeStore:
        def fulltext_search_facts(self, *, query: str, limit: int = 10):
            assert query
            return [
                {
                    "fact_id": "f_bm25_1",
                    "source": "e_alice",
                    "target": "e_acme",
                    "relation": "EMPLOYED_BY",
                    "fact": "Alice works at Acme",
                    "evidence": "user said so",
                    "timestamp": "2025-01-01T00:00:00+00:00",
                    "t_valid_from": "2025-01-01T00:00:00+00:00",
                    "t_valid_to": "",
                }
            ]

        def fulltext_search_entities(self, *, query: str, limit: int = 10):
            return [
                {
                    "entity_id": "e_alice",
                    "name": "Alice",
                    "summary": "A person",
                    "type_label": "Person",
                    "score": 10.0,
                }
            ]

        def scan_entities_with_embeddings(self, *, limit: int = 2000):
            return [
                {
                    "entity_id": "e_alice",
                    "name": "Alice",
                    "summary": "A person",
                    "type_label": "Person",
                    "embedding": [1.0, 0.0],
                },
                {
                    "entity_id": "e_acme",
                    "name": "Acme",
                    "summary": "A company",
                    "type_label": "Organization",
                    "embedding": [0.9, 0.1],
                },
            ]

        def export_semantic_subgraph(self, *, entity_ids, edge_limit: int = 600):
            # BFS channel will find a second fact through graph expansion.
            assert entity_ids
            return {
                "nodes": [
                    {
                        "id": "e_alice",
                        "name": "Alice",
                        "label": "Person",
                        "properties": {"summary": "A person"},
                    },
                    {
                        "id": "e_acme",
                        "name": "Acme",
                        "label": "Organization",
                        "properties": {"summary": "A company"},
                    },
                ],
                "edges": [
                    {
                        "fact_id": "f_bfs_1",
                        "source": "e_alice",
                        "target": "e_acme",
                        "relation": "EMPLOYED_BY",
                        "fact": "Alice is employed by Acme",
                        "evidence": "episode-1",
                        "timestamp": "2024-12-31T00:00:00+00:00",
                        "t_valid_from": "2024-12-31T00:00:00+00:00",
                        "t_valid_to": "",
                    }
                ],
            }

        def get_entities_by_ids(self, *, entity_ids):
            # Should not be required if nodes exist, but keep for robustness.
            return []

        def fulltext_search_communities(self, *, query: str, limit: int = 10):
            return [
                {
                    "community_id": "c1",
                    "name": "Employment",
                    "summary": "Facts about jobs",
                    "score": 3.0,
                }
            ]

    engine = GraphitiEngine(store=FakeStore())
    result = engine.search(
        query="Alice works at Acme",
        top_k=2,
        query_embedding=[1.0, 0.0],
        include_communities=True,
    )

    assert "[GraphitiRetrievedFacts]" in result.context
    assert "Alice" in result.context
    assert "Acme" in result.context
    assert "[GraphitiCommunities]" in result.context
    assert len(result.provenance) == 2
    assert all("fact_id" in p for p in result.provenance)
    assert all("mentions" in p for p in result.provenance)
    assert all("distance" in p for p in result.provenance)
