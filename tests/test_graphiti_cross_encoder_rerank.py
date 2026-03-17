from __future__ import annotations

from a_frame.memory.graph.graphiti_engine import GraphitiEngine


def test_graphiti_engine_cross_encoder_can_flip_order() -> None:
    class FakeStore:
        def fulltext_search_facts(self, *, query: str, limit: int = 10):
            # BM25 rank prefers f1 first.
            return [
                {
                    "fact_id": "f1",
                    "source": "e1",
                    "target": "e2",
                    "relation": "RELATED",
                    "fact": "First",
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
                    "fact": "Second",
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

    class FakeReranker:
        def score(self, *, query: str, passages):  # noqa: ANN001
            assert query
            # Prefer f2 strongly.
            return {"f1": 0.0, "f2": 1.0}

    engine = GraphitiEngine(
        store=FakeStore(),
        strict=False,
        cross_encoder=FakeReranker(),
        cross_encoder_top_n=2,
        cross_encoder_weight=10.0,
    )

    result = engine.search(query="q", top_k=2, query_embedding=None, include_communities=False)

    assert [p["fact_id"] for p in result.provenance] == ["f2", "f1"]
    assert result.provenance[0]["cross_encoder_score"] == 1.0
    assert result.provenance[1]["cross_encoder_score"] == 0.0
