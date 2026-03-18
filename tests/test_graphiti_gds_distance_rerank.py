from __future__ import annotations

from src.memory.graph.graphiti_engine import GraphitiEngine


def test_graphiti_engine_strict_uses_gds_distance_for_rerank() -> None:
    class FakeStore:
        def fulltext_search_facts(self, *, query: str, limit: int = 10):
            # Two facts, BM25 rank prefers far one first.
            return [
                {
                    "fact_id": "f_far",
                    "source": "e_far",
                    "target": "e_other",
                    "relation": "RELATED",
                    "fact": "Far fact",
                    "evidence": "",
                    "timestamp": "2026-01-02T00:00:00+00:00",
                    "t_valid_from": "2026-01-02T00:00:00+00:00",
                    "t_valid_to": "",
                },
                {
                    "fact_id": "f_near",
                    "source": "e2",
                    "target": "e3",
                    "relation": "RELATED",
                    "fact": "Near fact",
                    "evidence": "",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "t_valid_from": "2026-01-01T00:00:00+00:00",
                    "t_valid_to": "",
                },
            ]

        def fulltext_search_entities(self, *, query: str, limit: int = 10):
            # Provide a single anchor.
            return [{"entity_id": "e_anchor", "name": "Anchor", "score": 10.0}]

        def export_semantic_subgraph(self, *, entity_ids, edge_limit: int = 600):
            # Not used for facts in this test (edges empty), but still required for BFS channel.
            assert entity_ids
            return {
                "nodes": [
                    {"id": "e_anchor", "name": "Anchor", "label": "Entity", "properties": {}},
                    {"id": "e2", "name": "E2", "label": "Entity", "properties": {}},
                    {"id": "e3", "name": "E3", "label": "Entity", "properties": {}},
                    {"id": "e_far", "name": "Far", "label": "Entity", "properties": {}},
                    {"id": "e_other", "name": "Other", "label": "Entity", "properties": {}},
                ],
                "edges": [],
            }

        def get_entities_by_ids(self, *, entity_ids):
            return []

        def gds_min_distances_from_anchors(
            self,
            *,
            anchor_entity_ids: list[str],
            entity_ids: list[str],
            max_depth: int = 4,
        ) -> dict[str, int]:
            assert anchor_entity_ids == ["e_anchor"]
            # Make the near fact closer than the far fact.
            return {
                "e_anchor": 0,
                "e2": 1,
                "e3": 2,
                "e_far": 3,
                "e_other": 4,
            }

    engine = GraphitiEngine(store=FakeStore(), strict=True, gds_distance_max_depth=4)
    result = engine.search(query="q", top_k=2, query_embedding=None, include_communities=False)

    assert len(result.provenance) == 2
    # Near fact should outrank far fact due to GDS distance boost.
    assert result.provenance[0]["fact_id"] == "f_near"
    assert result.provenance[0]["distance"] == 1
    assert result.provenance[1]["fact_id"] == "f_far"
    assert result.provenance[1]["distance"] == 3
