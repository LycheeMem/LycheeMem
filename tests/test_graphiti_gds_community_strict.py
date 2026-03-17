from __future__ import annotations

from a_frame.memory.graph.graphiti_engine import GraphitiEngine


def test_graphiti_engine_strict_community_build_uses_gds() -> None:
    calls: dict[str, object] = {
        "gds_called": False,
        "upsert": [],
        "link": [],
    }

    class FakeStore:
        def fulltext_search_facts(self, *, query: str, limit: int = 10):
            return []

        def fulltext_search_entities(self, *, query: str, limit: int = 10):
            return [{"entity_id": "e1", "name": "E1", "summary": "", "type_label": ""}]

        def export_semantic_subgraph(self, *, entity_ids, edge_limit: int = 600):
            assert entity_ids
            return {
                "nodes": [
                    {"id": "e1", "name": "E1", "label": "Entity", "properties": {}},
                    {"id": "e2", "name": "E2", "label": "Entity", "properties": {}},
                ],
                "edges": [
                    {
                        "fact_id": "f1",
                        "source": "e1",
                        "target": "e2",
                        "relation": "RELATED",
                        "fact": "E1 related to E2",
                        "evidence": "",
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "t_valid_from": "2026-01-01T00:00:00+00:00",
                        "t_valid_to": "",
                    }
                ],
            }

        def fulltext_search_communities(self, *, query: str, limit: int = 10):
            return []

        def get_entity_embeddings_by_ids(self, *, entity_ids):
            return []

        def upsert_community(self, **kwargs):
            calls["upsert"].append(kwargs)
            return kwargs["community_id"]

        def link_entity_to_community(self, *, entity_id: str, community_id: str):
            calls["link"].append((entity_id, community_id))

        def get_entities_by_ids(self, *, entity_ids):
            return []

        def gds_label_propagation_groups(self, *, entity_ids: list[str], max_iterations: int = 10):
            calls["gds_called"] = True
            # Single community for both nodes.
            return {"1": ["e1", "e2"]}

        def gds_min_distances_from_anchors(
            self,
            *,
            anchor_entity_ids: list[str],
            entity_ids: list[str],
            max_depth: int = 4,
        ) -> dict[str, int]:
            return {"e1": 0, "e2": 1}

    engine = GraphitiEngine(store=FakeStore(), strict=True)
    r = engine.search(query="E1", top_k=1, query_embedding=None, include_communities=True)

    assert "[GraphitiCommunities]" in r.context
    assert calls["gds_called"] is True
    assert len(calls["upsert"]) >= 1
    linked_entities = {e for e, _ in calls["link"]}
    assert {"e1", "e2"}.issubset(linked_entities)
