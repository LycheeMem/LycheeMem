"""PR5: Community build/refresh minimal tests.

Avoid real Neo4j by using a FakeStore; ensure label-propagation communities are created
and linked when no communities exist for the query.
"""

from a_frame.memory.graph.graphiti_engine import GraphitiEngine


def test_search_builds_communities_when_none_exist():
    calls = {
        "upsert": [],
        "link": [],
    }

    class FakeStore:
        def fulltext_search_facts(self, *, query: str, limit: int = 10):
            return []

        def fulltext_search_entities(self, *, query: str, limit: int = 10):
            # Provide an anchor so BFS/communities can run.
            return [{"entity_id": "e1", "name": "E1", "summary": "", "type_label": ""}]

        def scan_entities_with_embeddings(self, *, limit: int = 2000):
            return []

        def export_semantic_subgraph(self, *, entity_ids, edge_limit: int = 600):
            # Two connected entities -> one community.
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
            # Return none so engine must build.
            return []

        def get_entity_embeddings_by_ids(self, *, entity_ids):
            # no embeddings
            return []

        def upsert_community(self, **kwargs):
            calls["upsert"].append(kwargs)
            return kwargs["community_id"]

        def link_entity_to_community(self, *, entity_id: str, community_id: str):
            calls["link"].append((entity_id, community_id))

        def get_entities_by_ids(self, *, entity_ids):
            return []

    engine = GraphitiEngine(store=FakeStore())
    r = engine.search(query="E1", top_k=1, query_embedding=None, include_communities=True)

    assert "<COMMUNITIES>" in r.context
    assert len(calls["upsert"]) >= 1
    # Must link both e1/e2
    linked_entities = {e for e, _ in calls["link"]}
    assert {"e1", "e2"}.issubset(linked_entities)
