"""PR3 closing: Graphiti 模式下 /memory/graph/search 走 store.export_semantic_subgraph。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core.factory import create_pipeline


class FakeLLM:
    def generate(self, messages, **kwargs):
        # API tests don't rely on actual chat quality here
        system_msg = messages[0]["content"] if messages else ""
        if "固化" in system_msg or "new_skills" in system_msg:
            return '{"new_skills": [], "should_extract_entities": false}'
        if "压缩" in system_msg:
            return "## Summary\ntest"
        return "ok"

    async def agenerate(self, messages, **kwargs):
        return self.generate(messages, **kwargs)


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 8


class FakeGraphitiStore:
    def __init__(self):
        self.last_entity_ids = None

    def fulltext_search_entities(self, *, query: str, limit: int = 10):
        # deterministic: always find Alice first
        return [
            {"entity_id": "e-alice", "name": "Alice", "score": 10.0},
            {"entity_id": "e-bob", "name": "Bob", "score": 9.0},
        ][:limit]

    def scan_entities_with_embeddings(self, *, limit: int = 2000):
        return []

    def export_semantic_subgraph(self, *, entity_ids: list[str], edge_limit: int = 200):
        self.last_entity_ids = list(entity_ids)
        # Return a tiny graph: Alice -> Bob
        return {
            "nodes": [
                {"id": "e-alice", "name": "Alice", "label": "Person", "properties": {}},
                {"id": "e-bob", "name": "Bob", "label": "Person", "properties": {}},
            ],
            "edges": [
                {
                    "source": "e-alice",
                    "target": "e-bob",
                    "relation": "KNOWS",
                    "confidence": 1.0,
                    "fact": "Alice knows Bob",
                    "evidence": "...",
                    "timestamp": "",
                    "source_session": "",
                    "episode_ids": ["ep-1"],
                }
            ],
        }


class FakeGraphitiEngine:
    def __init__(self, store: FakeGraphitiStore):
        self.store = store


def test_graphiti_graph_search_uses_subgraph_export():
    pipeline = create_pipeline(llm=FakeLLM(), embedder=FakeEmbedder())

    store = FakeGraphitiStore()
    pipeline.consolidator.graphiti_engine = FakeGraphitiEngine(store)

    app = create_app(pipeline)
    client = TestClient(app)

    resp = client.get("/memory/graph/search", params={"q": "Alice", "top_k": 1})
    assert resp.status_code == 200
    data = resp.json()

    assert store.last_entity_ids == ["e-alice"]

    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    # anchor marking is optional but should be present for the matched entity
    alice = next(n for n in data["nodes"] if n["id"] == "e-alice")
    assert alice["properties"].get("is_anchor") is True
