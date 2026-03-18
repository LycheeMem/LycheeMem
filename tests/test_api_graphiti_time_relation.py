"""PR4: API should route by-time/by-relation to Graphiti store when present."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core.factory import create_pipeline


class FakeLLM:
    def generate(self, messages, **kwargs):
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
    def search_facts_by_relation(self, *, relation: str, limit: int = 10):
        return [
            {
                "source": "e-a",
                "target": "e-b",
                "relation": relation,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ][:limit]

    def search_facts_by_time(
        self, *, since: str | None = None, until: str | None = None, limit: int = 10
    ):
        return [
            {
                "source": "e-a",
                "target": "e-b",
                "relation": "WORKS_FOR",
                "timestamp": since or "",
                "t_valid_from": since or "",
                "t_valid_to": until or "",
            }
        ][:limit]


class FakeGraphitiEngine:
    def __init__(self, store: FakeGraphitiStore):
        self.store = store


def test_graphiti_by_relation_uses_store():
    pipeline = create_pipeline(llm=FakeLLM(), embedder=FakeEmbedder())
    pipeline.consolidator.graphiti_engine = FakeGraphitiEngine(FakeGraphitiStore())

    app = create_app(pipeline)
    client = TestClient(app)

    resp = client.get("/memory/graph/by-relation", params={"relation": "KNOWS", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["edges"][0]["relation"] == "KNOWS"


def test_graphiti_by_time_uses_store():
    pipeline = create_pipeline(llm=FakeLLM(), embedder=FakeEmbedder())
    pipeline.consolidator.graphiti_engine = FakeGraphitiEngine(FakeGraphitiStore())

    app = create_app(pipeline)
    client = TestClient(app)

    resp = client.get(
        "/memory/graph/by-time",
        params={
            "since": "2026-01-01T00:00:00+00:00",
            "until": "2026-02-01T00:00:00+00:00",
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["edges"][0]["t_valid_from"] == "2026-01-01T00:00:00+00:00"
    assert data["edges"][0]["t_valid_to"] == "2026-02-01T00:00:00+00:00"
