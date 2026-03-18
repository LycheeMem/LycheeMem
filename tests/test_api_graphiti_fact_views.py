"""PR4: Facts view endpoints should route to Graphiti store when present.

Endpoints:
- GET /memory/graph/facts/active
- GET /memory/graph/facts/history

We keep this test store-only (no Neo4j) and deterministic.
"""

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
    def list_active_facts_for_subject(self, *, subject_entity_id: str, limit: int = 200):
        return [
            {
                "fact_id": "f1",
                "subject_entity_id": subject_entity_id,
                "object_entity_id": "e-b",
                "relation_type": "WORKS_FOR",
                "fact_text": "A works for B",
                "evidence_text": "...",
                "confidence": 0.9,
                "t_valid_from": "2026-01-01T00:00:00+00:00",
                "t_valid_to": "",
                "t_tx_created": "2026-01-01T00:00:00+00:00",
                "t_tx_expired": "",
            }
        ][:limit]

    def list_facts_for_subject(self, *, subject_entity_id: str, limit: int = 200):
        return [
            {
                "fact_id": "f_old",
                "subject_entity_id": subject_entity_id,
                "object_entity_id": "e-x",
                "relation_type": "WORKS_FOR",
                "fact_text": "A worked for X",
                "evidence_text": "...",
                "confidence": 0.8,
                "t_valid_from": "2025-01-01T00:00:00+00:00",
                "t_valid_to": "2026-01-01T00:00:00+00:00",
                "t_tx_created": "2025-01-01T00:00:00+00:00",
                "t_tx_expired": "2026-01-01T00:00:00+00:00",
            },
            {
                "fact_id": "f_new",
                "subject_entity_id": subject_entity_id,
                "object_entity_id": "e-b",
                "relation_type": "WORKS_FOR",
                "fact_text": "A works for B",
                "evidence_text": "...",
                "confidence": 0.9,
                "t_valid_from": "2026-01-01T00:00:00+00:00",
                "t_valid_to": "",
                "t_tx_created": "2026-01-01T00:00:00+00:00",
                "t_tx_expired": "",
            },
        ][:limit]


class FakeGraphitiEngine:
    def __init__(self, store: FakeGraphitiStore):
        self.store = store


def test_graphiti_facts_active_uses_store():
    pipeline = create_pipeline(llm=FakeLLM(), embedder=FakeEmbedder())
    pipeline.consolidator.graphiti_engine = FakeGraphitiEngine(FakeGraphitiStore())

    app = create_app(pipeline)
    client = TestClient(app)

    resp = client.get("/memory/graph/facts/active", params={"subject": "e-a", "top_k": 10})
    assert resp.status_code == 200
    data = resp.json()

    assert data["total"] == 1
    e0 = data["edges"][0]
    assert e0["source"] == "e-a"
    assert e0["relation"] == "WORKS_FOR"
    assert e0["t_valid_from"]


def test_graphiti_facts_history_uses_store():
    pipeline = create_pipeline(llm=FakeLLM(), embedder=FakeEmbedder())
    pipeline.consolidator.graphiti_engine = FakeGraphitiEngine(FakeGraphitiStore())

    app = create_app(pipeline)
    client = TestClient(app)

    resp = client.get("/memory/graph/facts/history", params={"subject": "e-a", "top_k": 10})
    assert resp.status_code == 200
    data = resp.json()

    assert data["total"] == 2
    assert {e["t_tx_expired"] for e in data["edges"]} >= {"", "2026-01-01T00:00:00+00:00"}
