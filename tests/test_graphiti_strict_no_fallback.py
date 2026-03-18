from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.agents.search_coordinator import SearchCoordinator
from src.api.server import create_app
from src.core.factory import create_pipeline
from src.memory.graph.graph_store import NetworkXGraphStore
from src.memory.procedural.skill_store import InMemorySkillStore


class FakeLLM:
    def generate(self, messages, **kwargs):
        # Keep it deterministic and unrelated to strict behavior.
        system_msg = messages[0]["content"] if messages else ""
        if "检索规划" in system_msg or "sub_queries" in system_msg:
            return '{"sub_queries": [], "reasoning": "test"}'
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


def test_search_coordinator_strict_graphiti_error_propagates() -> None:
    class ExplodingGraphitiEngine:
        strict = True

        def search(self, **kwargs):  # noqa: ANN003
            raise RuntimeError("graphiti failure")

    graph_store = NetworkXGraphStore()
    # Add a legacy triple that would otherwise be returned by fallback.
    graph_store.add(
        [
            {
                "subject": {"name": "Alice", "label": "Person"},
                "predicate": "knows",
                "object": {"name": "Bob", "label": "Person"},
            }
        ]
    )

    coordinator = SearchCoordinator(
        llm=FakeLLM(),
        embedder=FakeEmbedder(),
        graph_store=graph_store,
        skill_store=InMemorySkillStore(),
        graphiti_engine=ExplodingGraphitiEngine(),
    )

    with pytest.raises(RuntimeError, match="graphiti failure"):
        coordinator.run(user_query="Alice")


def test_api_graph_strict_graphiti_error_returns_500_not_legacy() -> None:
    class ExplodingStore:
        def export_semantic_graph(self):
            raise RuntimeError("export failed")

    class FakeGraphitiEngine:
        strict = True

        def __init__(self):
            self.store = ExplodingStore()

    pipeline = create_pipeline(llm=FakeLLM(), embedder=FakeEmbedder())

    # Populate legacy graph store; strict mode must not fall back to this.
    pipeline.search_coordinator.graph_store.add(
        [
            {
                "subject": {"name": "Alice", "label": "Person"},
                "predicate": "knows",
                "object": {"name": "Bob", "label": "Person"},
            }
        ]
    )

    pipeline.consolidator.graphiti_engine = FakeGraphitiEngine()

    app = create_app(pipeline)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/memory/graph")
    assert resp.status_code == 500


def test_api_graph_search_strict_graphiti_error_returns_500_not_legacy() -> None:
    class ExplodingStore:
        def fulltext_search_entities(self, *, query: str, limit: int = 10):
            return [{"entity_id": "e-alice", "name": "Alice", "score": 10.0}]

        def export_semantic_subgraph(self, *, entity_ids: list[str], edge_limit: int = 200):
            raise RuntimeError("subgraph export failed")

    class FakeGraphitiEngine:
        strict = True

        def __init__(self):
            self.store = ExplodingStore()

    pipeline = create_pipeline(llm=FakeLLM(), embedder=FakeEmbedder())
    pipeline.search_coordinator.graph_store.add(
        [
            {
                "subject": {"name": "Alice", "label": "Person"},
                "predicate": "knows",
                "object": {"name": "Bob", "label": "Person"},
            }
        ]
    )
    pipeline.consolidator.graphiti_engine = FakeGraphitiEngine()

    app = create_app(pipeline)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/memory/graph/search", params={"q": "Alice", "top_k": 1})
    assert resp.status_code == 500
