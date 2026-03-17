"""PR3: Graphiti 语义构建器单测（不依赖 Neo4j）。

覆盖：Entity/Fact 抽取 → resolution → upsert → Episode→Fact 证据链接。
"""

from __future__ import annotations

import json
from typing import Any

from a_frame.memory.graph.graphiti_semantic import GraphitiSemanticBuilder


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text: str):
        return [0.1] * 8


def _extract_current_message(system_prompt: str) -> str:
    start = system_prompt.find("<CURRENT MESSAGE>")
    end = system_prompt.find("</CURRENT MESSAGE>")
    if start == -1 or end == -1:
        return ""
    return system_prompt[start:end]


class FakeLLMGraphiti:
    """根据 system prompt 关键片段返回固定 JSON。"""

    def generate(self, messages, **kwargs):
        system_msg = messages[0]["content"] if messages else ""

        # Entity resolution
        if "entity resolution system" in system_msg:
            return json.dumps(
                {
                    "is_duplicate": False,
                    "existing_entity_id": None,
                    "name": "",
                    "summary": "",
                    "aliases": [],
                    "type_label": "",
                }
            )

        # Fact resolution
        if "fact deduplication system" in system_msg:
            return json.dumps(
                {
                    "is_duplicate": False,
                    "existing_fact_id": None,
                    "relation_type": "WORKS_FOR",
                    "fact_text": "张三在 Google 工作",
                    "evidence_text": "在 Google 工作",
                    "confidence": 0.9,
                }
            )

        current_msg = _extract_current_message(system_msg)

        # Fact extraction
        if "<ENTITIES>" in system_msg:
            if "张三" not in current_msg:
                return "[]"
            return json.dumps(
                [
                    {
                        "subject": "张三",
                        "object": "Google",
                        "relation_type": "WORKS_FOR",
                        "fact_text": "张三在 Google 工作",
                        "evidence_text": "在 Google 工作",
                        "confidence": 0.9,
                    }
                ]
            )

        # Entity extraction
        if "张三" not in current_msg:
            return "[]"
        return json.dumps(
            [
                {"name": "张三", "type_label": "Person", "summary": "", "aliases": []},
                {"name": "Google", "type_label": "Organization", "summary": "", "aliases": []},
            ]
        )


class FakeGraphitiStore:
    def __init__(self):
        self.entities: dict[str, dict[str, Any]] = {}
        self.facts: dict[str, dict[str, Any]] = {}
        self.episode_fact_links: list[tuple[str, str]] = []

        self.upsert_entity_calls = 0
        self.upsert_fact_calls = 0
        self.link_calls = 0

    def fulltext_search_entities(self, query: str, limit: int = 10):
        q = (query or "").lower()
        out = []
        for e in self.entities.values():
            if q and q in str(e.get("name") or "").lower():
                out.append(e)
        return out[:limit]

    def vector_search_entities(self, *, query_embedding, limit: int = 10):
        # Keep deterministic + simple for tests.
        return []

    def scan_entities_with_embeddings(self, limit: int = 2000):
        return list(self.entities.values())[:limit]

    def upsert_entity(
        self,
        *,
        entity_id: str,
        name: str,
        summary: str,
        aliases: list[str],
        type_label: str,
        embedding: list[float] | None,
        source_session: str,
        t_created: str,
    ):
        self.upsert_entity_calls += 1
        self.entities[entity_id] = {
            "entity_id": entity_id,
            "name": name,
            "summary": summary,
            "aliases": aliases,
            "type_label": type_label,
            "embedding": embedding,
            "source_session": source_session,
            "t_created": t_created,
        }

    def list_facts_between(self, *, subject_entity_id: str, object_entity_id: str, limit: int = 20):
        out = []
        for f in self.facts.values():
            if (
                f.get("subject_entity_id") == subject_entity_id
                and f.get("object_entity_id") == object_entity_id
            ):
                out.append(f)
        return out[:limit]

    def upsert_fact(
        self,
        *,
        fact_id: str,
        subject_entity_id: str,
        object_entity_id: str,
        relation_type: str,
        fact_text: str,
        evidence_text: str,
        embedding: list[float] | None = None,
        confidence: float,
        source_session: str,
        t_created: str,
        t_valid_from: str | None = None,
        t_valid_to: str | None = None,
        t_tx_created: str | None = None,
    ):
        self.upsert_fact_calls += 1
        self.facts[fact_id] = {
            "fact_id": fact_id,
            "subject_entity_id": subject_entity_id,
            "object_entity_id": object_entity_id,
            "relation_type": relation_type,
            "fact_text": fact_text,
            "evidence_text": evidence_text,
            "embedding": embedding,
            "confidence": confidence,
            "source_session": source_session,
            "t_created": t_created,
            "t_valid_from": t_valid_from,
            "t_valid_to": t_valid_to,
            "t_tx_created": t_tx_created,
        }

    def link_episode_to_fact(self, *, episode_id: str, fact_id: str):
        self.link_calls += 1
        self.episode_fact_links.append((episode_id, fact_id))


def test_graphiti_semantic_builder_ingest_user_turn():
    store = FakeGraphitiStore()
    builder = GraphitiSemanticBuilder(llm=FakeLLMGraphiti(), embedder=FakeEmbedder(), store=store)

    result = builder.ingest_user_turn(
        session_id="s1",
        episode_id="ep1",
        previous_turns=[{"role": "assistant", "content": "你好"}],
        current_turn={"role": "user", "content": "我叫张三，在 Google 工作"},
        reference_timestamp="2026-01-01T00:00:00+00:00",
    )

    assert result["entities_added"] == 2
    assert result["facts_added"] == 1

    assert store.upsert_entity_calls == 2
    assert store.upsert_fact_calls == 1
    assert store.link_calls == 1

    assert len(store.entities) == 2
    assert len(store.facts) == 1
    assert store.episode_fact_links == [("ep1", next(iter(store.facts.keys())))]
