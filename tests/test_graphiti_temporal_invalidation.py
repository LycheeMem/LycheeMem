"""PR4: Temporal + bi-temporal + invalidation (no Neo4j dependency).

Covers:
- t_valid_from/t_valid_to are written on Fact upsert
- a contradictory newer fact expires the older active fact (t_valid_to + t_tx_expired)

We keep the logic conservative and deterministic via FakeLLM.
"""

from __future__ import annotations

import json
from typing import Any

from src.memory.graph.graphiti_semantic import GraphitiSemanticBuilder


def _extract_reference_timestamp(system_prompt: str) -> str:
    start = system_prompt.find("<REFERENCE TIMESTAMP>")
    end = system_prompt.find("</REFERENCE TIMESTAMP>")
    if start == -1 or end == -1:
        return ""
    chunk = system_prompt[start:end]
    # crude: last non-empty line
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _extract_current_message(system_prompt: str) -> str:
    start = system_prompt.find("<CURRENT MESSAGE>")
    end = system_prompt.find("</CURRENT MESSAGE>")
    if start == -1 or end == -1:
        return ""
    return system_prompt[start:end]


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text: str):
        return [0.1] * 8


class FakeLLMTemporal:
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
            # Keep it deterministic but preserve key semantics for temporal tests.
            fact_text = ""
            evidence_text = ""
            if "曾经" in system_msg and "OpenAI" in system_msg:
                fact_text = "张三曾经在 OpenAI 工作"
                evidence_text = "曾经在 OpenAI 工作"
            elif "OpenAI" in system_msg:
                fact_text = "张三在 OpenAI 工作"
                evidence_text = "在 OpenAI 工作"
            elif "Google" in system_msg:
                fact_text = "张三在 Google 工作"
                evidence_text = "在 Google 工作"
            return json.dumps(
                {
                    "is_duplicate": False,
                    "existing_fact_id": None,
                    "relation_type": "WORKS_FOR",
                    "fact_text": fact_text,
                    "evidence_text": evidence_text,
                    "confidence": 0.9,
                }
            )

        # Temporal extraction
        if "temporal information extraction system" in system_msg:
            t_ref = _extract_reference_timestamp(system_msg)
            # Special case for non-overlap test: force a past validity window.
            if "曾经" in system_msg:
                return json.dumps(
                    {
                        "t_valid_from": "2025-01-01T00:00:00+00:00",
                        "t_valid_to": "2025-02-01T00:00:00+00:00",
                    }
                )

            return json.dumps(
                {"t_valid_from": t_ref or "2026-01-01T00:00:00+00:00", "t_valid_to": None}
            )

        current_msg = _extract_current_message(system_msg)

        # Fact extraction
        if "<ENTITIES>" in system_msg:
            if "Google" in current_msg:
                obj = "Google"
            elif "OpenAI" in current_msg:
                obj = "OpenAI"
            else:
                return "[]"
            fact_prefix = "张三曾经在 " if "曾经" in current_msg else "张三在 "
            evidence_prefix = "曾经在 " if "曾经" in current_msg else "在 "
            return json.dumps(
                [
                    {
                        "subject": "张三",
                        "object": obj,
                        "relation_type": "WORKS_FOR",
                        "fact_text": f"{fact_prefix}{obj} 工作",
                        "evidence_text": f"{evidence_prefix}{obj} 工作",
                        "confidence": 0.9,
                    }
                ]
            )

        # Entity extraction
        if "Google" in current_msg:
            return json.dumps(
                [
                    {"name": "张三", "type_label": "Person", "summary": "", "aliases": []},
                    {"name": "Google", "type_label": "Organization", "summary": "", "aliases": []},
                ]
            )
        if "OpenAI" in current_msg:
            return json.dumps(
                [
                    {"name": "张三", "type_label": "Person", "summary": "", "aliases": []},
                    {"name": "OpenAI", "type_label": "Organization", "summary": "", "aliases": []},
                ]
            )
        return "[]"


class FakeGraphitiStore:
    def __init__(self):
        self.entities: dict[str, dict[str, Any]] = {}
        self.facts: dict[str, dict[str, Any]] = {}
        self.episode_fact_links: list[tuple[str, str]] = []
        self.episode_entity_links: list[tuple[str, str]] = []

        self.expired: list[dict[str, Any]] = []

    def fulltext_search_entities(self, query: str, limit: int = 10):
        return []

    def vector_search_entities(self, *, query_embedding, limit: int = 10):
        return []

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

    def list_active_facts_for_subject_relation(
        self, *, subject_entity_id: str, relation_type: str, limit: int = 50
    ):
        rel = (relation_type or "").upper()
        out = []
        for f in self.facts.values():
            if (
                f.get("subject_entity_id") == subject_entity_id
                and (f.get("relation_type") or "").upper() == rel
                and not f.get("t_valid_to")
            ):
                out.append(f)
        return out[:limit]

    def expire_fact(self, *, fact_id: str, t_valid_to: str, t_tx_expired: str | None = None):
        f = self.facts.get(fact_id)
        if not f:
            return
        f["t_valid_to"] = t_valid_to
        f["t_tx_expired"] = t_tx_expired
        self.expired.append(
            {"fact_id": fact_id, "t_valid_to": t_valid_to, "t_tx_expired": t_tx_expired}
        )

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
        self.episode_fact_links.append((episode_id, fact_id))

    def link_episode_to_entity(self, *, episode_id: str, entity_id: str):
        self.episode_entity_links.append((episode_id, entity_id))


def test_temporal_invalidation_expires_old_fact():
    store = FakeGraphitiStore()
    builder = GraphitiSemanticBuilder(llm=FakeLLMTemporal(), embedder=FakeEmbedder(), store=store)

    # 1) First fact at t1
    t1 = "2026-01-01T00:00:00+00:00"
    r1 = builder.ingest_user_turn(
        session_id="s1",
        episode_id="ep1",
        previous_turns=[{"role": "assistant", "content": "你好"}],
        current_turn={"role": "user", "content": "我叫张三，在 Google 工作"},
        reference_timestamp=t1,
    )
    assert r1["facts_added"] == 1

    # Verify stored fact has t_valid_from
    f1_id = next(iter(store.facts.keys()))
    assert store.facts[f1_id]["t_valid_from"] == t1
    assert not store.facts[f1_id].get("t_valid_to")

    # 2) Contradictory newer fact at t2 should expire old
    t2 = "2026-02-01T00:00:00+00:00"
    r2 = builder.ingest_user_turn(
        session_id="s1",
        episode_id="ep2",
        previous_turns=[{"role": "assistant", "content": "好的"}],
        current_turn={"role": "user", "content": "我叫张三，现在在 OpenAI 工作"},
        reference_timestamp=t2,
    )
    assert r2["facts_added"] == 1
    assert r2["facts_expired"] >= 1

    assert store.facts[f1_id]["t_valid_to"] == t2
    assert store.facts[f1_id]["t_tx_expired"] == t2


def test_temporal_invalidation_skips_when_intervals_do_not_overlap():
    store = FakeGraphitiStore()
    builder = GraphitiSemanticBuilder(llm=FakeLLMTemporal(), embedder=FakeEmbedder(), store=store)

    # Active ongoing fact from 2026
    t1 = "2026-01-01T00:00:00+00:00"
    r1 = builder.ingest_user_turn(
        session_id="s1",
        episode_id="ep1",
        previous_turns=[{"role": "assistant", "content": "你好"}],
        current_turn={"role": "user", "content": "我叫张三，在 Google 工作"},
        reference_timestamp=t1,
    )
    assert r1["facts_added"] == 1
    f1_id = next(iter(store.facts.keys()))
    assert not store.facts[f1_id].get("t_valid_to")

    # Past fact window in 2025 should not expire the active 2026 one.
    t2 = "2026-03-01T00:00:00+00:00"
    r2 = builder.ingest_user_turn(
        session_id="s1",
        episode_id="ep2",
        previous_turns=[{"role": "assistant", "content": "好的"}],
        current_turn={"role": "user", "content": "我叫张三，曾经在 OpenAI 工作"},
        reference_timestamp=t2,
    )
    assert r2["facts_added"] == 1
    assert r2["facts_expired"] == 0
    assert store.facts[f1_id].get("t_valid_to") in (None, "")
