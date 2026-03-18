"""PR3: ConsolidatorAgent 在 Graphiti 模式下的增量/游标逻辑。

验证：
- Episode raw ingestion 只写入新增 turns（幂等）
- 语义抽取只对“最新 user turn”执行一次（幂等）
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.consolidator_agent import ConsolidatorAgent
from src.memory.graph.entity_extractor import EntityExtractor
from src.memory.graph.graph_store import NetworkXGraphStore
from src.memory.procedural.skill_store import InMemorySkillStore


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


class FakeLLMGraphitiAndConsolidate:
    def generate(self, messages, **kwargs):
        system_msg = messages[0]["content"] if messages else ""

        # Consolidation analysis
        if "记忆固化专家" in system_msg:
            return '{"new_skills": [], "should_extract_entities": false}'

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
        self.episode_entity_links: list[tuple[str, str]] = []

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
            "embedding": embedding,
            "t_valid_from": t_valid_from,
            "t_valid_to": t_valid_to,
            "t_tx_created": t_tx_created,
        }

    def link_episode_to_fact(self, *, episode_id: str, fact_id: str):
        self.link_calls += 1
        self.episode_fact_links.append((episode_id, fact_id))
        
    def link_episode_to_entity(self, *, episode_id: str, entity_id: str) -> None:
        # Paper: episodic edges Episode→Entity
        self.link_calls += 1
        self.episode_entity_links.append((episode_id, entity_id))


class FakeGraphitiEngine:
    def __init__(self, store: FakeGraphitiStore):
        self.store = store
        self.episodes: list[dict[str, Any]] = []

    def ingest_episode(
        self,
        *,
        session_id: str,
        turn_index: int,
        role: str,
        content: str,
        t_ref: str,
        episode_id: str,
    ):
        self.episodes.append(
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "role": role,
                "content": content,
                "t_ref": t_ref,
                "episode_id": episode_id,
            }
        )


def test_consolidator_graphiti_incremental_only_latest_user_turn():
    llm = FakeLLMGraphitiAndConsolidate()
    embedder = FakeEmbedder()

    store = FakeGraphitiStore()
    graphiti_engine = FakeGraphitiEngine(store)

    extractor = EntityExtractor(llm=llm, confidence_threshold=0.5)
    agent = ConsolidatorAgent(
        llm=llm,
        embedder=embedder,
        graph_store=NetworkXGraphStore(),
        skill_store=InMemorySkillStore(),
        entity_extractor=extractor,
        graphiti_engine=graphiti_engine,
    )

    turns = [
        {"role": "user", "content": "你好", "created_at": "2026-01-01T00:00:00+00:00"},
        {"role": "assistant", "content": "你好！", "created_at": "2026-01-01T00:00:01+00:00"},
        {
            "role": "user",
            "content": "我叫张三，在 Google 工作",
            "created_at": "2026-01-01T00:00:02+00:00",
        },
    ]

    result1 = agent.run(turns=turns, session_id="s1")
    assert result1["entities_added"] == 2
    assert result1["facts_added"] == 1

    # Episodes ingested for all turns
    assert len(graphiti_engine.episodes) == 3

    # Semantic builder ran once for latest user turn
    assert store.upsert_entity_calls == 2
    assert store.upsert_fact_calls == 1
    # Paper: Episode→Entity (2) + Episode→Fact (1)
    assert store.link_calls == 3

    # Second run should be idempotent (no new episodes, no new semantic build)
    result2 = agent.run(turns=turns, session_id="s1")
    assert result2["entities_added"] == 0
    assert result2["facts_added"] == 0

    assert len(graphiti_engine.episodes) == 3
    assert store.upsert_entity_calls == 2
    assert store.upsert_fact_calls == 1
    assert store.link_calls == 3
