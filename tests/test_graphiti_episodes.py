"""PR2: Graphiti Episode ingestion tests.

这些测试不依赖真实 Neo4j：
- 验证 Cypher 语句稳定（字符串级别）
- 验证 Engine/Consolidator 的调用链在启用时会写入 Episode
"""

from src.agents.consolidator_agent import ConsolidatorAgent
from src.memory.graph.entity_extractor import EntityExtractor
from src.memory.graph.graph_store import NetworkXGraphStore
from src.memory.graph.graphiti_engine import GraphitiEngine
from src.memory.graph.graphiti_neo4j_store import GraphitiNeo4jStore
from src.memory.procedural.skill_store import InMemorySkillStore


class FakeLLMNoAction:
    def generate(self, messages, **kwargs):
        system_msg = messages[0]["content"] if messages else ""
        if "实体抽取" in system_msg or "知识图谱" in system_msg:
            return "[]"
        return '{"new_skills": [], "should_extract_entities": false}'


class FakeEmbedder:
    def embed(self, texts):
        return [[0.5] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.5] * 8


def test_episode_upsert_cypher_contains_merge_and_fields():
    cypher = GraphitiNeo4jStore.episode_upsert_cypher()
    assert "MERGE (e:Episode {episode_id: $episode_id})" in cypher
    assert "e.session_id" in cypher
    assert "e.turn_index" in cypher
    assert "e.t_ref" in cypher


def test_graphiti_engine_ingest_episode_delegates_to_store():
    calls = []

    class FakeStore:
        def upsert_episode(self, **kwargs):
            calls.append(kwargs)
            return kwargs["episode_id"]

    engine = GraphitiEngine(store=FakeStore())
    eid = engine.ingest_episode(
        session_id="s1", turn_index=0, role="user", content="hi", t_ref="t0"
    )
    assert eid
    assert len(calls) == 1
    assert calls[0]["session_id"] == "s1"
    assert calls[0]["turn_index"] == 0


def test_consolidator_ingests_episodes_when_graphiti_engine_provided():
    ingested = []

    class FakeGraphitiEngine:
        def ingest_episode(self, **kwargs):
            ingested.append(kwargs)
            return kwargs["episode_id"]

    llm = FakeLLMNoAction()
    extractor = EntityExtractor(llm=llm)

    agent = ConsolidatorAgent(
        llm=llm,
        embedder=FakeEmbedder(),
        graph_store=NetworkXGraphStore(),
        skill_store=InMemorySkillStore(),
        entity_extractor=extractor,
        graphiti_engine=FakeGraphitiEngine(),
    )

    turns = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]

    result = agent.run(turns=turns, session_id="sess-1")
    assert result["entities_added"] == 0
    assert result["skills_added"] == 0
    assert len(ingested) == 2
    assert ingested[0]["session_id"] == "sess-1"
    assert ingested[0]["turn_index"] == 0
    assert ingested[0]["role"] == "user"
