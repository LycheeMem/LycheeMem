"""测试记忆固化 Agent。"""

from a_frame.agents.consolidator_agent import ConsolidatorAgent
from a_frame.memory.graph.entity_extractor import EntityExtractor
from a_frame.memory.graph.graph_store import NetworkXGraphStore
from a_frame.memory.procedural.skill_store import InMemorySkillStore


class FakeLLMConsolidate:
    """返回固化分析结果的假 LLM。"""

    def generate(self, messages, **kwargs):
        # 检查 system prompt 判断是固化还是实体抽取调用
        system_msg = messages[0]["content"] if messages else ""
        if "实体抽取" in system_msg or "知识图谱" in system_msg:
            return """[
  {"subject": {"name": "张三", "label": "Person"}, "predicate": "works_at", "object": {"name": "Google", "label": "Organization"}, "confidence": 0.9}
]"""
        return """{
  "new_skills": [
    {"intent": "用 Python 爬取网页", "tool_chain": [{"step": 1, "action": "requests.get", "details": "获取网页内容"}]}
  ],
  "should_extract_entities": true
}"""


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


class TestConsolidatorAgent:
    def setup_method(self):
        self.graph_store = NetworkXGraphStore()
        self.skill_store = InMemorySkillStore()

    def test_consolidate_with_new_skill_and_entities(self):
        llm = FakeLLMConsolidate()
        extractor = EntityExtractor(llm=llm, confidence_threshold=0.5)
        agent = ConsolidatorAgent(
            llm=llm,
            embedder=FakeEmbedder(),
            graph_store=self.graph_store,
            skill_store=self.skill_store,
            entity_extractor=extractor,
        )

        turns = [
            {"role": "user", "content": "帮我用 Python 爬取 example.com"},
            {"role": "assistant", "content": "好的，使用 requests 库..."},
        ]
        result = agent.run(turns=turns)

        assert result["skills_added"] == 1
        assert result["entities_added"] >= 1
        # 验证技能库确实有数据
        assert len(self.skill_store.get_all()) == 1

    def test_consolidate_no_action(self):
        llm = FakeLLMNoAction()
        extractor = EntityExtractor(llm=llm, confidence_threshold=0.5)
        agent = ConsolidatorAgent(
            llm=llm,
            embedder=FakeEmbedder(),
            graph_store=self.graph_store,
            skill_store=self.skill_store,
            entity_extractor=extractor,
        )

        turns = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        result = agent.run(turns=turns)

        assert result["skills_added"] == 0
        assert result["entities_added"] == 0

    def test_consolidate_empty_turns(self):
        llm = FakeLLMNoAction()
        extractor = EntityExtractor(llm=llm)
        agent = ConsolidatorAgent(
            llm=llm,
            embedder=FakeEmbedder(),
            graph_store=self.graph_store,
            skill_store=self.skill_store,
            entity_extractor=extractor,
        )
        result = agent.run(turns=[])
        assert result["entities_added"] == 0
        assert result["skills_added"] == 0
