"""测试检索协调器。"""

from a_frame.agents.search_coordinator import SearchCoordinator
from a_frame.memory.graph.graph_store import NetworkXGraphStore
from a_frame.memory.procedural.skill_store import InMemorySkillStore


class FakeLLM:
    """返回 HyDE 假设文档的假 LLM。"""

    def generate(self, messages, **kwargs):
        return "这是一个关于查询的假设性回答文档。"


class FakeEmbedder:
    """返回固定 embedding 的假 Embedder。"""

    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 8


class TestSearchCoordinator:
    def setup_method(self):
        self.graph_store = NetworkXGraphStore()
        self.skill_store = InMemorySkillStore()

        self.coordinator = SearchCoordinator(
            llm=FakeLLM(),
            embedder=FakeEmbedder(),
            graph_store=self.graph_store,
            skill_store=self.skill_store,
        )

    def test_empty_retrieval(self):
        result = self.coordinator.run(user_query="你好")
        assert result["retrieved_graph_memories"] == []
        assert result["retrieved_skills"] == []

    def test_graph_retrieval(self):
        self.graph_store.add([{
            "subject": {"name": "张三", "label": "Person"},
            "predicate": "works_at",
            "object": {"name": "Google", "label": "Organization"},
        }])

        result = self.coordinator.run(user_query="张三")
        assert len(result["retrieved_graph_memories"]) >= 1

    def test_graph_retrieval_empty(self):
        result = self.coordinator.run(user_query="不存在的实体")
        assert result["retrieved_graph_memories"] == []

    def test_skill_retrieval(self):
        self.skill_store.add([{
            "intent": "写一个爬虫",
            "embedding": [0.1] * 8,
            "doc_markdown": "# 写一个爬虫\n\n1. requests.get\n",
        }])

        result = self.coordinator.run(user_query="帮我写爬虫")
        assert len(result["retrieved_skills"]) >= 1

    def test_multi_source_retrieval(self):
        self.graph_store.add([{
            "subject": {"name": "测试", "label": "Concept"},
            "predicate": "is_a",
            "object": {"name": "实验", "label": "Concept"},
        }])

        result = self.coordinator.run(user_query="测试")
        assert len(result["retrieved_graph_memories"]) >= 1
        assert "retrieved_skills" in result
