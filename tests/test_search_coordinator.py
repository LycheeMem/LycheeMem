"""测试检索协调器。"""

from a_frame.agents.search_coordinator import SearchCoordinator
from a_frame.memory.graph.graph_store import NetworkXGraphStore
from a_frame.memory.procedural.skill_store import InMemorySkillStore
from a_frame.memory.sensory.buffer import SensoryBuffer


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
        self.sensory_buffer = SensoryBuffer(max_size=10)

        self.coordinator = SearchCoordinator(
            llm=FakeLLM(),
            embedder=FakeEmbedder(),
            graph_store=self.graph_store,
            skill_store=self.skill_store,
            sensory_buffer=self.sensory_buffer,
        )

    def test_no_retrieval(self):
        route = {"need_graph": False, "need_skills": False, "need_sensory": False}
        result = self.coordinator.run(user_query="你好", route=route)
        assert result["retrieved_graph_memories"] == []
        assert result["retrieved_skills"] == []
        assert result["retrieved_sensory"] == []

    def test_graph_retrieval(self):
        # 添加测试数据
        self.graph_store.add([{
            "subject": {"name": "张三", "label": "Person"},
            "predicate": "works_at",
            "object": {"name": "Google", "label": "Organization"},
        }])

        route = {"need_graph": True, "need_skills": False, "need_sensory": False}
        result = self.coordinator.run(user_query="张三", route=route)
        assert len(result["retrieved_graph_memories"]) >= 1

    def test_graph_retrieval_empty(self):
        route = {"need_graph": True, "need_skills": False, "need_sensory": False}
        result = self.coordinator.run(user_query="不存在的实体", route=route)
        assert result["retrieved_graph_memories"] == []

    def test_skill_retrieval(self):
        # 添加测试技能
        self.skill_store.add([{
            "intent": "写一个爬虫",
            "embedding": [0.1] * 8,
            "tool_chain": [{"step": 1, "action": "requests.get"}],
        }])

        route = {"need_graph": False, "need_skills": True, "need_sensory": False}
        result = self.coordinator.run(user_query="帮我写爬虫", route=route)
        assert len(result["retrieved_skills"]) >= 1

    def test_sensory_retrieval(self):
        self.sensory_buffer.push("最近的输入1")
        self.sensory_buffer.push("最近的输入2")

        route = {"need_graph": False, "need_skills": False, "need_sensory": True}
        result = self.coordinator.run(user_query="查看最近", route=route)
        assert len(result["retrieved_sensory"]) == 2

    def test_multi_source_retrieval(self):
        # 添加图谱数据
        self.graph_store.add([{
            "subject": {"name": "测试", "label": "Concept"},
            "predicate": "is_a",
            "object": {"name": "实验", "label": "Concept"},
        }])
        # 添加感觉数据
        self.sensory_buffer.push("感觉记忆条目")

        route = {"need_graph": True, "need_skills": False, "need_sensory": True}
        result = self.coordinator.run(user_query="测试", route=route)
        assert len(result["retrieved_graph_memories"]) >= 1
        assert len(result["retrieved_sensory"]) >= 1
