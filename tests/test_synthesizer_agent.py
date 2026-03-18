"""测试整合排序器。"""

from src.agents.synthesizer_agent import SynthesizerAgent


class FakeLLMSynthesizer:
    def generate(self, messages, **kwargs):
        return '{"kept_count": 2, "dropped_count": 0, "background_context": "张三在 Google 工作，住在北京。"}'


class FakeLLMBroken:
    def generate(self, messages, **kwargs):
        return "这是一段自然语言的整合结果，不是 JSON"


class TestSynthesizerAgent:
    def test_synthesize_with_graph_memories(self):
        synth = SynthesizerAgent(llm=FakeLLMSynthesizer())
        result = synth.run(
            user_query="张三的信息",
            retrieved_graph_memories=[
                {
                    "anchor": {"node_id": "张三", "label": "Person"},
                    "subgraph": {
                        "nodes": [{"id": "张三"}, {"id": "Google"}],
                        "edges": [{"source": "张三", "target": "Google", "relation": "works_at"}],
                    },
                }
            ],
        )
        assert "background_context" in result
        assert "张三" in result["background_context"]

    def test_synthesize_empty_input(self):
        synth = SynthesizerAgent(llm=FakeLLMSynthesizer())
        result = synth.run(user_query="你好")
        assert result["background_context"] == ""

    def test_synthesize_broken_json_fallback(self):
        synth = SynthesizerAgent(llm=FakeLLMBroken())
        result = synth.run(
            user_query="测试",
            retrieved_skills=[
                {"intent": "测试技能", "doc_markdown": "# 测试\n", "reusable": False}
            ],
        )
        # 应该 fallback 到原始 LLM 输出
        assert "background_context" in result
        assert len(result["background_context"]) > 0

    def test_format_fragments_skills(self):
        fragments = SynthesizerAgent._format_fragments(
            graph_memories=[],
            skills=[{"intent": "写爬虫", "doc_markdown": "# 写爬虫\n\n1. GET\n"}],
        )
        assert "[技能库]" in fragments
        assert "写爬虫" in fragments

    def test_format_fragments_mixed(self):
        fragments = SynthesizerAgent._format_fragments(
            graph_memories=[
                {
                    "anchor": {"node_id": "A"},
                    "subgraph": {
                        "nodes": [],
                        "edges": [{"source": "A", "target": "B", "relation": "r"}],
                    },
                }
            ],
            skills=[{"intent": "t", "doc_markdown": "# t\n"}],
        )
        assert "[知识图谱]" in fragments
        assert "[技能库]" in fragments

    def test_format_graph_edges_include_fact(self):
        fragments = SynthesizerAgent._format_fragments(
            graph_memories=[
                {
                    "anchor": {"node_id": "张三"},
                    "subgraph": {
                        "nodes": [],
                        "edges": [
                            {
                                "source": "张三",
                                "target": "Google",
                                "relation": "works_at",
                                "fact": "张三在 Google 工作",
                                "evidence": "user: 张三在哪工作？ assistant: 张三在 Google 工作",
                            }
                        ],
                    },
                }
            ],
            skills=[],
        )
        assert "事实: 张三在 Google 工作" in fragments

    def test_format_graph_memories_include_constructed_context(self):
        fragments = SynthesizerAgent._format_fragments(
            graph_memories=[
                {
                    "anchor": {"node_id": "graphiti_context"},
                    "subgraph": {"nodes": [], "edges": []},
                    "constructed_context": "<FACTS>\n- A r B (Date range: null - null)\n</FACTS>\n<ENTITIES>\n- A\n- B\n</ENTITIES>\n",
                }
            ],
            skills=[],
        )
        assert "构造上下文:" in fragments
        assert "<FACTS>" in fragments
