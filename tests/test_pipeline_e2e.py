"""Pipeline E2E 测试。

用 Fake LLM 和 Fake Embedder 完整跑通整个 LangGraph Pipeline。
流水线拓扑：wm_manager → search → synthesize → reason
"""

from a_frame.core.factory import create_pipeline
from a_frame.memory.graph.graphiti_engine import GraphitiSearchResult


class FakeLLMForPipeline:
    """模拟完整 Pipeline 中各 Agent 的 LLM 调用。

    根据 system prompt 关键词判断当前在哪个 Agent，返回对应的假响应。
    """

    def __init__(self):
        self.call_log: list[str] = []

    def generate(self, messages, **kwargs):
        system_msg = ""
        user_msg = ""
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"]
            elif m["role"] == "user":
                user_msg += m["content"]

        # 整合器（放在压缩器之前，因为 SYNTHESIS_SYSTEM_PROMPT 包含 "压缩" 子串）
        if (
            "记忆整合与法官" in system_msg
            or "Memory Synthesizer" in system_msg
            or "scored_fragments" in system_msg
        ):
            self.call_log.append("synthesizer")
            return '{"kept_count": 1, "dropped_count": 0, "background_context": "张三在 Google 工作。"}'

        # 压缩器
        if "压缩" in system_msg or "状态交接" in system_msg:
            self.call_log.append("compressor")
            return "## Intent Mapping\n测试\n## Progress Assessment\n完成\n## Recent Commands Analysis\n无"

        # HyDE (检索协调器)
        if "HyDE" in system_msg or "假设性回答" in system_msg or "锚点文本" in system_msg:
            self.call_log.append("hyde")
            return "这是一个关于查询的假设回答"

        # 检索规划
        if "检索规划" in system_msg or "sub_queries" in system_msg:
            self.call_log.append("planner")
            return '{"sub_queries": [{"source": "graph", "query": "子查询"}], "reasoning": "test"}'

        # 固化器
        if "固化" in system_msg or "记忆巩固" in system_msg or "new_skills" in system_msg:
            self.call_log.append("consolidator")
            return '{"new_skills": [], "should_extract_entities": false}'

        # 实体抽取
        if "实体抽取" in system_msg or "知识图谱实体" in system_msg:
            self.call_log.append("entity_extractor")
            return "[]"

        # 默认：推理器（最终回复）
        self.call_log.append("reasoner")
        return "这是 AI 的回复。"


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 8


class TestPipelineE2E:
    """测试完整 Pipeline 流程。"""

    def test_basic_response(self):
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder(), max_tokens=100_000)

        result = pipeline.run(user_query="你好", session_id="test-session-1")

        assert "final_response" in result
        assert len(result["final_response"]) > 0
        assert "reasoner" in llm.call_log

    def test_always_searches(self):
        """Pipeline 每次都走 search → synthesize → reason。"""
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        result = pipeline.run(user_query="你好", session_id="s1")

        assert "retrieved_graph_memories" in result
        assert "retrieved_skills" in result
        assert "background_context" in result
        # Synthesizer 节点会运行，但当检索片段为空时会直接返回空 context（不触发 LLM 调用）。
        assert "reasoner" in llm.call_log

    def test_session_persistence(self):
        """测试多轮对话的会话持久化。"""
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        r1 = pipeline.run(user_query="你好", session_id="s1")
        assert r1["final_response"]

        r2 = pipeline.run(user_query="继续", session_id="s1")
        assert r2["final_response"]

        turns = pipeline.wm_manager.session_store.get_turns("s1")
        assert len(turns) >= 4  # 2轮 × (user + assistant)

    def test_separate_sessions(self):
        """测试不同 session 互不干扰。"""
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        pipeline.run(user_query="会话A的消息", session_id="sa")
        pipeline.run(user_query="会话B的消息", session_id="sb")

        turns_a = pipeline.wm_manager.session_store.get_turns("sa")
        turns_b = pipeline.wm_manager.session_store.get_turns("sb")
        assert len(turns_a) == 2  # user + assistant
        assert len(turns_b) == 2


class TestPipelineWithGraphData:
    """测试图谱检索路径。"""

    def test_graph_retrieval(self):
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        pipeline.search_coordinator.graph_store.add(
            [
                {
                    "subject": {"name": "张三", "label": "Person"},
                    "predicate": "works_at",
                    "object": {"name": "Google", "label": "Organization"},
                }
            ]
        )

        result = pipeline.run(user_query="张三在哪里工作？", session_id="s-retrieval")

        assert "background_context" in result
        assert result["final_response"]
        assert len(result.get("retrieved_graph_memories", [])) >= 1
        assert "synthesizer" in llm.call_log
        assert "reasoner" in llm.call_log


class TestPipelineWithGraphitiSearch:
    """在不依赖 Neo4j 的情况下验证 Graphiti constructor context 能被 pipeline 消费。"""

    def test_graphiti_constructor_context_flow(self):
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        class FakeGraphitiEngine:
            def search(self, **kwargs):
                return GraphitiSearchResult(
                    context="<FACTS>\n- 张三在 Google 工作 (Date range: null - null)\n</FACTS>\n<ENTITIES>\n- 张三\n- Google\n</ENTITIES>\n",
                    provenance=[{"fact_id": "f1", "mentions": True, "distance": 1}],
                )

        pipeline.search_coordinator.graphiti_engine = FakeGraphitiEngine()

        result = pipeline.run(user_query="张三在哪里工作？", session_id="s-graphiti")
        assert result.get("background_context")
        assert result.get("final_response")
        assert "synthesizer" in llm.call_log
        assert "reasoner" in llm.call_log


class TestPipelineConsolidation:
    """测试固化 Agent 被触发。"""

    def test_consolidation_triggered(self):
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        result = pipeline.run(user_query="帮我做个任务", session_id="s-consolidate")

        assert result.get("consolidation_pending") is True
        assert "consolidator" in llm.call_log


class TestPipelineStateFields:
    """测试 Pipeline 返回的状态包含所有预期字段。"""

    def test_state_fields(self):
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        result = pipeline.run(user_query="你好", session_id="s-fields")

        assert "user_query" in result
        assert "session_id" in result
        assert "compressed_history" in result
        assert "wm_token_usage" in result
        assert "retrieved_graph_memories" in result
        assert "retrieved_skills" in result
        assert "background_context" in result
        assert "final_response" in result
