"""Pipeline E2E 测试。

用 Fake LLM 和 Fake Embedder 完整跑通整个 LangGraph Pipeline。
测试两条路径：
1. 需要检索的分支 (router → search → synthesize → reason)
2. 直接回答的分支 (router → reason)
"""

from a_frame.core.factory import create_pipeline


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

        # 路由器
        if "路由分析" in system_msg or "need_graph" in system_msg:
            self.call_log.append("router")
            # 根据用户查询决定路由
            if "张三" in user_msg or "实体" in user_msg:
                return '{"need_graph": true, "need_skills": false, "need_sensory": false, "reasoning": "实体查询"}'
            return '{"need_graph": false, "need_skills": false, "need_sensory": false, "reasoning": "简单对话"}'

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
        if "假设文档" in system_msg or "锚点文本" in system_msg:
            self.call_log.append("hyde")
            return "这是一个关于查询的假设回答"

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


class TestPipelineE2EDirectAnswer:
    """测试直接回答路径：router 判断不需要检索 → 直接 reason。"""

    def test_simple_greeting(self):
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder(), max_tokens=100_000)

        result = pipeline.run(user_query="你好", session_id="test-session-1")

        # 验证有最终回复
        assert "final_response" in result
        assert len(result["final_response"]) > 0

        # 验证路由决策为不检索
        assert result["route"]["need_graph"] is False
        assert result["route"]["need_skills"] is False

        # 验证调用链：router → reasoner（无 search/synthesize）
        assert "router" in llm.call_log
        assert "reasoner" in llm.call_log
        assert "hyde" not in llm.call_log  # 未走检索分支

    def test_session_persistence(self):
        """测试多轮对话的会话持久化。"""
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        # 第一轮
        r1 = pipeline.run(user_query="你好", session_id="s1")
        assert r1["final_response"]

        # 第二轮（同一session）
        r2 = pipeline.run(user_query="继续", session_id="s1")
        assert r2["final_response"]

        # 验证会话日志中有多轮
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


class TestPipelineE2EWithRetrieval:
    """测试需要检索的路径：router → search → synthesize → reason。"""

    def test_entity_query_triggers_retrieval(self):
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        # 预填充图谱数据
        pipeline.search_coordinator.graph_store.add([{
            "subject": {"name": "张三", "label": "Person"},
            "predicate": "works_at",
            "object": {"name": "Google", "label": "Organization"},
        }])

        result = pipeline.run(user_query="张三在哪里工作？", session_id="s-retrieval")

        # 验证走了检索路径
        assert result["route"]["need_graph"] is True
        assert "background_context" in result
        assert result["final_response"]

        # 验证图谱检索结果非空（双向匹配：节点名 "张三" 出现在查询中）
        assert len(result.get("retrieved_graph_memories", [])) >= 1

        # 验证调用链包含 router 和 reasoner
        assert "router" in llm.call_log
        assert "synthesizer" in llm.call_log
        assert "reasoner" in llm.call_log


class TestPipelineSensoryBuffer:
    """测试感觉缓冲区的更新。"""

    def test_user_input_pushed_to_sensory(self):
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        pipeline.run(user_query="第一条消息", session_id="sb-test")
        pipeline.run(user_query="第二条消息", session_id="sb-test")

        items = pipeline.sensory_buffer.get_recent()
        assert len(items) == 2
        assert items[0].content == "第一条消息"
        assert items[1].content == "第二条消息"


class TestPipelineConsolidation:
    """测试固化 Agent 被触发。"""

    def test_consolidation_triggered(self):
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        result = pipeline.run(user_query="帮我做个任务", session_id="s-consolidate")

        # consolidation_pending 应为 True
        assert result.get("consolidation_pending") is True
        # 固化器应该被调用
        assert "consolidator" in llm.call_log


class TestPipelineStateFields:
    """测试 Pipeline 返回的状态包含所有预期字段。"""

    def test_direct_answer_state_fields(self):
        llm = FakeLLMForPipeline()
        pipeline = create_pipeline(llm=llm, embedder=FakeEmbedder())

        result = pipeline.run(user_query="你好", session_id="s-fields")

        # 必须包含的核心字段
        assert "user_query" in result
        assert "session_id" in result
        assert "compressed_history" in result
        assert "wm_token_usage" in result
        assert "route" in result
        assert "final_response" in result
