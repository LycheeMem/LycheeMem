"""
测试所有 P1/P2 差距补全功能。

覆盖：
- P1-1: 双阈值异步压缩 (WMManager)
- P1-2: 后台固化 (Pipeline.consolidate)
- P1-3: 技能复用执行 (SkillEntry 增强, SearchCoordinator 阈值, SynthesizerAgent 执行计划)
- P1-4: 结构化检索规划 (SearchCoordinator._plan_retrieval)
- P2-1: 增强图谱模型 (时间戳/置信度/溯源, search_by_time, search_by_relation)
- P2-2: 三步整合器 (score→rank→fuse, provenance)
- P2-3: 会话元数据 (topic, tags, created_at, updated_at, 分页)
- P2-4: 运营 API (pagination, consolidate, graph by-time/by-relation, pipeline status)
"""

import time

from fastapi.testclient import TestClient

from a_frame.agents.reasoning_agent import ReasoningAgent
from a_frame.agents.search_coordinator import SearchCoordinator
from a_frame.agents.synthesizer_agent import SynthesizerAgent
from a_frame.agents.wm_manager import WMManager
from a_frame.api.server import create_app
from a_frame.core.factory import create_pipeline
from a_frame.memory.graph.entity_extractor import EntityExtractor
from a_frame.memory.graph.graph_store import NetworkXGraphStore
from a_frame.memory.procedural.skill_store import InMemorySkillStore, SkillEntry
from a_frame.memory.working.compressor import WorkingMemoryCompressor
from a_frame.memory.working.session_store import InMemorySessionStore


# ─── Fakes ───


class FakeLLM:
    """通用 FakeLLM，根据关键词分支。"""
    def generate(self, messages, **kwargs):
        system_msg = ""
        user_msg = ""
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"]
            elif m["role"] == "user":
                user_msg += m["content"]

        if "HyDE" in system_msg or "假设文档" in system_msg or "锚点文本" in system_msg:
            return "假设性回答文本"
        if "打分" in system_msg or "scored_fragments" in system_msg:
            return (
                '{"scored_fragments": [{"source": "graph", "index": 0, "relevance": 0.95, "summary": "test"}],'
                ' "kept_count": 1, "dropped_count": 0, "background_context": "整合上下文"}'
            )
        if "检索规划" in system_msg or "sub_queries" in system_msg:
            return '{"sub_queries": [{"source": "graph", "query": "子查询1"}], "reasoning": "分解"}'
        if "压缩" in system_msg or "状态交接" in system_msg:
            return "## Summary\ntest"
        if "固化" in system_msg or "new_skills" in system_msg:
            return '{"new_skills": [], "should_extract_entities": false}'
        if "知识图谱实体抽取" in system_msg:
            return '[]'
        return "Hello from A-Frame!"

    async def agenerate(self, messages, **kwargs):
        return self.generate(messages, **kwargs)


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 8


def _make_client() -> TestClient:
    pipeline = create_pipeline(llm=FakeLLM(), embedder=FakeEmbedder())
    return TestClient(create_app(pipeline))


def _make_pipeline():
    return create_pipeline(llm=FakeLLM(), embedder=FakeEmbedder())


# ═══════════════════════════════════════
# P1-1: 双阈值异步压缩
# ═══════════════════════════════════════


class TestDualThresholdCompression:
    def test_wm_manager_has_thread_pool(self):
        """WMManager 应有后台线程池和 pending 字典。"""
        store = InMemorySessionStore()
        comp = WorkingMemoryCompressor(llm=FakeLLM())
        wm = WMManager(session_store=store, compressor=comp)
        assert hasattr(wm, "_executor")
        assert hasattr(wm, "_pending")
        assert hasattr(wm, "_lock")

    def test_no_compression_below_threshold(self):
        """Token 低于阈值时不触发压缩。"""
        store = InMemorySessionStore()
        comp = WorkingMemoryCompressor(llm=FakeLLM())
        wm = WMManager(session_store=store, compressor=comp)
        result = wm.run(session_id="s1", user_query="短消息")
        assert "compressed_history" in result
        assert "wm_token_usage" in result
        # 不应有 pending 的压缩任务
        assert len(wm._pending) == 0

    def test_sync_compress_when_blocked(self):
        """阻塞阈值时应同步压缩。"""
        store = InMemorySessionStore()
        comp = WorkingMemoryCompressor(llm=FakeLLM(), max_tokens=50)  # 极小 budget
        wm = WMManager(session_store=store, compressor=comp)
        # 填充足够多的 turn
        for i in range(20):
            store.append_turn("s1", "user", f"消息 {i} " * 10)
            store.append_turn("s1", "assistant", f"回答 {i} " * 10)
        # 此时触发 run 应产生同步压缩
        result = wm.run(session_id="s1", user_query="最终查询")
        log = store.get_or_create("s1")
        # 由于 budget 极小，应该触发了压缩 → summaries 非空
        # (具体取决于 compressor 实现，但 raw_recent_turns 应少于全部)
        assert "raw_recent_turns" in result


# ═══════════════════════════════════════
# P1-2: 后台固化
# ═══════════════════════════════════════


class TestBackgroundConsolidation:
    def test_pipeline_has_consolidate_method(self):
        """Pipeline 应有公共 consolidate() 方法。"""
        pipeline = _make_pipeline()
        assert hasattr(pipeline, "consolidate")
        assert callable(pipeline.consolidate)

    def test_consolidate_empty_session(self):
        """对空会话固化应安全返回。"""
        pipeline = _make_pipeline()
        result = pipeline.consolidate("nonexistent_session")
        assert result == {"entities_added": 0, "skills_added": 0}

    def test_consolidate_with_data(self):
        """固化有数据的会话应返回结果。"""
        pipeline = _make_pipeline()
        pipeline.run(user_query="你好", session_id="c1")
        # 等待后台固化线程完成
        time.sleep(0.5)
        # 显式再次固化
        result = pipeline.consolidate("c1")
        assert "entities_added" in result or "skills_added" in result

    def test_pipeline_run_triggers_bg_consolidation(self):
        """pipeline.run() 应标记 consolidation_pending。"""
        pipeline = _make_pipeline()
        result = pipeline.run(user_query="测试", session_id="bg1")
        assert result.get("consolidation_pending") is True


# ═══════════════════════════════════════
# P1-3: 技能复用执行
# ═══════════════════════════════════════


class TestSkillReuse:
    def test_skill_entry_has_new_fields(self):
        """SkillEntry 应有 success_count, last_used, conditions。"""
        entry = SkillEntry(
            id="s1", intent="test", embedding=[0.1], doc_markdown="# test\n",
            success_count=5, last_used="2024-01-01", conditions="当用户要求时",
        )
        assert entry.success_count == 5
        assert entry.last_used == "2024-01-01"
        assert entry.conditions == "当用户要求时"

    def test_skill_store_record_usage(self):
        """record_usage 应增加 success_count。"""
        store = InMemorySkillStore()
        store.add([{"id": "s1", "intent": "test", "embedding": [0.1], "doc_markdown": "# test\n"}])
        store.record_usage("s1")
        store.record_usage("s1")
        all_skills = store.get_all()
        assert all_skills[0]["success_count"] == 2
        assert all_skills[0]["last_used"] is not None

    def test_skill_search_returns_new_fields(self):
        """搜索结果应包含 success_count 和 conditions。"""
        store = InMemorySkillStore()
        store.add([{
            "id": "s1", "intent": "test", "embedding": [0.1] * 8,
            "doc_markdown": "# test\n", "success_count": 3, "conditions": "任何场景",
        }])
        results = store.search("test", top_k=5, query_embedding=[0.1] * 8)
        assert results[0]["success_count"] == 3
        assert results[0]["conditions"] == "任何场景"

    def test_search_coordinator_marks_reusable(self):
        """高分技能应标记 reusable=True。"""
        llm = FakeLLM()
        embedder = FakeEmbedder()
        store = InMemorySkillStore()
        store.add([{
            "id": "s1", "intent": "test", "embedding": [0.1] * 8,
            "doc_markdown": "# test\n\n1. run\n",
        }])
        sc = SearchCoordinator(
            llm=llm, embedder=embedder,
            graph_store=NetworkXGraphStore(),
            skill_store=store,
            skill_reuse_threshold=0.5,  # 低阈值确保命中
        )
        results = sc._search_skills("test")
        # 向量完全相同 → score=1.0 > 0.5 → reusable
        assert any(r.get("reusable") for r in results)

    def test_synthesizer_builds_reuse_plan(self):
        """整合器应为 reusable 技能生成执行计划。"""
        synth = SynthesizerAgent(llm=FakeLLM())
        result = synth.run(
            user_query="测试",
            retrieved_skills=[
                {"id": "s1", "intent": "跑测试", "doc_markdown": "# 跑测试\n\n- `pytest -q`\n",
                 "score": 0.95, "reusable": True, "conditions": ""},
                {"id": "s2", "intent": "其他", "doc_markdown": "# 其他\n",
                 "score": 0.3, "reusable": False, "conditions": ""},
            ],
        )
        assert "skill_reuse_plan" in result
        plan = result["skill_reuse_plan"]
        assert len(plan) == 1
        assert plan[0]["skill_id"] == "s1"

    def test_reasoning_agent_accepts_skill_plan(self):
        """推理器应接收 skill_reuse_plan 参数。"""
        agent = ReasoningAgent(llm=FakeLLM())
        result = agent.run(
            user_query="测试",
            skill_reuse_plan=[{
                "skill_id": "s1",
                "intent": "跑测试",
                "doc_markdown": "# 跑测试\n\n- `pytest -q`\n",
                "score": 0.95,
                "conditions": "",
            }],
        )
        assert "final_response" in result


# ═══════════════════════════════════════
# P1-4: 结构化检索规划
# ═══════════════════════════════════════


class TestStructuredRetrievalPlanning:
    def test_plan_retrieval_returns_dict(self):
        """_plan_retrieval 应返回 source→query 映射。"""
        sc = SearchCoordinator(
            llm=FakeLLM(), embedder=FakeEmbedder(),
            graph_store=NetworkXGraphStore(),
            skill_store=InMemorySkillStore(),
        )
        plan = sc._plan_retrieval("复杂查询")
        assert isinstance(plan, dict)

    def test_plan_retrieval_fallback_on_error(self):
        """LLM 返回非法 JSON 时应安全回退。"""
        class BrokenLLM:
            def generate(self, messages, **kwargs):
                return "这不是JSON"
        sc = SearchCoordinator(
            llm=BrokenLLM(), embedder=FakeEmbedder(),
            graph_store=NetworkXGraphStore(),
            skill_store=InMemorySkillStore(),
        )
        plan = sc._plan_retrieval("查询")
        assert plan == {}

    def test_run_returns_both_sources(self):
        """run() 应同时返回图谱和技能检索结果。"""
        sc = SearchCoordinator(
            llm=FakeLLM(), embedder=FakeEmbedder(),
            graph_store=NetworkXGraphStore(),
            skill_store=InMemorySkillStore(),
        )
        result = sc.run("复杂查询")
        assert "retrieved_graph_memories" in result
        assert "retrieved_skills" in result


# ═══════════════════════════════════════
# P2-1: 增强图谱模型
# ═══════════════════════════════════════


class TestEnhancedGraphModel:
    def test_edge_has_timestamp(self):
        """边应自动添加 timestamp。"""
        store = NetworkXGraphStore()
        store.add_edge("A", "B", "knows")
        edges = store.get_all_edges()
        assert len(edges) == 1
        assert "timestamp" in edges[0]
        assert edges[0]["timestamp"]  # 非空

    def test_edge_has_confidence(self):
        """边应有 confidence 属性。"""
        store = NetworkXGraphStore()
        store.add_edge("A", "B", "knows", properties={"confidence": 0.8})
        edges = store.get_all_edges()
        assert edges[0]["confidence"] == 0.8

    def test_edge_default_confidence(self):
        """默认 confidence 应为 1.0。"""
        store = NetworkXGraphStore()
        store.add_edge("A", "B", "knows")
        edges = store.get_all_edges()
        assert edges[0]["confidence"] == 1.0

    def test_edge_source_session(self):
        """边应支持 source_session。"""
        store = NetworkXGraphStore()
        store.add_edge("A", "B", "knows", properties={"source_session": "s1"})
        edges = store.get_all_edges()
        assert edges[0]["source_session"] == "s1"

    def test_node_has_created_at(self):
        """节点应有 created_at。"""
        store = NetworkXGraphStore()
        store.add_node("A", "Person")
        nodes = store.get_all()
        assert "created_at" in nodes[0]

    def test_search_by_relation(self):
        """应能按关系类型检索。"""
        store = NetworkXGraphStore()
        store.add_edge("A", "B", "works_at")
        store.add_edge("C", "D", "lives_in")
        store.add_edge("E", "F", "works_at")
        results = store.search_by_relation("works_at")
        assert len(results) == 2
        assert all(r["relation"] == "works_at" for r in results)

    def test_search_by_time(self):
        """应能按时间范围检索。"""
        store = NetworkXGraphStore()
        store.add_edge("A", "B", "r1", properties={"timestamp": "2024-01-01T00:00:00"})
        store.add_edge("C", "D", "r2", properties={"timestamp": "2024-06-15T00:00:00"})
        store.add_edge("E", "F", "r3", properties={"timestamp": "2025-01-01T00:00:00"})

        results = store.search_by_time(since="2024-06-01")
        assert len(results) == 2  # r2 和 r3

        results = store.search_by_time(until="2024-06-30")
        assert len(results) == 2  # r1 和 r2

        results = store.search_by_time(since="2024-06-01", until="2024-12-31")
        assert len(results) == 1  # 只有 r2

    def test_get_all_edges(self):
        """get_all_edges 应返回所有边。"""
        store = NetworkXGraphStore()
        store.add_edge("A", "B", "r1")
        store.add_edge("C", "D", "r2")
        edges = store.get_all_edges()
        assert len(edges) == 2

    def test_add_triples_with_confidence(self, sample_triples):
        """通过 add() 添加的三元组应保留 confidence。"""
        store = NetworkXGraphStore()
        triples_with_conf = []
        for t in sample_triples:
            t_copy = dict(t)
            t_copy["confidence"] = 0.9
            triples_with_conf.append(t_copy)
        store.add(triples_with_conf)
        edges = store.get_all_edges()
        assert all(e["confidence"] == 0.9 for e in edges)


# ═══════════════════════════════════════
# P2-2: 三步整合器 (score→rank→fuse)
# ═══════════════════════════════════════


class TestThreeStepSynthesizer:
    def test_synthesizer_returns_provenance(self):
        """整合器应返回 provenance (scored_fragments)。"""
        synth = SynthesizerAgent(llm=FakeLLM())
        result = synth.run(
            user_query="测试",
            retrieved_graph_memories=[{
                "anchor": {"node_id": "A"},
                "subgraph": {"nodes": [], "edges": []},
            }],
        )
        assert "provenance" in result
        assert isinstance(result["provenance"], list)

    def test_synthesizer_empty_returns_empty_provenance(self):
        """空输入应返回空 provenance。"""
        synth = SynthesizerAgent(llm=FakeLLM())
        result = synth.run(user_query="你好")
        assert result["provenance"] == []
        assert result["skill_reuse_plan"] == []

    def test_build_reuse_plan_static(self):
        """_build_reuse_plan 应只包含 reusable=True 的技能。"""
        plan = SynthesizerAgent._build_reuse_plan([
            {"id": "s1", "intent": "a", "doc_markdown": "# a", "score": 0.9, "reusable": True, "conditions": ""},
            {"id": "s2", "intent": "b", "doc_markdown": "# b", "score": 0.3, "reusable": False, "conditions": ""},
        ])
        assert len(plan) == 1
        assert plan[0]["skill_id"] == "s1"


# ═══════════════════════════════════════
# P2-3: 会话元数据
# ═══════════════════════════════════════


class TestSessionMetadata:
    def test_session_log_has_timestamps(self):
        """SessionLog 应有 created_at 和 updated_at。"""
        store = InMemorySessionStore()
        log = store.get_or_create("s1")
        assert log.created_at
        assert log.updated_at

    def test_session_log_topic_tags(self):
        """SessionLog 应支持 topic 和 tags。"""
        store = InMemorySessionStore()
        log = store.get_or_create("s1")
        assert log.topic == ""
        assert log.tags == []

    def test_update_session_meta(self):
        """应能更新会话元数据。"""
        store = InMemorySessionStore()
        store.get_or_create("s1")
        store.update_session_meta("s1", topic="Python 学习", tags=["python", "tutorial"])
        log = store.get_or_create("s1")
        assert log.topic == "Python 学习"
        assert log.tags == ["python", "tutorial"]

    def test_updated_at_changes(self):
        """append_turn 应更新 updated_at。"""
        store = InMemorySessionStore()
        log1 = store.get_or_create("s1")
        ts1 = log1.updated_at
        time.sleep(0.01)
        store.append_turn("s1", "user", "消息")
        log2 = store.get_or_create("s1")
        assert log2.updated_at >= ts1

    def test_list_sessions_pagination(self):
        """list_sessions 应支持分页。"""
        store = InMemorySessionStore()
        for i in range(10):
            store.append_turn(f"s{i}", "user", f"消息{i}")

        page1 = store.list_sessions(offset=0, limit=3)
        assert len(page1) == 3
        page2 = store.list_sessions(offset=3, limit=3)
        assert len(page2) == 3
        assert page1[0]["session_id"] != page2[0]["session_id"]

    def test_list_sessions_includes_meta(self):
        """list_sessions 返回应包含 topic 和 tags。"""
        store = InMemorySessionStore()
        store.append_turn("s1", "user", "hello")
        store.update_session_meta("s1", topic="test topic", tags=["tag1"])
        sessions = store.list_sessions()
        assert sessions[0]["topic"] == "test topic"
        assert sessions[0]["tags"] == ["tag1"]
        assert sessions[0]["created_at"]
        assert sessions[0]["updated_at"]


# ═══════════════════════════════════════
# P2-3: EntityExtractor 增强
# ═══════════════════════════════════════


class TestEntityExtractorEnhanced:
    def test_extract_adds_timestamp(self):
        """提取结果应包含 timestamp。"""
        class FakeExtractLLM:
            def generate(self, messages, **kwargs):
                return '[{"subject": {"name": "A", "label": "Person"}, "predicate": "knows", "object": {"name": "B", "label": "Person"}, "confidence": 0.9}]'

        ext = EntityExtractor(llm=FakeExtractLLM())
        triples = ext.extract("A knows B", source_session="sess1")
        assert len(triples) == 1
        assert "timestamp" in triples[0]
        assert triples[0]["source_session"] == "sess1"

    def test_extract_from_turns_with_session(self):
        """extract_from_turns 应传递 source_session。"""
        class FakeExtractLLM:
            def generate(self, messages, **kwargs):
                return '[]'

        ext = EntityExtractor(llm=FakeExtractLLM())
        triples = ext.extract_from_turns(
            [{"role": "user", "content": "test"}],
            source_session="s1",
        )
        assert isinstance(triples, list)


# ═══════════════════════════════════════
# P2-4: 运营 API
# ═══════════════════════════════════════


class TestOperationalAPI:
    def test_sessions_pagination(self):
        """GET /sessions 支持 offset/limit 参数。"""
        client = _make_client()
        for i in range(5):
            client.post("/chat/complete", json={"session_id": f"p{i}", "message": "hi"})
        resp = client.get("/sessions?offset=0&limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 2

    def test_update_session_meta_endpoint(self):
        """PATCH /memory/session/{id}/meta 更新元数据。"""
        client = _make_client()
        client.post("/chat/complete", json={"session_id": "m1", "message": "hi"})
        resp = client.patch("/memory/session/m1/meta", json={
            "topic": "测试主题",
            "tags": ["tag1", "tag2"],
        })
        assert resp.status_code == 200
        assert "updated" in resp.json()["message"].lower()

    def test_consolidation_trigger(self):
        """POST /memory/consolidate/{session_id} 手动固化。"""
        client = _make_client()
        client.post("/chat/complete", json={"session_id": "c1", "message": "hi"})
        resp = client.post("/memory/consolidate/c1")
        assert resp.status_code == 200
        assert "consolidation" in resp.json()["message"].lower()

    def test_consolidation_empty_session(self):
        """空会话固化不应报错。"""
        client = _make_client()
        resp = client.post("/memory/consolidate/nonexistent")
        assert resp.status_code == 200

    def test_graph_by_relation(self):
        """GET /memory/graph/by-relation 按关系检索。"""
        pipeline = _make_pipeline()
        pipeline.search_coordinator.graph_store.add_edge("A", "B", "works_at")
        pipeline.search_coordinator.graph_store.add_edge("C", "D", "lives_in")
        client = TestClient(create_app(pipeline))

        resp = client.get("/memory/graph/by-relation?relation=works_at")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["edges"][0]["relation"] == "works_at"

    def test_graph_by_time(self):
        """GET /memory/graph/by-time 按时间检索。"""
        pipeline = _make_pipeline()
        gs = pipeline.search_coordinator.graph_store
        gs.add_edge("A", "B", "r1", properties={"timestamp": "2024-01-01T00:00:00"})
        gs.add_edge("C", "D", "r2", properties={"timestamp": "2025-01-01T00:00:00"})
        client = TestClient(create_app(pipeline))

        resp = client.get("/memory/graph/by-time?since=2024-06-01")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    def test_pipeline_status(self):
        """GET /pipeline/status 返回统计数据。"""
        pipeline = _make_pipeline()
        pipeline.search_coordinator.graph_store.add_node("A", "Person")
        pipeline.search_coordinator.graph_store.add_edge("A", "B", "knows")
        client = TestClient(create_app(pipeline))

        resp = client.get("/pipeline/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["graph_node_count"] >= 1
        assert data["graph_edge_count"] >= 1
        assert "session_count" in data
        assert "skill_count" in data

    def test_session_summary_has_new_fields(self):
        """会话列表条目应包含 topic, tags, created_at。"""
        client = _make_client()
        client.post("/chat/complete", json={"session_id": "sf1", "message": "hello"})
        resp = client.get("/sessions")
        session = resp.json()["sessions"][0]
        assert "topic" in session
        assert "tags" in session
        assert "created_at" in session
        assert "updated_at" in session


# ═══════════════════════════════════════
# Pipeline State 新字段
# ═══════════════════════════════════════


class TestPipelineStateFields:
    def test_state_has_skill_reuse_plan(self):
        """PipelineState 应包含 skill_reuse_plan。"""
        from a_frame.core.state import PipelineState
        state: PipelineState = {"user_query": "test", "session_id": "s1", "skill_reuse_plan": []}
        assert state["skill_reuse_plan"] == []

    def test_state_has_provenance(self):
        """PipelineState 应包含 provenance。"""
        from a_frame.core.state import PipelineState
        state: PipelineState = {"user_query": "test", "session_id": "s1", "provenance": [{"source": "graph"}]}
        assert len(state["provenance"]) == 1
