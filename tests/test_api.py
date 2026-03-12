"""API 端点测试。

使用 FastAPI TestClient + FakeLLM/Embedder 测试所有端点。
"""

from fastapi.testclient import TestClient

from a_frame.api.server import create_app
from a_frame.core.factory import create_pipeline


# ─── Fakes ───


class FakeLLM:
    def generate(self, messages, **kwargs):
        system_msg = ""
        user_msg = ""
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"]
            elif m["role"] == "user":
                user_msg += m["content"]

        if "路由分析" in system_msg or "need_graph" in system_msg:
            return '{"need_graph": false, "need_skills": false, "need_sensory": false, "reasoning": "test"}'
        if "信息整合" in system_msg:
            return '{"kept_count": 0, "dropped_count": 0, "background_context": ""}'
        if "压缩" in system_msg or "状态交接" in system_msg:
            return "## Summary\ntest"
        if "固化" in system_msg or "new_skills" in system_msg:
            return '{"new_skills": [], "should_extract_entities": false}'
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
    app = create_app(pipeline)
    return TestClient(app)


# ─── Health ───


class TestHealth:
    def test_health(self):
        client = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


# ─── Chat Complete ───


class TestChatComplete:
    def test_basic_chat(self):
        client = _make_client()
        resp = client.post("/chat/complete", json={
            "session_id": "test-1",
            "message": "你好",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "test-1"
        assert data["response"]  # non-empty
        assert "wm_token_usage" in data

    def test_session_persistence(self):
        client = _make_client()
        client.post("/chat/complete", json={"session_id": "s1", "message": "第一轮"})
        client.post("/chat/complete", json={"session_id": "s1", "message": "第二轮"})

        resp = client.get("/memory/session/s1")
        assert resp.status_code == 200
        data = resp.json()
        # 两轮对话 = 4 条 turn (user + assistant 各两条)
        assert data["turn_count"] == 4

    def test_validation_empty_message(self):
        client = _make_client()
        resp = client.post("/chat/complete", json={
            "session_id": "test-1",
            "message": "",
        })
        assert resp.status_code == 422  # pydantic validation

    def test_validation_missing_session(self):
        client = _make_client()
        resp = client.post("/chat/complete", json={"message": "hello"})
        assert resp.status_code == 422

    def test_trace_id_header(self):
        client = _make_client()
        resp = client.post(
            "/chat/complete",
            json={"session_id": "s1", "message": "hi"},
            headers={"X-Trace-ID": "my-trace-123"},
        )
        assert resp.headers.get("X-Trace-ID") == "my-trace-123"

    def test_auto_trace_id(self):
        client = _make_client()
        resp = client.post("/chat/complete", json={"session_id": "s1", "message": "hi"})
        assert "X-Trace-ID" in resp.headers


# ─── Chat SSE ───


class TestChatSSE:
    def test_sse_stream(self):
        client = _make_client()
        resp = client.post("/chat", json={
            "session_id": "sse-1",
            "message": "你好",
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # 解析 SSE 事件
        events = _parse_sse(resp.text)
        types = [e["type"] for e in events]
        assert "status" in types
        assert "answer" in types
        assert "done" in types

        # 最后一个 done 事件包含元数据
        done_event = [e for e in events if e["type"] == "done"][0]
        assert done_event["session_id"] == "sse-1"


# ─── Memory: Graph ───


class TestMemoryGraph:
    def test_empty_graph(self):
        client = _make_client()
        resp = client.get("/memory/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_graph_after_data(self):
        pipeline = create_pipeline(llm=FakeLLM(), embedder=FakeEmbedder())
        pipeline.search_coordinator.graph_store.add([{
            "subject": {"name": "Alice", "label": "Person"},
            "predicate": "knows",
            "object": {"name": "Bob", "label": "Person"},
        }])
        app = create_app(pipeline)
        client = TestClient(app)

        resp = client.get("/memory/graph")
        data = resp.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1


# ─── Memory: Skills ───


class TestMemorySkills:
    def test_empty_skills(self):
        client = _make_client()
        resp = client.get("/memory/skills")
        assert resp.status_code == 200
        data = resp.json()
        assert data["skills"] == []
        assert data["total"] == 0


# ─── Memory: Session ───


class TestMemorySession:
    def test_get_new_session(self):
        client = _make_client()
        resp = client.get("/memory/session/new-session")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "new-session"
        assert data["turns"] == []
        assert data["turn_count"] == 0

    def test_delete_session(self):
        client = _make_client()
        # 先创建一些数据
        client.post("/chat/complete", json={"session_id": "del-1", "message": "hello"})
        # 确认有数据
        resp = client.get("/memory/session/del-1")
        assert resp.json()["turn_count"] > 0

        # 删除
        resp = client.delete("/memory/session/del-1")
        assert resp.status_code == 200

        # 确认已清空
        resp = client.get("/memory/session/del-1")
        assert resp.json()["turn_count"] == 0


# ─── Pipeline Not Initialized ───


class TestNoopPipeline:
    def test_503_when_no_pipeline(self):
        app = create_app(pipeline=None)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/chat/complete", json={
            "session_id": "x",
            "message": "hi",
        })
        assert resp.status_code == 503


# ─── Helpers ───


def _parse_sse(text: str) -> list[dict]:
    """从 SSE 文本中解析事件列表。"""
    import json
    events = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events
