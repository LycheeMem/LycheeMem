from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.api.server import create_app
from tests.fakes import FakePipeline


def test_mcp_handshake():
    app = create_app(FakePipeline())
    client = TestClient(app)

    initialize = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )

    assert initialize.status_code == 200
    assert initialize.json()["result"]["protocolVersion"] == "2025-03-26"
    assert initialize.headers["Mcp-Session-Id"]

    session_id = initialize.headers["Mcp-Session-Id"]
    tools = client.post(
        "/mcp",
        headers={"Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    tool_names = [tool["name"] for tool in tools.json()["result"]["tools"]]

    assert "lychee_memory_search" in tool_names
    assert "lychee_memory_smart_search" in tool_names
    assert "lychee_memory_append_turn" in tool_names
    assert "lychee_memory_synthesize" in tool_names
    assert "lychee_memory_consolidate" in tool_names


def test_search_synthesize_token_reduction():
    app = create_app(FakePipeline())
    client = TestClient(app)

    initialize = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "init", "method": "initialize", "params": {}},
    )
    session_id = initialize.headers["Mcp-Session-Id"]

    search = client.post(
        "/mcp",
        headers={"Mcp-Session-Id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": "search",
            "method": "tools/call",
            "params": {
                "name": "lychee_memory_search",
                "arguments": {"query": "project continuity", "top_k": 2},
            },
        },
    )
    search_result = search.json()["result"]["structuredContent"]
    assert search_result["graph_results"]
    assert search_result["graph_results"][0]["anchor"]["node_id"] == "graphiti_context"
    assert search_result["graph_results"][0]["constructed_context"]

    synthesize = client.post(
        "/mcp",
        headers={"Mcp-Session-Id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": "synthesize",
            "method": "tools/call",
            "params": {
                "name": "lychee_memory_synthesize",
                "arguments": {
                    "user_query": "project continuity",
                    "graph_results": search_result["graph_results"],
                    "skill_results": search_result["skill_results"],
                },
            },
        },
    )
    synthesize_result = synthesize.json()["result"]["structuredContent"]

    raw_payload = json.dumps(
        search_result["graph_results"] + search_result["skill_results"],
        ensure_ascii=False,
    )
    assert synthesize_result["background_context"]
    assert len(synthesize_result["background_context"]) < len(raw_payload)
    assert synthesize_result["kept_count"] <= search_result["total"]


def test_smart_search_returns_synthesized_context():
    app = create_app(FakePipeline())
    client = TestClient(app)

    initialize = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "init", "method": "initialize", "params": {}},
    )
    session_id = initialize.headers["Mcp-Session-Id"]

    smart_search = client.post(
        "/mcp",
        headers={"Mcp-Session-Id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": "smart-search",
            "method": "tools/call",
            "params": {
                "name": "lychee_memory_smart_search",
                "arguments": {
                    "query": "project continuity",
                    "top_k": 2,
                    "synthesize": True,
                    "mode": "full",
                },
            },
        },
    )
    result = smart_search.json()["result"]["structuredContent"]

    assert result["graph_results"]
    assert result["skill_results"]
    assert result["synthesized"] is True
    assert result["background_context"]
    assert result["kept_count"] <= result["total"]


def test_smart_search_compact_mode_hides_raw_payload():
    app = create_app(FakePipeline())
    client = TestClient(app)

    initialize = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "init", "method": "initialize", "params": {}},
    )
    session_id = initialize.headers["Mcp-Session-Id"]

    smart_search = client.post(
        "/mcp",
        headers={"Mcp-Session-Id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": "smart-search-compact",
            "method": "tools/call",
            "params": {
                "name": "lychee_memory_smart_search",
                "arguments": {"query": "project continuity", "mode": "compact"},
            },
        },
    )
    result = smart_search.json()["result"]["structuredContent"]

    assert result["mode"] == "compact"
    assert result["synthesized"] is True
    assert result["background_context"]
    assert result["graph_results"] == []
    assert result["skill_results"] == []


def test_consolidate_reads_session():
    pipeline = FakePipeline()
    app = create_app(pipeline)
    client = TestClient(app)

    chat = client.post(
        "/chat/complete",
        json={"session_id": "session-1", "message": "remember this delivery preference"},
    )
    assert chat.status_code == 200

    initialize = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    session_id = initialize.headers["Mcp-Session-Id"]

    consolidate = client.post(
        "/mcp",
        headers={"Mcp-Session-Id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "lychee_memory_consolidate",
                "arguments": {
                    "session_id": "session-1",
                    "retrieved_context": "delivery preference background",
                    "background": False,
                },
            },
        },
    )
    result = consolidate.json()["result"]["structuredContent"]

    assert result["status"] == "done"
    assert pipeline.consolidator.calls
    assert len(pipeline.consolidator.calls[0]["turns"]) == 2
    assert pipeline.consolidator.calls[0]["session_id"] == "session-1"


def test_append_turn_bridge_enables_consolidate():
    pipeline = FakePipeline()
    app = create_app(pipeline)
    client = TestClient(app)

    initialize = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    session_id = initialize.headers["Mcp-Session-Id"]

    for role, content in [
        ("user", "以后项目文档默认中文。"),
        ("assistant", "已记录"),
    ]:
        append_turn = client.post(
            "/mcp",
            headers={"Mcp-Session-Id": session_id},
            json={
                "jsonrpc": "2.0",
                "id": f"append-{role}",
                "method": "tools/call",
                "params": {
                    "name": "lychee_memory_append_turn",
                    "arguments": {
                        "session_id": "bridged-session-1",
                        "role": role,
                        "content": content,
                    },
                },
            },
        )
        payload = append_turn.json()["result"]["structuredContent"]
        assert payload["status"] == "appended"
        assert payload["session_id"] == "bridged-session-1"

    consolidate = client.post(
        "/mcp",
        headers={"Mcp-Session-Id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": "consolidate",
            "method": "tools/call",
            "params": {
                "name": "lychee_memory_consolidate",
                "arguments": {
                    "session_id": "bridged-session-1",
                    "retrieved_context": "",
                    "background": False,
                },
            },
        },
    )
    result = consolidate.json()["result"]["structuredContent"]

    assert result["status"] == "done"
    assert pipeline.consolidator.calls
    assert len(pipeline.consolidator.calls[0]["turns"]) == 2
    assert pipeline.consolidator.calls[0]["session_id"] == "bridged-session-1"
