from __future__ import annotations

import sys
from pathlib import Path

import httpx


PLUGIN_DIR = Path(__file__).resolve().parents[1] / "openclaw-plugin" / "src"
sys.path.insert(0, str(PLUGIN_DIR))

from client import LycheeMemPluginClient  # type: ignore  # noqa: E402
from config import PluginConfig  # type: ignore  # noqa: E402


def test_plugin_health_check():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "version": "0.1.0"})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, timeout=5.0)
    client = LycheeMemPluginClient(
        PluginConfig(base_url="http://lychee.local", transport="http", timeout=5.0),
        http_client=http_client,
    )

    assert client.health_check()["status"] == "ok"


def test_plugin_search_and_synthesize_over_mcp():
    call_count = {"initialize": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read().decode()
        if request.url.path != "/mcp":
            raise AssertionError(f"unexpected path: {request.url.path}")

        if '"method":"initialize"' in payload or '"method": "initialize"' in payload:
            call_count["initialize"] += 1
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "mcp-session-1"},
                json={
                    "jsonrpc": "2.0",
                    "id": "init",
                    "result": {"protocolVersion": "2025-03-26"},
                },
            )
        if '"method":"initialized"' in payload or '"method": "initialized"' in payload:
            return httpx.Response(200, json={"jsonrpc": "2.0"})
        if '"name":"lychee_memory_search"' in payload or '"name": "lychee_memory_search"' in payload:
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": "lychee_memory_search",
                    "result": {
                        "structuredContent": {
                            "graph_results": [
                                {
                                    "anchor": {"node_id": "graphiti_context"},
                                    "subgraph": {"nodes": [], "edges": []},
                                    "constructed_context": "Very long structured memory bundle",
                                    "provenance": [{"fact_id": "fact-1"}],
                                }
                            ],
                            "skill_results": [{"id": "skill-1", "intent": "Prior workflow"}],
                            "total": 2,
                        }
                    },
                },
            )
        if '"name":"lychee_memory_synthesize"' in payload or '"name": "lychee_memory_synthesize"' in payload:
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": "lychee_memory_synthesize",
                    "result": {
                        "structuredContent": {
                            "background_context": "Short background",
                            "skill_reuse_plan": [{"skill_id": "skill-1"}],
                            "provenance": [{"fact_id": "fact-1"}],
                            "kept_count": 1,
                            "dropped_count": 1,
                        }
                    },
                },
            )
        raise AssertionError(f"unexpected payload: {payload}")

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, timeout=5.0)
    client = LycheeMemPluginClient(
        PluginConfig(base_url="http://lychee.local", transport="mcp", timeout=5.0),
        http_client=http_client,
    )

    search_result = client.search("long-term project context")
    synthesize_result = client.synthesize(
        user_query="long-term project context",
        graph_results=search_result["graph_results"],
        skill_results=search_result["skill_results"],
    )

    assert search_result["graph_results"][0]["anchor"]["node_id"] == "graphiti_context"
    assert search_result["total"] == 2
    assert synthesize_result["background_context"] == "Short background"
    assert call_count["initialize"] == 1
