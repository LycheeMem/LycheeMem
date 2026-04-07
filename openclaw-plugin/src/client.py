"""Thin OpenClaw client for LycheeMem MCP and HTTP endpoints."""

from __future__ import annotations

from typing import Any

import httpx

from config import PluginConfig, load_config


class LycheeMemPluginError(RuntimeError):
    """Raised when the plugin cannot reach or use LycheeMem."""


class LycheeMemPluginClient:
    """Small adapter that keeps orchestration in the plugin and memory logic in LycheeMem."""

    def __init__(
        self,
        config: PluginConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
    ):
        self.config = config or load_config()
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(timeout=self.config.timeout)
        self._mcp_session_id: str | None = None

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def health_check(self) -> dict[str, Any]:
        response = self._http.get(self.config.health_url, headers=self._headers())
        self._raise_for_status(response, "LycheeMem health check failed")
        return response.json()

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_graph: bool = True,
        include_skills: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "top_k": top_k,
            "include_graph": include_graph,
            "include_skills": include_skills,
        }
        if self.config.transport == "mcp":
            return self._call_mcp_tool("lychee_memory_search", payload)

        response = self._http.post(
            f"{self.config.base_url.rstrip('/')}/memory/search",
            json=payload,
            headers=self._headers(),
        )
        self._raise_for_status(response, "LycheeMem search failed")
        return response.json()

    def smart_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_graph: bool = True,
        include_skills: bool = True,
        synthesize: bool = True,
        mode: str = "full",
        include_provenance: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "top_k": top_k,
            "include_graph": include_graph,
            "include_skills": include_skills,
            "synthesize": synthesize,
            "mode": mode,
            "include_provenance": include_provenance,
        }
        if self.config.transport == "mcp":
            return self._call_mcp_tool("lychee_memory_smart_search", payload)

        response = self._http.post(
            f"{self.config.base_url.rstrip('/')}/memory/smart-search",
            json=payload,
            headers=self._headers(),
        )
        self._raise_for_status(response, "LycheeMem smart_search failed")
        return response.json()

    def synthesize(
        self,
        *,
        user_query: str,
        graph_results: list[dict[str, Any]],
        skill_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "user_query": user_query,
            "graph_results": graph_results,
            "skill_results": skill_results,
        }
        if self.config.transport == "mcp":
            return self._call_mcp_tool("lychee_memory_synthesize", payload)

        response = self._http.post(
            f"{self.config.base_url.rstrip('/')}/memory/synthesize",
            json=payload,
            headers=self._headers(),
        )
        self._raise_for_status(response, "LycheeMem synthesize failed")
        return response.json()

    def append_turn(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        token_count: int = 0,
    ) -> dict[str, Any]:
        payload = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "token_count": token_count,
        }
        if self.config.transport == "mcp":
            return self._call_mcp_tool("lychee_memory_append_turn", payload)

        response = self._http.post(
            f"{self.config.base_url.rstrip('/')}/memory/append-turn",
            json=payload,
            headers=self._headers(),
        )
        self._raise_for_status(response, "LycheeMem append_turn failed")
        return response.json()

    def consolidate(
        self,
        session_id: str,
        *,
        retrieved_context: str = "",
        background: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "session_id": session_id,
            "retrieved_context": retrieved_context,
            "background": background,
        }
        if self.config.transport == "mcp":
            return self._call_mcp_tool("lychee_memory_consolidate", payload)

        response = self._http.post(
            f"{self.config.base_url.rstrip('/')}/memory/consolidate",
            json=payload,
            headers=self._headers(),
        )
        self._raise_for_status(response, "LycheeMem consolidate failed")
        return response.json()

    def _call_mcp_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_mcp_initialized()
        response = self._http.post(
            self.config.mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": name,
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments,
                },
            },
            headers=self._mcp_headers(),
        )
        self._raise_for_status(response, f"MCP tool call failed: {name}")
        body = response.json()
        if "error" in body:
            raise LycheeMemPluginError(str(body["error"]))
        result = body.get("result", {})
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        return {}

    def _ensure_mcp_initialized(self) -> None:
        if self._mcp_session_id is not None:
            return
        response = self._http.post(
            self.config.mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {},
            },
            headers=self._headers(),
        )
        self._raise_for_status(response, "LycheeMem MCP initialize failed")
        self._mcp_session_id = response.headers.get("Mcp-Session-Id")
        if not self._mcp_session_id:
            raise LycheeMemPluginError("LycheeMem MCP did not return Mcp-Session-Id")
        initialized = self._http.post(
            self.config.mcp_url,
            json={
                "jsonrpc": "2.0",
                "method": "initialized",
                "params": {},
            },
            headers=self._mcp_headers(),
        )
        self._raise_for_status(initialized, "LycheeMem MCP initialization confirmation failed")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"
        return headers

    def _mcp_headers(self) -> dict[str, str]:
        headers = self._headers()
        if self._mcp_session_id:
            headers["Mcp-Session-Id"] = self._mcp_session_id
        return headers

    def _raise_for_status(self, response: httpx.Response, message: str) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LycheeMemPluginError(message) from exc
