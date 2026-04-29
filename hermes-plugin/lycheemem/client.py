"""Thin LycheeMem client for Hermes hooks and tools."""

from __future__ import annotations

from typing import Any
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import PluginConfig, load_config


class LycheeMemPluginError(RuntimeError):
    """Raised when the plugin cannot reach or use LycheeMem."""


class LycheeMemClient:
    """Small transport adapter for LycheeMem HTTP and MCP endpoints."""

    def __init__(self, config: PluginConfig | None = None):
        self.config = config or load_config()
        self._mcp_session_id: str | None = None

    def health_check(self) -> dict[str, Any]:
        return self._request_json("GET", self.config.health_url)

    def smart_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        session_id: str | None = None,
        include_graph: bool = True,
        include_skills: bool = True,
        synthesize: bool = True,
        mode: str = "compact",
        response_level: str = "minimal",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "include_graph": include_graph,
            "include_skills": include_skills,
            "synthesize": synthesize,
            "mode": mode,
            "response_level": response_level,
        }
        if session_id:
            payload["session_id"] = session_id
        if self.config.transport == "mcp":
            return self._call_mcp_tool("lychee_memory_smart_search", payload)
        return self._request_json(
            "POST",
            f"{self.config.base_url.rstrip('/')}/memory/smart-search",
            payload,
        )

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
        return self._request_json(
            "POST",
            f"{self.config.base_url.rstrip('/')}/memory/append-turn",
            payload,
        )

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
        return self._request_json(
            "POST",
            f"{self.config.base_url.rstrip('/')}/memory/consolidate",
            payload,
        )

    def _call_mcp_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_mcp_initialized()
        body = self._request_json(
            "POST",
            self.config.mcp_url,
            {
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
        if "error" in body:
            raise LycheeMemPluginError(str(body["error"]))
        result = body.get("result", {})
        if not isinstance(result, dict):
            return {"result": result}
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
        return result

    def _ensure_mcp_initialized(self) -> None:
        if self._mcp_session_id is not None:
            return
        init_request = Request(
            self.config.mcp_url,
            data=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "init",
                    "method": "initialize",
                    "params": {},
                }
            ).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(init_request, timeout=self.config.timeout) as response:
                response.read()
                self._mcp_session_id = response.headers.get("Mcp-Session-Id")
        except HTTPError as exc:
            raise LycheeMemPluginError(
                f"LycheeMem MCP initialize failed: HTTP {exc.code} {self._read_error_body(exc)}"
            ) from exc
        except URLError as exc:
            raise LycheeMemPluginError(
                f"LycheeMem MCP initialize failed: {exc.reason}"
            ) from exc

        if not self._mcp_session_id:
            raise LycheeMemPluginError(
                "LycheeMem MCP initialize failed: missing Mcp-Session-Id header"
            )

        self._request_json(
            "POST",
            self.config.mcp_url,
            {
                "jsonrpc": "2.0",
                "method": "initialized",
                "params": {},
            },
            headers=self._mcp_headers(),
        )

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        request_headers = self._headers()
        if headers:
            request_headers.update(headers)
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=data,
            headers=request_headers,
            method=method.upper(),
        )
        try:
            with urlopen(request, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise LycheeMemPluginError(
                f"{method.upper()} {url} failed: HTTP {exc.code} {self._read_error_body(exc)}"
            ) from exc
        except URLError as exc:
            raise LycheeMemPluginError(
                f"{method.upper()} {url} failed: {exc.reason}"
            ) from exc

        if not body:
            return {}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LycheeMemPluginError(
                f"{method.upper()} {url} returned invalid JSON"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}

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

    @staticmethod
    def _read_error_body(exc: HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        if not body:
            return ""
        if len(body) > 240:
            return body[:237] + "..."
        return body
