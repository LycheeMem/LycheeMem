"""LycheeMem MCP protocol handler."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from src.api.models import (
    MemoryAppendTurnRequest,
    MemoryConsolidateRequest,
    MemorySearchRequest,
    MemorySmartSearchRequest,
    MemorySynthesizeRequest,
)
from src.api.routers.memory import (
    run_memory_append_turn,
    run_memory_consolidate,
    run_memory_search,
    run_memory_smart_search,
    run_memory_synthesize,
)
from src.mcp.tools_schema import TOOLS_SCHEMA

MCP_PROTOCOL_VERSION = "2025-03-26"
MCP_SERVER_NAME = "lycheemem"
MCP_SERVER_VERSION = "0.1.0"
MCP_INSTRUCTIONS = (
    "LycheeMem is a structured long-term memory system for agents. "
    "Prefer lychee_memory_smart_search as the default recall path for agents. It performs search "
    "and can automatically synthesize a compact background_context in one tool call; compact mode "
    "is the recommended default for normal agent use. "
    "Use lychee_memory_search to retrieve relevant historical facts, entity relationships, "
    "project context, and reusable procedural knowledge when you explicitly want the raw retrieval "
    "payload. Use lychee_memory_synthesize after lychee_memory_search during development, analysis, "
    "or debugging when you want to inspect search and synthesis as separate stages. "
    "Use lychee_memory_append_turn after every completed dialogue turn when your host maintains its "
    "own transcript outside LycheeMem. Mirror the natural-language user turn and natural-language "
    "assistant reply into the same session_id even if you do not consolidate on that turn. Do not "
    "append raw tool invocations, tool arguments, tool outputs, or other orchestration-only traces "
    "unless the host explicitly wants those artifacts stored. "
    "Use lychee_memory_consolidate only after the relevant user and assistant turns have already "
    "been mirrored, and only when new facts, entities, preferences, relationships, or reusable "
    "procedures should be stored as long-term memory. "
    "LycheeMem is optimized for structured recall and long-term memory persistence rather than "
    "direct final-answer generation."
)


class LycheeMCPHandler:
    """Dispatches MCP JSON-RPC requests to the shared LycheeMem pipeline."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    async def handle(self, body: dict[str, Any], *, user_id: str = "") -> dict[str, Any]:
        if not isinstance(body, dict):
            return self._err(None, -32600, "Invalid request")

        method = body.get("method")
        req_id = body.get("id")

        if method == "initialize":
            return self._ok(
                req_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": MCP_SERVER_NAME,
                        "version": MCP_SERVER_VERSION,
                    },
                    "instructions": MCP_INSTRUCTIONS,
                },
            )
        if method == "initialized":
            return {"jsonrpc": "2.0"}
        if method == "tools/list":
            return self._ok(req_id, {"tools": TOOLS_SCHEMA})
        if method == "tools/call":
            return await self._dispatch_tool(req_id, body.get("params", {}), user_id=user_id)
        if method == "ping":
            return self._ok(req_id, {})
        return self._err(req_id, -32601, f"Method not found: {method}")

    async def _dispatch_tool(
        self,
        req_id: str | int | None,
        params: dict[str, Any],
        *,
        user_id: str = "",
    ) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}

        if not isinstance(arguments, dict):
            return self._err(req_id, -32602, "Tool arguments must be an object")

        try:
            if name == "lychee_memory_search":
                result = run_memory_search(
                    self.pipeline,
                    MemorySearchRequest.model_validate(arguments),
                    user_id=user_id,
                ).model_dump()
            elif name == "lychee_memory_smart_search":
                result = run_memory_smart_search(
                    self.pipeline,
                    MemorySmartSearchRequest.model_validate(arguments),
                    user_id=user_id,
                ).model_dump()
            elif name == "lychee_memory_append_turn":
                result = run_memory_append_turn(
                    self.pipeline,
                    MemoryAppendTurnRequest.model_validate(arguments),
                    user_id=user_id,
                ).model_dump()
            elif name == "lychee_memory_synthesize":
                result = run_memory_synthesize(
                    self.pipeline,
                    MemorySynthesizeRequest.model_validate(arguments),
                ).model_dump()
            elif name == "lychee_memory_consolidate":
                result = run_memory_consolidate(
                    self.pipeline,
                    MemoryConsolidateRequest.model_validate(arguments),
                    user_id=user_id,
                ).model_dump()
            else:
                return self._err(req_id, -32602, f"Unknown tool: {name}")
        except ValidationError as exc:
            return self._err(req_id, -32602, exc.errors())
        except Exception as exc:  # noqa: BLE001
            return self._err(req_id, -32603, str(exc))

        return self._ok(
            req_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False),
                    }
                ],
                "structuredContent": result,
                "isError": False,
            },
        )

    def _ok(self, req_id: str | int | None, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _err(self, req_id: str | int | None, code: int, message: Any) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": code,
                "message": message if isinstance(message, str) else json.dumps(message, ensure_ascii=False),
            },
        }
