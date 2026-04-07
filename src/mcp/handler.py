"""LycheeMem MCP protocol handler."""

from __future__ import annotations

import json
import logging
from pathlib import Path
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
    "and can automatically synthesize memory output in one tool call; full mode "
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


def _create_mcp_logger() -> logging.Logger:
    logger = logging.getLogger("src.mcp.tools")
    if logger.handlers:
        return logger

    repo_root = Path(__file__).resolve().parents[2]
    log_dir = repo_root / "appdata"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "mcp_tools.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _summarize_arguments(name: str | None, arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {"arguments_type": type(arguments).__name__}

    if name in {"lychee_memory_search", "lychee_memory_smart_search"}:
        return {
            "query": arguments.get("query"),
            "session_id": arguments.get("session_id"),
            "mode": arguments.get("mode"),
            "top_k": arguments.get("top_k"),
        }
    if name == "lychee_memory_append_turn":
        content = str(arguments.get("content", ""))
        return {
            "session_id": arguments.get("session_id"),
            "role": arguments.get("role"),
            "content_preview": content[:120],
            "content_length": len(content),
        }
    if name == "lychee_memory_consolidate":
        return {
            "session_id": arguments.get("session_id"),
            "background": arguments.get("background"),
            "retrieved_context_length": len(str(arguments.get("retrieved_context", ""))),
        }
    if name == "lychee_memory_synthesize":
        return {
            "user_query": arguments.get("user_query"),
            "graph_results_count": len(arguments.get("graph_results", []) or []),
            "skill_results_count": len(arguments.get("skill_results", []) or []),
        }
    return {"keys": sorted(arguments.keys())}


LOGGER = _create_mcp_logger()


class LycheeMCPHandler:
    """Dispatches MCP JSON-RPC requests to the shared LycheeMem pipeline."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    async def handle(self, body: dict[str, Any]) -> dict[str, Any]:
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
            return await self._dispatch_tool(req_id, body.get("params", {}))
        if method == "ping":
            return self._ok(req_id, {})
        return self._err(req_id, -32601, f"Method not found: {method}")

    async def _dispatch_tool(
        self,
        req_id: str | int | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}

        if not isinstance(arguments, dict):
            return self._err(req_id, -32602, "Tool arguments must be an object")

        summary = _summarize_arguments(name, arguments)
        LOGGER.info(
            "call.start req_id=%s user_id=%s tool=%s summary=%s",
            req_id,
            user_id or "-",
            name,
            json.dumps(summary, ensure_ascii=False),
        )

        try:
            if name == "lychee_memory_search":
                result = run_memory_search(
                    self.pipeline,
                    MemorySearchRequest.model_validate(arguments),
                ).model_dump()
            elif name == "lychee_memory_smart_search":
                result = run_memory_smart_search(
                    self.pipeline,
                    MemorySmartSearchRequest.model_validate(arguments),
                ).model_dump()
            elif name == "lychee_memory_append_turn":
                result = run_memory_append_turn(
                    self.pipeline,
                    MemoryAppendTurnRequest.model_validate(arguments),
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
                ).model_dump()
            else:
                LOGGER.warning("call.error req_id=%s user_id=%s tool=%s error=%s", req_id, user_id or "-", name, "Unknown tool")
                return self._err(req_id, -32602, f"Unknown tool: {name}")
        except ValidationError as exc:
            LOGGER.warning(
                "call.error req_id=%s user_id=%s tool=%s error=%s",
                req_id,
                user_id or "-",
                name,
                json.dumps(exc.errors(), ensure_ascii=False),
            )
            return self._err(req_id, -32602, exc.errors())
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "call.error req_id=%s user_id=%s tool=%s error=%s",
                req_id,
                user_id or "-",
                name,
                str(exc),
            )
            return self._err(req_id, -32603, str(exc))

        LOGGER.info(
            "call.ok req_id=%s user_id=%s tool=%s result_keys=%s",
            req_id,
            user_id or "-",
            name,
            sorted(result.keys()) if isinstance(result, dict) else type(result).__name__,
        )

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
