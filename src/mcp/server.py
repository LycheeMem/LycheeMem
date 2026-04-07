"""FastAPI mount for LycheeMem MCP endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.mcp.handler import LycheeMCPHandler


def _touch_session(registry: dict[str, dict[str, Any]], session_id: str) -> None:
    session = registry.setdefault(
        session_id,
        {
            "initialized": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    session["updated_at"] = datetime.now(timezone.utc).isoformat()


def register_mcp_routes(app: FastAPI, pipeline: Any) -> None:
    """Mount MCP routes onto the shared FastAPI app."""
    router = APIRouter()
    handler = LycheeMCPHandler(pipeline)
    sessions: dict[str, dict[str, Any]] = {}

    app.state.mcp_handler = handler
    app.state.mcp_sessions = sessions

    @router.get("/mcp")
    async def mcp_stream(request: Request):
        session_id = request.headers.get("Mcp-Session-Id") or uuid.uuid4().hex
        _touch_session(sessions, session_id)

        async def event_stream():
            yield ": lycheemem mcp keepalive\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Mcp-Session-Id": session_id,
            },
        )

    @router.post("/mcp")
    async def mcp_post(request: Request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
                status_code=400,
            )

        method = body.get("method") if isinstance(body, dict) else None
        session_id = request.headers.get("Mcp-Session-Id")
        if method == "initialize" and not session_id:
            session_id = uuid.uuid4().hex
        if session_id:
            _touch_session(sessions, session_id)

        payload = await handler.handle(body)
        if method == "initialized" and session_id in sessions:
            sessions[session_id]["initialized"] = True
        headers = {"Mcp-Session-Id": session_id} if session_id else {}
        return JSONResponse(payload, headers=headers)

    app.include_router(router)
