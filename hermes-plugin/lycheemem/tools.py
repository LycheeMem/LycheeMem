"""Schemas for the LycheeMem Hermes plugin tools."""

from __future__ import annotations

from typing import Any


LYCHEEMEM_HEALTH_SCHEMA: dict[str, Any] = {
    "name": "lycheemem_health",
    "description": "Check whether the configured LycheeMem server is reachable.",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


LYCHEEMEM_SMART_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "lycheemem_smart_search",
    "description": (
        "Recall relevant memory from LycheeMem. By default this returns a compact, "
        "agent-friendly payload centered on background_context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language memory query.",
            },
            "session_id": {
                "type": "string",
                "description": "Optional Hermes session id override.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum results to retrieve per source.",
            },
            "mode": {
                "type": "string",
                "enum": ["raw", "compact", "full"],
                "description": "LycheeMem smart-search mode override.",
            },
            "response_level": {
                "type": "string",
                "enum": ["minimal", "compact", "full"],
                "description": "Controls how much JSON LycheeMem returns.",
            },
            "include_graph": {
                "type": "boolean",
                "description": "Whether to search semantic memory.",
            },
            "include_skills": {
                "type": "boolean",
                "description": "Whether to search skill memory.",
            },
            "synthesize": {
                "type": "boolean",
                "description": "Whether to synthesize retrieved memory into background_context.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


LYCHEEMEM_APPEND_TURN_SCHEMA: dict[str, Any] = {
    "name": "lycheemem_append_turn",
    "description": "Append one conversation turn into LycheeMem's session store.",
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Optional Hermes session id override.",
            },
            "role": {
                "type": "string",
                "description": "Conversation role, usually user or assistant.",
            },
            "content": {
                "type": "string",
                "description": "Turn text to append.",
            },
            "token_count": {
                "type": "integer",
                "minimum": 0,
                "description": "Optional token count if already known.",
            },
        },
        "required": ["role", "content"],
        "additionalProperties": False,
    },
}


LYCHEEMEM_CONSOLIDATE_SCHEMA: dict[str, Any] = {
    "name": "lycheemem_consolidate",
    "description": "Persist newly appended turns into LycheeMem long-term memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Optional Hermes session id override.",
            },
            "retrieved_context": {
                "type": "string",
                "description": "Optional raw retrieved context used for novelty checks.",
            },
            "background": {
                "type": "boolean",
                "description": "Whether LycheeMem should consolidate asynchronously.",
            },
        },
        "additionalProperties": False,
    },
}


ALL_TOOLS = (
    ("lycheemem_health", LYCHEEMEM_HEALTH_SCHEMA, "Check LycheeMem connectivity.", ""),
    (
        "lycheemem_smart_search",
        LYCHEEMEM_SMART_SEARCH_SCHEMA,
        "Recall structured memory from LycheeMem.",
        "",
    ),
    (
        "lycheemem_append_turn",
        LYCHEEMEM_APPEND_TURN_SCHEMA,
        "Append one turn into the LycheeMem session bridge.",
        "",
    ),
    (
        "lycheemem_consolidate",
        LYCHEEMEM_CONSOLIDATE_SCHEMA,
        "Consolidate bridged turns into LycheeMem long-term memory.",
        "",
    ),
)


COMMAND_USAGE = (
    "Usage: /lycheemem status | /lycheemem health | /lycheemem recall <query> "
    "| /lycheemem consolidate"
)
