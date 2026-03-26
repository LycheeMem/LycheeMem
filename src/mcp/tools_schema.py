"""MCP tool schemas exposed by LycheeMem."""

TOOLS_SCHEMA = [
    {
        "name": "lychee_memory_search",
        "description": (
            "Retrieve relevant information from LycheeMem structured long-term memory. "
            "Use it for historical facts, entity relationships, long-running project context, "
            "and procedural knowledge recall. Returns structured retrieval results that can be "
            "passed directly to lychee_memory_synthesize; graph_results use a richer structure "
            "rather than a flat fact list."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language memory query.",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of results to return per source.",
                },
                "include_graph": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include graph memory: entities, relations, and facts.",
                },
                "include_skills": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include procedural skill memory.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "lychee_memory_synthesize",
        "description": (
            "Compress and fuse the structured retrieval results from lychee_memory_search into a "
            "concise background_context. Use it when downstream reasoning benefits from a shorter, "
            "higher-density memory summary instead of raw retrieved fragments."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "Current user query used for relevance scoring.",
                },
                "graph_results": {
                    "type": "array",
                    "description": "graph_results returned by lychee_memory_search.",
                },
                "skill_results": {
                    "type": "array",
                    "description": "skill_results returned by lychee_memory_search.",
                },
            },
            "required": ["user_query", "graph_results", "skill_results"],
        },
    },
    {
        "name": "lychee_memory_consolidate",
        "description": (
            "Persist new long-term memory after a conversation. Use it when the conversation "
            "introduced new facts, entities, preferences, relationships, or reusable procedures "
            "that should be stored for future retrieval."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Existing session ID that already contains persisted turns.",
                },
                "retrieved_context": {
                    "type": "string",
                    "default": "",
                    "description": "background_context from the current synthesize step, used for novelty checks.",
                },
                "background": {
                    "type": "boolean",
                    "default": True,
                    "description": "True runs consolidation asynchronously; false waits for completion.",
                },
            },
            "required": ["session_id"],
        },
    },
]
