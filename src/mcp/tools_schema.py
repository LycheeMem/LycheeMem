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
                    "description": "检索问题，自然语言。",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "每个来源最多返回条数。",
                },
                "include_graph": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否检索图谱记忆（实体、关系、事实）。",
                },
                "include_skills": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否检索技能库（程序性工作流）。",
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
                    "description": "当前用户问题，用于相关性打分。",
                },
                "graph_results": {
                    "type": "array",
                    "description": "来自 lychee_memory_search 的 graph_results。",
                },
                "skill_results": {
                    "type": "array",
                    "description": "来自 lychee_memory_search 的 skill_results。",
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
                    "description": "当前会话 ID。",
                },
                "retrieved_context": {
                    "type": "string",
                    "default": "",
                    "description": "本轮 synthesize 输出的 background_context，用于新颖性判断。",
                },
                "background": {
                    "type": "boolean",
                    "default": True,
                    "description": "true 表示后台异步执行，false 表示同步等待完成。",
                },
            },
            "required": ["session_id"],
        },
    },
]
