"""MCP tool schemas exposed by LycheeMem."""

TOOLS_SCHEMA = [
    {
        "name": "lychee_memory_smart_search",
        "description": (
            "Primary recall path for agents. Performs LycheeMem search with optional automatic "
            "synthesis in one tool call. Use compact mode by default when you want a concise "
            "background_context; use full or raw only when you need to inspect retrieval details."
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
                "synthesize": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否在检索后自动执行 synthesize，生成 background_context。",
                },
                "mode": {
                    "type": "string",
                    "default": "compact",
                    "description": "返回模式：raw 仅返回原始检索结果；full 返回原始结果和 synthesize 结果；compact 仅返回压缩后的结果。",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "lychee_memory_search",
        "description": (
            "Developer-facing raw retrieval tool. Retrieve relevant information from LycheeMem "
            "structured long-term memory when you explicitly want the unsynthesized graph_results "
            "and skill_results payload. For normal agent use, prefer lychee_memory_smart_search."
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
        "name": "lychee_memory_append_turn",
        "description": (
            "Append one natural-language conversation turn from an external host into LycheeMem's "
            "session store. Call it after every completed dialogue turn so both the user turn and "
            "the assistant reply are mirrored into the same session_id, even if you do not "
            "consolidate on that turn. Do not use it for raw tool invocations, tool arguments, tool "
            "outputs, or other orchestration-only traces unless the host explicitly wants those "
            "artifacts stored as memory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "LycheeMem session id used to accumulate mirrored host turns.",
                },
                "role": {
                    "type": "string",
                    "description": "Conversation role, typically user or assistant.",
                },
                "content": {
                    "type": "string",
                    "description": "Raw turn text to append into LycheeMem's session store.",
                },
                "token_count": {
                    "type": "integer",
                    "default": 0,
                    "description": "Optional token count if the host already knows it.",
                },
            },
            "required": ["session_id", "role", "content"],
        },
    },
    {
        "name": "lychee_memory_synthesize",
        "description": (
            "Developer-facing synthesis tool. Compress and fuse the structured retrieval results "
            "from lychee_memory_search into a concise background_context when you intentionally want "
            "to inspect search and synthesis as separate stages."
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
            "Persist new long-term memory after a conversation. Call it only after the relevant "
            "natural-language user and assistant turns have already been mirrored with "
            "lychee_memory_append_turn. Use it when the conversation introduced new facts, "
            "entities, preferences, relationships, or reusable procedures that should be stored "
            "for future retrieval."
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
