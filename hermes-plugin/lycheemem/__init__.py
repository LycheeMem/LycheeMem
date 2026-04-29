"""LycheeMem Hermes standalone plugin."""

from __future__ import annotations

import logging

from .runtime import LycheeMemHermesRuntime
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Register LycheeMem tools, hooks, and a slash command with Hermes."""
    runtime = LycheeMemHermesRuntime()

    handlers = {
        "lycheemem_health": runtime.handle_health,
        "lycheemem_smart_search": runtime.handle_smart_search,
        "lycheemem_append_turn": runtime.handle_append_turn,
        "lycheemem_consolidate": runtime.handle_consolidate,
    }

    for name, schema, description, emoji in ALL_TOOLS:
        ctx.register_tool(
            name=name,
            toolset="lycheemem",
            schema=schema,
            handler=handlers[name],
            description=description,
            emoji=emoji,
        )

    ctx.register_hook("on_session_start", runtime.on_session_start)
    ctx.register_hook("pre_llm_call", runtime.pre_llm_call)
    ctx.register_hook("post_llm_call", runtime.post_llm_call)
    ctx.register_hook("on_session_end", runtime.on_session_end)
    ctx.register_hook("on_session_finalize", runtime.on_session_finalize)

    ctx.register_command(
        name="lycheemem",
        handler=runtime.handle_command,
        description="Inspect LycheeMem status, recall, and consolidation from the current Hermes session.",
        args_hint="[status|health|recall <query>|consolidate]",
    )

    logger.info("LycheeMem Hermes plugin registered.")
