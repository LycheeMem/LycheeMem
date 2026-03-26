---
name: lycheemem
description: Teach OpenClaw when to use LycheeMem structured memory tools for long-term recall, retrieval compression, and background consolidation.
metadata:
  openclaw:
    requires:
      config:
        - plugin: lycheemem-tools
---

# LycheeMem OpenClaw Plugin

## Purpose

This plugin is a thin adapter between OpenClaw and LycheeMem. It does not replace `memory-core`, does not claim `plugins.slots.memory`, and does not duplicate LycheeMem algorithms.

Default plugin tool exposure:

- `lychee_memory_search`
- `lychee_memory_synthesize`
- `lychee_memory_consolidate`

## Use It For

- Historical facts the user mentioned earlier
- Long-running project context across sessions
- Entity and relationship recall
- Reusing procedural skills or workflows from prior work
- Compressing verbose retrieval results into a shorter `background_context` when needed

## Do Not Use It For

- Workspace rules already covered by `MEMORY.md` or `memory/*.md`
- Stable preferences already maintained in `memory-core`
- Replacing OpenClaw's built-in memory owner

## Trigger Guidance

- Prefer `lychee_memory_search` for complex recall questions such as "上次怎么处理的", "用户之前提过什么", "这个项目长期背景是什么".
- Use `lychee_memory_synthesize` only after `lychee_memory_search`, and only when the returned retrieval payload is too long or too repetitive to inject directly.
- Do not call OpenClaw `memory-core` search and `lychee_memory_search` for the same recall problem in the same turn.
- Use `lychee_memory_consolidate` at conversation wrap-up when the turn created new knowledge worth storing.
- Prefer `background=true` for `lychee_memory_consolidate` during normal agent operation so consolidation does not block the main reply.

## Recommended Pattern

The intended pattern is:

1. call `lychee_memory_search`
2. inspect whether the returned graph and skill payload is already compact enough
3. if it is too verbose, call `lychee_memory_synthesize`
4. inject only the needed structured memory or synthesized background into the main reasoning context
5. call `lychee_memory_consolidate` at the end only if new memory-worthy information appeared

This keeps OpenClaw in charge of the main reasoning loop while LycheeMem stays focused on long-term structured memory retrieval and persistence.
