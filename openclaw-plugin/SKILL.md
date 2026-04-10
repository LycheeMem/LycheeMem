# LycheeMem OpenClaw Plugin

## Purpose

This plugin is a thin adapter between OpenClaw and LycheeMem. It does not replace `memory-core`, does not claim `plugins.slots.memory`, and does not duplicate LycheeMem algorithms.

Collaboration model with OpenClaw:

- OpenClaw remains the host for the main reasoning loop, current-turn coordination, workspace guidance, and short-range conversational continuity.
- LycheeMem serves as the external structured long-term memory layer for cross-session recall, historical project background, entity relationships, and reusable procedures.
- Treat OpenClaw memory and LycheeMem as complementary layers, not competing memory owners.

Default plugin tool exposure:

- `lychee_memory_smart_search` (primary recall path, default lean response)
- `lychee_memory_search` (developer raw retrieval)
- `lychee_memory_append_turn`
- `lychee_memory_synthesize` (developer debugging)
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
- Re-answering the current turn with a second memory system when OpenClaw already has enough local context

## Coordination Rules

- Prefer OpenClaw's built-in memory and workspace context for same-session continuity, immediate local preferences, and repository-bound instructions.
- Prefer LycheeMem when the user is asking for longer-horizon context such as "上次这个项目怎么做的", "这个项目长期背景是什么", or "之前沉淀过哪些规则/关系/流程".
- Do not perform duplicate recall for the same question by calling both OpenClaw memory search and LycheeMem retrieval in the same turn unless the user explicitly wants a comparison.
- When LycheeMem returns a useful `background_context`, treat it as supplemental long-term context injected into OpenClaw's reasoning loop, not as a replacement for host memory.
- When OpenClaw already has enough local context to answer well, avoid unnecessary LycheeMem calls.

## Trigger Guidance

- Prefer `lychee_memory_smart_search` for recall questions such as "上次怎么处理的", "用户之前提过什么", "这个项目长期背景是什么". Treat it as the default recall path.
- Let `lychee_memory_smart_search` use full search mode internally but prefer a lean default response so the agent mainly receives synthesized `background_context`.
- Use `lychee_memory_search` only during development or debugging when you explicitly want the raw retrieval payload.
- When this plugin runs inside OpenClaw with host lifecycle integration enabled, assume the host usually mirrors natural-language user and assistant turns into LycheeMem automatically.
- In that host-integrated mode, do not manually call `lychee_memory_append_turn` from the model during normal operation, because it would duplicate the host-managed transcript mirror.
- If host lifecycle integration is unavailable, disabled, or you are debugging a non-standard flow, call `lychee_memory_append_turn` manually after each completed dialogue turn so the transcript can later be consolidated.
- Do not append raw tool invocations, tool arguments, tool outputs, scratchpad text, or other orchestration-only traces unless the user explicitly wants those artifacts stored as memory.
- Use `lychee_memory_synthesize` only after `lychee_memory_search`, and only for development or debugging when you want to inspect search and synthesis separately.
- Do not call OpenClaw `memory-core` search and `lychee_memory_search` for the same recall problem in the same turn.
- When this plugin runs inside OpenClaw with host lifecycle integration enabled, assume `/new`, `/reset`, and `/stop` boundaries may trigger `lychee_memory_consolidate` automatically with `background=true`.
- Important long-term signals such as explicit memory requests, defaults, stable preferences, rules, and project standards may also trigger proactive background consolidation before the next reset boundary.
- Use `lychee_memory_consolidate` manually at wrap-up when host automation is unavailable, disabled, or you are debugging explicit persistence behavior.
- Even in host-integrated mode, it is acceptable for the model to call `lychee_memory_consolidate` when it intentionally wants to persist important new long-term knowledge early. The model should still avoid manual `lychee_memory_append_turn` in that case.

## Recommended Pattern

The intended pattern is:

1. let OpenClaw evaluate whether its local memory, workspace instructions, and current-turn context are already sufficient
2. if longer-horizon recall is needed, call `lychee_memory_smart_search` with its default lean response
3. inject the returned `background_context` into the main reasoning context as supplemental long-term memory
4. answer in OpenClaw's normal reasoning loop
5. let the host lifecycle adapter mirror the natural-language user turn and assistant turn automatically when available; otherwise call `lychee_memory_append_turn` manually using the same `session_id`
6. do not append tool-call metadata or raw tool outputs by default
7. let host lifecycle boundaries trigger background consolidation when available, and allow proactive consolidation when the turn clearly introduces durable long-term knowledge; otherwise call `lychee_memory_consolidate` manually only if new memory-worthy information appeared in the mirrored natural-language turns
8. in host-integrated mode, if you choose to call `lychee_memory_consolidate` manually, do not precede it with extra model-driven `lychee_memory_append_turn` calls unless you are explicitly debugging transcript mirroring

Developer debugging path:
1. call `lychee_memory_search`
2. inspect the raw retrieval payload
3. call `lychee_memory_synthesize` if you want to inspect compression behavior separately

This keeps OpenClaw in charge of the main reasoning loop while LycheeMem stays focused on long-term structured memory retrieval and persistence.
