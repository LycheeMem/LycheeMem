---
name: lycheemem
description: Forceful operating rules for using LycheeMem as the primary structured long-term memory path inside OpenClaw.
requires:
  config:
    - plugins.entries.lycheemem-tools.enabled
---

# LycheeMem

## Mission

Use LycheeMem as the default structured long-term memory layer.

When the answer depends on prior conversations, historical facts, entity relationships, project background, preferences, procedures, or timeline reconstruction, prefer LycheeMem first.

Do not wait for the user to explicitly say `lycheemem`, `memory`, or `smart_search`.

## Default Priority

For long-horizon recall, use this order:

1. `lychee_memory_smart_search`
2. answer using the returned `background_context` or retrieval payload
3. `lychee_memory_consolidate` when important durable knowledge was added or clarified

Treat `lychee_memory_smart_search` as the primary recall tool.

Treat `lychee_memory_search` and `lychee_memory_synthesize` as debugging tools, not the normal path.

## When Smart Search Is Expected

Call `lychee_memory_smart_search` by default before answering if any of the following is true:

- the user asks about earlier dialogue, prior sessions, or historical context
- the user asks who/when/where/why/how about a person, relationship, event, preference, or decision that was not stated in the current message
- the answer requires reconstructing a timeline or resolving relative dates such as "昨天", "上周", "之前", "上次"
- the answer is about project standards, long-term project background, reusable workflows, or previously agreed rules
- the conversation is benchmark-like, memory-evaluation-like, or asks factual questions about prior dialogue turns
- there is any serious chance that host-local memory is incomplete, stale, or missing

In short: if the question is not answerable from the current message alone, try `lychee_memory_smart_search` first.

Use a lean default response by default, and switch to `response_level=full` only when you explicitly need retrieval details for debugging.

## When Not To Skip Smart Search

Do not skip `lychee_memory_smart_search` merely because:

- OpenClaw has some local workspace context
- you vaguely remember the answer
- `MEMORY.md` or `memory/*.md` might contain something related
- the user did not explicitly request a memory lookup

If the question is a factual recall question about prior dialogue, skipping `lychee_memory_smart_search` should be the exception, not the default.

## How To Use The Result

- Prefer the returned `background_context` when present
- use the retrieval result as supplemental long-term evidence
- if LycheeMem returns useful memory, answer from it directly instead of improvising from uncertain host memory
- if LycheeMem returns insufficient evidence, say so clearly instead of hallucinating

Do not call OpenClaw native memory search and LycheeMem retrieval for the same recall question unless the user explicitly wants comparison.

## Consolidation Rules

Use `lychee_memory_consolidate` more aggressively than before.

Call `lychee_memory_consolidate` with `background=true` when any of the following is true:

- the turn introduced a durable new fact, preference, rule, identity detail, relationship, or project standard
- a transcript/session/chunk of conversation was just ingested and should become long-term memory
- the user explicitly asked to remember, store, retain, or save something
- the conversation resolved an ambiguity and produced a stable final answer worth preserving
- a benchmark or evaluation workflow is intentionally feeding conversations into long-term memory

When in doubt, prefer one timely background consolidation over waiting too long and losing the memory.

## Append Turn Rules

If host lifecycle integration is enabled and working, assume natural-language user and assistant turns are usually mirrored automatically.

In that case:

- do not manually duplicate `lychee_memory_append_turn`
- but you may still call `lychee_memory_consolidate` after important memory-worthy turns

If host lifecycle integration is unavailable, disabled, or clearly not working:

- call `lychee_memory_append_turn` after each completed natural-language user turn and assistant turn
- then call `lychee_memory_consolidate` when important durable knowledge appeared

Do not append raw tool invocations, tool arguments, scratchpad, or raw tool outputs unless explicitly requested.

## Benchmark And Evaluation Guidance

In benchmark, QA, or dialogue-memory evaluation settings:

- assume factual recall questions should usually trigger `lychee_memory_smart_search`
- do not rely only on host-local memory for answers about prior dialogue
- after ingesting a conversation session or transcript chunk, prefer timely `lychee_memory_consolidate(background=true)` so later QA can retrieve the result

For benchmark-style recall, the intended pattern is:

1. ingest or mirror the conversation
2. consolidate important memory in the background
3. on each factual recall question, call `lychee_memory_smart_search`
4. answer from retrieved long-term memory

## Normal Operating Pattern

1. decide whether the question depends on prior dialogue or long-term memory
2. if yes, call `lychee_memory_smart_search` first
3. answer from the returned context
4. if the turn introduced durable new memory, call `lychee_memory_consolidate(background=true)`

## Debugging Path

Only during development or debugging:

1. call `lychee_memory_search`
2. inspect raw retrieval
3. call `lychee_memory_synthesize` if separate synthesis inspection is needed

Outside debugging, prefer `lychee_memory_smart_search`.
