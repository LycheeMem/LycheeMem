---
name: memory
description: Use LycheeMem as Claude Code's structured long-term memory for prior conversations, user preferences, project decisions, timelines, and durable facts.
allowed-tools:
  - mcp__lycheemem__lychee_memory_smart_search
  - mcp__lycheemem__lychee_memory_append_turn
  - mcp__lycheemem__lychee_memory_consolidate
  - mcp__lycheemem__lychee_memory_search
  - mcp__lycheemem__lychee_memory_synthesize
---

# LycheeMem Memory

Use LycheeMem as the primary structured long-term memory layer inside Claude Code.

When the answer depends on earlier conversations, historical facts, entity relationships, project background, user preferences, reusable workflows, or timeline reconstruction, prefer LycheeMem first.

Do not wait for the user to explicitly say `lycheemem`, `memory`, or `smart_search`.

## Default Priority

Use this order for long-horizon recall:

1. `mcp__lycheemem__lychee_memory_smart_search`
2. answer using the returned `background_context`
3. `mcp__lycheemem__lychee_memory_consolidate` when important durable knowledge was added or clarified

Treat `lychee_memory_smart_search` as the normal recall path.

Treat `lychee_memory_search` and `lychee_memory_synthesize` as debugging tools, not the normal path.

## When Smart Search Is Expected

Call `mcp__lycheemem__lychee_memory_smart_search` before answering when any of the following is true:

- the user asks about previous dialogue, prior sessions, or historical context
- the user asks who, when, where, why, or how about a person, relationship, event, preference, or decision that was not stated in the current message
- the answer requires reconstructing a timeline or resolving relative dates such as "yesterday", "last week", "before", "上次", or "之前"
- the answer is about project standards, long-term project background, reusable workflows, or previously agreed rules
- the conversation is benchmark-like, memory-evaluation-like, or asks factual questions about prior dialogue turns
- there is a meaningful chance that local session context is incomplete, stale, or missing

Use `response_level=minimal` by default so retrieval details stay concise unless you are debugging memory retrieval.

## How To Use The Result

- Prefer the returned `background_context` when present
- use raw retrieval results only as supplemental evidence
- if LycheeMem returns useful memory, answer from it directly instead of improvising
- if LycheeMem returns insufficient evidence, say so clearly instead of hallucinating

## Consolidation Rules

Call `mcp__lycheemem__lychee_memory_consolidate` when any of the following is true:

- the turn introduced a durable new fact, preference, rule, identity detail, relationship, or project standard
- a transcript, session, or chunk of conversation was just ingested and should become long-term memory
- the user explicitly asked to remember, store, retain, or save something
- the conversation resolved an ambiguity and produced a stable final answer worth preserving
- a benchmark or evaluation workflow is intentionally feeding conversations into long-term memory

If automatic hooks are enabled and working, normal user and assistant turns are mirrored automatically. In that case, avoid manually duplicating `lychee_memory_append_turn` unless the user is intentionally debugging the memory pipeline.

## Manual Invocation Pattern

When the user invokes this skill directly, interpret their arguments as a memory task:

- If the user asks to recall something, call `lychee_memory_smart_search`.
- If the user asks to remember something, append the turn and consolidate.
- If the user asks to debug retrieval, inspect smart-search output first, then use lower-level tools only if needed.
