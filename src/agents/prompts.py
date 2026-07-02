"""Agent-level LLM prompt templates."""

# ---------------------------------------------------------------------------
# SearchCoordinator
# ---------------------------------------------------------------------------

SEARCH_COORDINATOR_SYSTEM_PROMPT = """\
Analyze the user query for memory search and skill retrieval.

Return raw JSON only:
{
  "tentative_action": "",
  "known_constraints": [],
  "available_tools": [],
  "failure_signal": "",
  "looks_procedural": false,
  "hyde_doc": ""
}

Rules:
- `tentative_action`: the concrete task or operation the user wants to perform; empty for pure factual questions.
- `known_constraints`: explicit limits, requirements, preferences, environment details, or deadlines.
- `available_tools`: named tools, frameworks, APIs, libraries, services, or systems.
- `failure_signal`: exact error text, failed step, negative result, or symptom if present.
- `looks_procedural`: true for how-to, troubleshooting, task execution, setup, configuration, deployment, debugging, or reusable workflow requests.
- `hyde_doc`: only when procedural. Write 1-3 concise sentences as if an ideal solution had been completed. Include concrete tools, steps, constraints, and failure terms useful for vector search. Do not invent user-specific facts.
"""


# ---------------------------------------------------------------------------
# SynthesizerAgent
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """\
Score retrieved memory fragments and fuse the useful ones into compact context.

Scoring rules:
- Score relevance from 0.0 to 1.0.
- Prefer recall over precision. Keep partial, indirect, or category-matching evidence if it may help answer.
- Drop only fragments that are clearly unrelated or contradicted by stronger memory.
- For count/list/category questions, keep a fragment when its named object plausibly belongs to the requested category and the requested predicate is supported, even if the category word is absent.
- For aggregate questions, preserve each distinct candidate item/event and the evidence for including it.

Fusion rules:
- `background_context` should be dense factual prose, not a raw concatenation.
- Preserve exact names, objects, titles, counts, dates, locations, frequencies, prices, quoted text, and stated reasons.
- Do not generalize named objects into categories when the specific name matters.
- Do not recast assistant-only statements as user facts unless the user confirmed them.
- Keep time annotations distinct from storage time if both appear.
- Sort `scored_fragments` by descending relevance.

Return raw JSON only:
{
  "scored_fragments": [
    {"source": "semantic|skill", "index": 0, "relevance": 0.0, "summary": "brief support"}
  ],
  "kept_count": 0,
  "dropped_count": 0,
  "background_context": ""
}
"""


# ---------------------------------------------------------------------------
# ReasoningAgent
# ---------------------------------------------------------------------------

REASONING_SYSTEM_PROMPT = """\
You answer using retrieved memory and conversation history.

{history_section}

{background_section}

{skill_plan_section}

Rules:
- Check memory first. Prefer retrieved evidence over guesses.
- Check the subject and premise before answering. Do not attribute one person's fact to another person.
- Bridge wording differences when memory supports the same underlying fact, event, or category.
- For count/list/category questions:
  - Do not require the memory to repeat the category word from the question.
  - Decide whether each named object belongs to the requested category using ordinary world knowledge and memory context.
  - Include an item only when both the category match and the requested predicate/event are supported.
  - Count distinct supported items/events; avoid duplicate counting of the same evidence.
- For inferential questions, give the best-supported likely answer from observed behavior, values, goals, preferences, or events. Do not refuse only because the exact wording is absent.
- Preserve exact names, objects, counts, dates, frequencies, places, and proper nouns.
- If evidence is partial, answer the supported part and state the uncertainty.
- Say the information is unavailable only when no retrieved memory provides relevant evidence.
- Keep the final answer concise and direct.
"""


# ---------------------------------------------------------------------------
# ConsolidatorAgent
# ---------------------------------------------------------------------------

CONSOLIDATION_SYSTEM_PROMPT = """\
Extract reusable procedural skills from the conversation.

Return raw JSON only:
{
  "new_skills": [
    {
      "intent": "one-sentence task this skill helps with",
      "doc_markdown": "# Title\\n\\nReusable steps, commands, prerequisites, and pitfalls."
    }
  ]
}

Rules:
- Return `{"new_skills":[]}` when no reusable multi-step procedure is taught.
- Do not extract ordinary facts, preferences, one-off events, or unresolved failed attempts as skills.
- A skill must be reusable for a similar future task, not just a summary of this conversation.
- If an existing skill list is provided, avoid duplicates; update only when the conversation adds reusable new details.
- Make each skill self-contained: scenario, prerequisites, steps, commands if any, and pitfalls.
"""
