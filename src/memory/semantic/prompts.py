"""Prompt templates for compact semantic memory."""

# ---------------------------------------------------------------------------
# Compact Semantic Encoding
# ---------------------------------------------------------------------------

COMPACT_ENCODING_SYSTEM = """\
Extract atomic, lossless, user-grounded memory records from the conversation.

Input:
- <SESSION_DATE>: optional date anchor for resolving relative or incomplete dates.
- <REFERENCE_CONTEXT>: brief background notes from earlier dialogue, used only to understand references.
- <CURRENT_TURNS>: the only turns to extract from.

Requirements:
1. **Complete Coverage**: Generate enough records to capture all valuable facts, preferences, plans, events, relationships, constraints, procedures, failures, lessons, task details, corrections, and concrete state changes in <CURRENT_TURNS>.
2. **Atomic Records**: Each record contains one independent piece of information. Do not merge separate details just because they share a topic.
3. **Lossless Restatement**: `semantic_text` must be a complete standalone sentence, not a summary. Preserve exact names, objects, quantities, dates, places, reasons, outcomes, and before/after states.
4. **Force Disambiguation**: Resolve pronouns, aliases, vague references, implicit subjects, and omitted objects when evidence is available. If <SESSION_DATE> is provided, convert relative or incomplete dates to absolute dates in both `semantic_text` and `temporal`.
5. **User Grounding**: Store user-stated or user-confirmed information. Store assistant content only if it is personalized, accepted, confirmed, or necessary for a jointly established fact; skip generic explanations, templates, lists, and world knowledge.
6. **Precise Metadata**: `entities` includes people, locations, organizations, products, objects, projects, topics, and other searchable entities. Keep attribution in `source_role`.
7. **Low-Value Content**: Skip greetings, filler, repeated questions, and chatter with no durable information.
8. **Reference Boundary**: Use <REFERENCE_CONTEXT> only to understand <CURRENT_TURNS>; do not extract facts from it.
9. **Disambiguation Note**: Return `disambiguation_context` as a short note with only the resolved names, aliases, objects, dates, or open topics needed to understand later references. Use an empty string if nothing is needed.
- `evidence_turns` are 0-based indexes into <CURRENT_TURNS>.
- If there is no durable information, return `{"records":[], "disambiguation_context": ""}`.

Memory types:
- `fact`: stable factual information.
- `preference`: likes, dislikes, habits, or recurring choices.
- `event`: dated or one-off past/future events.
- `constraint`: requirements, limits, policies, or restrictions.
- `procedure`: reusable steps or workflows.
- `failure_pattern`: mistakes, failures, pitfalls, or lessons.
- `tool_affordance`: what a tool/system can or cannot do.


Output Format, return raw JSON only:
{
  "records": [
    {
      "memory_type": "fact|preference|event|constraint|procedure|failure_pattern|tool_affordance",
      "semantic_text": "standalone memory sentence",
      "entities": ["entity"],
      "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
      "tags": ["short keyword"],
      "evidence_turns": [0],
      "source_role": "user|assistant|both"
    }
  ],
  "disambiguation_context": "short free-text note for later reference resolution, or empty string"
}

Example:
Input:
<SESSION_DATE>2026-04-10</SESSION_DATE>
<REFERENCE_CONTEXT></REFERENCE_CONTEXT>
<CURRENT_TURNS>
user: I finally mailed the signed lease renewal for my Riverside studio yesterday. The new term starts on July 1, and the rent is $1,850 a month.
assistant: Great, I will remember that your Riverside studio lease renewal was mailed.
</CURRENT_TURNS>

Output:
{
  "records": [
    {
      "memory_type": "event",
      "semantic_text": "The user mailed the signed lease renewal for the user's Riverside studio on 2026-04-09.",
      "entities": ["Riverside studio", "signed lease renewal"],
      "temporal": {"t_ref": "2026-04-09", "t_valid_from": "2026-04-09", "t_valid_to": ""},
      "tags": ["lease renewal", "housing"],
      "evidence_turns": [0],
      "source_role": "user"
    },
    {
      "memory_type": "fact",
      "semantic_text": "The user's Riverside studio lease renewal starts a new term on 2026-07-01 with rent of $1,850 per month.",
      "entities": ["Riverside studio", "lease renewal", "$1,850 per month"],
      "temporal": {"t_ref": "2026-07-01", "t_valid_from": "2026-07-01", "t_valid_to": ""},
      "tags": ["lease term", "rent"],
      "evidence_turns": [0],
      "source_role": "user"
    }
  ],
  "disambiguation_context": "Riverside studio refers to the user's studio. The signed lease renewal is for the Riverside studio; the new term starts on 2026-07-01."
}
"""


NOVELTY_CHECK_SYSTEM = """\
Decide whether the conversation contains new durable information compared with existing records.

Use `has_novelty:true` for:
- New personal facts, preferences, plans, events, relationships, constraints, procedures, or tool knowledge.
- Corrections, updates, changed dates, changed decisions, or contradictions to existing records.
- New details that make an existing record more specific.

Use `has_novelty:false` only for:
- Pure lookup questions.
- Small talk with no durable information.
- Content already fully covered by existing records.

When unsure, choose `true`.

Return raw JSON only:
{
  "reason": "brief reason",
  "has_novelty": true
}
"""


# ---------------------------------------------------------------------------
# Retrieval Planning and Reflection
# ---------------------------------------------------------------------------

FEEDBACK_CLASSIFICATION_SYSTEM = """\
Classify whether the user is giving explicit feedback about the previous answer or action.

Labels:
- `positive`: success, satisfaction, resolution, approval.
- `negative`: failure, unresolved issue, frustration, error still present.
- `correction`: the user corrects a previous answer or gives the right answer.
- empty string: no clear feedback.

Return raw JSON only:
{
  "feedback": "positive|negative|correction|",
  "outcome": "success|fail|unknown"
}

Map positive to success, negative/correction to fail, and empty feedback to unknown.
"""


RETRIEVAL_PLANNING_SYSTEM = """\
Write a JSON plan for finding evidence relevant to the user request.

Inputs:
- <USER_QUERY>: current request.
- <RECENT_CONTEXT>: recent turns, if any.
- <ACTION_STATE>: optional task hints, constraints, missing slots, tools, or failure signals.

First decide the information need:
- What fact, event, preference, procedure, constraint, or evidence is needed?
- Is the user asking for one best fact, or for many matching facts/events?
- Are there explicit filters such as person, time, place, object class, project, owner, or condition?

Mode:
- `answer`: factual lookup or question answering.
- `action`: task execution, how-to, procedure, troubleshooting, tool choice.
- `mixed`: factual lookup plus task support.

Query generation:
- `semantic_queries` should be short phrases that could match stored text. Use 2-12 unless the query is trivial.
- Include direct phrases when useful, but also include alternative wording likely to appear in ordinary conversation.
- Preserve important entities, dates, people, places, and constraints where they are necessary.
- Do not make every query repeat every filter word. Some relevant evidence may mention the concrete fact/event without the category or location from the question.
- `pragmatic_queries` are for procedures, tools, constraints, failure patterns, and operational requirements. Leave empty for ordinary factual lookup.
- Use the same language as the user query when natural.

Aggregate/list/count planning:
- Set `is_aggregate_query:true` when answering requires collecting multiple facts/events: counts, lists, enumerations, comparisons, summaries, "which/what all", or "how many".
- `aggregate_target`: the object, event class, or fact set being counted/listed.
- `aggregate_constraints`: required predicates, filters, or conditions.
- For aggregate `semantic_queries`, generate evidence expressions rather than only keyword templates.
- Cover these query types when relevant:
  1. constrained evidence: target/filter plus predicate;
  2. unconstrained evidence: predicate/event wording without the target/filter;
  3. indirect evidence: resulting state, acquisition, removal, setup, recovery, completion, cancellation, replacement, repair, change, confirmation, or before/after wording implied by the predicate.
- Do not invent concrete answer entities unless they are explicitly present in the inputs.
- Do not rely on domain-specific examples. Generalize from the predicate and constraints.
- Self-check before returning: if most aggregate queries merely restate the same target and predicate words, replace several with indirect or unconstrained evidence expressions.

Return raw JSON only:
{
  "reason": "brief planning reason: mode, aggregate decision, target/constraints, query strategy",
  "mode": "answer|action|mixed",
  "semantic_queries": ["query"],
  "pragmatic_queries": [],
  "is_aggregate_query": false,
  "aggregate_target": "",
  "aggregate_constraints": []
}
"""


RETRIEVAL_ADEQUACY_CHECK_SYSTEM = """\
Judge whether the provided evidence is sufficient for the user query.

Rules:
- Use only the provided evidence, lookup plan, and action state.
- For a single factual question, sufficient means the core answer is directly or strongly supported.
- For inferential questions, sufficient means the evidence contains enough behavior, values, events, or facts to support a grounded inference.
- For aggregate questions, sufficient means there is broad enough coverage for the requested count/list, not just one relevant example.
- For action/mixed requests, check constraints, missing slots, tool choice basis, procedures, and failure-avoidance information.
- Prefer `is_sufficient:true` unless a critical gap is clear.
- When insufficient, describe the missing information in searchable terms.

Return raw JSON only:
{
  "missing_info": "",
  "is_sufficient": true,
  "missing_constraints": [],
  "missing_slots": [],
  "missing_affordances": [],
  "needs_failure_avoidance": false,
  "needs_tool_selection_basis": false
}
"""


RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM = """\
Generate supplementary lookup terms for gaps found by the adequacy check.

Rules:
- Target the specific missing evidence, not the whole original query again.
- Preserve necessary names, dates, places, objects, tools, and constraints.
- Add semantic queries for facts/events/preferences.
- Add pragmatic queries for tools/procedures/constraints/failures.
- Do not repeat queries that already produced the current evidence.
- Keep each list short: 0-4 items.

Return raw JSON only:
{
  "semantic_queries": [],
  "pragmatic_queries": [],
  "tool_hints": [],
  "required_constraints": [],
  "required_affordances": [],
  "missing_slots": []
}
"""


COMPOSITE_FILTER_SYSTEM = """\
Select summaries that may help answer the query.

Rules:
- Select a summary if any part could be relevant. Prefer recall over precision.
- For count/list/category questions, keep summaries that mention a plausible member, event, or condition even if the exact category word is absent.
- Put an id in `needs_detail` when the summary is relevant but lacks exact names, dates, values, steps, objects, or source details.
- `needs_detail` must be a subset of `selected_ids`.
- Exclude clearly unrelated summaries.

Return raw JSON only:
{
  "selected_ids": ["id"],
  "needs_detail": [],
  "reasoning": "brief reason"
}
"""
