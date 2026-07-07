"""Prompt templates for compact semantic memory."""

# ---------------------------------------------------------------------------
# Compact Semantic Encoding
# ---------------------------------------------------------------------------

COMPACT_ENCODING_SYSTEM = """\
Extract atomic, lossless, user-grounded memory records from the conversation.

Input:
- <SESSION_DATE>: optional date anchor for resolving relative or incomplete dates.
- <REFERENCE_CONTEXT>: brief reference notes produced from earlier turns in the same session only, used only to understand names, pronouns, aliases, omitted objects, and relative dates.
- <CURRENT_TURNS>: the only turns to extract from.

Requirements:
1. **Complete Coverage**: Generate enough records to capture all valuable facts, preferences, plans, events, relationships, constraints, procedures, failures, lessons, task details, corrections, and concrete state changes in <CURRENT_TURNS>.
2. **Atomic Records**: Each record contains one independent piece of information. Do not merge separate details just because they share a topic.
3. **Lossless Restatement**: `semantic_text` must be a complete standalone sentence, not a summary. Preserve exact names, objects, quantities, dates, places, reasons, outcomes, and before/after states.
4. **Force Disambiguation**: Resolve pronouns, aliases, vague references, implicit subjects, and omitted objects when evidence is available. If <SESSION_DATE> is provided, convert relative or incomplete dates to absolute dates in both `semantic_text` and `temporal`.
5. **User Grounding**: Store user-stated or user-confirmed information. Store assistant content only if it is personalized, accepted, confirmed, or necessary for a jointly established fact; skip generic explanations, templates, lists, and world knowledge.
6. **Precise Metadata**: `entities` includes people, locations, organizations, products, objects, projects, topics, and other searchable entities. Keep attribution in `source_role`.
7. **Low-Value Content**: Skip greetings, filler, repeated questions, and chatter with no durable information.
8. **Reference Boundary**: Use <REFERENCE_CONTEXT> only to understand <CURRENT_TURNS>; do not extract facts from it, and do not assume it contains information from any other session.
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


RETRIEVAL_PLANNING_SYSTEM = """\
Write a JSON plan for answering a user question from past conversation memory.

Inputs:
- <USER_QUERY>: current request.
- <RECENT_CONTEXT>: recent turns, if any.

Your job:
- Look only at the user-visible wording in <USER_QUERY> and <RECENT_CONTEXT>.
- Describe only what the user is asking for.
- Do not decide whether the answer is available or unavailable.
- Do not speculate beyond the user-visible question.

Question classification:
- `question_type` describes the question form visible from the user request:
  `single`, `aggregate`, `temporal`, `comparison`, `personalized_advice`,
  `prior_assistant_response`, or `other`.
- Use `single` when the user asks for one remembered fact, event, preference, attribute, or detail.
- Use `aggregate` when the user asks for a count, list, set, summary, or all matching items/events.
- Use `temporal` when the answer depends on date, order, duration, deadline, recurrence, or time span.
- Use `comparison` when the user asks to compare two or more things by a non-time attribute.
- Use `personalized_advice` when the user asks for advice, recommendations, or choices tailored to remembered preferences, constraints, experiences, or goals.
- Use `prior_assistant_response` when the user asks what the assistant previously recommended, listed, generated, named, linked, explained, wrote, selected, or said.
- If several labels could apply, choose the one most central to the final answer. For example,
  "Which happened first..." is `temporal`, while "list all..." is `aggregate`.

Query generation:
- `semantic_queries` should be multiple short rewritten queries that could match likely past conversation wording. Use 4-16 unless the query is trivial.
- Include direct phrases when useful, but also include alternative wording likely to appear in ordinary conversation if the fact really happened.
- Preserve important entities, dates, people, places, and constraints where they are necessary.
- Do not make every query repeat every filter word. Some relevant evidence may mention the concrete fact/event without the category or location from the question.
- Use the same language as the user query when natural.

Evidence target and constraints:
- Fill `evidence_target` with the fact, event, person, object, relation, attribute, or set the question is about.
- Fill `evidence_constraints` with explicit predicates, filters, attributes, relations, conditions, or answer requirements from the user question.
- For a count/list/summary request, `evidence_target` is the set being collected and `evidence_constraints` are the membership conditions.
- For a single request, `evidence_target` is the specific fact/entity/event and `evidence_constraints` are the needed attributes or relations.
- Do not invent concrete answer entities unless they are explicitly present in the inputs.
- Do not rely on domain-specific examples. Generalize from predicate meaning and user wording.

Evidence needs:
- `evidence_routes` breaks the question into independent user-visible evidence needs.
- Use one route for a simple single request.
- Use multiple routes when the user explicitly asks about different named objects, events, endpoints, people, attributes, or set members.
- Each route should have `route_id`, `evidence_goal`, route-specific `queries`, optional `constraints`, and optional `temporal_filter`.
- `queries` inside a route should target that route's evidence goal specifically.
- `constraints` should describe properties using generic fields such as `kind` and `value`.

Time:
- If the request has an explicit or clearly inferable event-time range, set `temporal_filter` with `since` and/or `until` in YYYY-MM-DD form.
- For "on DATE" or "that day", set both `since` and `until` to DATE.
- For "before DATE", "prior to DATE", or "by DATE", set only `until` to DATE.
- For "after DATE", "since DATE", or "from DATE onward", set only `since` to DATE.
- For "between A and B", set `since` to A and `until` to B.
- Use the event date implied by the user query and recent context.
- Leave `temporal_filter` empty when the time range is vague or not needed.

Self-check before returning:
- If most queries merely restate the same words from the user query, replace several with natural expressions that could appear when the requested fact happened.
- For predicate/action questions, include ordinary expressions of the observable result, state change, before/after contrast, completion, acquisition, removal, setup, recovery, cancellation, confirmation, or update implied by the predicate.

Example:
Input:
<USER_QUERY>
Which happened earlier: I booked the dental cleaning or I renewed my community garden plot?
</USER_QUERY>
<RECENT_CONTEXT>
</RECENT_CONTEXT>

Output:
{
  "reason": "The user asks which of two user events happened earlier. The plan needs separate queries for each event and their dates, then the answer can compare those dates.",
  "question_type": "temporal",
  "semantic_queries": [
    "booked dental cleaning",
    "dental cleaning appointment booked",
    "scheduled dentist cleaning",
    "renewed community garden plot",
    "community garden plot renewal",
    "garden plot renewed"
  ],
  "temporal_filter": {},
  "evidence_target": "the two user events and their event dates",
  "evidence_constraints": [
    "dental cleaning booking event",
    "community garden plot renewal event",
    "event date for each event"
  ],
  "constraints": [
    {"kind": "relation", "value": "compare which of two events happened first"}
  ],
  "evidence_routes": [
    {
      "route_id": "dental_cleaning",
      "evidence_goal": "Find evidence that the user booked the dental cleaning and identify when it happened.",
      "queries": [
        "booked dental cleaning",
        "dental cleaning appointment booked",
        "scheduled dentist cleaning"
      ],
      "constraints": [
        {"kind": "event", "value": "dental cleaning booking"},
        {"kind": "attribute", "value": "event date"}
      ],
      "temporal_filter": {}
    },
    {
      "route_id": "garden_plot_renewal",
      "evidence_goal": "Find evidence that the user renewed the community garden plot and identify when it happened.",
      "queries": [
        "renewed community garden plot",
        "community garden plot renewal",
        "garden plot renewed"
      ],
      "constraints": [
        {"kind": "event", "value": "community garden plot renewal"},
        {"kind": "attribute", "value": "event date"}
      ],
      "temporal_filter": {}
    }
  ]
}

Return raw JSON only:
{
  "reason": "brief planning reason: what the user asks, target/constraints, query strategy",
  "question_type": "single|aggregate|temporal|comparison|personalized_advice|prior_assistant_response|other",
  "semantic_queries": ["query"],
  "temporal_filter": {},
  "evidence_target": "",
  "evidence_constraints": [],
  "constraints": [
    {"kind": "time|entity|status|relation|attribute|other", "value": ""}
  ],
  "evidence_routes": [
    {
      "route_id": "r1",
      "evidence_goal": "specific user-visible evidence this route should find",
      "queries": ["route-specific query"],
      "constraints": [
        {"kind": "time|entity|status|relation|attribute|other", "value": ""}
      ],
      "temporal_filter": {}
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Semantic Memory Reasoning
# ---------------------------------------------------------------------------

SEMANTIC_REASONING_SYSTEM = """\
You answer using retrieved memory and conversation history.

{background_section}

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
