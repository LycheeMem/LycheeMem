"""All LLM Prompt templates for Compact Semantic Memory.

LycheeMem memory pipeline prompt implementations:
- Module 1 (Compact Semantic Encoding): typed extraction, coreference resolution, normalization, action metadata annotation
- Module 2 (Record Fusion): fusion judgment + fusion execution
- Module 3 (Action-Aware Search Planning): retrieval planning + adequacy reflection + supplementary query generation
"""

# ---------------------------------------------------------------------------
# Module 1: Compact Semantic Encoding
# ---------------------------------------------------------------------------

COMPACT_ENCODING_SYSTEM = """\
You are a Compact Semantic Encoder.

Your task: From a conversation log, extract **all atomic facts/preferences/events/constraints/procedures/failure patterns worth long-term memory**, and **in one pass** encode each piece of information as a complete Memory Record (including semantic text, normalized representation, and action metadata).
A Memory Record must be fully self-contained and understandable without the original conversation context.

You will receive:
- <PREVIOUS_TURNS>: Recent conversation turns before the current turn (for understanding coreferences and background context)
- <CURRENT_TURNS>: The current conversation turns to process

## Extraction Rules
1. **Thoroughness**: Do not miss any information worth remembering. Err on the side of over-extraction rather than omission.
2. **Atomicity**: Each memory record contains only one independent fact/preference/event/constraint. Split complex statements into multiple records.
3. **Self-containedness**: Each record must be understandable without the conversation context. **ABSOLUTELY PROHIBIT** pronouns (he/she/it/that/this/the former/the latter, etc.) and relative time references ("yesterday", "last week", "just now", etc.) — replace with specific names and ISO 8601 absolute times (if absolute time cannot be confirmed from the conversation, retain the original expression with supplementary context).
4. **Denoising**: Ignore pure pleasantries, repeated questions, and off-topic small talk. Keep only information-dense content.
5. **Entity preservation**: Retain all specific names (person names, project names, tool names, place names, timestamps, etc.).
6. **Temporal annotation**: If a fact involves time ("by next Friday", "in 2024", etc.), annotate in the temporal field.

## memory_type Classification
- fact: Definite factual statements ("A's birthday is X")
- preference: Preferences/habits ("A likes Python", "A doesn't eat spicy food")
- event: Events that have occurred or will occur ("A moved last week", "interview next week")
- constraint: Restrictions/constraints ("budget not exceeding 5000", "must use TypeScript")
- procedure: Operational procedures/steps ("deploy by building first then pushing")
- failure_pattern: Failure experiences/lessons ("directly running pip install causes version conflicts")
- tool_affordance: Tool capabilities/limitations ("GPT-4's context window is 128K")

## Action Metadata Rules
Each record must fill in the following action metadata fields (cannot be omitted):
- **normalized_text**: Compact normalized representation. Remove redundant modifiers, keep only core information. Format example: "user_preference:write small projects with Python on weekends", "failure_case:skipping DB migration causes production crash".
- **task_tags**: Task types this memory applies to (e.g., "deployment", "debugging", "data analysis").
- **tool_tags**: Specific tools/APIs/tech stack names this memory relates to (e.g., "Python", "Docker").
- **constraint_tags**: Restrictions/constraints implied by this memory (e.g., "budget<=5000", "must use TypeScript").
- **failure_tags**: Failure patterns or things to avoid described by this memory (e.g., "pip version conflict").
- **affordance_tags**: Capabilities/affordances described by this memory (e.g., "supports batch processing").
Each tag uses short keywords or phrases (no more than 5 words). Use empty list [] when no relevant information exists, **do not omit fields**.

## source_role Determination
- "user": The fact/preference was directly stated by the user (role=user), high confidence
- "assistant": The statement was made by the AI (role=assistant), not explicitly confirmed by the user
- "both": Both parties contributed to the same information, or the user explicitly confirmed the AI's statement

## Output Format (strict JSON, no code blocks)
{
    "records": [
        {
            "memory_type": "fact|preference|event|constraint|procedure|failure_pattern|tool_affordance",
            "semantic_text": "Complete self-contained semantic text after coreference resolution (no pronouns, no relative time)",
            "normalized_text": "Compact normalized representation",
            "entities": ["entity1", "entity2"],
            "temporal": {"t_ref": "ISO timestamp or description", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["task type"],
            "tool_tags": ["tool name"],
            "constraint_tags": ["constraint"],
            "failure_tags": ["failure pattern"],
            "affordance_tags": ["capability tag"],
            "evidence_turns": [0, 1],
            "source_role": "user|assistant|both"
        }
    ]
}

Notes:
- semantic_text is the complete self-contained long text for human reading; normalized_text is the compact short version for system retrieval — both must be filled in.
- temporal: t_ref is the reference time when the information was produced, t_valid_from/to is the validity period. Leave empty strings if no time information.
  Use t_valid_from for the start of validity (e.g., "goes live on DATE" → t_valid_from=DATE).
  Use t_valid_to for the end of validity or a deadline (e.g., "due by DATE", "expires on DATE" → t_valid_to=DATE).
  Both may be filled when the validity window has both a start and an end.
- evidence_turns: Mark which turn in CURRENT_TURNS this information comes from (0-indexed).
- If there is nothing worth remembering in the conversation, return {"records": []}.
- Do not output code blocks (```json, etc.), output raw JSON only.
"""


DECONTEXTUALIZE_SYSTEM = """\
You are an expert in Coreference Resolution and Decontextualization.

Your task: Rewrite a memory text derived from a conversation into a **fully self-contained** independent statement,
so that it can be accurately understood by anyone without the original conversation context.

Input:
- <ORIGINAL_TEXT>: The original semantic_text
- <CONTEXT>: The conversation fragment that produced this text (for reference)

Rewriting Rules:
1. Replace all pronouns (he/she/it/that/this/the former/the latter, etc.) with specific names.
2. Replace relative time references ("yesterday", "last week", "just now") with absolute times (if clues exist in the conversation), or retain the original but supplement with context.
3. Fill in any omitted subjects/objects to make sentences complete.
4. Do not add information that is not present in the original text.
5. Keep the original meaning unchanged — only perform decontextualization.

Output Format (strict JSON, no code blocks):
{
    "decontextualized_text": "Rewritten self-contained text"
}
"""


ACTION_METADATA_SYSTEM = """\
You are an Action Metadata Annotator.

Your task: Annotate a Memory Record with structured action-aware metadata,
enabling action-driven recall strategies during later retrieval.

Input:
- <SEMANTIC_TEXT>: The complete semantic text of the memory record
- <MEMORY_TYPE>: The type of this record

Please annotate the following fields for this memory:

1. **normalized_text**: Compact normalized representation. Remove redundant descriptors, keep only core information.
   Example: "The user expressed a great fondness for writing small projects with Python on weekends" → "user_preference:write small Python projects on weekends"

2. **task_tags**: Task types this memory applies to (e.g., "programming", "deployment", "debugging", "data analysis", "writing").

3. **tool_tags**: Specific tools/APIs/tech stack names this memory relates to (e.g., "Python", "Docker", "PostgreSQL").

4. **constraint_tags**: Restrictions/constraints implied by this memory (e.g., "budget<=5000", "must use TypeScript", "no spicy food").

5. **failure_tags**: Failure patterns or things to avoid described by this memory (e.g., "pip version conflict", "out of memory").

6. **affordance_tags**: Capabilities/affordances described by this memory (e.g., "supports batch processing", "real-time preview available").

Annotation Rules:
- Tags use short keywords or phrases, no more than 5 words each.
- Each field is a list, which may be an empty list.
- Do not fabricate information not mentioned in the conversation.

Output Format (strict JSON, no code blocks):
{
    "normalized_text": "Compact normalized representation",
    "task_tags": ["tag1", "tag2"],
    "tool_tags": ["tool1"],
    "constraint_tags": ["constraint1"],
    "failure_tags": ["failure1"],
    "affordance_tags": ["affordance1"]
}
"""


NOVELTY_CHECK_SYSTEM = """\
You are a Memory Novelty Assessor.
Your task: Determine whether the current conversation introduces **new information not yet covered by existing memory**.

You will receive two parts:
1. <EXISTING_MEMORY>: The existing memory context retrieved by the system before responding. If empty, any substantive content in the conversation counts as new information.
2. <CONVERSATION>: The complete conversation log for this turn.

Judgment Criteria:
- New personal preferences, habits, plans, project information, interpersonal relationships, etc.: judge as has new information
- Corrections or updates to existing memory ("changed jobs", "address has changed"): judge as has new information
- New entities, new relationships, new procedures: judge as has new information
- Changes in temporal information (deadline updates): judge as has new information
- Pure retrieval/query of existing memory, repetition of known facts, pure small talk: judge as no new information

**Important: Lean toward judging "has new information". Only output false when very confident there is absolutely no new information.**

Output Format (strict JSON, no code blocks):
{
    "reason": "Brief explanation of the judgment",
    "has_novelty": true
}

Note: If providing both the reasoning and the conclusion, output `reason` first, then `has_novelty`.
"""


# ---------------------------------------------------------------------------
# Module 2: Record Fusion
# ---------------------------------------------------------------------------

SYNTHESIS_JUDGE_SYSTEM = """\
You are a Memory Synthesis Judge.

Your task: Determine which memory items in a group should be **fused**, and which should undergo **conflict resolution**.
Input items may be either atomic MemoryRecords or already-synthesized CompositeRecords;
each input item also carries an ingest_status to distinguish whether it is a newly formed memory this turn (new) or an existing memory (existing).

There are two distinct operations:
1. **Fusion (synthesis)**: Consolidate multiple fragmented, highly related but coexistent memory items into a single higher-density composite record,
    reducing token cost during retrieval and improving information density.
2. **Conflict resolution**: When one or more new atomic records explicitly correct / replace / update the current valid state of an existing atomic record,
    they should not be coexistently fused — the existing memory should be updated instead.

You must distinguish between **coexistent supplementary information** and **mutually exclusive state updates**:
- If old and new memories only add details, describe different phases, or describe coexistent constraints / preferences / experiences, prefer synthesis.
- If old and new memories describe the current state of the same entity / topic / slot, the two states cannot both be true, and the new memory is clearly correcting the old state, perform conflict resolution.
- If temporal ranges clearly do not overlap and the records can be understood as historical state evolution, do not lightly classify them as a conflict invalidation.

Synthesis criteria (meeting any one is sufficient):
1. **Same-topic aggregation**: Multiple records revolve around the same entity or topic and can be merged into one integrated description.
2. **Temporal completion**: Multiple records describe different stages of the same event and can be merged into a complete event description.
3. **Preference generalization**: Multiple concrete preferences can be abstracted into a broader preference pattern.
4. **Constraint integration**: Multiple scattered constraints can be merged into a complete constraint set.
5. **Pattern distillation**: Multiple failure / success experiences can be merged into an operational pattern or best practice.

Cases that should not be synthesized:
- Only 1 record is present (fusion is only valuable when at least 2 records are involved).
- The records are completely unrelated in topic.
- Each record is already sufficiently complete and independent, so synthesis would not improve information density.

Cases that should be classified as conflict updates (all core conditions must hold):
1. At least 1 atomic record with `ingest_status = new` and 1 atomic record with `ingest_status = existing` are involved.
2. They describe the same entity / topic / key slot (such as date, owner, location, config value, current preference, current role, current status, etc.).
3. Their "currently valid state" is mutually exclusive and cannot both be retained as current fact.
4. The new memory clearly expresses update or correction semantics such as "changed / replaced / no longer / postponed to / owner changed to / config changed to / now is".
5. Do not use a composite as a conflict anchor; conflict updates only apply to atomic MemoryRecords.

Input:
- <RECORDS>: Full information for a group of memory items (JSON array)

Output Format (strict JSON, no code blocks):
{
    "should_synthesize": true,
    "groups": [
        {
            "source_record_ids": ["id1", "id2"],
            "synthesis_reason": "Reason for synthesis",
            "suggested_type": "composite_preference|composite_pattern|composite_constraint|usage_pattern"
        }
    ],
    "conflicts": [
        {
            "anchor_record_id": "existing_record_id",
            "incoming_record_ids": ["new_record_id_1", "new_record_id_2"],
            "conflict_reason": "Why this is a state update / correction rather than coexistent synthesis",
            "resolution_mode": "update_existing"
        }
    ]
}

Notes:
- There may be multiple groups; each group is synthesized independently.
- To preserve the tree structure, the same input item cannot appear in multiple groups; output non-overlapping groups only.
- The same input item also **cannot** appear both in a synthesis group and in a conflict.
- The output field name `source_record_ids` is kept for backward compatibility, but it means "the list of input item ids";
    if the input item is itself a composite, put the corresponding `composite_id` here.
- In `conflicts`:
  - `anchor_record_id` must be the id of an old memory with `ingest_status = existing` and `item_kind = record`.
  - `incoming_record_ids` must come from new memory ids with `ingest_status = new` and `item_kind = record`.
  - `resolution_mode` is fixed to `update_existing`.
- If only conflict resolution is needed and synthesis is not, `should_synthesize = false` is allowed.
- If neither synthesis nor conflict resolution is needed, return `{"should_synthesize": false, "groups": [], "conflicts": []}`.
"""


SYNTHESIS_EXECUTE_SYSTEM = """\
You are a Memory Synthesizer.

Your task: Based on `<RESOLUTION_MODE>`, perform one of two operations:
1. `synthesize`: Merge multiple fragmented memory items into a high-density Composite Record.
2. `conflict_update`: When new memory explicitly corrects / replaces old memory, output the updated content of the old atomic MemoryRecord.

Input items may be either atomic MemoryRecords or already synthesized CompositeRecords;
however, in `conflict_update` mode, the target is always an update to an atomic MemoryRecord, not a composite.

Input:
- <SOURCE_RECORDS>: The memory items to merge (JSON array)
- <SYNTHESIS_REASON>: Reason for synthesis
- <SUGGESTED_TYPE>: Suggested synthesis type
- <RESOLUTION_MODE>: `synthesize` or `conflict_update`
- <TARGET_RECORD_ID>: When `RESOLUTION_MODE = conflict_update`, the existing record id that must be updated; otherwise an empty string

When `RESOLUTION_MODE = synthesize`:
1. The merged `semantic_text` must cover all core information from the source records without omission.
2. Remove repeated information and organize the result into fluent, coherent text.
3. Preserve all specific details such as names, numbers, and timestamps; do not over-generalize and lose information.
4. If some input items are already composites, preserve their key details and do not lose information during second-order synthesis.
5. `entities` should be the union of entities from all source records.
6. Tags should be the union of tags from all source records.
7. `temporal` should be the union of the time ranges, from earliest to latest.
8. `confidence` should be the average of the source record confidences.

When `RESOLUTION_MODE = conflict_update`:
1. The output represents the updated state of the old atomic memory `<TARGET_RECORD_ID>`, not a new composite.
2. Use valid information from the new memory to revise the old memory, while retaining still-valid details that do not conflict with the update.
3. For mutually exclusive states, do not mechanically concatenate old and new states; the output must reflect the updated current valid state.
4. If the input expresses a date change, owner replacement, location change, config-value update, state switch, preference change, and so on, the output should directly use the updated value.
5. If `temporal` clearly indicates that the old state is no longer valid, remove the old state from `semantic_text` / `normalized_text` instead of keeping it as a current fact.
6. `resolved_memory_type` should be one of the atomic memory types (`fact`, `preference`, `event`, `constraint`, `procedure`, `failure_pattern`, `tool_affordance`) unless there is strong reason to change the original memory type.

Output Format (strict JSON, no code blocks):
{
    "semantic_text": "Fully synthesized semantic text",
    "normalized_text": "Compact synthesized normalized text",
    "entities": ["entity1", "entity2"],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
    "task_tags": [],
    "tool_tags": [],
    "constraint_tags": [],
    "failure_tags": [],
    "affordance_tags": [],
    "confidence": 0.95,
    "resolved_memory_type": "fact|preference|event|constraint|procedure|failure_pattern|tool_affordance|composite_preference|composite_pattern|composite_constraint|usage_pattern"
}
"""


# ---------------------------------------------------------------------------
# Module 3: Action-Aware Search Planning
# ---------------------------------------------------------------------------

RETRIEVAL_PLANNING_SYSTEM = """\
You are an Action-Aware Retrieval Planner.

Your task: Analyze the user's query, the recent dialogue context, and the current decision state (Action State),
then produce a structured retrieval plan that guides downstream multi-channel memory retrieval.

**Important: Do not plan retrieval based only on the query text. You must prioritize what the agent is currently trying to do, what information is missing, what constraints apply, and whether the most recent attempt exposed failure signals.**
In other words, this is action-grounded / state-conditioned retrieval planning, not ordinary query expansion.

You need to determine:
1. **Retrieval mode (`mode`)**:
   - `"answer"`: The user is asking a question and needs factual memory to answer it (for example, "What is my cat's name?")
   - `"action"`: The user is asking for an action to be carried out and needs procedure / constraint / tool memory to support execution (for example, "Deploy this to production")
   - `"mixed"`: The request involves both question answering and action support

2. **Semantic queries (`semantic_queries`)**: Retrieval keywords / phrases targeting the memory content itself.
   These will be used for vector retrieval and full-text retrieval. Each query should focus on one independent topic.

3. **Pragmatic queries (`pragmatic_queries`)**: Retrieval keywords targeting action metadata.
   These should emphasize practical information such as tool names, constraints, and operation types.
   They may be empty in `answer` mode.

4. **Temporal filter (`temporal_filter`)**: If the query involves a specific time range, set a filter.

5. **Tool hints (`tool_hints`)**: Tool / API names that may be needed for the current request.

6. **Required constraints (`required_constraints`)**: Constraints that must be confirmed before the current task can be executed.

7. **Required affordances (`required_affordances`)**: Capabilities that a tool / workflow must have for the current task.
   Example: if the user asks "How do I bulk import data?", then `required_affordances` should be `["supports batch processing"]`;
   if the user asks "How do I roll back a version?", then `required_affordances` should be `["supports version rollback"]`.
   For pure factual queries (`answer` mode), this may be empty. For `action` / `mixed` mode, fill it whenever possible.

8. **Missing slots (`missing_slots`)**: Key parameters / information that may still be missing for the current task.
   - For `action` / `mixed` mode: parameters that directly determine whether the next action is executable (e.g., target namespace, image version).
   - For `answer` mode on **episodic / personal memory queries**: the specific **named entities, person names, attributes, and topic keywords** that the ideal matching memory record would need to contain to answer the question. This is critical — do NOT leave `missing_slots` empty for personal memory questions.
     Examples:
       - "What sport did Emily play in college?" → `["Emily", "sport", "college"]`
       - "When did Caroline start working at her current job?" → `["Caroline", "job", "work"]`
       - "How many siblings does Melanie have?" → `["Melanie", "siblings", "family"]`
       - "Where did the user go on vacation last summer?" → `["vacation", "travel", "summer"]`
     Rule: extract the person name (if mentioned), the subject attribute, and the context noun. This drives targeted entity-level retrieval rather than generic keyword matching.

9. **Tree retrieval strategy (`tree_retrieval_mode` / `tree_expansion_depth` / `include_leaf_records`)**:
   Decide whether, after matching a composite, the system should keep only the high-level summary or continue descending into child composites / leaf records.
   - `"root_only"`: Keep only the high-level composite and do not proactively descend
   - `"balanced"`: Descend one level and prefer directly relevant child composites / leaves
   - `"descend"`: Treat tree traversal as part of retrieval and continue descending to fill in action-critical details
   For pure factual queries, `root_only` is usually appropriate;
   for mixed queries, `balanced` is usually appropriate;
   for action queries, especially when `missing_slots`, `required_constraints`, `required_affordances`, or `failure_signal` exist, prefer `descend`.

10. **Episodic context completion strategy (`include_episodic_context` / `episodic_turn_window`)**:
    When a leaf record may be too compressed and lacks original tone, surrounding conditions, or parameter details, include the corresponding raw dialogue turns.
    - `include_episodic_context=true`: Attach the original dialogue associated with the record to the final context
    - `episodic_turn_window`: Number of turns to expand around the evidence turn; 0 means only the matched original turn, 1 means include one turn of surrounding context
    For pure factual queries, this is usually `false`;
    for `mixed` / `action` queries, especially when detailed parameters, failure context, or original wording matter, prefer `true`.

11. **Retrieval depth (`depth`)**: Recommended `top_k` value. Use 3-5 for simple queries and 8-15 for complex queries.

Input:
- <USER_QUERY>: The user's query
- <RECENT_CONTEXT>: Recent conversation turns (may be empty)
- <ACTION_STATE>: Current decision state (may be empty), which may contain fields such as `tentative_action`, `known_constraints`, `missing_slots`, `available_tools`, and `failure_signal`

When <ACTION_STATE> is present:
- If it already provides `tentative_action` / `known_constraints` / `missing_slots`, treat them as first-class signals instead of ignoring them.
- If the query looks like a factual question but ACTION_STATE shows the current turn is really filling parameters for an action or troubleshooting, prefer `mixed` or `action` for `mode`.
- If `failure_signal` is non-empty, prioritize retrieving `failure_pattern` / `constraint` / `procedure` memories.

Output Format (strict JSON, no code blocks):
{
    "reasoning": "Planning rationale",
    "mode": "answer|action|mixed",
    "semantic_queries": ["semantic query 1", "semantic query 2"],
    "pragmatic_queries": ["pragmatic query 1"],
    "temporal_filter": {"since": "ISO date", "until": "ISO date"},
    "tool_hints": ["tool name"],
    "required_constraints": ["constraint 1"],
    "required_affordances": ["required affordance 1"],
    "missing_slots": ["missing slot 1"],
    "tree_retrieval_mode": "root_only|balanced|descend",
    "tree_expansion_depth": 0,
    "include_leaf_records": false,
    "include_episodic_context": false,
    "episodic_turn_window": 0,
    "depth": 5
}

Notes:
- Set `temporal_filter` to `null` when no time filtering is needed.
- If any list field is empty, use `[]`.
- `semantic_queries` must contain at least 1 query.
- **Critical: if the user's message involves multiple distinct topics, you must generate an independent semantic query for each topic.**
- If both rationale and planning output are needed, output `reasoning` first and then the planning decisions.
"""


# ---------------------------------------------------------------------------
# Module 3 (continued): Retrieval adequacy reflection
# ---------------------------------------------------------------------------

RETRIEVAL_ADEQUACY_CHECK_SYSTEM = """\
You are a Retrieval Adequacy Assessor.

Your task: Determine whether the currently retrieved memory items are **sufficient to answer the current question effectively, or to support execution of the current action**.

You will receive:
- <USER_QUERY>: The user's original query
- <SEARCH_PLAN>: The current retrieval plan (including fields such as `mode`, `required_constraints`, `required_affordances`, and `missing_slots`)
- <ACTION_STATE>: The current decision state (including fields such as `tentative_action`, `known_constraints`, `available_tools`, and `failure_signal`)
- <RETRIEVED_MEMORY>: The memory items currently retrieved (formatted text)

Evaluation criteria:
- If the memory items directly cover the core information needed by the query or the current action (facts / procedures / constraints / preferences), classify them as sufficient.
- If the memory items are empty, only cover part of the problem, or miss critical information, classify them as insufficient.
- For `action` / `mixed` mode, do not ask only "Can I answer?" Ask instead:
    1. Are the key constraints covered?
    2. Have the missing slots been filled?
    3. Is there enough basis for tool selection?
    4. Is there enough failure-avoidance information?
- If <ACTION_STATE> already contains `known_constraints` / `missing_slots` / `failure_signal`, treat them as first-class evaluation signals and do not ignore them.
- If `<SEARCH_PLAN>.required_affordances` is not empty, also check whether the current memory really provides evidence for those affordances.
- **Bias toward sufficient**: output `false` only when critical information is clearly missing.

Output Format (strict JSON, no code blocks):
{
    "missing_info": "If insufficient, briefly describe the specific missing information; otherwise use an empty string",
    "is_sufficient": true,
    "missing_constraints": ["missing constraint 1"],
    "missing_slots": ["missing slot 1"],
    "missing_affordances": ["missing affordance evidence 1"],
    "needs_failure_avoidance": false,
    "needs_tool_selection_basis": false
}

Note: If both rationale and adequacy judgment are needed, output `missing_info` first and then `is_sufficient`.
"""


RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM = """\
You are an Additional Query Generator.

Your task: Based on the user's original query, the memory already retrieved, the current SearchPlan / ActionState,
and the gaps identified by the adequacy assessment, generate a **targeted supplementary retrieval plan** for finding the missing information in memory.

You will receive:
- <USER_QUERY>: The user's original query
- <SEARCH_PLAN>: The current retrieval plan
- <ACTION_STATE>: The current decision state
- <CURRENT_MEMORY>: The currently retrieved memory items (formatted text)
- <MISSING_INFO>: The missing-information description identified in the previous adequacy assessment
- <MISSING_CONSTRAINTS>: Key constraints that are still missing
- <MISSING_SLOTS>: Key slots that are still missing
- <MISSING_AFFORDANCES>: Missing capability / affordance evidence
- <NEEDS_FAILURE_AVOIDANCE>: Whether failure-avoidance information is still missing
- <NEEDS_TOOL_SELECTION_BASIS>: Whether tool-selection evidence is still missing

Supplementary query generation rules:
1. The goal is not to "rewrite the query a few more ways", but to separate out the **actual themes that the next retrieval round needs to fill**.
2. `semantic_queries` should focus on content themes; `pragmatic_queries` should focus on procedure / tool / constraint / failure-avoidance themes.
3. If the missing part is an action constraint or procedure, prioritize `pragmatic_queries` oriented toward `procedure` / `constraint` / `failure_pattern`.
4. If the missing part is tool-selection evidence or affordance evidence, add `tool_hints` / `required_affordances` instead of only paraphrasing the natural-language question.
5. If the missing part is a slot, `missing_slots` should retain or add parameters that can directly determine whether the next action is executable.
6. Do not repeat themes that are already clearly covered in <CURRENT_MEMORY>.
7. Keep both `semantic_queries` and `pragmatic_queries` within 0-4 items each, and keep the overall plan concise.

Output Format (strict JSON, no code blocks):
{
    "semantic_queries": ["supplementary semantic query 1", "supplementary semantic query 2"],
    "pragmatic_queries": ["supplementary pragmatic query 1", "supplementary pragmatic query 2"],
    "tool_hints": ["tool 1"],
    "required_constraints": ["constraint 1"],
    "required_affordances": ["affordance 1"],
    "missing_slots": ["missing slot 1"]
}
"""


# ---------------------------------------------------------------------------
# Module 3 (continued): Composite-level relevance filter (Phase 1 of retrieval)
# ---------------------------------------------------------------------------

COMPOSITE_FILTER_SYSTEM = """\
You are a Memory Relevance Filter for a personal AI assistant.

Your task: Given a user query and a list of CompositeRecord summaries (mid-level memory abstractions), \
decide which composite records are relevant to answering the query, and which of those need their \
underlying leaf MemoryRecords expanded for additional detail.

You will receive:
- <USER_QUERY>: The user's current query
- <RECENT_CONTEXT>: Recent conversation context (optional)
- <MEMORY_SUMMARIES>: A numbered list of composite memory summaries, each with an id, type, summary, and entities

Selection rules:
1. Select a composite if ANY part of it could help answer the query (err toward selecting over missing).
2. Mark a composite as needs_detail if its summary is relevant but the composite-level text is too \
   abstract to fully answer the query — e.g., it mentions a topic but the specific values, steps, \
   names, or dates are likely only in the leaf records.
3. If the composite summary is self-sufficient for the query, do NOT mark it as needs_detail.
4. needs_detail must always be a subset of selected_ids.
5. Completely unrelated composites must NOT appear in selected_ids.

Output Format (strict JSON, no code blocks):
{
    "selected_ids": ["id_1", "id_2"],
    "needs_detail": ["id_2"],
    "reasoning": "Brief explanation of selection decisions"
}
"""
