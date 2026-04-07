"""All LLM Prompt templates for Compact Semantic Memory.

LycheeMem memory pipeline prompt implementations:
- Module 1 (Compact Semantic Encoding): typed extraction, coreference resolution, normalization, action metadata annotation
- Module 2 (Record Fusion): fusion judgment + fusion execution
- Module 3 (Action-Aware Search Planning): retrieval planning + adequacy reflection + supplementary query generation
"""

from __future__ import annotations

import os

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
- evidence_turns: Mark which turn in CURRENT_TURNS this information comes from (0-indexed).
- If there is nothing worth remembering in the conversation, return {"records": []}.
- Do not output code blocks (```json, etc.), output raw JSON only.

---

## Example 1: Multi-type conversation with coreference resolution, temporal annotation, and action metadata

<PREVIOUS_TURNS>
user: Is the project under a lot of pressure lately?
assistant: I'm ready to help at any time, what can I assist you with?
</PREVIOUS_TURNS>
<CURRENT_TURNS>
user: The DataFlow project goes live next Friday (2026-02-20). The tech stack is Python 3.11 + FastAPI, deployed on our self-built K8s cluster. I personally prefer Go, but most of the team is familiar with Python so we went with that.
assistant: Understood, Python + FastAPI + K8s is a mature combination. There are a few key points to note before going live...
user: Last time we released UserService we skipped the database migration pre-check, which directly caused production table structure mismatch, service startup failure, and rollback took two hours. We absolutely cannot repeat that mistake.
</CURRENT_TURNS>

Expected output:
{
    "records": [
        {
            "memory_type": "event",
            "semantic_text": "The user's company's DataFlow project is scheduled to go live on 2026-02-20 (Friday). The tech stack is Python 3.11 and FastAPI, deployed on the company's self-built Kubernetes cluster.",
            "normalized_text": "event:DataFlow goes live 2026-02-20, stack Python3.11+FastAPI+K8s",
            "entities": ["DataFlow", "Python 3.11", "FastAPI", "Kubernetes"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": "2026-02-20"},
            "task_tags": ["project management", "deployment"],
            "tool_tags": ["Python 3.11", "FastAPI", "Kubernetes"],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "preference",
            "semantic_text": "The user personally prefers Go as a programming language, but uses Python in the DataFlow project to align with the team (most team members are familiar with Python).",
            "normalized_text": "user_preference:prefers Go, uses Python in DataFlow project due to team constraints",
            "entities": ["Go", "Python", "DataFlow"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["programming"],
            "tool_tags": ["Go", "Python"],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "failure_pattern",
            "semantic_text": "The user's team skipped the database migration pre-check when releasing UserService, causing production table structure mismatch and service startup failure. Rollback took two hours.",
            "normalized_text": "failure_case:UserService release skipped DB migration pre-check, production table mismatch, startup failed, rollback took 2 hours",
            "entities": ["UserService"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["deployment", "database migration", "incident handling"],
            "tool_tags": ["UserService"],
            "constraint_tags": [],
            "failure_tags": ["skipped DB migration pre-check", "production table structure mismatch", "service startup failure"],
            "affordance_tags": [],
            "evidence_turns": [1],
            "source_role": "user"
        },
        {
            "memory_type": "constraint",
            "semantic_text": "The user's team requires a database migration pre-check to be completed before every deployment, to prevent production table structure mismatch failures.",
            "normalized_text": "constraint:must execute database migration pre-check before every deployment",
            "entities": [],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["deployment", "database migration"],
            "tool_tags": [],
            "constraint_tags": ["must execute DB migration pre-check before deployment"],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [1],
            "source_role": "user"
        }
    ]
}

## Example 2: Pure knowledge query, no memory material needed

<PREVIOUS_TURNS>
(none)
</PREVIOUS_TURNS>
<CURRENT_TURNS>
user: How do you write a list comprehension in Python?
assistant: List comprehension syntax is [expr for item in iterable if condition], for example [x*2 for x in range(10) if x%2==0].
</CURRENT_TURNS>

Expected output:
{"records": []}

## Example 3: Team assignment + project plan + deadline

<PREVIOUS_TURNS>
(none)
</PREVIOUS_TURNS>
<CURRENT_TURNS>
user: Our recommendation system project's next iteration (due 3.31) needs to complete user profile feature dimension expansion. I, Wang Ming, and Zhao Lin are responsible for this module. Division of work: Wang Ming handles new behavior feature event tracking design and data upload, Zhao Lin handles the ETL processing logic for the features, and I need to complete the feature adaptation and offline training validation for the recommendation model. The design review must be completed within the first 2 days.
assistant: Got it, I've noted it down. The recommendation system's next iteration members are you, Wang Ming, and Zhao Lin. Feature dimension expansion project, deadline 3.31.
</CURRENT_TURNS>

Expected output:
{
    "records": [
        {
            "memory_type": "event",
            "semantic_text": "The recommendation system project's next iteration (deadline 3.31) needs to complete the user profile feature dimension expansion module. The design review must be completed within the first 2 days.",
            "normalized_text": "event:recommendation system iteration-feature dimension expansion-deadline 3.31-design review within 2 days",
            "entities": ["recommendation system", "user profile"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": "3.31"},
            "task_tags": ["project management", "product iteration"],
            "tool_tags": [],
            "constraint_tags": ["deadline 3.31", "design review within 2 days"],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "fact",
            "semantic_text": "The recommendation system project feature dimension expansion iteration is handled by three developers: the user, Wang Ming, and Zhao Lin.",
            "normalized_text": "fact:recommendation system feature expansion team members - user + Wang Ming + Zhao Lin",
            "entities": ["Wang Ming", "Zhao Lin"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["project management"],
            "tool_tags": [],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "fact",
            "semantic_text": "In the recommendation system feature dimension expansion project, Wang Ming is responsible for designing new behavior feature event tracking and implementing data upload to backend services.",
            "normalized_text": "assignment:Wang Ming handles behavior feature event tracking design + data upload - recommendation system project",
            "entities": ["Wang Ming"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["frontend development", "project management"],
            "tool_tags": [],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "fact",
            "semantic_text": "In the recommendation system feature dimension expansion project, Zhao Lin is responsible for the ETL processing logic of the raw event feature data, cleaning and normalizing the collected data for model training use.",
            "normalized_text": "assignment:Zhao Lin handles feature ETL processing - recommendation system project",
            "entities": ["Zhao Lin"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["data processing", "project management"],
            "tool_tags": ["ETL"],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "fact",
            "semantic_text": "In the recommendation system feature dimension expansion project, the user is responsible for adapting the new features in the recommendation model, including model parameter tuning, offline training, and online/offline performance validation.",
            "normalized_text": "assignment:user handles feature adaptation + offline training + performance validation - recommendation system project",
            "entities": [],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["machine learning", "project management"],
            "tool_tags": [],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        }
    ]
}
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

---

## Example 1: Personal pronoun + relative time resolution

<ORIGINAL_TEXT>He said yesterday that the project was shut down</ORIGINAL_TEXT>
<CONTEXT>
user: Engineer Zhang came to notify me today
assistant: Does Engineer Zhang refer to Zhang Ming?
user: Yes, it's Zhang Ming. He told me yesterday (2026-01-10) that the ApolloX project has been shut down due to budget cuts.
</CONTEXT>

Expected output:
{"decontextualized_text": "Technical lead Zhang Ming informed the user on 2026-01-10 that the ApolloX project has been officially shut down due to budget cuts."}

## Example 2: Demonstrative pronoun + omitted subject resolution

<ORIGINAL_TEXT>This limitation is because the service over there doesn't support more than 50 concurrent connections</ORIGINAL_TEXT>
<CONTEXT>
user: Why can't the API gateway's throughput scale up?
assistant: This limitation is because the service over there doesn't support more than 50 concurrent requests.
user: Does "over there" refer to the downstream PaymentService?
assistant: Yes, PaymentService has a concurrency limit of 50 requests.
</CONTEXT>

Expected output:
{"decontextualized_text": "The API gateway's throughput is limited by the downstream PaymentService, which does not support more than 50 concurrent requests."}
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

---

## Example 1: memory_type = "procedure" (operational procedure)

<SEMANTIC_TEXT>When deploying the DataFlow service to the Kubernetes cluster, the following steps must be executed in order: (1) Run a database migration dry-run to confirm no schema conflicts; (2) Push the new version image via Helm upgrade; (3) Continuously observe Pod health for at least 5 minutes to confirm stability.</SEMANTIC_TEXT>
<MEMORY_TYPE>procedure</MEMORY_TYPE>

Expected output:
{
    "normalized_text": "deployment procedure:DataFlow to K8s, steps: DB migration dry-run -> Helm upgrade -> Pod observation at least 5 min",
    "task_tags": ["deployment", "release", "operations"],
    "tool_tags": ["Kubernetes", "Helm", "DataFlow"],
    "constraint_tags": ["must run DB migration dry-run first", "Pod observation no less than 5 minutes"],
    "failure_tags": [],
    "affordance_tags": ["supports rolling upgrade", "Helm-managed version rollback"]
}

## Example 2: memory_type = "failure_pattern" (failure pattern)

<SEMANTIC_TEXT>The user's team skipped the database migration pre-check when releasing UserService, causing production table structure mismatch and service startup failure. Rollback took two hours.</SEMANTIC_TEXT>
<MEMORY_TYPE>failure_pattern</MEMORY_TYPE>

Expected output:
{
    "normalized_text": "failure_case:UserService release skipped DB migration pre-check, production table mismatch, startup failed, rollback 2 hours",
    "task_tags": ["deployment", "database migration", "incident handling"],
    "tool_tags": ["UserService"],
    "constraint_tags": [],
    "failure_tags": ["skipped DB migration pre-check", "production table structure mismatch", "service startup failure"],
    "affordance_tags": []
}

## Example 3: memory_type = "tool_affordance" (tool capability)

<SEMANTIC_TEXT>LanceDB supports storing multiple vectors of different dimensions for the same record (multi-vector), allowing independent vector indexes to be built separately for semantic_text and normalized_text, with the ability to select which vector column to use during retrieval.</SEMANTIC_TEXT>
<MEMORY_TYPE>tool_affordance</MEMORY_TYPE>

Expected output:
{
    "normalized_text": "LanceDB:supports multi-vector, independent vector indexes per text field",
    "task_tags": ["vector retrieval", "storage design"],
    "tool_tags": ["LanceDB"],
    "constraint_tags": [],
    "failure_tags": [],
    "affordance_tags": ["supports multi-vector storage", "column-selective vector retrieval", "independent per-field vector indexing"]
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

---

## Example 1: Has new information (new facts + not covered by existing memory)

<EXISTING_MEMORY>
- The user prefers using Python for daily development.
- The user's project uses the FastAPI framework.
</EXISTING_MEMORY>
<CONVERSATION>
user: I changed jobs. I'm now at ByteEdge working on recommendation systems, mainly writing Go.
assistant: Congratulations! Go is indeed well-suited for high-concurrency recommendation systems.
</CONVERSATION>

Expected output:
{"reason": "User changed jobs (ByteEdge company), new role (recommendation systems), primary language switched to Go — all three are new information not covered by existing memory.", "has_novelty": true}

## Example 2: No new information (pure query of existing memory, AI restates facts)

<EXISTING_MEMORY>
- DataFlow project is scheduled to go live on 2026-02-20, with stack Python 3.11 + FastAPI + Kubernetes.
- A database migration pre-check must be performed before deployment.
</EXISTING_MEMORY>
<CONVERSATION>
user: Remind me, when does DataFlow go live?
assistant: Based on the information you mentioned earlier, the DataFlow project is scheduled to go live on 2026-02-20.
user: OK, thanks.
</CONVERSATION>

Expected output:
{"reason": "The conversation is only the user querying the already-recorded DataFlow launch date. The AI restated existing memory content without introducing any new facts or updates.", "has_novelty": false}

## Example 3: Has new information (update/correction of existing memory)

<EXISTING_MEMORY>
- The user planned to attend PyCon China on 2026-03-01.
</EXISTING_MEMORY>
<CONVERSATION>
user: The PyCon date has changed, it's been pushed back to April 15th.
assistant: Understood, I'll note that PyCon China has been postponed to 2026-04-15.
</CONVERSATION>

Expected output:
{"reason": "PyCon China's date was updated from 2026-03-01 to 2026-04-15, correcting existing memory — this constitutes valid new information.", "has_novelty": true}
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

---

## Example 1: Multiple preferences / constraints for the same entity should be synthesized

<RECORDS>
[
    {"record_id": "u1", "memory_type": "preference", "normalized_text": "user_preference:use Go for backend development", "tool_tags": ["Go"], "task_tags": ["backend development"]},
    {"record_id": "u2", "memory_type": "constraint", "normalized_text": "constraint:frontend projects must use TypeScript, no plain JavaScript", "tool_tags": ["TypeScript"], "task_tags": ["frontend development"]},
    {"record_id": "u3", "memory_type": "preference", "normalized_text": "user_preference:use Docker for containerization instead of virtual machines", "tool_tags": ["Docker"], "task_tags": ["deployment"]},
    {"record_id": "u4", "memory_type": "event", "normalized_text": "event:user attended Go conference on 2026-02-10", "tool_tags": ["Go"], "task_tags": []}
]
</RECORDS>

Expected output:
{
    "should_synthesize": true,
    "groups": [
        {
            "source_record_ids": ["u1", "u2", "u3"],
            "synthesis_reason": "u1/u2/u3 all describe the user's and team's technology-stack preferences and hard constraints. Merging them into one overview record improves retrieval density and reduces redundant context.",
            "suggested_type": "composite_preference"
        }
    ]
}

Explanation: `u4` is an independent event record. It does not sufficiently overlap with the preference / constraint topic group, aside from a shared tool tag, so it should not be included.

## Example 2: Unrelated topics should not be synthesized

<RECORDS>
[
    {"record_id": "v1", "memory_type": "fact", "normalized_text": "fact:user's cat is named Mochi and is an orange British Shorthair", "tool_tags": [], "task_tags": []},
    {"record_id": "v2", "memory_type": "procedure", "normalized_text": "deployment_procedure:DataFlow to K8s, steps:DB migration dry-run -> Helm upgrade -> observe Pods for at least 5 minutes", "tool_tags": ["Kubernetes", "Helm"], "task_tags": ["deployment"]}
]
</RECORDS>

Expected output:
{
    "should_synthesize": false,
    "groups": [],
    "conflicts": []
}

## Example 3: New memory corrects old memory and should trigger a conflict update

<RECORDS>
[
    {"record_id": "e_old", "item_kind": "record", "ingest_status": "existing", "memory_type": "event", "normalized_text": "event:PyCon China takes place on 2026-03-01", "semantic_text": "The user plans to attend PyCon China on 2026-03-01.", "entities": ["PyCon China"], "task_tags": ["scheduling"], "tool_tags": [], "constraint_tags": [], "failure_tags": [], "affordance_tags": []},
    {"record_id": "e_new", "item_kind": "record", "ingest_status": "new", "memory_type": "event", "normalized_text": "event:PyCon China postponed to 2026-04-15", "semantic_text": "The date of PyCon China changed to 2026-04-15.", "entities": ["PyCon China"], "task_tags": ["scheduling"], "tool_tags": [], "constraint_tags": [], "failure_tags": [], "affordance_tags": []}
]
</RECORDS>

Expected output:
{
    "should_synthesize": false,
    "groups": [],
    "conflicts": [
        {
            "anchor_record_id": "e_old",
            "incoming_record_ids": ["e_new"],
            "conflict_reason": "Both records describe the current date of the same event. 2026-04-15 is clearly a correction of the original 2026-03-01, not supplementary coexistent information — the old memory should be updated.",
            "resolution_mode": "update_existing"
        }
    ]
}
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

---

## Example 1: Synthesizing technology-stack preferences and constraints

<SOURCE_RECORDS>
[
  {
    "record_id": "u1", "memory_type": "preference",
    "semantic_text": "The user personally prefers Go as a programming language, but uses Python in the DataFlow project to align with the team.",
    "entities": ["Go", "Python", "DataFlow"],
    "tool_tags": ["Go", "Python"], "task_tags": ["backend development"],
    "constraint_tags": [], "failure_tags": [], "affordance_tags": [],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""}, "confidence": 0.90
  },
  {
    "record_id": "u2", "memory_type": "constraint",
    "semantic_text": "The user's team requires frontend projects to use TypeScript and forbids plain JavaScript.",
    "entities": ["TypeScript", "JavaScript"],
    "tool_tags": ["TypeScript"], "task_tags": ["frontend development"],
    "constraint_tags": ["must use TypeScript"], "failure_tags": [], "affordance_tags": [],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""}, "confidence": 0.95
  },
  {
    "record_id": "u3", "memory_type": "preference",
    "semantic_text": "The user prefers Docker for service containerization and does not use traditional virtual-machine deployment.",
    "entities": ["Docker"],
    "tool_tags": ["Docker"], "task_tags": ["deployment", "containerization"],
    "constraint_tags": [], "failure_tags": [], "affordance_tags": [],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""}, "confidence": 0.88
  }
]
</SOURCE_RECORDS>
<SYNTHESIS_REASON>All three records describe the user's and team's technology-stack preferences and constraints. Merging them into a single overview improves retrieval density.</SYNTHESIS_REASON>
<SUGGESTED_TYPE>composite_preference</SUGGESTED_TYPE>
<RESOLUTION_MODE>synthesize</RESOLUTION_MODE>
<TARGET_RECORD_ID></TARGET_RECORD_ID>

Expected output:
{
    "semantic_text": "The user personally prefers Go for backend development, but uses Python in the DataFlow project because the team mainly works in Python. The team requires TypeScript for frontend projects and forbids plain JavaScript. For deployment, the user prefers Docker-based containerization instead of traditional virtual machines.",
    "normalized_text": "tech stack:backend Go (DataFlow uses Python), frontend TypeScript required, deployment uses Docker",
    "entities": ["Go", "Python", "DataFlow", "TypeScript", "JavaScript", "Docker"],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
    "task_tags": ["backend development", "frontend development", "deployment", "containerization"],
    "tool_tags": ["Go", "Python", "TypeScript", "Docker"],
    "constraint_tags": ["must use TypeScript"],
    "failure_tags": [],
    "affordance_tags": [],
    "confidence": 0.91,
    "resolved_memory_type": "composite_preference"
}

## Example 2: Conflict update where a new date corrects an old date

<SOURCE_RECORDS>
[
    {
        "record_id": "e_old", "item_kind": "record", "memory_type": "event",
        "semantic_text": "The user plans to attend PyCon China on 2026-03-01.",
        "normalized_text": "event:PyCon China takes place on 2026-03-01",
        "entities": ["PyCon China"],
        "task_tags": ["scheduling"], "tool_tags": [], "constraint_tags": [],
        "failure_tags": [], "affordance_tags": [],
        "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": "2026-03-01"}, "confidence": 0.93
    },
    {
        "record_id": "e_new", "item_kind": "record", "memory_type": "event",
        "semantic_text": "PyCon China has been rescheduled to 2026-04-15.",
        "normalized_text": "event:PyCon China postponed to 2026-04-15",
        "entities": ["PyCon China"],
        "task_tags": ["scheduling"], "tool_tags": [], "constraint_tags": [],
        "failure_tags": [], "affordance_tags": [],
        "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": "2026-04-15"}, "confidence": 0.96
    }
]
</SOURCE_RECORDS>
<SYNTHESIS_REASON>The new record explicitly corrects the PyCon China date from 2026-03-01 to 2026-04-15, so the old memory should be updated.</SYNTHESIS_REASON>
<SUGGESTED_TYPE>event</SUGGESTED_TYPE>
<RESOLUTION_MODE>conflict_update</RESOLUTION_MODE>
<TARGET_RECORD_ID>e_old</TARGET_RECORD_ID>

Expected output:
{
    "semantic_text": "The user plans to attend PyCon China on 2026-04-15.",
    "normalized_text": "event:PyCon China takes place on 2026-04-15",
    "entities": ["PyCon China"],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": "2026-04-15"},
    "task_tags": ["scheduling"],
    "tool_tags": [],
    "constraint_tags": [],
    "failure_tags": [],
    "affordance_tags": [],
    "confidence": 0.95,
    "resolved_memory_type": "event"
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
    These slots should be parameters that can directly determine whether the next action is executable.

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

---

## Example 1: `answer` mode (factual query)

<USER_QUERY>What technology stack does the DataFlow project use?</USER_QUERY>
<RECENT_CONTEXT>(none)</RECENT_CONTEXT>
<ACTION_STATE>{"current_subgoal":"Understand the DataFlow technology stack","tentative_action":"","missing_slots":[],"known_constraints":[],"available_tools":[],"failure_signal":"","token_budget":0}</ACTION_STATE>

Expected output:
{
    "reasoning": "The user is asking for the technical facts of a specific project. This is pure information retrieval, so no action-supporting memory is needed and a small depth is sufficient.",
    "mode": "answer",
    "semantic_queries": ["DataFlow technology stack", "frameworks and languages used by DataFlow"],
    "pragmatic_queries": [],
    "temporal_filter": null,
    "tool_hints": [],
    "required_constraints": [],
    "required_affordances": [],
    "missing_slots": [],
    "tree_retrieval_mode": "root_only",
    "tree_expansion_depth": 0,
    "include_leaf_records": false,
    "include_episodic_context": false,
    "episodic_turn_window": 0,
    "depth": 5
}

## Example 2: `action` mode (performing an operation)

<USER_QUERY>Deploy the DataFlow service to production.</USER_QUERY>
<RECENT_CONTEXT>
user: DataFlow is going live today.
assistant: Understood. I will help prepare the deployment.
</RECENT_CONTEXT>
<ACTION_STATE>{"current_subgoal":"Deploy the DataFlow service to production","tentative_action":"Deploy DataFlow to production","missing_slots":[],"known_constraints":[],"available_tools":["Kubernetes","Helm"],"failure_signal":"","token_budget":12000}</ACTION_STATE>

Expected output:
{
    "reasoning": "The user is asking to execute a deployment directly. The system needs the full deployment procedure, tool constraints, and historical failure experience, so a higher depth is appropriate to cover all relevant constraints.",
    "mode": "action",
    "semantic_queries": ["DataFlow deployment procedure", "Kubernetes production deployment steps", "pre-deployment checklist"],
    "pragmatic_queries": ["database migration pre-check", "Helm upgrade", "K8s Pod health check"],
    "temporal_filter": null,
    "tool_hints": ["Kubernetes", "Helm", "Docker"],
    "required_constraints": ["must run DB migration dry-run first"],
    "required_affordances": ["supports rolling upgrade", "supports version rollback"],
    "missing_slots": ["target namespace", "image version"],
    "tree_retrieval_mode": "descend",
    "tree_expansion_depth": 2,
    "include_leaf_records": true,
    "include_episodic_context": true,
    "episodic_turn_window": 1,
    "depth": 10
}

## Example 3: `mixed` mode (factual query plus action guidance)

<USER_QUERY>What problems did we hit the last time we deployed UserService, and how should we avoid them for this DataFlow deployment?</USER_QUERY>
<RECENT_CONTEXT>(none)</RECENT_CONTEXT>
<ACTION_STATE>{"current_subgoal":"Avoid risk in the current DataFlow deployment","tentative_action":"Investigate and avoid release risk before deployment","missing_slots":[],"known_constraints":[],"available_tools":["Kubernetes","Helm"],"failure_signal":"Recent releases have failure risk","token_budget":12000}</ACTION_STATE>

Expected output:
{
    "reasoning": "The user is asking both about historical failure facts (UserService lessons) and action guidance (how to avoid them for DataFlow), so this is mixed mode. A higher depth is needed to cover both failure-pattern memory and operational-procedure memory.",
    "mode": "mixed",
    "semantic_queries": ["UserService deployment failure lessons", "DataFlow deployment precautions", "historical lessons from production releases"],
    "pragmatic_queries": ["database migration issues", "service startup failure", "deployment rollback procedure"],
    "temporal_filter": null,
    "tool_hints": ["Kubernetes", "Helm"],
    "required_constraints": [],
    "required_affordances": ["supports version rollback", "supports migration pre-check"],
    "missing_slots": [],
    "tree_retrieval_mode": "balanced",
    "tree_expansion_depth": 1,
    "include_leaf_records": true,
    "include_episodic_context": true,
    "episodic_turn_window": 1,
    "depth": 12
}
"""


def get_semantic_output_language() -> str:
    """Return the configured semantic-memory output language."""
    raw = os.getenv("LYCHEEMEM_SEMANTIC_OUTPUT_LANGUAGE", "en").strip().lower()
    if raw in {"zh", "zh-cn", "zh_hans", "chinese"}:
        return "zh"
    return "en"


def get_compact_encoding_system() -> str:
    """Return the encoding prompt with a lightweight output-language override."""
    language = get_semantic_output_language()
    if language == "zh":
        return (
            COMPACT_ENCODING_SYSTEM
            + "\n\n额外要求：\n"
            + "- `semantic_text` 与 `normalized_text` 默认使用中文输出。\n"
            + "- 若能确定绝对日期，优先写入 ISO 8601 日期。"
        )
    return (
        COMPACT_ENCODING_SYSTEM
        + "\n\nAdditional requirements:\n"
        + "- Write `semantic_text` and `normalized_text` in English.\n"
        + "- Preserve person names, entity names, and ISO 8601 dates exactly.\n"
        + "- When relative dates can be resolved from context, normalize them to absolute dates."
    )


def get_synthesis_execute_system() -> str:
    """Return the composite-synthesis prompt with a matching output-language override."""
    language = get_semantic_output_language()
    if language == "zh":
        return (
            SYNTHESIS_EXECUTE_SYSTEM
            + "\n\n额外要求：\n"
            + "- 输出中的 `semantic_text` 与 `normalized_text` 默认使用中文。\n"
            + "- 合成时保留原始记录中的所有绝对日期与时间范围。"
        )
    return (
        SYNTHESIS_EXECUTE_SYSTEM
        + "\n\nAdditional requirements:\n"
        + "- Write `semantic_text` and `normalized_text` in English.\n"
        + "- Preserve absolute dates, time ranges, and named entities from the source records."
    )


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

---

## Example 1: Sufficient (retrieval directly covers the query)

<USER_QUERY>What technology stack does the DataFlow project use?</USER_QUERY>
<SEARCH_PLAN>{"mode":"answer","required_constraints":[],"required_affordances":[],"missing_slots":[]}</SEARCH_PLAN>
<ACTION_STATE>{"tentative_action":"","known_constraints":[],"available_tools":[],"failure_signal":""}</ACTION_STATE>
<RETRIEVED_MEMORY>
[1] (event, score=0.912) entities=[DataFlow, Python 3.11, FastAPI, Kubernetes]
The user's company's DataFlow project is scheduled to go live on 2026-02-20. The stack is Python 3.11 and FastAPI, and the deployment target is the company's self-managed Kubernetes cluster.
</RETRIEVED_MEMORY>

Expected output:
{"missing_info": "", "is_sufficient": true, "missing_constraints": [], "missing_slots": [], "missing_affordances": [], "needs_failure_avoidance": false, "needs_tool_selection_basis": false}

## Example 2: Insufficient (only an event record is present and concrete procedural constraints are missing)

<USER_QUERY>Deploy the DataFlow service to production.</USER_QUERY>
<SEARCH_PLAN>{"mode":"action","required_constraints":["must run DB migration dry-run first"],"required_affordances":["supports version rollback"],"missing_slots":["target namespace","image version"]}</SEARCH_PLAN>
<ACTION_STATE>{"tentative_action":"Deploy DataFlow to production","known_constraints":[],"available_tools":["Kubernetes","Helm"],"failure_signal":"Recent releases have failure risk"}</ACTION_STATE>
<RETRIEVED_MEMORY>
[1] (event, score=0.872) entities=[DataFlow]
The user's company's DataFlow project is scheduled to go live on 2026-02-20, and the stack is Python 3.11 and FastAPI.
</RETRIEVED_MEMORY>

Expected output:
{"missing_info": "Critical execution information is missing, including concrete deployment steps, database migration constraints, rollback support evidence, image version, and target namespace.", "is_sufficient": false, "missing_constraints": ["must run DB migration dry-run first"], "missing_slots": ["target namespace", "image version"], "missing_affordances": ["supports version rollback"], "needs_failure_avoidance": true, "needs_tool_selection_basis": true}

## Example 3: Insufficient (no results retrieved)

<USER_QUERY>What problem did I run into the last time I configured a Redis cluster?</USER_QUERY>
<SEARCH_PLAN>{"mode":"mixed","required_constraints":[],"required_affordances":[],"missing_slots":[]}</SEARCH_PLAN>
<ACTION_STATE>{"tentative_action":"Investigate historical Redis cluster failures","known_constraints":[],"available_tools":["Redis"],"failure_signal":"Redis cluster configuration abnormality"}</ACTION_STATE>
<RETRIEVED_MEMORY>
(no retrieved results)
</RETRIEVED_MEMORY>

Expected output:
{"missing_info": "No historical records, failure patterns, or troubleshooting evidence were found for Redis cluster configuration.", "is_sufficient": false, "missing_constraints": [], "missing_slots": [], "missing_affordances": [], "needs_failure_avoidance": true, "needs_tool_selection_basis": false}
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

---

## Example 1: An action query is missing procedure and constraints

<USER_QUERY>Deploy the DataFlow service to production.</USER_QUERY>
<SEARCH_PLAN>{"mode":"action","tool_hints":["Kubernetes","Helm"],"required_constraints":["must run DB migration dry-run first"],"required_affordances":["supports version rollback"],"missing_slots":["target namespace","image version"]}</SEARCH_PLAN>
<ACTION_STATE>{"tentative_action":"Deploy DataFlow to production","known_constraints":[],"available_tools":["Kubernetes","Helm"],"failure_signal":"Recent releases have failure risk"}</ACTION_STATE>
<CURRENT_MEMORY>
[1] (event) The DataFlow project is scheduled to go live on 2026-02-20, and the stack is Python 3.11 + FastAPI + K8s.
</CURRENT_MEMORY>
<MISSING_INFO>Concrete deployment steps (Helm/kubectl commands), database migration constraints, historical failure experience, and rollback procedure are missing.</MISSING_INFO>
<MISSING_CONSTRAINTS>["must run DB migration dry-run first"]</MISSING_CONSTRAINTS>
<MISSING_SLOTS>["target namespace","image version"]</MISSING_SLOTS>
<MISSING_AFFORDANCES>["supports version rollback"]</MISSING_AFFORDANCES>
<NEEDS_FAILURE_AVOIDANCE>true</NEEDS_FAILURE_AVOIDANCE>
<NEEDS_TOOL_SELECTION_BASIS>true</NEEDS_TOOL_SELECTION_BASIS>

Expected output:
{
    "semantic_queries": [
        "DataFlow deployment procedure",
        "DataFlow release failure lessons"
    ],
    "pragmatic_queries": [
        "Kubernetes Helm deployment steps",
        "database migration pre-check deployment constraint",
        "version rollback procedure"
    ],
    "tool_hints": ["Kubernetes", "Helm"],
    "required_constraints": ["must run DB migration dry-run first"],
    "required_affordances": ["supports version rollback"],
    "missing_slots": ["target namespace", "image version"]
}

## Example 2: A fact query is missing specific detail

<USER_QUERY>What problem did I run into the last time I configured a Redis cluster?</USER_QUERY>
<SEARCH_PLAN>{"mode":"mixed","tool_hints":["Redis"],"required_constraints":[],"required_affordances":[],"missing_slots":[]}</SEARCH_PLAN>
<ACTION_STATE>{"tentative_action":"Recall historical Redis Cluster failures for troubleshooting","known_constraints":[],"available_tools":["Redis"],"failure_signal":"Redis cluster configuration abnormality"}</ACTION_STATE>
<CURRENT_MEMORY>
[1] (tool_affordance) Redis supports Cluster mode and requires at least 6 nodes (3 masters and 3 replicas).
</CURRENT_MEMORY>
<MISSING_INFO>No record was found about the user's own Redis cluster configuration problem; the current result is only generic tool information.</MISSING_INFO>
<MISSING_CONSTRAINTS>[]</MISSING_CONSTRAINTS>
<MISSING_SLOTS>[]</MISSING_SLOTS>
<MISSING_AFFORDANCES>[]</MISSING_AFFORDANCES>
<NEEDS_FAILURE_AVOIDANCE>true</NEEDS_FAILURE_AVOIDANCE>
<NEEDS_TOOL_SELECTION_BASIS>false</NEEDS_TOOL_SELECTION_BASIS>

Expected output:
{
    "semantic_queries": [
        "Redis cluster configuration failure experience",
        "Redis Cluster troubleshooting record"
    ],
    "pragmatic_queries": [
        "Redis node connection issue troubleshooting experience",
        "Redis Cluster configuration error failure pattern"
    ],
    "tool_hints": ["Redis"],
    "required_constraints": [],
    "required_affordances": [],
    "missing_slots": []
}
"""
