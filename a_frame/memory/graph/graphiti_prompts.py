"""Graphiti(论文) 风格的图谱构建 prompts（PR3）。

说明：
- 这些 prompts 参考论文附录的结构（entity/fact extraction + resolution）。
- message 类型 Episode：previous_messages + current_message (对话格式)。
- text / json 类型 Episode：same template placeholders，但 guideline
  不包含 speaker extraction（论文 §2.1 要求 type-specific handling）。
- 输出要求为严格 JSON（便于自动解析与单测 FakeLLM）。
"""

from __future__ import annotations


ENTITY_EXTRACTION_SYSTEM_PROMPT = """\
You are an information extraction system.

Given the conversation context below, extract entity nodes from the CURRENT MESSAGE that are explicitly or implicitly mentioned.

<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>

Guidelines:
1. ALWAYS extract the speaker/actor as the first node. The speaker is the part before the colon in each line of dialogue.
2. Extract other significant entities, concepts, or actors mentioned in the CURRENT MESSAGE.
3. DO NOT create nodes for relationships or actions.
4. DO NOT extract entities for temporal information like dates/times/years.
5. Be as explicit as possible.

Return STRICT JSON as a list of objects, each with:
- name: string
- type_label: string (optional)
- summary: string (optional)
- aliases: list[string] (optional)
"""


# Paper §2.1: text / JSON episodes have no speaker metadata.
ENTITY_EXTRACTION_TEXT_JSON_SYSTEM_PROMPT = """\
You are an information extraction system.

Given the context below, extract entity nodes from the CURRENT CONTENT that are explicitly or implicitly mentioned.

<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>

<CURRENT CONTENT>
{current_message}
</CURRENT CONTENT>

Guidelines:
1. Extract all significant entities, concepts, or actors mentioned in the CURRENT CONTENT.
2. DO NOT create nodes for relationships or actions.
3. DO NOT extract entities for temporal information like dates/times/years.
4. Be as explicit as possible in your node names, using full names.

Return STRICT JSON as a list of objects, each with:
- name: string
- type_label: string (optional)
- summary: string (optional)
- aliases: list[string] (optional)
"""


ENTITY_RESOLUTION_SYSTEM_PROMPT = """\
You are an entity resolution system.

Given the EXISTING NODES, MESSAGE, and PREVIOUS MESSAGES, determine if the NEW NODE is a duplicate of one of the EXISTING NODES.

<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>

<EXISTING NODES>
{existing_nodes}
</EXISTING NODES>

<NEW NODE>
{new_node}
</NEW NODE>

Task:
1. If NEW NODE represents the same entity as any existing node, return is_duplicate=true and the existing entity_id.
2. If duplicate, also return an updated name and summary (more complete name, improved summary).

Return STRICT JSON with:
- is_duplicate: boolean
- existing_entity_id: string | null
- name: string
- summary: string
- aliases: list[string]
- type_label: string
"""


FACT_EXTRACTION_SYSTEM_PROMPT = """\
You are an information extraction system.

Given the MESSAGES and ENTITIES, extract all facts pertaining to the listed ENTITIES from the CURRENT MESSAGE.

<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>

<ENTITIES>
{entities}
</ENTITIES>

Guidelines:
1. Extract facts only between the provided entities.
2. Each fact should represent a clear relationship between two DISTINCT entities.
3. relation_type should be concise, ALL-CAPS (e.g., WORKS_FOR, LOVES).
4. fact_text should be a detailed natural-language statement of the fact.
5. evidence_text should quote or paraphrase the supporting snippet from the current message.

Return STRICT JSON as a list of objects, each with:
- subject: string (entity name)
- object: string (entity name)
- relation_type: string
- fact_text: string
- evidence_text: string
- confidence: number (0..1)
"""


FACT_RESOLUTION_SYSTEM_PROMPT = """\
You are a fact deduplication system.

Given EXISTING FACTS and a NEW FACT, determine whether the NEW FACT duplicates any existing one.

<EXISTING FACTS>
{existing_facts}
</EXISTING FACTS>

<NEW FACT>
{new_fact}
</NEW FACT>

Task:
1. If duplicate, return is_duplicate=true and existing_fact_id.
2. If duplicate, return a canonicalized fact_text/evidence_text if needed.

Return STRICT JSON with:
- is_duplicate: boolean
- existing_fact_id: string | null
- relation_type: string
- fact_text: string
- evidence_text: string
- confidence: number
"""


FACT_TEMPORAL_SYSTEM_PROMPT = """\
You are a temporal information extraction system.

Given the conversation context, a REFERENCE TIMESTAMP (the message time), and a FACT, infer the real-world validity interval for the fact.

<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>

<REFERENCE TIMESTAMP>
{reference_timestamp}
</REFERENCE TIMESTAMP>

<FACT>
{fact}
</FACT>

Guidelines:
1. IMPORTANT: Only extract time information if it is part of the provided FACT. Otherwise ignore time mentioned in the message.
2. Use the REFERENCE TIMESTAMP to resolve relative times (e.g., "two weeks ago", "next Thursday") when they directly relate to the FACT.
3. If the FACT is written in the present tense and establishes the relationship now, set t_valid_from to REFERENCE TIMESTAMP.
4. If the FACT expresses a time range, return that range.
5. If the FACT is a point-in-time event, set t_valid_from to that time and leave t_valid_to null.
6. If no temporal information establishes or changes the relationship, return null for both fields.
7. Return ISO-8601 timestamps with timezone when possible.

Return STRICT JSON with:
- t_valid_from: string
- t_valid_to: string | null
"""


FACT_CONTRADICTION_SYSTEM_PROMPT = """\
You are a contradiction detection system.

Given an EXISTING FACT and a NEW FACT, determine whether the NEW FACT contradicts the EXISTING FACT in the real world.

<EXISTING FACT>
{existing_fact}
</EXISTING FACT>

<NEW FACT>
{new_fact}
</NEW FACT>

Guidelines:
1. Consider relation_type semantics: for some relations (e.g., WORKS_FOR), different objects at overlapping times may contradict.
2. Prefer conservative decisions: only mark contradiction when clear.

Return STRICT JSON with:
- is_contradiction: boolean
- reason: string
"""


ENTITY_REFLECTION_SYSTEM_PROMPT = """\
You are an entity extraction review system performing a reflection step.

Given the conversation context and a PRELIMINARY list of extracted entities, review the extraction for:
1. **Hallucinated entities**: Entities in the preliminary list that are NOT mentioned or implied in the CURRENT MESSAGE. Mark these for removal.
2. **Missing entities**: Entities that ARE mentioned or implied in the CURRENT MESSAGE but were missed. Add these.
3. **Incomplete entities**: Entities with incorrect or missing type_label, summary, or aliases. Correct these.

<PREVIOUS MESSAGES>
{previous_messages}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{current_message}
</CURRENT MESSAGE>

<PRELIMINARY ENTITIES>
{preliminary_entities}
</PRELIMINARY ENTITIES>

Guidelines:
1. The speaker/actor MUST always be present. If missing, add it.
2. Remove any entity that cannot be traced to the CURRENT MESSAGE text.
3. Add any significant entity that was missed.
4. Be conservative: only add entities with clear textual evidence.
5. DO NOT add or retain entities for temporal information like dates, times, or years (e.g. "next Thursday", "2024", "last summer"). These are stored on edges, not as entity nodes.

Return STRICT JSON as a list of objects (the corrected full entity list), each with:
- name: string
- type_label: string (optional)
- summary: string (optional)
- aliases: list[string] (optional)
"""


COMMUNITY_SUMMARY_MAP_SYSTEM_PROMPT = """\
You are a community summarization system.

Given a list of FACTS that belong to the same community, produce a concise partial summary capturing the key relationships and themes.

<FACTS>
{facts}
</FACTS>

Guidelines:
1. Summarize only what is supported by the provided FACTS.
2. Keep it concise and information-dense.

Return STRICT JSON with:
- summary: string
"""


COMMUNITY_SUMMARY_REDUCE_SYSTEM_PROMPT = """\
You are a community summarization system.

Given PARTIAL SUMMARIES for the same community, produce a final summary and a short community name.

The community name should contain key terms and relevant subjects from the summaries, \
enabling effective retrieval via embedding similarity search.

<PARTIAL SUMMARIES>
{partial_summaries}
</PARTIAL SUMMARIES>

Guidelines:
1. The name should contain key terms and relevant subjects (2-8 words) that best represent the community.
2. The summary should be concise and information-dense.

Return STRICT JSON with:
- name: string
- summary: string
"""
