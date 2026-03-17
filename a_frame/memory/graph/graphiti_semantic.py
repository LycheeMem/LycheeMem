"""Graphiti(论文) 风格的语义子图构建：Entity/Fact 的抽取与消歧/去重（PR3）。

设计目标（PR3）：
- 输入：最近 n=4 turns 上下文 + 当前 user turn + reference timestamp
- 输出：在 Neo4j 中写入 Entity/Fact（Fact-node 模型）并建立 Episode→Fact 证据索引

注意：
- 本模块不依赖 FastAPI/Pipeline；由 ConsolidatorAgent 在后台调用。
- LLM 输出必须是严格 JSON；解析失败时应安全降级为空结果。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
import datetime
from typing import Any

from a_frame.embedder.base import BaseEmbedder
from a_frame.llm.base import BaseLLM
from a_frame.memory.graph.graphiti_prompts import (
    COMMUNITY_SUMMARY_MAP_SYSTEM_PROMPT,
    COMMUNITY_SUMMARY_REDUCE_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_TEXT_JSON_SYSTEM_PROMPT,
    ENTITY_REFLECTION_SYSTEM_PROMPT,
    ENTITY_RESOLUTION_SYSTEM_PROMPT,
    FACT_EXTRACTION_SYSTEM_PROMPT,
    FACT_CONTRADICTION_SYSTEM_PROMPT,
    FACT_RESOLUTION_SYSTEM_PROMPT,
    FACT_TEMPORAL_SYSTEM_PROMPT,
)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# Paper §2.2.1 / Appendix §6.1.1: "DO NOT create nodes for temporal information
# like dates, times or years".
# Matches bare years (4-digit), date strings (YYYY-MM-DD, MM/DD/YYYY etc.),
# ISO timestamps, relative-time phrases like "next Thursday" / "last summer",
# and pure month/day names without a subject noun.
_TEMPORAL_ENTITY_RE = re.compile(
    r'^('  # full-name matches only
    # bare 4-digit year
    r'\d{4}'
    # ISO date / datetime
    r'|\d{4}-\d{2}-\d{2}(T[\d:Z.+-]+)?'
    # US / EU numeric dates
    r'|\d{1,2}[/\-\.]\d{1,2}([/\-\.]\d{2,4})?'
    # relative-time phrases: "last/next/this + time-word"
    r'|(?:last|next|this|past|coming)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december|summer|winter|fall|spring|quarter)'
    # bare time expressions: "two weeks ago", "in three days", etc.
    r'|(?:in\s+)?(?:a\s+)?(?:few|couple\s+of|\d+)\s+(?:second|minute|hour|day|week|month|year)s?(?:\s+ago)?'
    # just a month name alone
    r'|(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:,?\s*\d{4})?'
    r')$',
    re.IGNORECASE,
)


def _is_temporal_entity(name: str) -> bool:
    """Return True if *name* looks like a bare date/time/year token that should
    be stored on edges rather than as an entity node (paper §2.2.1)."""
    return bool(_TEMPORAL_ENTITY_RE.match(name.strip()))


def _safe_json_loads(text: str) -> Any:
    """尽可能安全地解析 JSON。

    兼容一些 LLM 常见输出：
    - 前后多余空白
    - 以 code fence 包裹（```json ... ```）
    """

    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _format_turns(turns: list[dict[str, Any]]) -> str:
    role_map = {
        "user": "User",
        "assistant": "Assistant",
        "system": "System",
    }

    lines: list[str] = []
    for t in turns or []:
        if not t:
            continue
        role = str(t.get("role") or "")
        content = str(t.get("content") or "")
        actor = role_map.get(role, role or "User")
        lines.append(f"{actor}: {content}")
    return "\n".join(lines)


def _format_current_message(turn: dict[str, Any]) -> str:
    """Format a single episode turn into a current_message string.

    Paper §2.1: message / text / JSON each need specific handling.
    - message: 'Actor: content' dialogue format (speaker extracted as entity).
    - text:    raw text with no speaker prefix.
    - json:    raw serialised JSON with no speaker prefix.
    """
    role = str(turn.get("role") or "")
    content = str(turn.get("content") or "")
    if role in ("text", "json"):
        return content
    actor = {"user": "User", "assistant": "Assistant", "system": "System"}.get(role, role or "User")
    return f"{actor}: {content}"


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@dataclass(slots=True)
class ResolvedEntity:
    entity_id: str
    name: str
    summary: str = ""
    aliases: list[str] | None = None
    type_label: str = ""
    # True when entity resolution identified a duplicate and the LLM produced
    # a merged summary (paper §2.2.1 "generate an updated name and summary").
    # Callers should pass force_summary=True to upsert_entity so the
    # LLM-fused summary is never discarded by the length-comparison safety-net.
    is_duplicate: bool = False


@dataclass(slots=True)
class ExtractedFact:
    subject_entity_id: str
    object_entity_id: str
    relation_type: str
    fact_text: str
    evidence_text: str
    confidence: float = 1.0


class GraphitiSemanticBuilder:
    """Graphiti 语义子图构建器。

    依赖 store 提供：
    - upsert_entity / candidate search
    - upsert_fact / list facts between entities
    - link_episode_to_fact
    """

    def __init__(
        self,
        *,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        store: Any,
        entity_candidate_top_k: int = 8,
        entity_embedding_threshold: float = 0.82,
        entity_embedding_scan_limit: int = 2000,
    ):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.entity_candidate_top_k = entity_candidate_top_k
        self.entity_embedding_threshold = entity_embedding_threshold
        self.entity_embedding_scan_limit = entity_embedding_scan_limit

    def _call_llm(self, *, system_prompt: str, user_content: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.llm.generate(messages)

    def _update_community_summary(self, community_id: str) -> None:
        """Paper §2.3: update community summary after dynamic extension.

        Fetches intra-community facts and runs map-reduce summarization
        to regenerate the community name and summary, then re-embeds.
        """
        if not hasattr(self.store, "get_community_member_facts"):
            return
        if not hasattr(self.store, "upsert_community"):
            return
        facts = self.store.get_community_member_facts(
            community_id=community_id, limit=50
        )
        if not facts:
            return

        # Map phase
        chunk_size = 10
        partials: list[str] = []
        for i in range(0, len(facts), chunk_size):
            chunk = facts[i : i + chunk_size]
            prompt = COMMUNITY_SUMMARY_MAP_SYSTEM_PROMPT.format(
                facts="\n".join(f"- {x}" for x in chunk)
            )
            raw = self._call_llm(system_prompt=prompt, user_content="Return JSON only.")
            data = _safe_json_loads(raw)
            if isinstance(data, dict) and str(data.get("summary") or "").strip():
                partials.append(str(data["summary"]).strip())

        # Reduce phase
        reduce_prompt = COMMUNITY_SUMMARY_REDUCE_SYSTEM_PROMPT.format(
            partial_summaries="\n".join(f"- {s}" for s in partials) if partials else ""
        )
        raw = self._call_llm(system_prompt=reduce_prompt, user_content="Return JSON only.")
        data = _safe_json_loads(raw)
        if not isinstance(data, dict):
            return
        name = str(data.get("name") or "").strip()
        summary = str(data.get("summary") or "").strip()
        if not summary:
            return

        # Re-embed community name (paper §2.3)
        embedding = None
        if name:
            try:
                embedding = self.embedder.embed_query(name)
            except Exception:
                pass

        self.store.upsert_community(
            community_id=community_id,
            name=name,
            summary=summary,
            embedding=embedding,
        )

    def _entity_id_from_name(self, name: str) -> str:
        return _sha1(name.strip().lower())

    def extract_entities(
        self,
        *,
        previous_turns: list[dict[str, Any]],
        current_turn: dict[str, Any],
    ) -> list[dict[str, Any]]:
        previous_messages = _format_turns(previous_turns)
        current_message = _format_current_message(current_turn)

        # Paper §2.1: text/json episodes have no speaker — use dedicated prompt.
        role = str(current_turn.get("role") or "")
        extraction_prompt = (
            ENTITY_EXTRACTION_TEXT_JSON_SYSTEM_PROMPT
            if role in ("text", "json")
            else ENTITY_EXTRACTION_SYSTEM_PROMPT
        )
        prompt = extraction_prompt.format(
            previous_messages=previous_messages,
            current_message=current_message,
        )
        raw = self._call_llm(system_prompt=prompt, user_content="Return JSON only.")
        data = _safe_json_loads(raw)
        if not isinstance(data, list):
            return []
        out: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            out.append(
                {
                    "name": name,
                    "type_label": str(item.get("type_label") or "").strip(),
                    "summary": str(item.get("summary") or "").strip(),
                    "aliases": item.get("aliases") if isinstance(item.get("aliases"), list) else [],
                }
            )
        return out

    def reflect_entities(
        self,
        *,
        previous_turns: list[dict[str, Any]],
        current_turn: dict[str, Any],
        preliminary_entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Paper §2.2.1: Reflexion technique to minimize hallucinations and enhance coverage.

        Takes preliminary extraction results and runs a second LLM pass to:
        - Remove hallucinated entities not grounded in the CURRENT MESSAGE
        - Add missing entities that were overlooked
        - Correct incomplete type_label / summary / aliases
        """
        if not preliminary_entities:
            return []

        previous_messages = _format_turns(previous_turns)
        current_message = _format_current_message(current_turn)

        prompt = ENTITY_REFLECTION_SYSTEM_PROMPT.format(
            previous_messages=previous_messages,
            current_message=current_message,
            preliminary_entities=json.dumps(preliminary_entities, ensure_ascii=False),
        )
        raw = self._call_llm(system_prompt=prompt, user_content="Return JSON only.")
        data = _safe_json_loads(raw)
        if not isinstance(data, list):
            # Reflection failed to parse; fall back to preliminary results.
            return preliminary_entities

        out: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            # Paper §2.2.1 / Appendix §6.1.1: "DO NOT create nodes for temporal
            # information like dates, times or years".
            # The reflection prompt carries this instruction, but LLMs can still
            # slip temporal tokens through.  Guard programmatically by rejecting
            # names that look like bare dates, years, or time expressions.
            if _is_temporal_entity(name):
                continue
            out.append(
                {
                    "name": name,
                    "type_label": str(item.get("type_label") or "").strip(),
                    "summary": str(item.get("summary") or "").strip(),
                    "aliases": item.get("aliases") if isinstance(item.get("aliases"), list) else [],
                }
            )
        # If reflection returned empty but preliminary was not, keep preliminary.
        return out if out else preliminary_entities

    def resolve_entity(
        self,
        *,
        previous_turns: list[dict[str, Any]],
        current_turn: dict[str, Any],
        new_entity: dict[str, Any],
    ) -> ResolvedEntity:
        name = str(new_entity.get("name") or "").strip()
        if not name:
            return ResolvedEntity(entity_id=self._entity_id_from_name(""), name="")

        # Candidate search (hybrid): fulltext + vector index (Neo4j native) (best-effort)
        candidates: list[dict[str, Any]] = []
        try:
            candidates.extend(
                self.store.fulltext_search_entities(query=name, limit=self.entity_candidate_top_k)
            )
        except Exception:
            pass

        try:
            q_emb = self.embedder.embed_query(name)
            if hasattr(self.store, "vector_search_entities"):
                rows = self.store.vector_search_entities(
                    query_embedding=q_emb,
                    limit=max(self.entity_candidate_top_k * 3, 20),
                )
                for r in rows:
                    try:
                        score = float(r.get("score") or 0.0)
                    except Exception:
                        score = 0.0
                    if score >= float(self.entity_embedding_threshold):
                        candidates.append(r)
            elif hasattr(self.store, "scan_entities_with_embeddings"):
                # Legacy fallback when vector index is unavailable.
                emb_candidates = self.store.scan_entities_with_embeddings(
                    limit=self.entity_embedding_scan_limit
                )
                scored = []
                for c in emb_candidates:
                    emb = c.get("embedding")
                    if isinstance(emb, list):
                        score = _cosine(q_emb, emb)
                        if score >= self.entity_embedding_threshold:
                            scored.append((score, c))
                scored.sort(key=lambda x: x[0], reverse=True)
                for _, c in scored[: self.entity_candidate_top_k]:
                    candidates.append(c)
        except Exception:
            pass

        # Dedup candidates by entity_id
        seen = set()
        uniq_candidates = []
        for c in candidates:
            eid = str(c.get("entity_id") or "").strip()
            if not eid or eid in seen:
                continue
            seen.add(eid)
            uniq_candidates.append(
                {
                    "entity_id": eid,
                    "name": c.get("name") or "",
                    "summary": c.get("summary") or "",
                    "aliases": c.get("aliases") or [],
                    "type_label": c.get("type_label") or "",
                }
            )

        previous_messages = _format_turns(previous_turns)
        current_message = _format_current_message(current_turn)

        prompt = ENTITY_RESOLUTION_SYSTEM_PROMPT.format(
            previous_messages=previous_messages,
            current_message=current_message,
            existing_nodes=json.dumps(uniq_candidates, ensure_ascii=False),
            new_node=json.dumps(new_entity, ensure_ascii=False),
        )
        raw = self._call_llm(system_prompt=prompt, user_content="Return JSON only.")
        data = _safe_json_loads(raw)
        if not isinstance(data, dict):
            # fallback: create new entity id
            return ResolvedEntity(
                entity_id=self._entity_id_from_name(name),
                name=name,
                summary=str(new_entity.get("summary") or ""),
                aliases=list(new_entity.get("aliases") or []),
                type_label=str(new_entity.get("type_label") or ""),
            )

        is_dup = bool(data.get("is_duplicate"))
        existing_id = str(data.get("existing_entity_id") or "").strip() or None
        resolved_name = str(data.get("name") or name).strip()
        resolved_aliases = (
            data.get("aliases")
            if isinstance(data.get("aliases"), list)
            else list(new_entity.get("aliases") or [])
        )
        resolved_type = str(data.get("type_label") or new_entity.get("type_label") or "").strip()

        # Paper §2.2.1: when the LLM confirms a duplicate it is also asked to
        # "generate an updated name and summary" having seen both the existing
        # entity summary (in existing_nodes) and the new extraction.
        # We therefore prefer data["summary"] (the LLM fusion) over the raw
        # new_entity summary, but only fall back to the latter when the LLM
        # returned nothing.
        llm_fused_summary = str(data.get("summary") or "").strip()
        if is_dup and llm_fused_summary:
            # Genuine LLM-merged summary: mark for unconditional write-back.
            resolved_summary = llm_fused_summary
            force_summary = True
        else:
            # New entity or LLM returned empty summary: use extraction summary;
            # let the Cypher length-guard decide on concurrent-write conflicts.
            resolved_summary = str(data.get("summary") or new_entity.get("summary") or "").strip()
            force_summary = False

        if is_dup and existing_id:
            entity_id = existing_id
        else:
            entity_id = self._entity_id_from_name(resolved_name)

        return ResolvedEntity(
            entity_id=entity_id,
            name=resolved_name,
            summary=resolved_summary,
            aliases=resolved_aliases,
            type_label=resolved_type,
            is_duplicate=force_summary,
        )

    def extract_facts(
        self,
        *,
        previous_turns: list[dict[str, Any]],
        current_turn: dict[str, Any],
        entities: list[ResolvedEntity],
    ) -> list[dict[str, Any]]:
        if not entities:
            return []

        previous_messages = _format_turns(previous_turns)
        current_message = _format_current_message(current_turn)
        entity_payload = [
            {
                "entity_id": e.entity_id,
                "name": e.name,
                "summary": e.summary,
                "type_label": e.type_label,
            }
            for e in entities
        ]

        prompt = FACT_EXTRACTION_SYSTEM_PROMPT.format(
            previous_messages=previous_messages,
            current_message=current_message,
            entities=json.dumps(entity_payload, ensure_ascii=False),
        )
        raw = self._call_llm(system_prompt=prompt, user_content="Return JSON only.")
        data = _safe_json_loads(raw)
        if not isinstance(data, list):
            return []

        out: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            subj = str(item.get("subject") or "").strip()
            obj = str(item.get("object") or "").strip()
            rel = str(item.get("relation_type") or "").strip()
            fact_text = str(item.get("fact_text") or "").strip()
            if not subj or not obj or not rel or not fact_text:
                continue
            out.append(
                {
                    "subject": subj,
                    "object": obj,
                    "relation_type": rel,
                    "fact_text": fact_text,
                    "evidence_text": str(item.get("evidence_text") or "").strip(),
                    "confidence": float(item.get("confidence") or 1.0),
                }
            )
        return out

    def resolve_fact(
        self,
        *,
        existing_facts: list[dict[str, Any]],
        new_fact: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = FACT_RESOLUTION_SYSTEM_PROMPT.format(
            existing_facts=json.dumps(existing_facts, ensure_ascii=False),
            new_fact=json.dumps(new_fact, ensure_ascii=False),
        )
        raw = self._call_llm(system_prompt=prompt, user_content="Return JSON only.")
        data = _safe_json_loads(raw)
        if not isinstance(data, dict):
            return {
                "is_duplicate": False,
                "existing_fact_id": None,
                "relation_type": new_fact.get("relation_type") or "",
                "fact_text": new_fact.get("fact_text") or "",
                "evidence_text": new_fact.get("evidence_text") or "",
                "confidence": float(new_fact.get("confidence") or 1.0),
            }

        return {
            "is_duplicate": bool(data.get("is_duplicate")),
            "existing_fact_id": (str(data.get("existing_fact_id") or "").strip() or None),
            "relation_type": str(
                data.get("relation_type") or new_fact.get("relation_type") or ""
            ).strip(),
            "fact_text": str(data.get("fact_text") or new_fact.get("fact_text") or "").strip(),
            "evidence_text": str(
                data.get("evidence_text") or new_fact.get("evidence_text") or ""
            ).strip(),
            "confidence": float(data.get("confidence") or new_fact.get("confidence") or 1.0),
        }

    def extract_fact_temporal(
        self,
        *,
        previous_turns: list[dict[str, Any]],
        current_turn: dict[str, Any],
        reference_timestamp: str,
        fact: dict[str, Any],
    ) -> dict[str, str | None]:
        previous_messages = _format_turns(previous_turns)
        current_message = _format_current_message(current_turn)

        prompt = FACT_TEMPORAL_SYSTEM_PROMPT.format(
            previous_messages=previous_messages,
            current_message=current_message,
            reference_timestamp=reference_timestamp,
            fact=json.dumps(fact, ensure_ascii=False),
        )
        raw = self._call_llm(system_prompt=prompt, user_content="Return JSON only.")
        data = _safe_json_loads(raw)
        if not isinstance(data, dict):
            return {"t_valid_from": None, "t_valid_to": None}

        t_from = data.get("t_valid_from")
        if t_from is not None:
            t_from = str(t_from).strip() or None
        t_to = data.get("t_valid_to")
        if t_to is not None:
            t_to = str(t_to).strip() or None
        return {"t_valid_from": t_from, "t_valid_to": t_to}

    def is_contradiction(self, *, existing_fact: dict[str, Any], new_fact: dict[str, Any]) -> bool:
        prompt = FACT_CONTRADICTION_SYSTEM_PROMPT.format(
            existing_fact=json.dumps(existing_fact, ensure_ascii=False),
            new_fact=json.dumps(new_fact, ensure_ascii=False),
        )
        raw = self._call_llm(system_prompt=prompt, user_content="Return JSON only.")
        data = _safe_json_loads(raw)
        if not isinstance(data, dict):
            return False
        return bool(data.get("is_contradiction"))

    @staticmethod
    def _relation_group(relation_type: str) -> str:
        """将 relation_type 归并为“语义组”，用于扩大矛盾候选集。

        目标：比 “同 subject + 同 relation” 更宽，但仍保持保守。
        """

        rel = (relation_type or "").strip().upper()
        groups: dict[str, set[str]] = {
            "EMPLOYMENT": {"WORKS_FOR", "EMPLOYED_BY", "HAS_TITLE"},
            "LOCATION": {"LIVES_IN", "LOCATED_IN", "BORN_IN"},
            "OWNERSHIP": {"OWNS"},
        }
        for g, rels in groups.items():
            if rel in rels:
                return g
        return rel

    @staticmethod
    def _contradiction_heuristic(
        *,
        new_relation_type: str,
        old_relation_type: str,
        old_object: str,
        new_object: str,
    ) -> bool:
        # Minimal conservative heuristic: some relation groups are typically mutually exclusive
        # for a given subject at overlapping time.
        exclusive = {
            "WORKS_FOR",
            "LIVES_IN",
            "LOCATED_IN",
            "HAS_TITLE",
            "OWNS",
            "EMPLOYED_BY",
        }
        new_rel = (new_relation_type or "").strip().upper()
        old_rel = (old_relation_type or "").strip().upper()
        if new_rel not in exclusive or old_rel not in exclusive:
            return False
        if GraphitiSemanticBuilder._relation_group(
            new_rel
        ) != GraphitiSemanticBuilder._relation_group(old_rel):
            return False
        return bool(old_object and new_object and old_object != new_object)

    @staticmethod
    def _parse_iso(ts: str | None) -> datetime.datetime | None:
        if not ts:
            return None
        s = str(ts).strip()
        if not s:
            return None
        # tolerate Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt

    @classmethod
    def _intervals_overlap(
        cls,
        *,
        a_from: str | None,
        a_to: str | None,
        b_from: str | None,
        b_to: str | None,
    ) -> bool:
        """Validity interval overlap check.

        Treat missing *_to as open interval.
        Paper §2.2.3: when both facts lack t_valid_from they are
        temporally indeterminate — treat as overlapping so that
        semantic contradiction can still trigger invalidation.
        If only one side has t_valid_from, we cannot determine
        overlap and return False (conservative).
        """

        a_from_s = (a_from or "").strip()
        a_to_s = (a_to or "").strip()
        b_from_s = (b_from or "").strip()
        b_to_s = (b_to or "").strip()

        # Both lack temporal info → indeterminate; assume overlap so
        # contradiction-based invalidation can still proceed.
        if not a_from_s and not b_from_s:
            return True

        # Only one side has temporal info → cannot determine overlap.
        if not a_from_s or not b_from_s:
            return False

        # Fast path: attempt datetime comparison
        a_from_dt = cls._parse_iso(a_from_s)
        a_to_dt = cls._parse_iso(a_to_s) if a_to_s else None
        b_from_dt = cls._parse_iso(b_from_s)
        b_to_dt = cls._parse_iso(b_to_s) if b_to_s else None
        if a_from_dt and b_from_dt:
            a_to_dt = a_to_dt or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)
            b_to_dt = b_to_dt or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)
            return a_from_dt <= b_to_dt and b_from_dt <= a_to_dt

        # Fallback: lexical compare for ISO strings
        a_to_cmp = a_to_s or "9999-12-31T23:59:59+00:00"
        b_to_cmp = b_to_s or "9999-12-31T23:59:59+00:00"
        return a_from_s <= b_to_cmp and b_from_s <= a_to_cmp

    def ingest_user_turn(
        self,
        *,
        session_id: str,
        episode_id: str,
        previous_turns: list[dict[str, Any]],
        current_turn: dict[str, Any],
        reference_timestamp: str,
        episode_type: str = "message",
    ) -> dict[str, int]:
        """从单个 Episode 里抽取 Entity/Fact 并写入 store。

        Paper §2.1: Episodes can be one of three core types — message, text,
        or JSON.  Each type requires specific handling during graph construction.
        """

        if not hasattr(self.store, "link_episode_to_entity"):
            raise RuntimeError(
                "Graphiti semantic build requires store.link_episode_to_entity() (Episode→Entity episodic edges)"
            )

        # ── Episode-type-aware preprocessing (Paper §2.1) ──
        # "Episodes can be one of three core types: message, text, or JSON.
        #  While each type requires specific handling during graph construction…"
        ep_type = str(episode_type or "message").strip().lower()
        if ep_type == "text":
            # text episodes: raw text with no speaker/actor metadata.
            # Wrap content as-is; previous_turns still use dialogue format
            # (they may contain earlier messages).  No speaker extraction.
            current_turn = {
                "role": "text",
                "content": str(current_turn.get("content") or ""),
            }
        elif ep_type == "json":
            # json episodes: structured data.  Serialize as pretty-printed JSON
            # so the LLM can reason about structured fields.
            raw_content = current_turn.get("content")
            if isinstance(raw_content, (dict, list)):
                formatted = json.dumps(raw_content, ensure_ascii=False, indent=2)
            else:
                formatted = str(raw_content or "")
            current_turn = {"role": "json", "content": formatted}
        # else: 'message' — no transformation needed (existing dialogue format).

        # 1) entity extraction + reflexion (paper §2.2.1) + resolution + upsert
        extracted_entities = self.extract_entities(
            previous_turns=previous_turns, current_turn=current_turn
        )
        # Paper §2.2.1: "we employ a reflection technique inspired by reflexion[12]
        # to minimize hallucinations and enhance extraction coverage."
        extracted_entities = self.reflect_entities(
            previous_turns=previous_turns,
            current_turn=current_turn,
            preliminary_entities=extracted_entities,
        )
        resolved: list[ResolvedEntity] = []
        for ent in extracted_entities:
            r = self.resolve_entity(
                previous_turns=previous_turns, current_turn=current_turn, new_entity=ent
            )
            if not r.name or not r.entity_id:
                continue
            try:
                emb = self.embedder.embed_query(r.name)
            except Exception:
                emb = None

            self.store.upsert_entity(
                entity_id=r.entity_id,
                name=r.name,
                summary=r.summary,
                aliases=r.aliases or [],
                type_label=r.type_label,
                embedding=emb,
                source_session=session_id,
                t_created=reference_timestamp,
                # Paper §2.2.1: when is_duplicate=True the summary is the
                # LLM-fused version; write it unconditionally.
                force_summary=r.is_duplicate,
            )

            # Paper: episodic edges linking Episode → referenced semantic entities.
            self.store.link_episode_to_entity(episode_id=episode_id, entity_id=r.entity_id)
            resolved.append(r)

        # 1.5) Ingestion-time community dynamic extension (paper §2.3)
        # "When the system adds a new entity node to the graph, it surveys the
        # communities of neighboring nodes. The system assigns the new node to the
        # community held by the plurality of its neighbors, then updates the
        # community summary and graph accordingly."
        if hasattr(self.store, "get_neighbor_communities") and hasattr(
            self.store, "link_entity_to_community"
        ):
            for r in resolved:
                try:
                    neighbor_comms = self.store.get_neighbor_communities(entity_id=r.entity_id)
                    if neighbor_comms:
                        # Assign to the community with the highest neighbor count (plurality).
                        best = neighbor_comms[0]
                        best_community_id = str(best.get("community_id") or "").strip()
                        if best_community_id:
                            self.store.link_entity_to_community(
                                entity_id=r.entity_id, community_id=best_community_id
                            )
                            # Paper §2.3: "then updates the community summary
                            # and graph accordingly."
                            self._update_community_summary(best_community_id)
                except Exception:
                    # Best-effort; community assignment should not block entity ingestion.
                    continue

        # 2) fact extraction + resolution + upsert
        extracted_facts = self.extract_facts(
            previous_turns=previous_turns, current_turn=current_turn, entities=resolved
        )

        name_to_id = {e.name: e.entity_id for e in resolved}
        facts_added = 0
        facts_expired = 0
        for f in extracted_facts:
            subj_name = f.get("subject")
            obj_name = f.get("object")
            subj_id = name_to_id.get(subj_name)
            obj_id = name_to_id.get(obj_name)
            if not subj_id or not obj_id or subj_id == obj_id:
                continue

            existing = []
            try:
                # Paper §2.2.2: "The hybrid search for relevant edges is
                # constrained to edges existing between the same entity pairs
                # as the proposed new edge."  — includes both directions.
                forward = self.store.list_facts_between(
                    subject_entity_id=subj_id,
                    object_entity_id=obj_id,
                    limit=20,
                )
                reverse = (
                    self.store.list_facts_between(
                        subject_entity_id=obj_id,
                        object_entity_id=subj_id,
                        limit=20,
                    )
                    if subj_id != obj_id
                    else []
                )
                seen_ids: set[str] = set()
                for rec in forward + reverse:
                    fid = str(rec.get("fact_id") or "").strip()
                    if fid and fid not in seen_ids:
                        existing.append(rec)
                        seen_ids.add(fid)
            except Exception:
                existing = []

            resolved_fact = self.resolve_fact(existing_facts=existing, new_fact=f)
            fact_id = resolved_fact.get("existing_fact_id")
            if not fact_id:
                # deterministic id for idempotency under re-processing
                fact_id = _sha1(
                    f"{subj_id}|{resolved_fact.get('relation_type', '')}|{obj_id}|{resolved_fact.get('fact_text', '')}".lower()
                )

            temporal = self.extract_fact_temporal(
                previous_turns=previous_turns,
                current_turn=current_turn,
                reference_timestamp=reference_timestamp,
                fact=resolved_fact,
            )
            t_valid_from = temporal.get("t_valid_from")
            t_valid_to = temporal.get("t_valid_to")

            # Invalidation: expire older active facts that contradict this one.
            # Paper §2.2.3: "The system employs an LLM to compare new edges against
            # semantically related existing edges to identify potential contradictions."
            relation_type = str(resolved_fact.get("relation_type") or "").strip().upper()

            # Compute fact embedding early — needed for both upsert and semantic invalidation.
            fact_embedding = None
            try:
                fact_for_embed = (
                    f"{subj_name or ''} --{relation_type}--> {obj_name or ''}: {resolved_fact.get('fact_text') or ''}"
                ).strip()
                fact_embedding = self.embedder.embed_query(fact_for_embed)
            except Exception:
                fact_embedding = None

            if hasattr(self.store, "list_active_facts_for_subject_relation") and hasattr(
                self.store, "expire_fact"
            ):
                # Phase 1: Same-subject active facts (original scope, relation-group filtered).
                try:
                    if hasattr(self.store, "list_active_facts_for_subject"):
                        active = self.store.list_active_facts_for_subject(
                            subject_entity_id=subj_id, limit=200
                        )
                    else:
                        active = self.store.list_active_facts_for_subject_relation(
                            subject_entity_id=subj_id, relation_type=relation_type, limit=50
                        )
                except Exception:
                    active = []

                new_group = self._relation_group(relation_type)
                active = [
                    a
                    for a in (active or [])
                    if self._relation_group(str(a.get("relation_type") or "")) == new_group
                ]

                # Phase 2 (paper §2.2.3): Semantic similarity search across ALL entity pairs.
                # "The system employs an LLM to compare new edges against semantically
                # related existing edges to identify potential contradictions."
                if (
                    fact_embedding is not None
                    and hasattr(self.store, "search_active_facts_by_embedding")
                ):
                    try:
                        semantic_candidates = self.store.search_active_facts_by_embedding(
                            query_embedding=fact_embedding, limit=20
                        )
                        # Merge into active, dedup by fact_id.
                        existing_fids = {
                            str(a.get("fact_id") or "").strip() for a in active
                        }
                        for sc in semantic_candidates:
                            sc_fid = str(sc.get("fact_id") or "").strip()
                            if sc_fid and sc_fid not in existing_fids:
                                active.append(sc)
                                existing_fids.add(sc_fid)
                    except Exception:
                        pass

                new_payload = {
                    "subject_entity_id": subj_id,
                    "object_entity_id": obj_id,
                    "relation_type": relation_type,
                    "fact_text": resolved_fact.get("fact_text") or "",
                    "t_valid_from": t_valid_from,
                    "t_valid_to": t_valid_to,
                }

                for old in active:
                    try:
                        old_fact_id = str(old.get("fact_id") or "").strip()
                        old_object = str(old.get("object_entity_id") or "").strip()
                        old_relation = str(old.get("relation_type") or "").strip().upper()
                        if not old_fact_id or old_fact_id == fact_id:
                            continue

                        # Only invalidate if validity intervals overlap and overlap is determinable.
                        old_from = str(old.get("t_valid_from") or "").strip() or None
                        old_to = str(old.get("t_valid_to") or "").strip() or None
                        new_from = str(t_valid_from).strip() if t_valid_from else None
                        new_to = str(t_valid_to).strip() if t_valid_to else None
                        if not self._intervals_overlap(
                            a_from=old_from,
                            a_to=old_to,
                            b_from=new_from,
                            b_to=new_to,
                        ):
                            continue

                        if not self._contradiction_heuristic(
                            new_relation_type=relation_type,
                            old_relation_type=old_relation,
                            old_object=old_object,
                            new_object=obj_id,
                        ):
                            # Heuristic says "not clearly exclusive"; try LLM contradiction (best-effort)
                            if not self.is_contradiction(existing_fact=old, new_fact=new_payload):
                                continue

                        self.store.expire_fact(
                            fact_id=old_fact_id,
                            t_valid_to=t_valid_from,
                            t_tx_expired=reference_timestamp,
                        )
                        facts_expired += 1
                    except Exception:
                        continue

            # Paper §2.2.2 hyper-edge: canonical_fact_hash groups semantically
            # identical facts across different entity pairs.
            canonical_text = (resolved_fact.get("fact_text") or "").strip().lower()
            canonical_fact_hash = _sha1(canonical_text) if canonical_text else None

            self.store.upsert_fact(
                fact_id=fact_id,
                subject_entity_id=subj_id,
                object_entity_id=obj_id,
                relation_type=relation_type,
                fact_text=resolved_fact.get("fact_text") or "",
                evidence_text=resolved_fact.get("evidence_text") or "",
                embedding=fact_embedding,
                confidence=float(resolved_fact.get("confidence") or 1.0),
                source_session=session_id,
                canonical_fact_hash=canonical_fact_hash,
                t_created=reference_timestamp,
                t_valid_from=t_valid_from,
                t_valid_to=t_valid_to,
                t_tx_created=reference_timestamp,
            )
            self.store.link_episode_to_fact(episode_id=episode_id, fact_id=fact_id)
            facts_added += 1

        return {
            "entities_added": len(resolved),
            "facts_added": facts_added,
            "facts_expired": facts_expired,
        }
