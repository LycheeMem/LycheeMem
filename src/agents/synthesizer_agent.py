"""
整合排序器 (Memory Synthesizer & Ranker)。

对多源召回的记忆片段进行：
- LLM-as-judge 二元有效性打分
- 去重与聚类融合
- 输出精炼的 Background Context
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agents.base_agent import BaseAgent
from src.llm.base import BaseLLM

SYNTHESIS_SYSTEM_PROMPT = """\
You are a strict Memory Synthesizer and Judge.
You will receive the user's current task and several raw memory fragments retrieved from different memory sources (`semantic` / `skill`).

Your tasks:
1. Evaluate the absolute contribution of each memory fragment to the current task.
2. Discard fragments that are clearly irrelevant or low value.
3. Deduplicate and fuse the retained high-value fragments into a dense `background_context`.

Scoring and threshold strategy:
- You may first score contribution mentally on a 0-10 scale, then normalize it to 0.0-1.0 in the `relevance` field.
- Rough equivalence: 6/10 ≈ 0.6.
- Try to discard fragments with relevance < 0.6 and keep only genuinely useful content.

Reply strictly in JSON using exactly the following structure and field names:
{
    "scored_fragments": [
        {"source": "semantic|skill", "index": 0, "relevance": 0.95, "summary": "Condensed key point of this fragment"}
    ],
    "kept_count": number_of_kept_fragments,
    "dropped_count": number_of_dropped_fragments,
    "background_context": "Integrated background knowledge text that can be injected directly as system context"
}

Rules:
- If all fragments are irrelevant, `background_context` must be an empty string.
- `background_context` should be a dense fused text. Do not simply concatenate originals; compress and rewrite the information.
- Keep facts accurate and do not invent information absent from the retrieved fragments.
- If fragments include time annotations, explicitly use them and distinguish between memory write time (`created_at`) and event / fact time (`temporal`).
- During synthesis, do not incorrectly merge facts from different time periods into one tense or conclusion. For constraints, states, failures, and task progress, prioritize the time conditions most relevant to the current problem.
- Sort `scored_fragments` by `relevance` in descending order. Use `summary` to briefly describe the fragment's core information.

## Example (for reference only, do not copy verbatim)

User query:
    "Review the historical issues in this project related to 'user-service timeout' so I can avoid repeating the same mistakes in this investigation."

Fragments from different memory sources (already prepared by the system):
- [semantic] Fragment 0: A historical failure record in semantic memory says, "Last time the overall timeout was caused by a slow downstream payment-service."
- [skill] Fragment 1: A skill exists in the skill library with the intent "troubleshoot user-service timeout", containing a Markdown skill document with steps, commands, and notes.

Expected JSON output:
{
    "scored_fragments": [
        {"source": "skill", "index": 1, "relevance": 0.95, "summary": "A specialized multi-step troubleshooting skill for user-service timeout exists and its checking order can be reused directly."},
        {"source": "semantic", "index": 0, "relevance": 0.85, "summary": "Historical semantic memory indicates that the timeout on 2024-01-15 was mainly caused by downstream payment-service slowdown."}
    ],
    "kept_count": 2,
    "dropped_count": 0,
    "background_context": "Historical information shows that user-service timeouts were strongly related to payment-service performance issues, and there is already a mature troubleshooting skill that includes checking gateway QPS, user-service error rate, and downstream dependency status. For this investigation, reuse that troubleshooting order first and pay special attention to payment-service and gateway traffic so the same failure pattern is not repeated."
}
"""


class SynthesizerAgent(BaseAgent):
    """整合排序器：将多源检索结果融合为 Background Context。"""

    _SEMANTIC_SOURCE_ALIASES = {
        "graph": "semantic",
        "graphiti": "semantic",
        "graphiti_retrieval": "semantic",
        "graphiti_context": "semantic",
        "graphiti_community": "semantic",
        "compact_semantic": "semantic",
        "record": "semantic",
        "composite": "semantic",
        "semantic": "semantic",
        "skill": "skill",
    }

    def __init__(self, llm: BaseLLM):
        super().__init__(llm=llm, prompt_template=SYNTHESIS_SYSTEM_PROMPT)

    def run(
        self,
        user_query: str,
        retrieved_graph_memories: list[dict[str, Any]] | None = None,
        retrieved_skills: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """将多源检索结果整合为 background_context + 技能复用计划。

        三步流程：score → rank → fuse
        对标记了 reusable=True 的技能，输出结构化执行计划。

        Returns:
            dict 包含：background_context, skill_reuse_plan, provenance
        """
        skills = retrieved_skills or []

        fragments = self._collect_fragments(
            retrieved_graph_memories or [],
            skills,
        )

        # 构建可复用技能执行计划
        skill_reuse_plan = self._build_reuse_plan(skills)

        if not fragments:
            return {
                "background_context": "",
                "skill_reuse_plan": skill_reuse_plan,
                "provenance": [],
                "kept_count": 0,
                "dropped_count": 0,
                "input_fragment_count": 0,
            }

        fragments_text = self._format_fragments(fragments)
        user_content = f"User query: {user_query}\n\nRetrieved memory fragments:\n{fragments_text}"
        response = self._call_llm(
            user_content,
            system_content=self.prompt_template,
            add_time_basis=True,
        )

        try:
            parsed = self._parse_json(response)
            provenance = self._materialize_provenance(
                parsed.get("scored_fragments", []),
                fragments,
            )
            kept_count = len(provenance)
            dropped_count = max(0, len(fragments) - kept_count)
            return {
                "background_context": parsed.get("background_context", ""),
                "skill_reuse_plan": skill_reuse_plan,
                "provenance": provenance,
                "kept_count": kept_count,
                "dropped_count": dropped_count,
                "input_fragment_count": len(fragments),
            }
        except (ValueError, KeyError):
            fallback_provenance = self._fallback_provenance(fragments)
            return {
                "background_context": response,
                "skill_reuse_plan": skill_reuse_plan,
                "provenance": fallback_provenance,
                "kept_count": len(fallback_provenance),
                "dropped_count": max(0, len(fragments) - len(fallback_provenance)),
                "input_fragment_count": len(fragments),
            }

    @staticmethod
    def _build_reuse_plan(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从标记了 reusable=True 的技能构建"可复用技能文档列表"。"""
        plan = []
        for skill in skills:
            if skill.get("reusable"):
                plan.append(
                    {
                        "skill_id": skill.get("id", ""),
                        "intent": skill.get("intent", ""),
                        "doc_markdown": skill.get("doc_markdown", ""),
                        "score": skill.get("score", 0),
                        "conditions": skill.get("conditions", ""),
                    }
                )
        return plan

    @staticmethod
    def _format_fragments(
        fragments: list[dict[str, Any]],
    ) -> str:
        """将不同来源的检索结果格式化为统一文本。"""
        sections = []

        semantic_fragments = [f for f in fragments if f.get("source") == "semantic"]
        if semantic_fragments:
            lines = ["[Semantic Memory]"]
            for frag in semantic_fragments:
                idx = int(frag.get("index", 0))
                memory_type = str(frag.get("memory_type", "") or "unknown")
                semantic_source_type = str(frag.get("semantic_source_type", "") or "semantic")
                retrieval_score = float(frag.get("retrieval_score", 0.0) or 0.0)
                text = str(frag.get("display_text") or frag.get("semantic_text") or "").strip()
                text = text.replace("\r\n", "\n")
                if len(text) > 800:
                    text = text[:800] + "…"

                lines.append(
                    f"  Fragment {idx}: source={semantic_source_type}, memory_type={memory_type}, retrieval_score={retrieval_score:.3f}"
                )
                time_info = SynthesizerAgent._format_fragment_time_info(frag)
                if time_info:
                    lines.append(f"    Time: {time_info}")
                lines.append(f"    Content: {text}")

                entities = frag.get("entities") or []
                if entities:
                    preview = ", ".join(str(e) for e in entities[:8])
                    lines.append(f"    Entities: {preview}")
            sections.append("\n".join(lines))

        skill_fragments = [f for f in fragments if f.get("source") == "skill"]
        if skill_fragments:
            lines = ["[Skill Library]"]
            for frag in skill_fragments:
                idx = int(frag.get("index", 0))
                intent = str(frag.get("intent", "") or "?")
                doc = str(frag.get("doc_markdown", "") or "")
                doc_preview = doc.replace("\n", " ").strip()[:240]
                lines.append(f"  Fragment {idx}: intent={intent}, document={doc_preview}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _collect_fragments(
        self,
        graph_memories: list[dict[str, Any]],
        skills: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """将 wrapper 结果打平成可供 LLM 打分的独立片段。"""
        fragments: list[dict[str, Any]] = []

        for mem in graph_memories:
            anchor = mem.get("anchor", {}) or {}
            provenance = mem.get("provenance")
            node_id = str(anchor.get("node_id") or "")

            if node_id == "compact_context" and isinstance(provenance, list) and provenance:
                for item in provenance:
                    if not isinstance(item, dict):
                        continue
                    semantic_text = str(item.get("semantic_text") or "").strip()
                    display_text = str(item.get("display_text") or semantic_text).strip()
                    if not semantic_text:
                        continue
                    fragments.append(
                        {
                            "source": "semantic",
                            "semantic_source_type": str(item.get("source") or "record") or "record",
                            "record_id": str(item.get("record_id") or ""),
                            "memory_type": str(item.get("memory_type") or ""),
                            "semantic_text": semantic_text,
                            "display_text": display_text,
                            "created_at": str(item.get("created_at") or ""),
                            "temporal": item.get("temporal") or {},
                            "episode_refs": list(item.get("episode_refs") or []),
                            "entities": list(item.get("entities") or []),
                            "retrieval_score": float(item.get("score") or 0.0),
                            "score_breakdown": item.get("score_breakdown") or {},
                        }
                    )
                continue

            constructed = str(mem.get("constructed_context") or "").strip()
            if not constructed:
                continue
            fragments.append(
                {
                    "source": "semantic",
                    "semantic_source_type": "context",
                    "record_id": str(anchor.get("node_id") or "semantic_context"),
                    "memory_type": str(anchor.get("label") or "context"),
                    "semantic_text": constructed,
                    "display_text": constructed,
                    "entities": [],
                    "retrieval_score": float(anchor.get("score") or 0.0),
                    "score_breakdown": {},
                }
            )

        for skill in skills:
            intent = str(skill.get("intent") or "").strip()
            doc = str(skill.get("doc_markdown") or "").strip()
            display_text = doc or intent
            if not display_text:
                continue
            fragments.append(
                {
                    "source": "skill",
                    "skill_id": str(skill.get("id") or skill.get("skill_id") or ""),
                    "intent": intent,
                    "doc_markdown": doc,
                    "display_text": display_text,
                    "retrieval_score": float(skill.get("score") or 0.0),
                }
            )

        for idx, frag in enumerate(fragments):
            frag["index"] = idx

        return fragments

    def _materialize_provenance(
        self,
        scored_fragments: Any,
        fragments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """将 LLM 评分结果与真实 fragment 元数据对齐。"""
        if not isinstance(scored_fragments, list):
            return []

        index_map = {
            int(f.get("index", -1)): f
            for f in fragments
        }
        provenance: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int, str]] = set()

        for item in scored_fragments:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index", -1))
            except (TypeError, ValueError):
                continue
            fragment = index_map.get(idx)
            if fragment is None:
                continue

            source = self._normalize_fragment_source(item.get("source"), fragment)
            summary = str(item.get("summary") or fragment.get("display_text") or "").strip()
            relevance = float(item.get("relevance") or 0.0)
            unique_id = str(fragment.get("record_id") or fragment.get("skill_id") or f"fragment:{idx}")
            dedupe_key = (source, idx, unique_id)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            provenance.append(
                self._provenance_from_fragment(
                    fragment,
                    source=source,
                    relevance=relevance,
                    summary=summary,
                )
            )

        provenance.sort(key=lambda x: float(x.get("relevance") or 0.0), reverse=True)
        return provenance

    def _fallback_provenance(self, fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """LLM 输出异常时，退化为按检索分数展示前几个片段。"""
        ranked = sorted(
            fragments,
            key=lambda f: float(f.get("retrieval_score") or 0.0),
            reverse=True,
        )
        fallback = []
        for frag in ranked[:8]:
            fallback.append(
                self._provenance_from_fragment(
                    frag,
                    source=str(frag.get("source") or "semantic"),
                    relevance=float(frag.get("retrieval_score") or 0.0),
                    summary=str(frag.get("display_text") or "")[:160],
                )
            )
        return fallback

    def _normalize_fragment_source(
        self,
        raw_source: Any,
        fragment: dict[str, Any],
    ) -> str:
        fragment_source = self._SEMANTIC_SOURCE_ALIASES.get(
            str(fragment.get("source") or "semantic").strip().lower(),
            "semantic",
        )
        value = str(raw_source or "").strip().lower()
        normalized = self._SEMANTIC_SOURCE_ALIASES.get(value)
        if normalized and normalized == fragment_source:
            return normalized
        return fragment_source

    @staticmethod
    def _provenance_from_fragment(
        fragment: dict[str, Any],
        *,
        source: str,
        relevance: float,
        summary: str,
    ) -> dict[str, Any]:
        base = {
            "source": source,
            "index": int(fragment.get("index", 0)),
            "relevance": relevance,
            "summary": summary,
        }
        if source == "skill":
            base.update(
                {
                    "skill_id": str(fragment.get("skill_id") or ""),
                    "intent": str(fragment.get("intent") or ""),
                    "score": float(fragment.get("retrieval_score") or 0.0),
                }
            )
            return base

        base.update(
            {
                "record_id": str(fragment.get("record_id") or ""),
                "memory_type": str(fragment.get("memory_type") or ""),
                "semantic_source_type": str(fragment.get("semantic_source_type") or "semantic"),
                "score": float(fragment.get("retrieval_score") or 0.0),
                "score_breakdown": fragment.get("score_breakdown") or {},
                "semantic_text": str(fragment.get("semantic_text") or ""),
                "display_text": str(fragment.get("display_text") or fragment.get("semantic_text") or ""),
                "created_at": str(fragment.get("created_at") or ""),
                "temporal": fragment.get("temporal") or {},
                "episode_refs": list(fragment.get("episode_refs") or []),
                "entities": list(fragment.get("entities") or []),
            }
        )
        return base

    @staticmethod
    def _format_fragment_time_info(fragment: dict[str, Any]) -> str:
        parts: list[str] = []

        created_at = SynthesizerAgent._compact_timestamp(fragment.get("created_at"))
        if created_at:
            parts.append(f"memory_written={created_at}")

        temporal = fragment.get("temporal") or {}
        temporal_text = SynthesizerAgent._format_temporal_dict(temporal)
        if temporal_text:
            parts.append(f"event_time={temporal_text}")

        episode_refs = fragment.get("episode_refs") or []
        episode_time_text = SynthesizerAgent._format_episode_time_refs(episode_refs)
        if episode_time_text:
            parts.append(f"source_dialogue_time={episode_time_text}")

        return "; ".join(parts)

    @staticmethod
    def _format_temporal_dict(temporal: Any) -> str:
        if not isinstance(temporal, dict):
            return ""
        parts: list[str] = []
        for key in sorted(temporal.keys()):
            value = str(temporal.get(key) or "").strip()
            if not value:
                continue
            parts.append(f"{key}={SynthesizerAgent._compact_timestamp(value)}")
        return ", ".join(parts)

    @staticmethod
    def _format_episode_time_refs(episode_refs: Any) -> str:
        if not isinstance(episode_refs, list):
            return ""
        values: list[str] = []
        seen: set[str] = set()
        for ref in episode_refs[:4]:
            if not isinstance(ref, dict):
                continue
            created_at = SynthesizerAgent._compact_timestamp(ref.get("created_at"))
            if not created_at or created_at in seen:
                continue
            seen.add(created_at)
            values.append(created_at)
        return ", ".join(values)

    @staticmethod
    def _compact_timestamp(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = text.replace(" ", "T")
        return re.sub(r"\.\d+(?=(?:Z|[+-]\d{2}:?\d{2})?$)", "", text)
