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
from src.agents.prompts import SYNTHESIS_SYSTEM_PROMPT
from src.evolve.prompt_registry import get_prompt
from src.llm.base import BaseLLM, set_llm_call_source


class SynthesizerAgent(BaseAgent):
    """整合排序器：将多源检索结果融合为 Background Context。"""

    _MAX_REUSE_SKILLS = 2

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
        set_llm_call_source("synthesis_scoring")
        response = self._call_llm(
            user_content,
            system_content=get_prompt("synthesis", self.prompt_template),
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
        reusable_skills = [
            skill for skill in skills
            if skill.get("reusable")
        ]
        reusable_skills.sort(
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )

        plan = []
        for skill in reusable_skills[:SynthesizerAgent._MAX_REUSE_SKILLS]:
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
