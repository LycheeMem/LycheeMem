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
你是严苛的记忆整合与判断器（Memory Synthesizer & Judge）。
你将收到用户当前的任务需求，以及从不同记忆源（semantic / skill）召回的若干原始记忆片段。

你的任务：
1. 为每一段记忆评估其对当前任务的绝对贡献度；
2. 将明显无关或价值很低的记忆丢弃；
3. 将保留下来的高价值记忆去重、融合为一段高密度的 Background Context。

打分与阈值策略：
- 你可以先在脑中按 0-10 分评估贡献度，然后归一化为 0.0-1.0 写入 `relevance` 字段；
- 约等价：6/10 ≈ 0.6；
- 请尽量丢弃 relevance < 0.6 的片段，只保留真正有用的内容。

请严格以 JSON 格式回复，结构如下（字段名必须保持一致）：
{
    "scored_fragments": [
        {"source": "semantic|skill", "index": 0, "relevance": 0.95, "summary": "精炼后的该片段要点"}
    ],
    "kept_count": 保留的片段数,
    "dropped_count": 丢弃的片段数,
    "background_context": "整合后的背景知识文本（直接可用于注入系统的上文）"
}

规则：
- 如果全部片段都不相关，background_context 必须是空字符串；
- background_context 应是高密度的融合文本，不要简单拼接原文，要进行信息压缩和改写；
- 保持事实准确，不虚构检索结果中不存在的信息；
- 如果片段带有“时间”标注，必须显式利用这些时间信息，区分“记忆写入时间(created_at)”与“事件/事实时间(temporal)”；
- 整合时不得把不同时间段的事实错误混合成同一时态结论；涉及约束、状态、故障、任务进度时，优先保留与当前问题最相关的时间条件；
- scored_fragments 按 relevance 降序排列，summary 用简短中文概括该片段的核心信息。

## 示例（仅供参考，不要原样抄写）

用户查询：
    "帮我回顾一下这个项目里和 'user-service 超时' 相关的历史问题，避免我这次排查踩坑。"

来自不同记忆源的片段（由系统预先整理好给你）：
- [semantic] 片段 0：语义记忆中存在一条历史故障记录，备注为 "上次因为下游 payment-service 慢导致整体超时"。
- [skill] 片段 1：技能库中有一条技能，其 intent 为 "排查 user-service 超时问题"，包含一份 Markdown 技能文档（步骤、命令、注意事项等）。

期望的 JSON 输出示例：
{
    "scored_fragments": [
        {"source": "skill", "index": 1, "relevance": 0.95, "summary": "历史上专门用于排查 user-service 超时问题的多步排查技能，可直接复用其检查顺序。"},
        {"source": "semantic", "index": 0, "relevance": 0.85, "summary": "历史语义记忆表明 2024-01-15 的超时主要由下游 payment-service 变慢引起。"}
    ],
    "kept_count": 2,
    "dropped_count": 0,
    "background_context": "历史信息显示：user-service 的超时曾与 payment-service 性能问题高度相关，并且你之前已经有一套成熟的排查技能（包含查看网关 QPS、user-service 错误率以及下游依赖状态等步骤）。建议本次排查优先复用该技能的步骤顺序，并重点关注 payment-service 与网关流量情况，以避免重复踩入相同故障模式。"
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
        user_content = f"用户查询：{user_query}\n\n检索到的记忆片段：\n{fragments_text}"
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
            lines = ["[语义记忆]"]
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
                    f"  片段{idx}: 来源={semantic_source_type}, memory_type={memory_type}, 检索分数={retrieval_score:.3f}"
                )
                time_info = SynthesizerAgent._format_fragment_time_info(frag)
                if time_info:
                    lines.append(f"    时间: {time_info}")
                lines.append(f"    内容: {text}")

                entities = frag.get("entities") or []
                if entities:
                    preview = ", ".join(str(e) for e in entities[:8])
                    lines.append(f"    实体: {preview}")
            sections.append("\n".join(lines))

        skill_fragments = [f for f in fragments if f.get("source") == "skill"]
        if skill_fragments:
            lines = ["[技能库]"]
            for frag in skill_fragments:
                idx = int(frag.get("index", 0))
                intent = str(frag.get("intent", "") or "?")
                doc = str(frag.get("doc_markdown", "") or "")
                doc_preview = doc.replace("\n", " ").strip()[:240]
                lines.append(f"  片段{idx}: 意图={intent}, 文档={doc_preview}")
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
            parts.append(f"记忆写入={created_at}")

        temporal = fragment.get("temporal") or {}
        temporal_text = SynthesizerAgent._format_temporal_dict(temporal)
        if temporal_text:
            parts.append(f"事件时间={temporal_text}")

        episode_refs = fragment.get("episode_refs") or []
        episode_time_text = SynthesizerAgent._format_episode_time_refs(episode_refs)
        if episode_time_text:
            parts.append(f"原始对话时间={episode_time_text}")

        return "；".join(parts)

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
