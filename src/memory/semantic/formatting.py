"""Context formatting and debug payload helpers for CompactSemanticEngine."""

from __future__ import annotations

import re
from typing import Any

from src.memory.semantic.scorer import ScoredCandidate
from src.utils.time_utils import normalize_date_key


class RetrievalFormattingMixin:
    @staticmethod
    def _format_time_label(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        date = normalize_date_key(raw)
        if date:
            return date
        return re.sub(r"\.\d+(?=(?:Z|[+-]\d{2}:?\d{2})?$)", "", raw)

    @staticmethod
    def _debug_candidate_payload(candidates: list[dict[str, Any]], *, max_text: int = 500) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for item in candidates:
            text = str(
                item.get("display_text")
                or item.get("semantic_text")
                or item.get("normalized_text")
                or ""
            ).strip()
            if len(text) > max_text:
                text = text[:max_text] + "..."
            payload.append({
                "id": item.get("id") or item.get("record_id") or item.get("episode_id") or "",
                "source": item.get("source", ""),
                "memory_type": item.get("memory_type", ""),
                "source_role": item.get("source_role", ""),
                "field_score": item.get("field_score", ""),
                "semantic_distance": item.get("semantic_distance", ""),
                "retrieval_score": item.get("retrieval_score", ""),
                "rerank_score": item.get("rerank_score", ""),
                "cross_encoder_score": item.get("cross_encoder_score", ""),
                "matched_queries": item.get("matched_queries", []),
                "matched_channels": item.get("matched_channels", []),
                "matched_evidence_nodes": item.get("matched_evidence_nodes", []),
                "created_at": item.get("created_at", ""),
                "source_dialogue_time": item.get("source_dialogue_time", ""),
                "source_session": item.get("source_session", ""),
                "evidence_turn_range": item.get("evidence_turn_range", []),
                "anchor_ids": item.get("anchor_ids", []),
                "source_turn_count": len(item.get("source_turns") or []),
                "entities": item.get("entities", []),
                "tags": item.get("tags", []),
                "temporal": item.get("temporal", {}),
                "text": text,
            })
        return payload

    @classmethod
    def _debug_scored_payload(cls, scored: list[ScoredCandidate], *, max_text: int = 500) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for candidate in scored:
            item = dict(candidate.data)
            rows = cls._debug_candidate_payload([item], max_text=max_text)
            row = rows[0] if rows else {}
            row.update({
                "id": candidate.id,
                "source": candidate.source,
                "final_score": candidate.final_score,
                "score_breakdown": candidate.score_breakdown,
                "episode_refs": item.get("episode_refs", []),
                "episodic_context": str(item.get("episodic_context") or "")[:max_text],
            })
            payload.append(row)
        return payload

    def _format_context(self, scored: list[ScoredCandidate]) -> str:
        """将 top-k 候选格式化为 LLM 可注入的文本。"""
        if not scored:
            return ""

        parts: list[str] = []
        for i, sc in enumerate(scored, 1):
            d = sc.data
            text = d.get("display_text") or d.get("semantic_text", d.get("normalized_text", ""))
            header = f"[{i}]"
            detail_lines: list[str] = []

            temporal = d.get("temporal") or {}
            if isinstance(temporal, dict):
                time_labels: list[str] = []
                for key in ("t_ref", "t_valid_from", "t_valid_to", "start", "end"):
                    value = temporal.get(key)
                    if not value:
                        continue
                    label = self._format_time_label(value)
                    if label and label not in time_labels:
                        time_labels.append(label)
                if time_labels:
                    detail_lines.append(f"Time: {' | '.join(time_labels)}")

            if d.get("source") == "episode":
                source_time = self._date_key(d.get("source_dialogue_time") or d.get("created_at"))
                if not source_time:
                    source_time = self._format_time_label(
                        str(d.get("source_dialogue_time") or d.get("created_at") or "")
                    )
                if source_time:
                    detail_lines.append(f"Conversation time: {source_time}")

            body_lines = [header]
            body_lines.extend(detail_lines)
            body_lines.append(str(text).strip())
            parts.append("\n".join(line for line in body_lines if line))

        return "\n\n".join(parts)

    @staticmethod
    def _build_provenance(scored: list[ScoredCandidate]) -> list[dict[str, Any]]:
        """构建溯源信息。"""
        provenance = []
        for sc in scored:
            d = sc.data
            provenance.append({
                "record_id": sc.id,
                "source": sc.source,
                "memory_type": d.get("memory_type", ""),
                "semantic_source_type": sc.source,
                "score": sc.final_score,
                "score_breakdown": sc.score_breakdown,
                "rerank_score": d.get("rerank_score", ""),
                "cross_encoder_score": d.get("cross_encoder_score", ""),
                "matched_queries": d.get("matched_queries", []),
                "semantic_text": d.get("semantic_text", ""),
                "display_text": d.get("display_text") or d.get("semantic_text", ""),
                "created_at": d.get("created_at", ""),
                "source_dialogue_time": d.get("source_dialogue_time", ""),
                "temporal": d.get("temporal", {}),
                "episodic_context": d.get("episodic_context", ""),
                "episode_refs": d.get("episode_refs", []),
                "entities": d.get("entities", []),
                "matched_evidence_nodes": d.get("matched_evidence_nodes", []),
                "matched_routes": d.get("matched_routes", []),
                "primary_route_id": d.get("primary_route_id", ""),
                "primary_route_goal": d.get("primary_route_goal", ""),
                "route_rank": d.get("route_rank", ""),
                "source_session": d.get("source_session", ""),
                "source_role": d.get("source_role", ""),
                "evidence_turn_range": d.get("evidence_turn_range", []),
            })
        return provenance

