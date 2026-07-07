"""Candidate normalization and date helpers for retrieval."""

from __future__ import annotations

import json
from typing import Any

from src.memory.semantic.models import MemoryRecord
from src.utils.time_utils import extract_date_keys, normalize_date_key


class RetrievalCandidateUtilsMixin:
    @classmethod
    def _candidate_turn_bounds(cls, item: dict[str, Any]) -> tuple[int | None, int | None]:
        turns = item.get("evidence_turn_range")
        if not isinstance(turns, list):
            return None, None
        values: list[int] = []
        for value in turns:
            try:
                values.append(int(value))
            except (TypeError, ValueError):
                continue
        if not values:
            return None, None
        return min(values), max(values)

    @classmethod
    def _merge_candidate_dicts(
        cls,
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for item in list(primary or []) + list(secondary or []):
            cid = cls._candidate_id(item)
            if not cid:
                continue
            existing = by_id.get(cid)
            if existing is None:
                by_id[cid] = dict(item)
                continue
            existing_score = cls._safe_float(existing.get("field_score"), 0.0)
            item_score = cls._safe_float(item.get("field_score"), 0.0)
            if item_score > existing_score:
                merged = dict(item)
                for key in ("matched_queries", "matched_channels", "matched_evidence_nodes"):
                    merged[key] = list(existing.get(key) or []) + list(item.get(key) or [])
                by_id[cid] = merged
                existing = merged
            else:
                for key in ("matched_queries", "matched_channels", "matched_evidence_nodes"):
                    current = existing.get(key)
                    if not isinstance(current, list):
                        current = []
                        existing[key] = current
                    for value in item.get(key) or []:
                        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                        seen = {
                            json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)
                            for x in current
                        }
                        if marker not in seen:
                            current.append(value)
        return list(by_id.values())

    @staticmethod
    def _date_key(value: Any) -> str:
        return normalize_date_key(value)

    @classmethod
    def _candidate_dates(cls, item: dict[str, Any]) -> list[str]:
        dates: list[str] = []
        temporal = item.get("temporal")
        if isinstance(temporal, dict):
            for key in (
                "t_ref",
                "t_valid_from",
                "t_valid_to",
                "valid_from",
                "valid_to",
                "start",
                "end",
            ):
                date = cls._date_key(temporal.get(key))
                if date:
                    dates.append(date)
        dates.extend(extract_date_keys(cls._candidate_text(item)))
        return list(dict.fromkeys(dates))

    @classmethod
    def _candidate_date_spans(cls, item: dict[str, Any]) -> list[tuple[str, str]]:
        return cls._candidate_event_date_spans(item) or cls._candidate_source_date_spans(item)

    @classmethod
    def _candidate_event_date_spans(cls, item: dict[str, Any]) -> list[tuple[str, str]]:
        temporal = item.get("temporal")
        spans: list[tuple[str, str]] = []

        if isinstance(temporal, dict):
            explicit_start = cls._date_key(temporal.get("start"))
            explicit_end = cls._date_key(temporal.get("end"))
            if explicit_start or explicit_end:
                start = explicit_start or explicit_end
                end = explicit_end or explicit_start
                spans.append((start, end))

            t_ref = cls._date_key(temporal.get("t_ref"))
            valid_start = cls._date_key(temporal.get("t_valid_from") or temporal.get("valid_from"))
            valid_end = cls._date_key(temporal.get("t_valid_to") or temporal.get("valid_to"))
            if valid_start or valid_end:
                start = valid_start or t_ref or valid_end
                end = valid_end or t_ref or valid_start
                if start and end and start > end:
                    start, end = end, start
                spans.append((start, end))
            elif t_ref:
                spans.append((t_ref, t_ref))

        spans.extend((date, date) for date in extract_date_keys(cls._candidate_text(item)))
        return cls._dedupe_spans(spans)

    @classmethod
    def _candidate_source_date_spans(cls, item: dict[str, Any]) -> list[tuple[str, str]]:
        source = str(item.get("source") or "").strip().lower()
        if source != "episode":
            return []
        date = cls._date_key(item.get("source_dialogue_time") or item.get("created_at"))
        return [(date, date)] if date else []

    @staticmethod
    def _dedupe_spans(spans: list[tuple[str, str]]) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for start, end in spans:
            if not start or not end:
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    @staticmethod
    def _date_in_range(date: str, *, since: str, until: str) -> bool:
        if since and date < since:
            return False
        if until and date > until:
            return False
        return True

    @staticmethod
    def _date_range_overlaps(start: str, end: str, *, since: str, until: str) -> bool:
        if since and end < since:
            return False
        if until and start > until:
            return False
        return True

    def _merge_record_candidate(
        self,
        candidate_by_id: dict[str, dict[str, Any]],
        record: MemoryRecord,
        *,
        score: float,
        matched_query: str,
        channel: str,
        node: dict[str, Any] | None,
    ) -> None:
        record_id = record.record_id
        if not record_id:
            return
        candidate = candidate_by_id.get(record_id)
        if candidate is None:
            candidate = self._record_to_candidate(record)
            candidate["field_score"] = 0.0
            candidate["matched_queries"] = []
            candidate["matched_channels"] = []
            candidate["matched_evidence_nodes"] = []
            candidate["retrieval_score"] = 0.0
            candidate_by_id[record_id] = candidate
        candidate["field_score"] = min(
            1.0,
            self._safe_float(candidate.get("field_score"), 0.0) + max(0.0, score),
        )
        candidate["semantic_distance"] = min(
            self._safe_float(candidate.get("semantic_distance"), 1.0),
            1.0 - min(1.0, self._safe_float(candidate.get("field_score"), 0.0)),
        )
        self._append_unique(candidate, "matched_queries", matched_query)
        self._append_unique(candidate, "matched_channels", channel)
        if node:
            self._append_unique(candidate, "matched_evidence_nodes", {
                "node_id": node.get("node_id"),
                "node_type": node.get("node_type"),
                "key": node.get("key"),
                "label": node.get("label"),
            })

    @staticmethod
    def _append_unique(target: dict[str, Any], key: str, value: Any) -> None:
        if value in ("", None):
            return
        current = target.get(key)
        if not isinstance(current, list):
            current = []
            target[key] = current
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        existing = {
            json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            for item in current
        }
        if marker not in existing:
            current.append(value)

    @staticmethod
    def _record_to_candidate(record: MemoryRecord) -> dict[str, Any]:
        """MemoryRecord → scorer 需要的 candidate dict。"""
        return {
            "id": record.record_id,
            "source": "record",
            "semantic_distance": 0.5,  # 默认，由调用方覆盖
            "memory_type": record.memory_type,
            "semantic_text": record.semantic_text,
            "normalized_text": record.normalized_text,
            "tags": record.tags,
            "created_at": record.created_at,
            "evidence_turn_range": record.evidence_turn_range,
            "source_session": record.source_session,
            "source_role": record.source_role,
            "temporal": record.temporal,
            "entities": record.entities,
        }



