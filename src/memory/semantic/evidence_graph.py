"""Fielded evidence graph construction.

This organizer replaces embedding-cluster summary nodes. It never links records
because their sentence embeddings are close. It only builds index nodes from
fields already produced by the encoder: entities, tags, temporal metadata,
source session, and evidence turns.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from src.embedder.base import set_embedding_call_source
from src.memory.semantic.models import MemoryRecord
from src.memory.semantic.sqlite_store import SQLiteSemanticStore
from src.utils.time_utils import extract_date_keys, month_key_from_date
from src.memory.semantic.vector_index import LanceVectorIndex


class FieldedEvidenceOrganizer:
    """Build entity/tag/time/event-frame evidence indexes without LLM calls."""

    def __init__(
        self,
        *,
        sqlite_store: SQLiteSemanticStore,
        vector_index: LanceVectorIndex,
        event_turn_span: int = 4,
    ) -> None:
        self._sqlite = sqlite_store
        self._vector = vector_index
        self._event_turn_span = max(1, int(event_turn_span or 4))

    def organize_on_ingest(self, records: list[MemoryRecord]) -> dict[str, Any]:
        if not records:
            return {"evidence_nodes_upserted": 0, "by_type": {}}

        nodes_by_id: dict[str, dict[str, Any]] = {}
        for record in records:
            self._add_record_nodes(record, nodes_by_id)
        self._add_event_frame_nodes(records, nodes_by_id)

        nodes = list(nodes_by_id.values())
        if not nodes:
            return {"evidence_nodes_upserted": 0, "by_type": {}}

        self._sqlite.upsert_evidence_nodes(nodes)
        set_embedding_call_source("evidence_graph")
        self._vector.upsert_evidence_nodes_batch(nodes)

        by_type: dict[str, int] = {}
        for node in nodes:
            node_type = str(node.get("node_type") or "")
            by_type[node_type] = by_type.get(node_type, 0) + 1
        return {
            "evidence_nodes_upserted": len(nodes),
            "by_type": by_type,
        }

    def _add_record_nodes(
        self,
        record: MemoryRecord,
        nodes_by_id: dict[str, dict[str, Any]],
    ) -> None:
        entities = self._clean_terms(record.entities)
        tags = self._clean_terms(record.tags)
        if record.memory_type:
            tags = self._merge_unique(tags + [record.memory_type])
        turn_ref = self._turn_ref(record)
        temporal_span = self._temporal_span(record.temporal)

        for entity in entities:
            key = self._normalize_key(entity)
            if not key:
                continue
            self._merge_node(
                nodes_by_id,
                node_type="entity",
                key=key,
                label=entity,
                record=record,
                entities=[entity],
                tags=tags,
                temporal=temporal_span,
                turn_ref=turn_ref,
                search_text=self._entity_search_text(entity, record, tags),
            )

        for tag in tags:
            key = self._normalize_key(tag)
            if not key:
                continue
            self._merge_node(
                nodes_by_id,
                node_type="tag",
                key=key,
                label=tag,
                record=record,
                entities=entities,
                tags=[tag],
                temporal=temporal_span,
                turn_ref=turn_ref,
                search_text=self._tag_search_text(tag, record, entities),
            )

        for entity in entities:
            entity_key = self._normalize_key(entity)
            if not entity_key:
                continue
            for tag in tags:
                tag_key = self._normalize_key(tag)
                if not tag_key:
                    continue
                edge_key = f"{entity_key}::{tag_key}"
                self._merge_node(
                    nodes_by_id,
                    node_type="entity_tag",
                    key=edge_key,
                    label=f"{entity} / {tag}",
                    record=record,
                    entities=[entity],
                    tags=[tag],
                    temporal=temporal_span,
                    turn_ref=turn_ref,
                    search_text=self._edge_search_text(entity, tag, record),
                )

        for date_key, label in self._temporal_keys(record.temporal):
            self._merge_node(
                nodes_by_id,
                node_type="temporal",
                key=date_key,
                label=label,
                record=record,
                entities=entities,
                tags=tags,
                temporal=temporal_span,
                turn_ref=turn_ref,
                search_text=self._temporal_search_text(label, record, entities, tags),
            )

    def _add_event_frame_nodes(
        self,
        records: list[MemoryRecord],
        nodes_by_id: dict[str, dict[str, Any]],
    ) -> None:
        groups: dict[tuple[str, int], list[MemoryRecord]] = defaultdict(list)
        for record in records:
            session_id = str(record.source_session or "").strip()
            if not session_id:
                continue
            turns = [int(t) for t in (record.evidence_turn_range or []) if isinstance(t, int)]
            if not turns:
                continue
            bucket = min(turns) // self._event_turn_span
            groups[(session_id, bucket)].append(record)

        for (session_id, bucket), group in groups.items():
            turn_values = [
                int(t)
                for record in group
                for t in (record.evidence_turn_range or [])
                if isinstance(t, int)
            ]
            if not turn_values:
                continue
            start_turn, end_turn = min(turn_values), max(turn_values)
            frame_key = f"{session_id}:{bucket}:{start_turn}-{end_turn}"
            entities = self._merge_unique(
                term for record in group for term in self._clean_terms(record.entities)
            )
            tags = self._merge_unique(
                term
                for record in group
                for term in self._clean_terms(record.tags) + [record.memory_type]
                if term
            )
            temporal = self._merge_temporals([record.temporal for record in group])
            search_text = self._event_frame_search_text(
                session_id=session_id,
                start_turn=start_turn,
                end_turn=end_turn,
                records=group,
                entities=entities,
                tags=tags,
            )
            node_id = self._node_id("event_frame", frame_key)
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "node_type": "event_frame",
                "key": frame_key,
                "label": f"{session_id}:{start_turn}-{end_turn}",
                "search_text": search_text,
                "record_ids": self._merge_unique(record.record_id for record in group),
                "session_ids": [session_id],
                "entities": entities,
                "tags": tags,
                "temporal": self._temporal_span(temporal),
                "evidence_turn_ranges": [
                    {"session_id": session_id, "turns": [start_turn, end_turn]}
                ],
                "metadata": {"turn_start": start_turn, "turn_end": end_turn},
                "created_at": self._now_iso(),
                "updated_at": self._now_iso(),
            }

    def _merge_node(
        self,
        nodes_by_id: dict[str, dict[str, Any]],
        *,
        node_type: str,
        key: str,
        label: str,
        record: MemoryRecord,
        entities: list[str],
        tags: list[str],
        temporal: dict[str, Any],
        turn_ref: dict[str, Any] | None,
        search_text: str,
    ) -> None:
        node_id = self._node_id(node_type, key)
        now = self._now_iso()
        incoming = {
            "node_id": node_id,
            "node_type": node_type,
            "key": key,
            "label": label,
            "search_text": search_text,
            "record_ids": [record.record_id],
            "session_ids": [record.source_session] if record.source_session else [],
            "entities": entities,
            "tags": tags,
            "temporal": temporal,
            "evidence_turn_ranges": [turn_ref] if turn_ref else [],
            "metadata": {},
            "created_at": now,
            "updated_at": now,
        }
        existing = nodes_by_id.get(node_id)
        if existing is None:
            nodes_by_id[node_id] = incoming
            return

        if search_text and search_text not in str(existing.get("search_text") or ""):
            existing["search_text"] = (
                str(existing.get("search_text") or "").strip() + "\n" + search_text
            ).strip()
        existing["record_ids"] = self._merge_unique(existing.get("record_ids", []), [record.record_id])
        existing["session_ids"] = self._merge_unique(existing.get("session_ids", []), incoming["session_ids"])
        existing["entities"] = self._merge_unique(existing.get("entities", []), entities)
        existing["tags"] = self._merge_unique(existing.get("tags", []), tags)
        existing["temporal"] = self._merge_temporals([existing.get("temporal", {}), temporal])
        existing["evidence_turn_ranges"] = self._merge_unique(
            existing.get("evidence_turn_ranges", []),
            incoming["evidence_turn_ranges"],
        )
        existing["updated_at"] = now

    @staticmethod
    def _entity_search_text(entity: str, record: MemoryRecord, tags: list[str]) -> str:
        return "\n".join([
            f"Entity: {entity}",
            f"Tags: {', '.join(tags)}" if tags else "Tags:",
            f"Record: {record.semantic_text}",
        ])

    @staticmethod
    def _tag_search_text(tag: str, record: MemoryRecord, entities: list[str]) -> str:
        return "\n".join([
            f"Tag: {tag}",
            f"Entities: {', '.join(entities)}" if entities else "Entities:",
            f"Record: {record.semantic_text}",
        ])

    @staticmethod
    def _edge_search_text(entity: str, tag: str, record: MemoryRecord) -> str:
        return "\n".join([
            f"Entity: {entity}",
            f"Tag: {tag}",
            f"Evidence: {record.semantic_text}",
        ])

    @staticmethod
    def _temporal_search_text(
        label: str,
        record: MemoryRecord,
        entities: list[str],
        tags: list[str],
    ) -> str:
        return "\n".join([
            f"Time: {label}",
            f"Entities: {', '.join(entities)}" if entities else "Entities:",
            f"Tags: {', '.join(tags)}" if tags else "Tags:",
            f"Record: {record.semantic_text}",
        ])

    @staticmethod
    def _event_frame_search_text(
        *,
        session_id: str,
        start_turn: int,
        end_turn: int,
        records: list[MemoryRecord],
        entities: list[str],
        tags: list[str],
    ) -> str:
        lines = [
            f"Session: {session_id}",
            f"Turn range: {start_turn}-{end_turn}",
            f"Entities: {', '.join(entities)}" if entities else "Entities:",
            f"Tags: {', '.join(tags)}" if tags else "Tags:",
            "Records:",
        ]
        lines.extend(f"- {record.semantic_text}" for record in records[:12])
        return "\n".join(lines)

    @staticmethod
    def _clean_terms(values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            raw_items = [values]
        elif isinstance(values, (list, tuple, set)):
            raw_items = list(values)
        else:
            raw_items = [values]
        return FieldedEvidenceOrganizer._merge_unique(
            str(item or "").strip()
            for item in raw_items
            if str(item or "").strip()
        )

    @staticmethod
    def _normalize_key(value: str) -> str:
        text = str(value or "").casefold().strip()
        text = re.sub(r"[\s\-_:/\\|,;，；：]+", " ", text)
        text = re.sub(r"[^\w\u4e00-\u9fff ]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _temporal_keys(temporal: Any) -> list[tuple[str, str]]:
        if not isinstance(temporal, dict):
            return []
        dates: list[str] = []
        for key in ("t_ref", "t_valid_from", "t_valid_to"):
            dates.extend(extract_date_keys(temporal.get(key)))
        results: list[tuple[str, str]] = []
        for day in FieldedEvidenceOrganizer._merge_unique(dates):
            month = month_key_from_date(day)
            results.append((f"day:{day}", day))
            if month:
                results.append((f"month:{month}", month))
        return results

    @staticmethod
    def _temporal_span(temporal: Any) -> dict[str, Any]:
        if not isinstance(temporal, dict):
            return {}
        if temporal.get("start") or temporal.get("end"):
            start = extract_date_keys(temporal.get("start"))
            end = extract_date_keys(temporal.get("end"))
            start_value = start[0] if start else str(temporal.get("start") or "").strip()
            end_value = end[0] if end else str(temporal.get("end") or "").strip()
            if start_value or end_value:
                return {"start": start_value or end_value, "end": end_value or start_value}
        dates = [
            date
            for key in ("t_ref", "t_valid_from", "t_valid_to")
            for date in extract_date_keys(temporal.get(key))
        ]
        if not dates:
            return {}
        return {"start": min(dates), "end": max(dates)}

    @staticmethod
    def _merge_temporals(temporals: list[Any]) -> dict[str, Any]:
        starts: list[str] = []
        ends: list[str] = []
        for temporal in temporals:
            span = FieldedEvidenceOrganizer._temporal_span(temporal)
            if span.get("start"):
                starts.append(str(span["start"]))
            if span.get("end"):
                ends.append(str(span["end"]))
        if not starts and not ends:
            return {}
        return {
            "start": min(starts) if starts else "",
            "end": max(ends) if ends else "",
        }

    @staticmethod
    def _turn_ref(record: MemoryRecord) -> dict[str, Any] | None:
        turns = [
            int(t)
            for t in (record.evidence_turn_range or [])
            if isinstance(t, int)
        ]
        if not record.source_session or not turns:
            return None
        return {"session_id": record.source_session, "turns": [min(turns), max(turns)]}

    @staticmethod
    def _merge_unique(*items: Any) -> list[Any]:
        result: list[Any] = []
        seen: set[str] = set()
        for item_group in items:
            if item_group is None:
                continue
            if isinstance(item_group, (str, bytes)):
                iterable = [item_group]
            else:
                try:
                    iterable = list(item_group)
                except TypeError:
                    iterable = [item_group]
            for item in iterable:
                if item in (None, ""):
                    continue
                key = repr(item).casefold()
                if key in seen:
                    continue
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _node_id(node_type: str, key: str) -> str:
        digest = hashlib.sha256(f"{node_type}:{key}".encode("utf-8")).hexdigest()[:24]
        return f"ev_{node_type}_{digest}"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
