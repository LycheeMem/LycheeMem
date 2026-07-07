"""Evidence-node, record, and temporal recall helpers."""

from __future__ import annotations

import re
from typing import Any

from src.memory.semantic.models import MemoryRecord


class RecallRetrievalMixin:
    def _recall_records_from_evidence_nodes_multi(
        self,
        candidate_by_id: dict[str, dict[str, Any]],
        *,
        query: str,
        query_vector: list[float] | None,
        limit_per_type: int,
        record_cache: dict[str, MemoryRecord | None],
    ) -> int:
        """Recall entity/tag/entity_tag evidence with one ANN call per query.

        The previous implementation searched the same evidence_nodes LanceDB
        table three times for the same query vector and only changed the
        Python-side node_type filter. This preserves the same evidence channels
        while avoiding duplicate ANN work.
        """
        if not str(query or "").strip() or query_vector is None:
            return 0

        channel_by_type = {
            "entity": "entity_ann",
            "tag": "tag_ann",
            "entity_tag": "entity_tag_ann",
            "event_frame": "event_frame_ann",
        }
        per_type_limit = max(1, int(limit_per_type or 1))
        try:
            hits = self._vector.search_evidence_nodes(
                query,
                node_types=list(channel_by_type),
                limit=per_type_limit * len(channel_by_type),
                query_vector=query_vector,
            )
        except Exception:
            return 1

        kept_by_type = {node_type: 0 for node_type in channel_by_type}
        hit_by_id: dict[str, dict[str, Any]] = {}
        for hit in hits:
            node_type = str(hit.get("node_type") or "")
            channel = channel_by_type.get(node_type)
            if not channel:
                continue
            if kept_by_type[node_type] >= per_type_limit:
                continue
            node_id = str(hit.get("node_id") or "").strip()
            if not node_id:
                continue
            kept_by_type[node_type] += 1
            score = self._distance_to_retrieval_score(
                self._safe_float(hit.get("_distance"), 1.0)
            )
            current = hit_by_id.get(node_id)
            if current is None or score > self._safe_float(current.get("score"), 0.0):
                hit_by_id[node_id] = {
                    "score": score,
                    "distance": hit.get("_distance"),
                    "channel": channel,
                }

        nodes = self._sqlite.get_evidence_nodes_by_ids(list(hit_by_id.keys()))
        self._prefetch_records_from_ids(
            (
                str(record_id or "")
                for node in nodes
                for record_id in (node.get("record_ids") or [])
            ),
            record_cache,
        )

        for node in nodes:
            node_id = str(node.get("node_id") or "").strip()
            hit_info = hit_by_id.get(node_id, {})
            node_type = str(node.get("node_type") or "")
            node_score = self._safe_float(hit_info.get("score"), 0.0)
            weighted = node_score * self._evidence_node_weight(node_type)
            channel = str(hit_info.get("channel") or channel_by_type.get(node_type) or "evidence_ann")
            for record_id in node.get("record_ids") or []:
                record = self._get_cached_record(str(record_id or ""), record_cache)
                if record is None or record.expired:
                    continue
                self._merge_record_candidate(
                    candidate_by_id,
                    record,
                    score=weighted,
                    matched_query=query,
                    channel=channel,
                    node=node,
                )
        return 1

    def _recall_records_from_evidence_nodes(
        self,
        candidate_by_id: dict[str, dict[str, Any]],
        *,
        query: str,
        query_vector: list[float] | None,
        node_types: list[str],
        channel: str,
        limit: int,
        record_cache: dict[str, MemoryRecord | None],
    ) -> None:
        """Recall evidence nodes by ANN and expand them to linked records."""
        if not str(query or "").strip():
            return
        if query_vector is None:
            return
        try:
            hits = self._vector.search_evidence_nodes(
                query,
                node_types=node_types,
                limit=limit,
                query_vector=query_vector,
            )
        except Exception:
            return

        hit_by_id: dict[str, dict[str, Any]] = {}
        for hit in hits:
            node_id = str(hit.get("node_id") or "").strip()
            if not node_id:
                continue
            score = self._distance_to_retrieval_score(
                self._safe_float(hit.get("_distance"), 1.0)
            )
            current = hit_by_id.get(node_id)
            if current is None or score > self._safe_float(current.get("score"), 0.0):
                hit_by_id[node_id] = {"score": score, "distance": hit.get("_distance")}

        nodes = self._sqlite.get_evidence_nodes_by_ids(list(hit_by_id.keys()))
        self._prefetch_records_from_ids(
            (
                str(record_id or "")
                for node in nodes
                for record_id in (node.get("record_ids") or [])
            ),
            record_cache,
        )

        for node in nodes:
            node_id = str(node.get("node_id") or "").strip()
            hit_info = hit_by_id.get(node_id, {})
            node_type = str(node.get("node_type") or "")
            node_score = self._safe_float(hit_info.get("score"), 0.0)
            weighted = node_score * self._evidence_node_weight(node_type)
            for record_id in node.get("record_ids") or []:
                record = self._get_cached_record(str(record_id or ""), record_cache)
                if record is None or record.expired:
                    continue
                self._merge_record_candidate(
                    candidate_by_id,
                    record,
                    score=weighted,
                    matched_query=query,
                    channel=channel,
                    node=node,
                )

    def _recall_records_from_temporal_filter(
        self,
        candidate_by_id: dict[str, dict[str, Any]],
        *,
        temporal_filter: dict[str, str] | None,
        record_cache: dict[str, MemoryRecord | None],
        limit: int,
    ) -> None:
        """Recall records through temporal evidence nodes before semantic ANN.

        Temporal constraints are exact structured evidence, not a semantic query
        variant. This gives questions like "before 2023-05-30" or "after
        2023-06-01" an independent recall path.
        """
        since = self._date_key((temporal_filter or {}).get("since"))
        until = self._date_key((temporal_filter or {}).get("until"))
        if not since and not until:
            return

        try:
            nodes = self._sqlite.get_evidence_nodes_by_type("temporal", limit=100_000)
        except Exception:
            return
        if not nodes:
            return

        matched_nodes: list[tuple[dict[str, Any], float]] = []
        for node in nodes:
            score = self._temporal_node_match_score(node, since=since, until=until)
            if score <= 0.0:
                continue
            matched_nodes.append((node, score))

        matched_nodes.sort(key=lambda item: item[1], reverse=True)
        matched_nodes = matched_nodes[: max(1, int(limit or 1))]
        self._prefetch_records_from_ids(
            (
                str(record_id or "")
                for node, _score in matched_nodes
                for record_id in (node.get("record_ids") or [])
            ),
            record_cache,
        )

        for node, score in matched_nodes:
            matched_query = self._temporal_query_label(since=since, until=until)
            for record_id in node.get("record_ids") or []:
                record = self._get_cached_record(str(record_id or ""), record_cache)
                if record is None or record.expired:
                    continue
                self._merge_record_candidate(
                    candidate_by_id,
                    record,
                    score=score,
                    matched_query=matched_query,
                    channel="temporal_range",
                    node=node,
                )

    def _temporal_node_match_score(
        self,
        node: dict[str, Any],
        *,
        since: str,
        until: str,
    ) -> float:
        start, end = self._node_temporal_span(node)
        if not start and not end:
            return 0.0
        start = start or end
        end = end or start
        if not self._date_range_overlaps(start, end, since=since, until=until):
            return 0.0

        key = str(node.get("key") or "")
        if key.startswith("day:"):
            return 0.42
        if key.startswith("month:"):
            return 0.28
        return 0.34

    def _node_temporal_span(self, node: dict[str, Any]) -> tuple[str, str]:
        key = str(node.get("key") or "")
        day_match = re.search(r"day:(\d{4}-\d{2}-\d{2})", key)
        if day_match:
            day = day_match.group(1)
            return day, day

        month_match = re.search(r"month:(\d{4}-\d{2})", key)
        if month_match:
            month = month_match.group(1)
            return f"{month}-01", self._month_end(month)

        temporal = node.get("temporal")
        if isinstance(temporal, dict):
            start = self._date_key(temporal.get("start"))
            end = self._date_key(temporal.get("end"))
            if start or end:
                return start, end
        return "", ""

    @staticmethod
    def _month_end(month: str) -> str:
        year_text, month_text = month.split("-", 1)
        year = int(year_text)
        month_num = int(month_text)
        if month_num == 12:
            return f"{year}-12-31"
        # next_month = month_num + 1
        import calendar

        last_day = calendar.monthrange(year, month_num)[1]
        return f"{year}-{month_num:02d}-{last_day:02d}"

    @staticmethod
    def _temporal_query_label(*, since: str, until: str) -> str:
        if since and until and since == until:
            return f"time = {since}"
        if since and until:
            return f"time between {since} and {until}"
        if since:
            return f"time >= {since}"
        return f"time <= {until}"

    @staticmethod
    def _evidence_node_weight(node_type: str) -> float:
        weights = {
            "entity": 0.34,
            "tag": 0.32,
            "entity_tag": 0.38,
            "event_frame": 0.42,
            "temporal": 0.36,
        }
        return weights.get(str(node_type or ""), 0.30)

    def _direct_record_search(
        self,
        query: str,
        top_k: int,
        *,
        query_vector: list[float] | None = None,
        record_cache: dict[str, MemoryRecord | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Normalized-vector recall for MemoryRecord."""
        if query_vector is None:
            return []
        limit = max(top_k, 20)
        cache = record_cache if record_cache is not None else {}
        try:
            hits = self._vector.search(
                query,
                column="normalized_vector",
                limit=limit,
                query_vector=query_vector,
            )
        except Exception:
            return []

        self._prefetch_records_from_ids(
            (str(hit.get("record_id", "") or "") for hit in hits),
            cache,
        )

        candidates: list[dict[str, Any]] = []
        for hit in hits:
            uid = str(hit.get("record_id", "") or "").strip()
            if not uid:
                continue
            full = self._get_cached_record(uid, cache)
            if not full:
                continue
            candidate = self._record_to_candidate(full)
            candidate["semantic_distance"] = self._safe_float(hit.get("_distance"), 1.0)
            candidate["matched_channels"] = ["record_normalized_ann"]
            candidates.append(candidate)

        return candidates


