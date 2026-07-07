"""Candidate boosting and episodic expansion helpers."""

from __future__ import annotations

from typing import Any

from src.memory.semantic.models import EvidenceRoute
from src.memory.semantic.retrieval.strategy import RetrievalStrategy


class RetrievalExpansionMixin:
    def _materialize_source_windows_from_anchors(
        self,
        anchors: list[dict[str, Any]],
        *,
        strategy: RetrievalStrategy,
        route: EvidenceRoute,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Turn semantic anchors into original-dialogue evidence windows.

        MemoryRecord / evidence-node hits are useful as pointers, but they are
        often too compressed to be final evidence. This step uses each anchor's
        source_session + evidence_turn_range to recover the raw conversation
        window that actually contains the answer details.
        """
        if (
            not getattr(strategy, "source_window_enabled", True)
            or self._session_store is None
            or not anchors
        ):
            return []
        get_window = getattr(self._session_store, "get_turn_window", None)
        if not callable(get_window):
            return []

        ordered = sorted(
            anchors,
            key=lambda item: (
                self._anchor_score(item),
                self._safe_float(item.get("retrieval_score"), 0.0),
            ),
            reverse=True,
        )
        del top_k  # The strategy owns the per-route source-window budget.
        anchor_limit = max(
            1,
            min(
                len(ordered),
                int(getattr(strategy, "source_window_anchor_limit_per_route", 24) or 24),
            ),
        )
        window_size = max(0, int(getattr(strategy, "source_window_size", 1) or 0))
        windows: list[dict[str, Any]] = []
        seen_anchor_ids: set[str] = set()

        for anchor in ordered[:anchor_limit]:
            anchor_id = self._candidate_id(anchor)
            if anchor_id and anchor_id in seen_anchor_ids:
                continue
            if anchor_id:
                seen_anchor_ids.add(anchor_id)
            session_id = str(anchor.get("source_session") or "").strip()
            turn_start, turn_end = self._candidate_turn_bounds(anchor)
            if not session_id or turn_start is None or turn_end is None:
                continue
            try:
                raw_turns = get_window(
                    session_id,
                    int(turn_start),
                    int(turn_end),
                    window=window_size,
                )
            except Exception:
                continue
            clean_turns = self._clean_session_turns(raw_turns)
            if not clean_turns:
                continue
            item = self._make_source_window_candidate(
                session_id=session_id,
                turns=clean_turns,
                anchor=anchor,
                route=route,
                strategy=strategy,
            )
            if item:
                windows.append(item)

        return self._merge_source_windows(
            windows,
            merge_gap=max(0, int(getattr(strategy, "source_window_merge_gap", 1) or 0)),
        )

    def _make_source_window_candidate(
        self,
        *,
        session_id: str,
        turns: list[dict[str, Any]],
        anchor: dict[str, Any],
        route: EvidenceRoute,
        strategy: RetrievalStrategy,
    ) -> dict[str, Any] | None:
        if not turns:
            return None
        turn_indices = [
            int(turn["turn_index"])
            for turn in turns
            if "turn_index" in turn
        ]
        if not turn_indices:
            return None
        start, end = min(turn_indices), max(turn_indices)
        anchor_id = self._candidate_id(anchor)
        anchor_score = self._anchor_score(anchor)
        retrieval_score = max(
            anchor_score,
            self._safe_float(anchor.get("retrieval_score"), 0.0),
        )
        text = self._format_source_window_text(
            anchor_text=self._candidate_text(anchor),
            turns=turns,
            query_texts=self._source_window_query_texts(anchor=anchor, route=route),
            max_chars=max(800, int(getattr(strategy, "source_window_max_chars", 2800) or 2800)),
        )
        source_time = str(anchor.get("source_dialogue_time") or anchor.get("created_at") or "")
        if not source_time:
            source_time = next(
                (
                    str(turn.get("created_at") or "")
                    for turn in turns
                    if str(turn.get("created_at") or "").strip()
                ),
                "",
            )
        matched_channels = list(anchor.get("matched_channels") or [])
        if "source_window" not in matched_channels:
            matched_channels.append("source_window")

        return {
            "id": f"source_window:{session_id}:{start}-{end}",
            "source": "episode",
            "memory_type": "source_window",
            "semantic_text": text,
            "normalized_text": text,
            "display_text": text,
            "semantic_distance": max(0.0, min(1.0, 1.0 - retrieval_score)),
            "field_score": min(1.0, max(0.0, anchor_score) + 0.04),
            "retrieval_score": retrieval_score,
            "source_session": session_id,
            "source_role": "both",
            "role": "both",
            "evidence_turn_range": list(range(start, end + 1)),
            "entities": list(anchor.get("entities") or []),
            "tags": self._merge_any_unique(anchor.get("tags") or [], ["source_window"]),
            "temporal": dict(anchor.get("temporal") or {}),
            "created_at": source_time,
            "source_dialogue_time": source_time,
            "matched_queries": list(anchor.get("matched_queries") or []),
            "matched_channels": matched_channels,
            "matched_evidence_nodes": list(anchor.get("matched_evidence_nodes") or []),
            "matched_routes": list(anchor.get("matched_routes") or []),
            "primary_route_id": str(route.route_id or ""),
            "primary_route_goal": str(route.evidence_goal or ""),
            "anchor_ids": [anchor_id] if anchor_id else [],
            "anchor_texts": [self._candidate_text(anchor)] if self._candidate_text(anchor) else [],
            "anchor_sources": [str(anchor.get("source") or "")],
            "source_turns": turns,
            "source_weight_override": self._safe_float(strategy.episode_source_weight, 0.88),
        }

    def _merge_source_windows(
        self,
        windows: list[dict[str, Any]],
        *,
        merge_gap: int,
    ) -> list[dict[str, Any]]:
        if not windows:
            return []
        ordered = sorted(
            windows,
            key=lambda item: (
                str(item.get("source_session") or ""),
                self._candidate_turn_bounds(item)[0] or 0,
                self._candidate_turn_bounds(item)[1] or 0,
            ),
        )
        merged: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for item in ordered:
            session_id = str(item.get("source_session") or "")
            start, end = self._candidate_turn_bounds(item)
            if start is None or end is None:
                continue
            if current is None:
                current = dict(item)
                continue
            cur_session = str(current.get("source_session") or "")
            cur_start, cur_end = self._candidate_turn_bounds(current)
            if (
                session_id == cur_session
                and cur_start is not None
                and cur_end is not None
                and start <= cur_end + max(0, merge_gap) + 1
            ):
                current = self._merge_two_source_windows(current, item)
            else:
                merged.append(current)
                current = dict(item)

        if current is not None:
            merged.append(current)
        merged.sort(
            key=lambda item: (
                self._safe_float(item.get("field_score"), 0.0),
                self._safe_float(item.get("retrieval_score"), 0.0),
            ),
            reverse=True,
        )
        return merged

    def _merge_two_source_windows(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(left)
        session_id = str(merged.get("source_session") or right.get("source_session") or "")
        turns_by_index: dict[int, dict[str, Any]] = {}
        for turn in list(left.get("source_turns") or []) + list(right.get("source_turns") or []):
            try:
                turn_index = int(turn.get("turn_index"))
            except (TypeError, ValueError):
                continue
            turns_by_index[turn_index] = dict(turn)
        turns = [turns_by_index[index] for index in sorted(turns_by_index)]
        if turns:
            start = int(turns[0]["turn_index"])
            end = int(turns[-1]["turn_index"])
            merged["id"] = f"source_window:{session_id}:{start}-{end}"
            merged["evidence_turn_range"] = list(range(start, end + 1))
            merged["source_turns"] = turns
            anchor_texts = self._merge_any_unique(
                left.get("anchor_texts") or [],
                right.get("anchor_texts") or [],
            )
            merged["semantic_text"] = self._format_source_window_text(
                anchor_text=" | ".join(anchor_texts[:4]),
                turns=turns,
                query_texts=self._merge_any_unique(
                    left.get("matched_queries") or [],
                    right.get("matched_queries") or [],
                    anchor_texts,
                ),
                max_chars=max(
                    len(str(left.get("display_text") or "")),
                    len(str(right.get("display_text") or "")),
                    1200,
                ),
            )
            merged["normalized_text"] = merged["semantic_text"]
            merged["display_text"] = merged["semantic_text"]
        merged["field_score"] = min(
            1.0,
            max(
                self._safe_float(left.get("field_score"), 0.0),
                self._safe_float(right.get("field_score"), 0.0),
            )
            + 0.03,
        )
        merged["retrieval_score"] = max(
            self._safe_float(left.get("retrieval_score"), 0.0),
            self._safe_float(right.get("retrieval_score"), 0.0),
        )
        merged["semantic_distance"] = max(
            0.0,
            min(1.0, 1.0 - self._safe_float(merged.get("retrieval_score"), 0.0)),
        )
        for key in (
            "matched_queries",
            "matched_channels",
            "matched_evidence_nodes",
            "matched_routes",
            "anchor_ids",
            "anchor_texts",
            "anchor_sources",
            "entities",
            "tags",
        ):
            merged[key] = self._merge_any_unique(left.get(key) or [], right.get(key) or [])
        if not merged.get("source_dialogue_time"):
            merged["source_dialogue_time"] = right.get("source_dialogue_time", "")
        if not merged.get("created_at"):
            merged["created_at"] = right.get("created_at", "")
        return merged

    @staticmethod
    def _format_source_window_text(
        *,
        anchor_text: str,
        turns: list[dict[str, Any]],
        query_texts: list[str] | None = None,
        max_chars: int = 2800,
    ) -> str:
        lines: list[str] = []
        anchor = str(anchor_text or "").strip()
        if anchor:
            lines.append(f"Anchor: {anchor}")
        lines.append("Original conversation:")
        turn_lines = RetrievalExpansionMixin._source_window_turn_lines(
            turns=turns,
            query_texts=list(query_texts or []) + ([anchor] if anchor else []),
            max_chars=max_chars,
        )
        lines.extend(turn_lines)
        return "\n".join(lines).strip()

    @staticmethod
    def _source_window_turn_lines(
        *,
        turns: list[dict[str, Any]],
        query_texts: list[str],
        max_chars: int,
    ) -> list[str]:
        rendered: list[tuple[str, list[str]]] = []
        for turn in turns:
            role = str(turn.get("role") or "unknown").strip().lower() or "unknown"
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            try:
                turn_index = int(turn.get("turn_index"))
                prefix = f"[{turn_index} {role}]"
            except (TypeError, ValueError):
                prefix = f"[{role}]"
            split_lines = [
                line.strip()
                for line in content.splitlines()
                if line.strip()
            ]
            if not split_lines:
                split_lines = [content]
            rendered.append((prefix, split_lines))

        full_lines = [
            f"{prefix}: {line}"
            for prefix, split_lines in rendered
            for line in split_lines
        ]
        full_text = "\n".join(full_lines)
        if len(full_text) <= max_chars:
            return full_lines

        query_terms = RetrievalExpansionMixin._source_window_terms(query_texts)
        selected_keys: set[tuple[int, int]] = set()
        scored: list[tuple[float, int, int]] = []
        for turn_pos, (_prefix, split_lines) in enumerate(rendered):
            for line_pos, line in enumerate(split_lines):
                line_key = (turn_pos, line_pos)
                lower = line.casefold()
                if "session date" in lower:
                    selected_keys.add(line_key)
                    continue
                line_terms = RetrievalExpansionMixin._source_window_terms([line])
                overlap = len(query_terms & line_terms)
                score = float(overlap)
                if overlap:
                    score += min(2.0, len(line_terms) / 80.0)
                if score > 0:
                    scored.append((score, turn_pos, line_pos))

        scored.sort(reverse=True)
        for _score, turn_pos, line_pos in scored[:12]:
            for neighbor in (line_pos - 1, line_pos, line_pos + 1):
                if 0 <= neighbor < len(rendered[turn_pos][1]):
                    selected_keys.add((turn_pos, neighbor))

        if not selected_keys:
            for turn_pos, (_prefix, split_lines) in enumerate(rendered):
                for line_pos in range(min(4, len(split_lines))):
                    selected_keys.add((turn_pos, line_pos))

        excerpt_lines: list[str] = []
        used_chars = 0
        last_turn_pos: int | None = None
        last_line_pos: int | None = None
        for turn_pos, line_pos in sorted(selected_keys):
            prefix, split_lines = rendered[turn_pos]
            if line_pos >= len(split_lines):
                continue
            if (
                excerpt_lines
                and (
                    last_turn_pos != turn_pos
                    or last_line_pos is None
                    or line_pos > last_line_pos + 1
                )
            ):
                omitted = "[...]"
                if used_chars + len(omitted) + 1 > max_chars:
                    break
                excerpt_lines.append(omitted)
                used_chars += len(omitted) + 1
            line = f"{prefix}: {split_lines[line_pos]}"
            if used_chars + len(line) + 1 > max_chars:
                break
            excerpt_lines.append(line)
            used_chars += len(line) + 1
            last_turn_pos = turn_pos
            last_line_pos = line_pos

        return excerpt_lines or full_lines[:1]

    @staticmethod
    def _source_window_terms(texts: list[str]) -> set[str]:
        import re

        stop = {
            "the", "and", "for", "with", "that", "this", "from", "what", "when",
            "where", "which", "who", "how", "did", "does", "has", "have", "had",
            "was", "were", "are", "is", "about", "into", "your", "their", "they",
            "them", "you", "jon", "gina", "user", "assistant", "anchor", "original",
            "conversation",
        }
        terms: set[str] = set()
        for text in texts:
            for token in re.findall(r"[A-Za-z0-9_]+", str(text or "").casefold()):
                if len(token) < 3 or token in stop:
                    continue
                terms.add(token)
        return terms

    def _source_window_query_texts(
        self,
        *,
        anchor: dict[str, Any],
        route: EvidenceRoute,
    ) -> list[str]:
        texts: list[str] = []
        if route.evidence_goal:
            texts.append(str(route.evidence_goal))
        texts.extend(str(query) for query in (route.queries or []))
        texts.extend(str(query) for query in (anchor.get("matched_queries") or []))
        for item in route.constraints or []:
            if isinstance(item, dict) and item.get("value"):
                texts.append(str(item.get("value")))
        anchor_text = self._candidate_text(anchor)
        if anchor_text:
            texts.append(anchor_text)
        return texts

    def _anchor_score(self, item: dict[str, Any]) -> float:
        return max(
            self._safe_float(item.get("rerank_score"), 0.0),
            self._safe_float(item.get("field_score"), 0.0),
            self._safe_float(item.get("retrieval_score"), 0.0),
            self._distance_to_retrieval_score(
                self._safe_float(item.get("semantic_distance"), 1.0)
            ),
        )

    @staticmethod
    def _merge_any_unique(*items: Any) -> list[Any]:
        result: list[Any] = []
        seen: set[str] = set()
        for group in items:
            if group is None:
                continue
            if isinstance(group, (str, bytes)):
                iterable = [group]
            else:
                try:
                    iterable = list(group)
                except TypeError:
                    iterable = [group]
            for item in iterable:
                if item in (None, ""):
                    continue
                marker = repr(item).casefold()
                if marker in seen:
                    continue
                seen.add(marker)
                result.append(item)
        return result

    def _apply_temporal_boost(
        self,
        candidates: list[dict[str, Any]],
        temporal_filter: dict[str, str] | None,
    ) -> list[dict[str, Any]]:
        """Use planner time constraints only as a recall/ranking signal.

        Temporal evidence is not reliable enough to gate semantic recall. A time
        match can boost a candidate, but a time mismatch must not remove it.
        """
        if not candidates or not temporal_filter:
            return candidates
        since = self._date_key(temporal_filter.get("since"))
        until = self._date_key(temporal_filter.get("until"))
        if not since and not until:
            return candidates

        boosted_any = False
        boosted_candidates: list[dict[str, Any]] = []
        for item in candidates:
            event_spans = self._candidate_event_date_spans(item)
            source_spans = self._candidate_source_date_spans(item)
            event_match = bool(event_spans) and any(
                self._date_range_overlaps(start, end, since=since, until=until)
                for start, end in event_spans
            )
            source_match = not event_spans and bool(source_spans) and any(
                self._date_range_overlaps(start, end, since=since, until=until)
                for start, end in source_spans
            )
            if event_match or source_match:
                boosted_any = True
                boosted = dict(item)
                boosted["field_score"] = min(
                    1.0,
                    self._safe_float(boosted.get("field_score"), 0.0)
                    + (0.08 if event_match else 0.04),
                )
                self._append_unique(
                    boosted,
                    "matched_channels",
                    "temporal_boost" if event_match else "source_time_boost",
                )
                boosted_candidates.append(boosted)
                continue
            boosted_candidates.append(item)

        if not boosted_any:
            return candidates
        return sorted(
            boosted_candidates,
            key=lambda item: self._safe_float(item.get("field_score"), 0.0),
            reverse=True,
        )

    @classmethod
    def _raw_turn_field_score(
        cls,
        retrieval_score: float,
        item: dict[str, Any],
        strategy: RetrievalStrategy,
    ) -> float:
        base = 0.34 + 0.42 * max(0.0, min(1.0, float(retrieval_score or 0.0)))
        role_bonus = cls._raw_turn_role_bonus(item, strategy)
        score = base * max(0.0, strategy.raw_turn_score_multiplier) + role_bonus
        return max(0.0, min(1.0, score))

    @staticmethod
    def _raw_turn_role_bonus(
        item: dict[str, Any],
        strategy: RetrievalStrategy,
    ) -> float:
        role = str(item.get("source_role") or item.get("role") or "").strip().lower()
        if role == "assistant":
            return max(0.0, strategy.assistant_turn_bonus)
        if role == "user":
            return max(0.0, strategy.user_turn_bonus)
        return 0.0

    def _apply_strategy_candidate_boosts(
        self,
        candidates: list[dict[str, Any]],
        strategy: RetrievalStrategy,
    ) -> list[dict[str, Any]]:
        boosted: list[dict[str, Any]] = []
        for item in candidates:
            enriched = dict(item)
            source = str(enriched.get("source") or "record").strip().lower()
            field_score = self._safe_float(enriched.get("field_score"), 0.0)
            if source == "episode":
                field_score = max(
                    field_score,
                    self._raw_turn_field_score(
                        self._safe_float(enriched.get("retrieval_score"), 0.0),
                        enriched,
                        strategy,
                    ),
                )
                enriched["source_weight_override"] = strategy.episode_source_weight
            else:
                field_score += self._record_strategy_bonus(enriched, strategy)
            enriched["field_score"] = max(0.0, min(1.0, field_score))
            boosted.append(enriched)
        return sorted(
            boosted,
            key=lambda item: self._safe_float(item.get("field_score"), 0.0),
            reverse=True,
        )

    @staticmethod
    def _record_strategy_bonus(
        item: dict[str, Any],
        strategy: RetrievalStrategy,
    ) -> float:
        bonus = 0.0
        source_role = str(item.get("source_role") or "").strip().lower()
        memory_type = str(item.get("memory_type") or "").strip().lower()
        if strategy.question_type == "prior_assistant_response" and source_role in {"assistant", "both"}:
            bonus += 0.08
        if strategy.question_type == "personalized_advice" and memory_type in {
            "preference",
            "constraint",
            "event",
            "failure_pattern",
        }:
            bonus += 0.05
        if strategy.question_type == "temporal" and item.get("temporal"):
            bonus += 0.04
        if strategy.question_type in {"aggregate", "comparison"} and memory_type == "event":
            bonus += 0.03
        return bonus

    def _expand_session_neighborhood(
        self,
        candidates: list[dict[str, Any]],
        *,
        strategy: RetrievalStrategy,
        route: EvidenceRoute,
    ) -> list[dict[str, Any]]:
        if (
            self._session_store is None
            or strategy.expansion_window <= 0
            or strategy.expansion_limit_per_route <= 0
            or not candidates
        ):
            return []
        get_window = getattr(self._session_store, "get_turn_window", None)
        if not callable(get_window):
            return []

        expanded: list[dict[str, Any]] = []
        seen: set[str] = set()
        role_filter = {
            str(role).strip().lower()
            for role in (strategy.expansion_roles or ())
            if str(role).strip()
        }
        anchors = candidates[: max(1, min(len(candidates), strategy.expansion_limit_per_route))]
        for anchor in anchors:
            session_id = str(anchor.get("source_session") or "").strip()
            turn_start, turn_end = self._candidate_turn_bounds(anchor)
            if not session_id or turn_start is None or turn_end is None:
                continue
            try:
                turns = get_window(
                    session_id,
                    int(turn_start),
                    int(turn_end),
                    window=strategy.expansion_window,
                )
            except Exception:
                continue
            anchor_score = max(
                self._safe_float(anchor.get("rerank_score"), 0.0),
                self._safe_float(anchor.get("field_score"), 0.0),
                self._safe_float(anchor.get("retrieval_score"), 0.0),
            )
            for turn in turns:
                role = str(turn.get("role") or "unknown").strip().lower()
                if role_filter and role not in role_filter:
                    continue
                content = str(turn.get("content") or "").strip()
                if not content or bool(turn.get("deleted", False)):
                    continue
                try:
                    turn_index = int(turn.get("turn_index"))
                except (TypeError, ValueError):
                    continue
                episode_id = self._make_episode_id(session_id, turn_index)
                if episode_id in seen:
                    continue
                seen.add(episode_id)
                display = f"[{role}]: {content}"
                score = min(
                    1.0,
                    strategy.expansion_score
                    + 0.10 * max(0.0, min(1.0, anchor_score))
                    + self._raw_turn_role_bonus({"source_role": role}, strategy),
                )
                item = {
                    "id": episode_id,
                    "source": "episode",
                    "memory_type": "raw_turn",
                    "semantic_text": display,
                    "normalized_text": content,
                    "display_text": display,
                    "semantic_distance": max(0.0, min(1.0, 1.0 - score)),
                    "field_score": score,
                    "retrieval_score": score,
                    "source_session": session_id,
                    "source_role": role,
                    "role": role,
                    "evidence_turn_range": [turn_index],
                    "entities": [],
                    "tags": ["same_session_context"],
                    "temporal": {},
                    "created_at": str(turn.get("created_at") or ""),
                    "source_dialogue_time": str(turn.get("created_at") or ""),
                    "matched_queries": list(anchor.get("matched_queries") or []),
                    "matched_channels": ["same_session_window"],
                    "neighbor_anchor_id": self._candidate_id(anchor),
                    "neighbor_anchor_route": str(route.route_id or ""),
                    "source_weight_override": strategy.episode_source_weight,
                }
                expanded.append(item)
                if len(expanded) >= strategy.expansion_limit_per_route:
                    return expanded
        return expanded

    def _build_assistant_answer_windows(
        self,
        candidates: list[dict[str, Any]],
        *,
        strategy: RetrievalStrategy,
        route: EvidenceRoute,
    ) -> list[dict[str, Any]]:
        if (
            strategy.question_type != "prior_assistant_response"
            or self._session_store is None
            or not candidates
        ):
            return []
        get_window = getattr(self._session_store, "get_turn_window", None)
        if not callable(get_window):
            return []

        anchors = [
            item
            for item in candidates
            if str(item.get("source") or "").strip().lower() == "episode"
            and str(item.get("memory_type") or "").strip().lower() != "assistant_answer_window"
        ]
        anchors.sort(
            key=lambda item: (
                self._safe_float(item.get("rerank_score"), 0.0),
                self._safe_float(item.get("field_score"), 0.0),
                self._safe_float(item.get("retrieval_score"), 0.0),
            ),
            reverse=True,
        )
        anchor_limit = max(8, int(strategy.expansion_limit_per_route or 0) * 2)
        anchors = anchors[:anchor_limit]

        windows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for anchor in anchors:
            session_id = str(anchor.get("source_session") or "").strip()
            turn_start, turn_end = self._candidate_turn_bounds(anchor)
            if not session_id or turn_start is None or turn_end is None:
                continue
            try:
                turns = get_window(
                    session_id,
                    int(turn_start),
                    int(turn_end),
                    window=max(1, int(strategy.expansion_window or 1)),
                )
            except Exception:
                continue
            clean_turns = self._clean_session_turns(turns)
            if not clean_turns:
                continue
            assistant_turns = self._assistant_window_target_turns(
                clean_turns,
                anchor_role=str(anchor.get("source_role") or anchor.get("role") or ""),
                anchor_start=int(turn_start),
                anchor_end=int(turn_end),
            )
            anchor_score = max(
                self._safe_float(anchor.get("rerank_score"), 0.0),
                self._safe_float(anchor.get("field_score"), 0.0),
                self._safe_float(anchor.get("retrieval_score"), 0.0),
            )
            for assistant_turn in assistant_turns:
                assistant_idx = int(assistant_turn["turn_index"])
                window_id = f"episode_window:{session_id}:{assistant_idx}"
                if window_id in seen:
                    continue
                seen.add(window_id)
                previous_user = self._previous_user_turn(clean_turns, assistant_idx)
                item = self._make_assistant_answer_window_candidate(
                    session_id=session_id,
                    assistant_turn=assistant_turn,
                    previous_user=previous_user,
                    anchor=anchor,
                    route=route,
                    anchor_score=anchor_score,
                    strategy=strategy,
                )
                if item:
                    windows.append(item)
        return windows

    @staticmethod
    def _clean_session_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for turn in turns or []:
            if bool(turn.get("deleted", False)):
                continue
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            try:
                turn_index = int(turn.get("turn_index"))
            except (TypeError, ValueError):
                continue
            cleaned.append({
                "turn_index": turn_index,
                "role": str(turn.get("role") or "unknown").strip().lower(),
                "content": content,
                "created_at": str(turn.get("created_at") or ""),
            })
        cleaned.sort(key=lambda item: int(item["turn_index"]))
        return cleaned

    @staticmethod
    def _assistant_window_target_turns(
        turns: list[dict[str, Any]],
        *,
        anchor_role: str,
        anchor_start: int,
        anchor_end: int,
    ) -> list[dict[str, Any]]:
        role = str(anchor_role or "").strip().lower()
        if role == "assistant":
            return [
                turn
                for turn in turns
                if turn.get("role") == "assistant"
                and int(turn.get("turn_index", -1)) == int(anchor_start)
            ]
        if role == "user":
            result: list[dict[str, Any]] = []
            for turn in turns:
                if turn.get("role") != "assistant":
                    continue
                if int(turn.get("turn_index", -1)) <= int(anchor_end):
                    continue
                result.append(turn)
                if len(result) >= 2:
                    break
            return result
        return [
            turn
            for turn in turns
            if turn.get("role") == "assistant"
            and int(anchor_start) <= int(turn.get("turn_index", -1)) <= int(anchor_end)
        ]

    @staticmethod
    def _previous_user_turn(
        turns: list[dict[str, Any]],
        assistant_index: int,
    ) -> dict[str, Any] | None:
        previous: dict[str, Any] | None = None
        for turn in turns:
            try:
                turn_index = int(turn.get("turn_index"))
            except (TypeError, ValueError):
                continue
            if turn_index >= int(assistant_index):
                break
            if turn.get("role") == "user":
                previous = turn
        return previous

    def _make_assistant_answer_window_candidate(
        self,
        *,
        session_id: str,
        assistant_turn: dict[str, Any],
        previous_user: dict[str, Any] | None,
        anchor: dict[str, Any],
        route: EvidenceRoute,
        anchor_score: float,
        strategy: RetrievalStrategy,
    ) -> dict[str, Any] | None:
        try:
            assistant_idx = int(assistant_turn.get("turn_index"))
        except (TypeError, ValueError):
            return None
        assistant_text = str(assistant_turn.get("content") or "").strip()
        if not assistant_text:
            return None

        display_parts: list[str] = []
        evidence_turns: list[int] = []
        if previous_user is not None:
            try:
                user_idx = int(previous_user.get("turn_index"))
                evidence_turns.append(user_idx)
            except (TypeError, ValueError):
                pass
            user_text = str(previous_user.get("content") or "").strip()
            if user_text:
                display_parts.append(f"[user]: {user_text}")
        evidence_turns.append(assistant_idx)
        display_parts.append(f"[assistant]: {assistant_text}")
        display = "\n".join(display_parts)

        score = min(
            1.0,
            max(0.0, float(anchor_score or 0.0))
            + 0.04
            + max(0.0, strategy.assistant_turn_bonus),
        )
        matched_channels = list(anchor.get("matched_channels") or [])
        if "assistant_answer_window" not in matched_channels:
            matched_channels.append("assistant_answer_window")
        return {
            "id": f"episode_window:{session_id}:{assistant_idx}",
            "source": "episode",
            "memory_type": "assistant_answer_window",
            "semantic_text": display,
            "normalized_text": assistant_text,
            "display_text": display,
            "semantic_distance": max(0.0, min(1.0, 1.0 - score)),
            "field_score": score,
            "retrieval_score": max(
                score,
                self._safe_float(anchor.get("retrieval_score"), 0.0),
            ),
            "source_session": session_id,
            "source_role": "assistant",
            "role": "assistant",
            "evidence_turn_range": evidence_turns,
            "entities": [],
            "tags": ["assistant_answer_window"],
            "temporal": {},
            "created_at": str(assistant_turn.get("created_at") or ""),
            "source_dialogue_time": str(assistant_turn.get("created_at") or ""),
            "matched_queries": list(anchor.get("matched_queries") or []),
            "matched_channels": matched_channels,
            "matched_evidence_nodes": list(anchor.get("matched_evidence_nodes") or []),
            "neighbor_anchor_id": self._candidate_id(anchor),
            "neighbor_anchor_route": str(route.route_id or ""),
            "answer_turn_index": assistant_idx,
            "source_weight_override": strategy.episode_source_weight,
        }


