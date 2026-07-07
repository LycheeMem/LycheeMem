"""Online embedding-based semantic chunking for memory consolidation.

The chunker is intentionally LLM-free. It keeps one open chunk per session and
decides, when a new exchange arrives, whether that exchange still belongs to
the open chunk or should start a new one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import re
import threading
from typing import Any

from src.embedder.base import BaseEmbedder, set_embedding_call_source


@dataclass(slots=True)
class SemanticChunk:
    """A finalized chunk ready for CompactSemanticEncoder."""

    turns: list[dict[str, Any]]
    turn_index_offset: int
    turn_end_index: int
    reason: str
    token_count: int
    exchange_count: int
    boundary_score: float = 0.0
    boundary_probability: float = 0.0


@dataclass(slots=True)
class ChunkingResult:
    finalized_chunks: list[SemanticChunk] = field(default_factory=list)
    pending_token_count: int = 0
    pending_exchange_count: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class _Exchange:
    turns: list[dict[str, Any]]
    start_turn_index: int
    end_turn_index: int
    text: str
    token_count: int
    force_finalize: bool = False


@dataclass(slots=True)
class _PendingChunk:
    turns: list[dict[str, Any]] = field(default_factory=list)
    turn_index_offset: int = 0
    turn_end_index: int = -1
    vectors: list[list[float]] = field(default_factory=list)
    token_count: int = 0
    exchange_count: int = 0
    distance_history: deque[float] = field(default_factory=lambda: deque(maxlen=64))

    def is_empty(self) -> bool:
        return not self.turns


class OnlineSemanticChunker:
    """Online chunker based on local embedding distribution change.

    Length is used as a soft local preference before the hard cap:
    - below 1k tokens, closing is discouraged unless semantics strongly shift;
    - from 1k to 2k, closing pressure rises quickly;
    - around the target size, the open chunk is finalized proactively;
    - a chunk is closed before appending an exchange that would exceed 2k.
    """

    DEFAULT_MIN_CHUNK_TOKENS = 1000
    DEFAULT_TARGET_CHUNK_TOKENS = 1500
    DEFAULT_MAX_CHUNK_TOKENS = 2000

    def __init__(
        self,
        *,
        embedder: BaseEmbedder,
        cut_probability: float = 0.55,
        max_embedding_chars: int = 4000,
        min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
        target_chunk_tokens: int = DEFAULT_TARGET_CHUNK_TOKENS,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    ) -> None:
        self._embedder = embedder
        self._cut_probability = max(0.05, min(0.95, float(cut_probability)))
        self._max_embedding_chars = max(256, int(max_embedding_chars or 4000))
        self._max_chunk_tokens = max(512, int(max_chunk_tokens or self.DEFAULT_MAX_CHUNK_TOKENS))
        self._min_chunk_tokens = max(128, min(
            self._max_chunk_tokens,
            int(min_chunk_tokens or self.DEFAULT_MIN_CHUNK_TOKENS),
        ))
        self._target_chunk_tokens = max(
            self._min_chunk_tokens,
            min(
                self._max_chunk_tokens,
                int(target_chunk_tokens or self.DEFAULT_TARGET_CHUNK_TOKENS),
            ),
        )
        self._states: dict[str, _PendingChunk] = {}
        self._lock = threading.Lock()

    def add_turns(
        self,
        *,
        session_id: str,
        turns: list[dict[str, Any]],
        turn_index_offset: int,
        flush: bool = False,
    ) -> ChunkingResult:
        """Add new turns and return chunks that should now be encoded."""

        sid = str(session_id or "").strip()
        if not sid:
            sid = "__default__"

        exchanges = self._split_exchanges(
            turns,
            turn_index_offset=max(0, int(turn_index_offset or 0)),
        )
        pre_decisions: list[dict[str, Any]] = []
        with self._lock:
            existing = self._states.get(sid)
            pending_end = (
                existing.turn_end_index + 1
                if existing is not None and not existing.is_empty()
                else -1
            )
        if pending_end >= 0:
            fresh_exchanges: list[_Exchange] = []
            for exchange in exchanges:
                if exchange.start_turn_index < pending_end:
                    pre_decisions.append({
                        "action": "skip_duplicate_pending_exchange",
                        "turn_index": exchange.start_turn_index,
                        "token_count": exchange.token_count,
                    })
                    continue
                fresh_exchanges.append(exchange)
            exchanges = fresh_exchanges

        vectors = self._embed_exchanges(exchanges)

        with self._lock:
            state = self._states.get(sid)
            if state is None:
                state = _PendingChunk()
                self._states[sid] = state

            finalized: list[SemanticChunk] = []
            decisions: list[dict[str, Any]] = list(pre_decisions)

            for exchange, vector in zip(exchanges, vectors):
                if (
                    not state.is_empty()
                    and exchange.start_turn_index <= state.turn_end_index
                ):
                    decisions.append({
                        "action": "skip_duplicate_pending_exchange",
                        "turn_index": exchange.start_turn_index,
                        "token_count": exchange.token_count,
                    })
                    continue

                if state.is_empty():
                    self._start_state(state, exchange, vector)
                    decisions.append({
                        "action": "start_chunk",
                        "token_count": exchange.token_count,
                        "turn_index": exchange.start_turn_index,
                    })
                    if self._should_finalize_after_append(state, exchange):
                        reason = self._post_append_finalize_reason(state, exchange)
                        decisions.append(self._finalize_after_append_decision(state, exchange, reason))
                        finalized.append(self._finalize_state(
                            state,
                            reason=reason,
                            boundary_score=0.0,
                            boundary_probability=1.0,
                        ))
                    continue

                if exchange.force_finalize:
                    decisions.append({
                        "action": "cut_before_exchange",
                        "cut": True,
                        "reason": "oversized_turn_fragment",
                        "score": 0.0,
                        "probability": 1.0,
                        "semantic_surprise": None,
                        "centroid_similarity": None,
                        "tail_similarity": None,
                        "cohesion_drop": None,
                        "length_hazard": None,
                        "turn_hazard": None,
                        "chunk_tokens_before": state.token_count,
                        "new_exchange_tokens": exchange.token_count,
                        "chunk_exchanges_before": state.exchange_count,
                        "turn_index": exchange.start_turn_index,
                    })
                    finalized.append(self._finalize_state(
                        state,
                        reason="oversized_turn_fragment",
                        boundary_score=0.0,
                        boundary_probability=1.0,
                    ))
                    self._start_state(state, exchange, vector)
                    finalized.append(self._finalize_state(
                        state,
                        reason="oversized_turn_fragment",
                        boundary_score=0.0,
                        boundary_probability=1.0,
                    ))
                    continue

                if self._would_exceed_hard_cap(state, exchange):
                    decisions.append({
                        "action": "cut_before_exchange",
                        "cut": True,
                        "reason": "capacity_limit",
                        "score": 0.0,
                        "probability": 1.0,
                        "semantic_surprise": None,
                        "centroid_similarity": None,
                        "tail_similarity": None,
                        "cohesion_drop": None,
                        "length_hazard": None,
                        "turn_hazard": None,
                        "chunk_tokens_before": state.token_count,
                        "new_exchange_tokens": exchange.token_count,
                        "chunk_exchanges_before": state.exchange_count,
                        "turn_index": exchange.start_turn_index,
                    })
                    finalized.append(self._finalize_state(
                        state,
                        reason="capacity_limit",
                        boundary_score=0.0,
                        boundary_probability=1.0,
                    ))
                    self._start_state(state, exchange, vector)
                    if self._should_finalize_after_append(state, exchange):
                        reason = self._post_append_finalize_reason(state, exchange)
                        decisions.append(self._finalize_after_append_decision(state, exchange, reason))
                        finalized.append(self._finalize_state(
                            state,
                            reason=reason,
                            boundary_score=0.0,
                            boundary_probability=1.0,
                        ))
                    continue

                decision = self._boundary_decision(state, exchange, vector)
                decisions.append(decision)
                state.distance_history.append(float(decision["semantic_surprise"]))

                if bool(decision["cut"]):
                    finalized.append(self._finalize_state(
                        state,
                        reason=str(decision["reason"]),
                        boundary_score=float(decision["score"]),
                        boundary_probability=float(decision["probability"]),
                    ))
                    self._start_state(state, exchange, vector)
                    if self._should_finalize_after_append(state, exchange):
                        reason = self._post_append_finalize_reason(state, exchange)
                        decisions.append(self._finalize_after_append_decision(state, exchange, reason))
                        finalized.append(self._finalize_state(
                            state,
                            reason=reason,
                            boundary_score=0.0,
                            boundary_probability=1.0,
                        ))
                else:
                    self._append_state(state, exchange, vector)
                    if self._should_finalize_after_append(state, exchange):
                        reason = self._post_append_finalize_reason(state, exchange)
                        decisions.append(self._finalize_after_append_decision(state, exchange, reason))
                        finalized.append(self._finalize_state(
                            state,
                            reason=reason,
                            boundary_score=0.0,
                            boundary_probability=1.0,
                        ))

            if flush and not state.is_empty():
                finalized.append(self._finalize_state(
                    state,
                    reason="session_flush",
                    boundary_score=0.0,
                    boundary_probability=1.0,
                ))

            if flush and state.is_empty():
                self._states.pop(sid, None)
            elif state.is_empty() and not state.distance_history:
                self._states.pop(sid, None)

            return ChunkingResult(
                finalized_chunks=finalized,
                pending_token_count=state.token_count,
                pending_exchange_count=state.exchange_count,
                decisions=decisions,
            )

    def reset_session(self, session_id: str) -> None:
        sid = str(session_id or "").strip() or "__default__"
        with self._lock:
            self._states.pop(sid, None)

    def reset_all(self) -> None:
        with self._lock:
            self._states.clear()

    def pending_summary(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                sid: {
                    "token_count": state.token_count,
                    "exchange_count": state.exchange_count,
                    "turn_count": len(state.turns),
                }
                for sid, state in self._states.items()
                if not state.is_empty()
            }

    def _embed_exchanges(self, exchanges: list[_Exchange]) -> list[list[float]]:
        if not exchanges:
            return []
        texts = [self._truncate_for_embedding(exchange.text) for exchange in exchanges]
        set_embedding_call_source("semantic_chunking")
        return self._embedder.embed(texts)

    def _split_exchanges(
        self,
        turns: list[dict[str, Any]],
        *,
        turn_index_offset: int,
    ) -> list[_Exchange]:
        exchanges: list[_Exchange] = []
        current: list[dict[str, Any]] = []
        current_start = turn_index_offset

        def close_current() -> None:
            nonlocal current
            if not current:
                return
            text = self._format_turns(current)
            if text:
                exchange = _Exchange(
                    turns=[dict(turn) for turn in current],
                    start_turn_index=current_start,
                    end_turn_index=current_start + len(current) - 1,
                    text=text,
                    token_count=self._estimate_tokens(text),
                )
                exchanges.extend(self._split_exchange_by_capacity(exchange))
            current = []

        for local_index, turn in enumerate(turns or []):
            role = str(turn.get("role") or "").strip().lower()
            if role == "user" and current:
                close_current()
                current_start = turn_index_offset + local_index
            elif not current:
                current_start = turn_index_offset + local_index
            current.append(dict(turn))
        close_current()
        return exchanges

    def _split_exchange_by_capacity(self, exchange: _Exchange) -> list[_Exchange]:
        """Split oversized user-bounded exchanges at turn boundaries when possible."""
        if exchange.token_count <= self._max_chunk_tokens:
            return [exchange]
        if len(exchange.turns) <= 1:
            return self._split_long_single_turn_exchange(exchange)

        pieces: list[_Exchange] = []
        current: list[dict[str, Any]] = []
        current_start = exchange.start_turn_index
        current_tokens = 0

        def close_piece() -> None:
            nonlocal current, current_tokens, current_start
            if not current:
                return
            text = self._format_turns(current)
            if text:
                pieces.append(_Exchange(
                    turns=[dict(turn) for turn in current],
                    start_turn_index=current_start,
                    end_turn_index=current_start + len(current) - 1,
                    text=text,
                    token_count=max(1, current_tokens),
                ))
            current = []
            current_tokens = 0

        for local_index, turn in enumerate(exchange.turns):
            absolute_index = exchange.start_turn_index + local_index
            turn_text = self._format_turns([turn])
            turn_tokens = self._estimate_tokens(turn_text)
            if turn_tokens > self._max_chunk_tokens:
                close_piece()
                single_turn_exchange = _Exchange(
                    turns=[dict(turn)],
                    start_turn_index=absolute_index,
                    end_turn_index=absolute_index,
                    text=turn_text,
                    token_count=turn_tokens,
                )
                pieces.extend(self._split_long_single_turn_exchange(single_turn_exchange))
                current_start = absolute_index + 1
                continue
            if current and current_tokens + turn_tokens > self._max_chunk_tokens:
                close_piece()
                current_start = absolute_index
            elif not current:
                current_start = absolute_index
            current.append(dict(turn))
            current_tokens += turn_tokens

        close_piece()
        return pieces or [exchange]

    def _split_long_single_turn_exchange(self, exchange: _Exchange) -> list[_Exchange]:
        """Split one oversized turn into encoder-only fragments with the original turn index."""
        if not exchange.turns:
            return [exchange]
        turn = dict(exchange.turns[0])
        content = str(turn.get("content", ""))
        if not content:
            return [exchange]

        pieces: list[_Exchange] = []
        buffer: list[str] = []

        def close_fragment() -> None:
            nonlocal buffer
            fragment = "".join(buffer).strip()
            if not fragment:
                buffer = []
                return
            fragment_turn = dict(turn)
            fragment_turn["content"] = fragment
            text = self._format_turns([fragment_turn])
            pieces.append(_Exchange(
                turns=[fragment_turn],
                start_turn_index=exchange.start_turn_index,
                end_turn_index=exchange.end_turn_index,
                text=text,
                token_count=self._estimate_tokens(text),
                force_finalize=True,
            ))
            buffer = []

        for char in content:
            if buffer:
                candidate_turn = dict(turn)
                candidate_turn["content"] = "".join(buffer) + char
                if self._estimate_tokens(self._format_turns([candidate_turn])) > self._max_chunk_tokens:
                    close_fragment()
            buffer.append(char)

        close_fragment()
        return pieces or [exchange]

    @staticmethod
    def _format_turns(turns: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"{turn.get('role', 'unknown')}: {turn.get('content', '')}".strip()
            for turn in turns
            if str(turn.get("content", "")).strip()
        ).strip()

    def _truncate_for_embedding(self, text: str) -> str:
        if len(text) <= self._max_embedding_chars:
            return text
        head = self._max_embedding_chars // 2
        tail = self._max_embedding_chars - head
        return f"{text[:head]}\n...\n{text[-tail:]}"

    def _start_state(
        self,
        state: _PendingChunk,
        exchange: _Exchange,
        vector: list[float],
    ) -> None:
        state.turns = [dict(turn) for turn in exchange.turns]
        state.turn_index_offset = exchange.start_turn_index
        state.turn_end_index = exchange.end_turn_index
        state.vectors = [self._normalize_vector(vector)]
        state.token_count = exchange.token_count
        state.exchange_count = 1

    def _append_state(
        self,
        state: _PendingChunk,
        exchange: _Exchange,
        vector: list[float],
    ) -> None:
        state.turns.extend(dict(turn) for turn in exchange.turns)
        state.turn_end_index = exchange.end_turn_index
        state.vectors.append(self._normalize_vector(vector))
        state.token_count += exchange.token_count
        state.exchange_count += 1

    def _finalize_state(
        self,
        state: _PendingChunk,
        *,
        reason: str,
        boundary_score: float,
        boundary_probability: float,
    ) -> SemanticChunk:
        chunk = SemanticChunk(
            turns=[dict(turn) for turn in state.turns],
            turn_index_offset=state.turn_index_offset,
            turn_end_index=state.turn_end_index,
            reason=reason,
            token_count=state.token_count,
            exchange_count=state.exchange_count,
            boundary_score=boundary_score,
            boundary_probability=boundary_probability,
        )
        history = state.distance_history
        state.turns = []
        state.turn_index_offset = 0
        state.turn_end_index = -1
        state.vectors = []
        state.token_count = 0
        state.exchange_count = 0
        state.distance_history = history
        return chunk

    def _boundary_decision(
        self,
        state: _PendingChunk,
        exchange: _Exchange,
        vector: list[float],
    ) -> dict[str, Any]:
        emb = self._normalize_vector(vector)
        centroid = self._centroid(state.vectors)
        tail = self._centroid(state.vectors[-2:])
        centroid_sim = self._dot(emb, centroid)
        tail_sim = self._dot(emb, tail)
        semantic_surprise = max(0.0, 1.0 - max(centroid_sim, tail_sim))

        old_cohesion = self._cohesion(state.vectors)
        new_cohesion = self._cohesion(state.vectors + [emb])
        cohesion_drop = max(0.0, old_cohesion - new_cohesion)

        robust_signal = self._robust_z(semantic_surprise, list(state.distance_history))
        absolute_signal = self._absolute_surprise_signal(semantic_surprise)
        cohesion_signal = min(2.5, cohesion_drop / 0.06) if cohesion_drop > 0 else 0.0
        length_signal = self._length_hazard(state.token_count)
        turn_signal = self._turn_hazard(state.exchange_count)

        score = (
            0.65 * max(absolute_signal, 0.45 * robust_signal)
            + 0.45 * cohesion_signal
            + 0.90 * length_signal
            + 0.30 * turn_signal
        )
        probability = self._sigmoid(-1.10 + score)
        cut = probability >= self._cut_probability

        reason = "append"
        if cut:
            reason = "semantic_boundary"

        return {
            "action": "cut_before_exchange" if cut else "append_exchange",
            "cut": cut,
            "reason": reason,
            "score": round(score, 6),
            "probability": round(probability, 6),
            "semantic_surprise": round(semantic_surprise, 6),
            "centroid_similarity": round(centroid_sim, 6),
            "tail_similarity": round(tail_sim, 6),
            "cohesion_drop": round(cohesion_drop, 6),
            "length_hazard": round(length_signal, 6),
            "turn_hazard": round(turn_signal, 6),
            "chunk_tokens_before": state.token_count,
            "new_exchange_tokens": exchange.token_count,
            "chunk_exchanges_before": state.exchange_count,
            "turn_index": exchange.start_turn_index,
        }

    @staticmethod
    def _length_hazard(tokens: int) -> float:
        """Soft local capacity prior for the 1k-2k target band."""
        t = max(0, int(tokens or 0))
        if t < 700:
            return -1.30
        if t < 1000:
            return -0.80 + 0.80 * ((t - 700) / 300.0)
        if t < 1500:
            return 0.45 + 1.45 * ((t - 1000) / 500.0)
        if t < 2000:
            return 1.90 + 0.90 * ((t - 1500) / 500.0)
        return 3.00

    def _would_exceed_hard_cap(self, state: _PendingChunk, exchange: _Exchange) -> bool:
        return state.token_count + exchange.token_count > self._max_chunk_tokens

    def _should_finalize_after_append(self, state: _PendingChunk, exchange: _Exchange) -> bool:
        if state.is_empty():
            return False
        if exchange.force_finalize:
            return True
        if state.token_count >= self._max_chunk_tokens:
            return True
        return state.token_count >= self._target_chunk_tokens

    def _post_append_finalize_reason(self, state: _PendingChunk, exchange: _Exchange) -> str:
        if exchange.force_finalize:
            return "oversized_turn_fragment"
        if state.token_count > self._max_chunk_tokens:
            return "oversized_exchange"
        if state.token_count >= self._max_chunk_tokens:
            return "capacity_limit"
        return "target_length"

    def _finalize_after_append_decision(
        self,
        state: _PendingChunk,
        exchange: _Exchange,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "action": "finalize_after_exchange",
            "cut": True,
            "reason": reason,
            "score": 0.0,
            "probability": 1.0,
            "semantic_surprise": None,
            "centroid_similarity": None,
            "tail_similarity": None,
            "cohesion_drop": None,
            "length_hazard": None,
            "turn_hazard": None,
            "chunk_tokens_before": max(0, state.token_count - exchange.token_count),
            "new_exchange_tokens": exchange.token_count,
            "chunk_exchanges_before": max(0, state.exchange_count - 1),
            "chunk_tokens_after": state.token_count,
            "turn_index": exchange.start_turn_index,
        }

    @staticmethod
    def _turn_hazard(exchanges: int) -> float:
        n = max(0, int(exchanges or 0))
        if n <= 1:
            return -0.85
        if n == 2:
            return -0.15
        if n == 3:
            return 0.15
        return min(1.0, 0.30 + 0.15 * (n - 4))

    @staticmethod
    def _absolute_surprise_signal(value: float) -> float:
        return max(-1.0, min(2.5, (float(value) - 0.20) / 0.14))

    @staticmethod
    def _robust_z(value: float, history: list[float]) -> float:
        if len(history) < 5:
            return 0.0
        ordered = sorted(float(x) for x in history)
        median = OnlineSemanticChunker._median(ordered)
        deviations = sorted(abs(x - median) for x in ordered)
        mad = OnlineSemanticChunker._median(deviations) or 1e-6
        return max(-2.0, min(4.0, (float(value) - median) / (1.4826 * mad)))

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        mid = len(values) // 2
        if len(values) % 2:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2.0

    @staticmethod
    def _normalize_vector(vector: list[float]) -> list[float]:
        values = [float(x) for x in (vector or [])]
        norm = math.sqrt(sum(x * x for x in values))
        if norm <= 0.0:
            return values
        return [x / norm for x in values]

    @staticmethod
    def _centroid(vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        mean = [0.0] * dim
        used = 0
        for vector in vectors:
            if len(vector) != dim:
                continue
            used += 1
            for idx, value in enumerate(vector):
                mean[idx] += float(value)
        if used <= 0:
            return []
        return OnlineSemanticChunker._normalize_vector([x / used for x in mean])

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(float(a) * float(b) for a, b in zip(left, right))

    @classmethod
    def _cohesion(cls, vectors: list[list[float]]) -> float:
        if len(vectors) <= 1:
            return 1.0
        centroid = cls._centroid(vectors)
        if not centroid:
            return 0.0
        return sum(cls._dot(vector, centroid) for vector in vectors) / len(vectors)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 50.0:
            return 1.0
        if value <= -50.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Cheap mixed Chinese/English token estimate for local capacity control."""
        raw = str(text or "")
        cjk = len(re.findall(r"[\u4e00-\u9fff]", raw))
        words = len(re.findall(r"[A-Za-z0-9_]+", raw))
        other = max(0, len(raw) - cjk - sum(len(m.group(0)) for m in re.finditer(r"[A-Za-z0-9_]+", raw)))
        return max(1, cjk + words + other // 4)
