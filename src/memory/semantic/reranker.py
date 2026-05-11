"""Plan-conditioned semantic reranking helpers.

The module is intentionally domain-agnostic. It only consumes the retrieval
plan produced by the planner plus generic candidate metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RerankCandidate:
    """A candidate prepared for cross-encoder reranking."""

    id: str
    source: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    retrieval_score: float = 0.0


@dataclass(frozen=True)
class RerankResult:
    """Rerank output for one candidate."""

    id: str
    rerank_score: float
    cross_encoder_score: float


class SemanticReranker:
    """Interface for semantic rerankers."""

    def rerank(
        self,
        *,
        query: str,
        retrieval_plan: Mapping[str, Any],
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankResult]:
        raise NotImplementedError


class LocalCrossEncoderReranker(SemanticReranker):
    """Local sentence-transformers CrossEncoder reranker."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size or 16))
        self.max_length = max(32, int(max_length or 512))
        self._model: Any | None = None

    def rerank(
        self,
        *,
        query: str,
        retrieval_plan: Mapping[str, Any],
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankResult]:
        if not candidates:
            return []

        model = self._ensure_model()
        query_text = self._format_query(query, retrieval_plan)
        pairs = [
            (query_text, self._format_candidate(candidate))
            for candidate in candidates
        ]
        raw_scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        results: list[RerankResult] = []
        for candidate, raw in zip(candidates, raw_scores):
            raw_score = self._to_float(raw)
            results.append(RerankResult(
                id=candidate.id,
                rerank_score=self._normalize_score(raw_score),
                cross_encoder_score=raw_score,
            ))
        return results

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for LocalCrossEncoderReranker. "
                "Install the local-embed or rerank optional dependency."
            ) from exc

        device_arg = None if self.device == "auto" else self.device
        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=device_arg,
        )
        return self._model

    @staticmethod
    def _format_query(query: str, retrieval_plan: Mapping[str, Any]) -> str:
        parts = [f"User query: {str(query or '').strip()}"]
        mode = str(retrieval_plan.get("mode") or "").strip()
        if mode:
            parts.append(f"Retrieval mode: {mode}")
        semantic_queries = LocalCrossEncoderReranker._string_list(
            retrieval_plan.get("semantic_queries")
        )
        if semantic_queries:
            parts.append(f"Semantic queries: {' | '.join(semantic_queries)}")
        aggregate_target = str(retrieval_plan.get("aggregate_target") or "").strip()
        if aggregate_target:
            parts.append(f"Aggregate target: {aggregate_target}")
        aggregate_constraints = LocalCrossEncoderReranker._string_list(
            retrieval_plan.get("aggregate_constraints")
        )
        if aggregate_constraints:
            parts.append(f"Aggregate constraints: {' | '.join(aggregate_constraints)}")
        return "\n".join(parts)

    @staticmethod
    def _format_candidate(candidate: RerankCandidate) -> str:
        metadata = candidate.metadata or {}
        parts = [f"Source: {candidate.source}"]
        matched_queries = LocalCrossEncoderReranker._string_list(
            metadata.get("matched_queries")
        )
        if matched_queries:
            parts.append(f"Matched queries: {' | '.join(matched_queries)}")
        entities = LocalCrossEncoderReranker._string_list(metadata.get("entities"))
        if entities:
            parts.append(f"Entities: {', '.join(entities)}")
        temporal = metadata.get("temporal")
        if temporal:
            parts.append(
                "Temporal: "
                + json.dumps(temporal, ensure_ascii=False, default=str)
            )
        text = str(candidate.text or "").strip()
        parts.append(f"Text: {text}")
        return "\n".join(parts)

    @staticmethod
    def _string_list(value: Any, *, limit: int = 12) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, Sequence):
            items = list(value)
        else:
            items = [value]

        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _to_float(value: Any) -> float:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                return 0.0
            value = value[-1]
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalize_score(value: float) -> float:
        if math.isnan(value):
            return 0.0
        if 0.0 <= value <= 1.0:
            return value
        if value >= 50.0:
            return 1.0
        if value <= -50.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-value))
