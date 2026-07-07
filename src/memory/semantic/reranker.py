"""Plan-conditioned semantic reranking helpers.

The module is intentionally domain-agnostic. It only consumes the retrieval
plan produced by the planner plus generic candidate metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import threading
from typing import Any, Mapping, Sequence

import httpx


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
        self._load_lock = threading.Lock()
        self._predict_semaphore = threading.BoundedSemaphore(
            max(1, self._env_int("LYCHEE_LOCAL_RERANKER_CONCURRENCY", 1))
        )

    def rerank(
        self,
        *,
        query: str,
        retrieval_plan: Mapping[str, Any],
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankResult]:
        if not candidates:
            return []

        query_text = self._format_query(query, retrieval_plan)
        pairs = [
            (query_text, self._format_candidate(candidate))
            for candidate in candidates
        ]
        with self._predict_semaphore:
            model = self._ensure_model()
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
        with self._load_lock:
            if self._model is not None:
                return self._model
            return self._load_model_locked()

    def _load_model_locked(self) -> Any:
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
        plan_traits = []
        for key in ("question_type",):
            value = str(retrieval_plan.get(key) or "").strip()
            if value:
                plan_traits.append(f"{key}={value}")
        if plan_traits:
            parts.append("Question traits: " + " | ".join(plan_traits))
        semantic_queries = LocalCrossEncoderReranker._string_list(
            retrieval_plan.get("semantic_queries")
        )
        if semantic_queries:
            parts.append(f"Semantic queries: {' | '.join(semantic_queries)}")
        evidence_target = str(
            retrieval_plan.get("evidence_target")
            or ""
        ).strip()
        if evidence_target:
            parts.append(f"Evidence target: {evidence_target}")
        evidence_constraints = LocalCrossEncoderReranker._string_list(
            retrieval_plan.get("evidence_constraints")
        )
        if evidence_constraints:
            parts.append(f"Evidence constraints: {' | '.join(evidence_constraints)}")
        constraints = LocalCrossEncoderReranker._format_constraints(
            retrieval_plan.get("constraints")
        )
        if constraints:
            parts.append(f"Constraints: {' | '.join(constraints)}")
        routes = retrieval_plan.get("evidence_routes")
        if isinstance(routes, Sequence) and not isinstance(routes, (str, bytes)):
            route_parts = []
            for route in routes[:3]:
                if not isinstance(route, Mapping):
                    continue
                route_id = str(route.get("route_id") or "").strip()
                goal = str(route.get("evidence_goal") or "").strip()
                if goal:
                    route_parts.append(f"{route_id}: {goal}" if route_id else goal)
            if route_parts:
                parts.append(f"Evidence routes: {' | '.join(route_parts)}")
        temporal_filter = retrieval_plan.get("temporal_filter")
        if isinstance(temporal_filter, Mapping) and temporal_filter:
            parts.append(
                "Temporal filter: "
                + json.dumps(temporal_filter, ensure_ascii=False, default=str)
            )
        return "\n".join(parts)

    @staticmethod
    def _format_constraints(value: Any, *, limit: int = 8) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
        result: list[str] = []
        for item in items[:limit]:
            if isinstance(item, Mapping):
                kind = str(item.get("kind") or "other").strip()
                text = str(item.get("value") or item.get("text") or "").strip()
                if text:
                    result.append(f"{kind}: {text}" if kind else text)
            else:
                text = str(item or "").strip()
                if text:
                    result.append(text)
        return result

    @staticmethod
    def _format_candidate(candidate: RerankCandidate) -> str:
        metadata = candidate.metadata or {}
        parts = []
        text = str(candidate.text or "").strip()
        if text:
            parts.append(f"Text: {text}")
        parts.append(f"Source: {candidate.source}")
        memory_type = str(metadata.get("memory_type") or "").strip()
        if memory_type:
            parts.append(f"Memory type: {memory_type}")
        source_session = str(metadata.get("source_session") or "").strip()
        if source_session:
            parts.append(f"Source session: {source_session}")
        source_dialogue_time = str(metadata.get("source_dialogue_time") or "").strip()
        if source_dialogue_time:
            parts.append(f"Source dialogue time: {source_dialogue_time}")
        evidence_turn_range = metadata.get("evidence_turn_range")
        if evidence_turn_range:
            parts.append(
                "Evidence turns: "
                + json.dumps(evidence_turn_range, ensure_ascii=False, default=str)
            )
        matched_channels = LocalCrossEncoderReranker._string_list(
            metadata.get("matched_channels")
        )
        if matched_channels:
            parts.append(f"Matched channels: {' | '.join(matched_channels)}")
        entities = LocalCrossEncoderReranker._string_list(metadata.get("entities"))
        if entities:
            parts.append(f"Entities: {', '.join(entities)}")
        tags = LocalCrossEncoderReranker._string_list(metadata.get("tags"))
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        temporal = metadata.get("temporal")
        if temporal:
            parts.append(
                "Temporal: "
                + json.dumps(temporal, ensure_ascii=False, default=str)
            )
        evidence_nodes = LocalCrossEncoderReranker._format_evidence_nodes(
            metadata.get("matched_evidence_nodes")
        )
        if evidence_nodes:
            parts.append(f"Matched evidence nodes: {' | '.join(evidence_nodes)}")
        retrieval_score = metadata.get("retrieval_score", candidate.retrieval_score)
        if retrieval_score not in ("", None):
            parts.append(
                f"Retrieval score: {LocalCrossEncoderReranker._to_float(retrieval_score):.4f}"
            )
        return "\n".join(parts)

    @staticmethod
    def _format_evidence_nodes(value: Any, *, limit: int = 8) -> list[str]:
        if value is None:
            return []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = value
        else:
            items = [value]
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            if isinstance(item, Mapping):
                node_type = str(item.get("node_type") or "").strip()
                label = str(
                    item.get("label") or item.get("key") or item.get("node_id") or ""
                ).strip()
                if not label:
                    continue
                text = f"{node_type}:{label}" if node_type else label
            else:
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

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, "") or default)
        except (TypeError, ValueError):
            return default


class RemoteHTTPReranker(SemanticReranker):
    """Call a shared reranker server (OpenAI-compatible /v1/rerank).

    Compatible with vLLM, TEI, Xinference, and other serving frameworks that
    expose the standard ``POST /v1/rerank`` endpoint.
    """

    def __init__(
        self,
        *,
        api_base: str,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or None
        self.timeout = timeout
        self._local = threading.local()

    def rerank(
        self,
        *,
        query: str,
        retrieval_plan: Mapping[str, Any],
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankResult]:
        if not candidates:
            return []
        query_text = LocalCrossEncoderReranker._format_query(query, retrieval_plan)
        documents = [
            LocalCrossEncoderReranker._format_candidate(candidate)
            for candidate in candidates
        ]
        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query_text,
            "documents": documents,
            "top_n": len(candidates),
        }
        data = self._post("/v1/rerank", payload)

        items = self._result_items(data)
        results: list[RerankResult] = []
        for item in items:
            idx = self._result_index(item)
            if idx < 0 or idx >= len(candidates):
                continue
            raw_score = self._result_score(item)
            results.append(RerankResult(
                id=candidates[idx].id,
                rerank_score=LocalCrossEncoderReranker._normalize_score(raw_score),
                cross_encoder_score=raw_score,
            ))
        return results

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self._url(path)
        client = self._client()
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    def _url(self, path: str) -> str:
        """Build an OpenAI-compatible endpoint URL.

        Both ``http://host`` and ``http://host/v1`` are accepted as api_base.
        """
        cleaned_path = "/" + str(path or "").lstrip("/")
        if self.api_base.endswith("/v1") and cleaned_path.startswith("/v1/"):
            cleaned_path = cleaned_path[3:]
        return f"{self.api_base}{cleaned_path}"

    @staticmethod
    def _result_items(data: Any) -> list[Any]:
        if isinstance(data, list):
            return data
        if not isinstance(data, Mapping):
            return []
        items = data.get("data")
        if items is None:
            items = data.get("results")
        if items is None:
            items = data.get("rerank_results")
        if isinstance(items, list):
            return items
        return []

    @staticmethod
    def _result_index(item: Any) -> int:
        if not isinstance(item, Mapping):
            return -1
        for key in ("index", "document_index", "doc_index"):
            if key in item:
                try:
                    return int(item.get(key))
                except (TypeError, ValueError):
                    return -1
        return -1

    @staticmethod
    def _result_score(item: Any) -> float:
        if not isinstance(item, Mapping):
            return LocalCrossEncoderReranker._to_float(item)
        for key in ("relevance_score", "relevance", "score", "rerank_score"):
            if key in item:
                return LocalCrossEncoderReranker._to_float(item.get(key))
        return 0.0

    def _client(self) -> httpx.Client:
        client = getattr(self._local, "client", None)
        if client is None:
            client = httpx.Client(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=self._env_int("LYCHEE_RERANKER_HTTP_MAX_CONNECTIONS", 32),
                    max_keepalive_connections=self._env_int(
                        "LYCHEE_RERANKER_HTTP_MAX_KEEPALIVE", 16
                    ),
                ),
            )
            self._local.client = client
        return client

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, "") or default)
        except (TypeError, ValueError):
            return default
