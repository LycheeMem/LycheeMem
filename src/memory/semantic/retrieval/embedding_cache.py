"""Query embedding and record prefetch caches for retrieval."""

from __future__ import annotations

import threading
from typing import Any

from src.embedder.base import set_embedding_call_source
from src.memory.semantic.models import MemoryRecord


class RetrievalEmbeddingCacheMixin:
    @staticmethod
    def _query_vector_key(text: str) -> str:
        return str(text or "").strip().casefold()

    def _build_search_query_vector_cache(
        self,
        *,
        query_texts: list[str],
        original_query: str,
        original_query_embedding: list[float] | None,
    ) -> dict[str, list[float]]:
        """Batch embed all ANN query texts once per search call.

        The same query vector is reused across evidence-node ANN, record semantic
        ANN, record normalized ANN, and raw-turn ANN. This avoids the previous
        query_variants × channels embedding explosion.
        """
        cache: dict[str, list[float]] = {}
        original_key = self._query_vector_key(original_query)
        if original_query_embedding is not None and original_key:
            cache[original_key] = original_query_embedding
            self._put_cached_query_embedding(original_key, original_query_embedding)

        unique_texts: list[str] = []
        seen: set[str] = set(cache)
        for text in query_texts:
            cleaned = str(text or "").strip()
            key = self._query_vector_key(cleaned)
            if not cleaned or key in seen:
                continue
            seen.add(key)
            unique_texts.append(cleaned)

        if not unique_texts:
            return cache

        texts_to_embed: list[str] = []
        wait_for: list[tuple[str, str, threading.Event]] = []
        for text in unique_texts:
            key = self._query_vector_key(text)
            cached, event, owner = self._claim_query_embedding(key)
            if cached is not None:
                cache[key] = cached
            elif owner:
                texts_to_embed.append(text)
            elif event is not None:
                wait_for.append((text, key, event))

        if not texts_to_embed and not wait_for:
            return cache

        if texts_to_embed:
            embedded_keys = [self._query_vector_key(text) for text in texts_to_embed]
            try:
                set_embedding_call_source("semantic_search_query")
                vectors = self._embedder.embed_queries(texts_to_embed)
                for text, vector in zip(texts_to_embed, vectors):
                    key = self._query_vector_key(text)
                    cache[key] = vector
                    self._put_cached_query_embedding(key, vector)
            finally:
                for key in embedded_keys:
                    self._release_query_embedding(key)

        for _text, key, event in wait_for:
            event.wait(timeout=30.0)
            cached = self._get_cached_query_embedding(key)
            if cached is not None:
                cache[key] = cached
        return cache

    def _claim_query_embedding(
        self,
        key: str,
    ) -> tuple[list[float] | None, threading.Event | None, bool]:
        if not key:
            return None, None, False
        with self._query_embedding_cache_lock:
            vector = self._query_embedding_cache.get(key)
            if vector is not None:
                self._query_embedding_cache.move_to_end(key)
                return vector, None, False
            event = self._query_embedding_inflight.get(key)
            if event is not None:
                return None, event, False
            event = threading.Event()
            self._query_embedding_inflight[key] = event
            return None, event, True

    def _release_query_embedding(self, key: str) -> None:
        if not key:
            return
        with self._query_embedding_cache_lock:
            event = self._query_embedding_inflight.pop(key, None)
            if event is not None:
                event.set()

    def _get_cached_query_embedding(self, key: str) -> list[float] | None:
        if not key:
            return None
        with self._query_embedding_cache_lock:
            vector = self._query_embedding_cache.get(key)
            if vector is None:
                return None
            self._query_embedding_cache.move_to_end(key)
            return vector

    def _put_cached_query_embedding(self, key: str, vector: list[float]) -> None:
        if not key or vector is None:
            return
        with self._query_embedding_cache_lock:
            self._query_embedding_cache[key] = vector
            self._query_embedding_cache.move_to_end(key)
            while len(self._query_embedding_cache) > self._query_embedding_cache_max:
                self._query_embedding_cache.popitem(last=False)

    def _get_cached_record(
        self,
        record_id: str,
        record_cache: dict[str, MemoryRecord | None],
    ) -> MemoryRecord | None:
        rid = str(record_id or "").strip()
        if not rid:
            return None
        if rid not in record_cache:
            record_cache[rid] = self._sqlite.get_record(rid)
        return record_cache[rid]

    def _prefetch_records_from_ids(
        self,
        record_ids: Any,
        record_cache: dict[str, MemoryRecord | None],
    ) -> None:
        missing: list[str] = []
        seen: set[str] = set()
        for record_id in record_ids:
            rid = str(record_id or "").strip()
            if not rid or rid in record_cache or rid in seen:
                continue
            seen.add(rid)
            missing.append(rid)
        if not missing:
            return
        try:
            records = self._sqlite.get_records_by_ids(missing)
        except Exception:
            return
        for rid in missing:
            record_cache[rid] = records.get(rid)


