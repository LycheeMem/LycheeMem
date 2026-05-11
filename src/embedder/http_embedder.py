"""HTTP client for the project-local embedding server."""

from __future__ import annotations

import time
from typing import Any

import httpx

from src.embedder.base import BaseEmbedder


class HTTPEmbeddingServerEmbedder(BaseEmbedder):
    """Call a shared embedding server instead of loading a local model per process."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str = "local-embed",
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or None
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0 = time.perf_counter()
        data = self._post("/embeddings", {"model": self.model, "input": texts})
        latency_ms = (time.perf_counter() - t0) * 1000
        self._accumulate_usage(len(texts), 0, latency_ms)
        return self._extract_embeddings(data)

    def embed_query(self, text: str) -> list[float]:
        t0 = time.perf_counter()
        data = self._post("/embeddings/query", {"model": self.model, "input": text})
        latency_ms = (time.perf_counter() - t0) * 1000
        self._accumulate_usage(1, 0, latency_ms)
        embeddings = self._extract_embeddings(data)
        return embeddings[0] if embeddings else []

    @property
    def dimension(self) -> int:
        data = self._get("/health")
        dim = data.get("dim") or data.get("embedding_dim")
        return int(dim or 0)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _url(self, path: str) -> str:
        if self.api_base.endswith("/v1"):
            return f"{self.api_base}{path}"
        return f"{self.api_base}/v1{path}"

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self._url(path), headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    def _get(self, path: str) -> dict[str, Any]:
        url = self._url(path)
        if path == "/health" and self.api_base.endswith("/v1"):
            url = f"{self.api_base[:-3]}/health"
        elif path == "/health":
            url = f"{self.api_base}/health"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _extract_embeddings(data: dict[str, Any]) -> list[list[float]]:
        items = data.get("data") or []
        embeddings: list[list[float]] = []
        for item in items:
            if isinstance(item, dict):
                vec = item.get("embedding")
            else:
                vec = getattr(item, "embedding", None)
            if vec is not None:
                embeddings.append(list(vec))
        return embeddings
