"""统一 LiteLLM Embedder——所有 provider 的 embedding 调用层。

model 格式遵循 litellm 约定（参考 https://docs.litellm.ai/docs/embedding/supported_embedding）：
  - "openai/text-embedding-3-small"
  - "gemini/gemini-embedding-2-preview"
  - "mistral/mistral-embed"
  等等。

对于支持 task_type 的 provider（如 Gemini）：
  - embed()       使用 task_type       （默认 RETRIEVAL_DOCUMENT）
  - embed_query() 使用 query_task_type （默认 RETRIEVAL_QUERY）
非 Gemini/VertexAI provider 不传 task_type，避免意外报错。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import litellm

from src.embedder.base import BaseEmbedder

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 600.0
_DEFAULT_RETRY_ATTEMPTS = 100
_DEFAULT_RETRY_BACKOFF_SECONDS = 3.0

# litellm 全局配置在 src.llm.litellm_llm 中统一设置；
# 此处仅保留一个保底 suppress，确保单独使用 Embedder 时也生效。
litellm.telemetry = False
litellm.suppress_debug_info = True
litellm.set_verbose = False


class LiteLLMEmbedder(BaseEmbedder):
    """通过 LiteLLM 统一调用任意 provider 的 Embedding 接口。"""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        dimensions: int | None = None,
        task_type: str | None = "RETRIEVAL_DOCUMENT",
        query_task_type: str | None = "RETRIEVAL_QUERY",
        **extra_kwargs: Any,
    ) -> None:
        self.model = model
        self._api_key = api_key or None
        self._api_base = api_base or None
        self._dimensions = dimensions
        self._task_type = task_type
        self._query_task_type = query_task_type
        self._timeout = _DEFAULT_TIMEOUT_SECONDS
        self._retry_attempts = _DEFAULT_RETRY_ATTEMPTS
        self._retry_backoff_seconds = _DEFAULT_RETRY_BACKOFF_SECONDS
        self._extra = extra_kwargs
        # task_type 仅对 Gemini / Vertex AI 有意义，其他 provider 不传。
        _m = model.lower()
        self._supports_task_type: bool = _m.startswith("gemini/") or _m.startswith("vertex_ai/")

    def _build_kwargs(self, *, task_type: str | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {**self._extra}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._timeout is not None and self._timeout > 0:
            kwargs["timeout"] = self._timeout
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        if task_type and self._supports_task_type:
            kwargs["task_type"] = task_type
        # 对非 Gemini/VertexAI provider，显式指定 encoding_format="float"。
        # litellm 在部分代码路径下会将 encoding_format 默认为空字符串，
        # 导致严格 OpenAI 兼容接口返回 400 错误（要求 'float' 或 'base64'）。
        if not self._supports_task_type and "encoding_format" not in kwargs:
            kwargs["encoding_format"] = "float"
        return kwargs

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        """Return False for client/config errors that retrying will not fix."""
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            return True
        return code not in {400, 401, 403, 404, 422}

    def _sleep_before_retry(self, attempt: int) -> None:
        if self._retry_backoff_seconds <= 0:
            return
        time.sleep(self._retry_backoff_seconds * (2 ** max(0, attempt - 1)))

    @staticmethod
    def _extract_embedding(item: Any) -> list[float]:
        """兼容 litellm 返回 dict 或对象两种格式。"""
        if isinstance(item, dict):
            return item["embedding"]
        return item.embedding

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding（文档侧）。"""
        if not texts:
            return []
        t0 = time.perf_counter()
        kwargs = self._build_kwargs(task_type=self._task_type)
        last_exc: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                resp = litellm.embedding(
                    model=self.model,
                    input=texts,
                    **kwargs,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= self._retry_attempts or not self._is_retryable_exception(exc):
                    raise
                logger.warning(
                    "Embedding call failed; retrying model=%s attempt=%d/%d error=%s",
                    self.model,
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                self._sleep_before_retry(attempt)
        else:
            raise last_exc or RuntimeError("Embedding call failed")
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        self._accumulate_usage(len(texts), tokens, latency_ms)
        return [self._extract_embedding(item) for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        """单条查询 embedding（查询侧）。"""
        vectors = self.embed_queries([text])
        if not vectors:
            return []
        return vectors[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """批量生成查询 embedding（查询侧）。"""
        if not texts:
            return []
        t0 = time.perf_counter()
        kwargs = self._build_kwargs(task_type=self._query_task_type)
        last_exc: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                resp = litellm.embedding(
                    model=self.model,
                    input=texts,
                    **kwargs,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= self._retry_attempts or not self._is_retryable_exception(exc):
                    raise
                logger.warning(
                    "Embedding query failed; retrying model=%s attempt=%d/%d error=%s",
                    self.model,
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                self._sleep_before_retry(attempt)
        else:
            raise last_exc or RuntimeError("Embedding query failed")
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        self._accumulate_usage(len(texts), tokens, latency_ms)
        return [self._extract_embedding(item) for item in resp.data]
