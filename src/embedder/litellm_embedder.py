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

from typing import Any

import litellm

from src.embedder.base import BaseEmbedder

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
    def _extract_embedding(item: Any) -> list[float]:
        """兼容 litellm 返回 dict 或对象两种格式。"""
        if isinstance(item, dict):
            return item["embedding"]
        return item.embedding

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding（文档侧）。"""
        resp = litellm.embedding(
            model=self.model,
            input=texts,
            **self._build_kwargs(task_type=self._task_type),
        )
        return [self._extract_embedding(item) for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        """单条查询 embedding（查询侧）。"""
        resp = litellm.embedding(
            model=self.model,
            input=[text],
            **self._build_kwargs(task_type=self._query_task_type),
        )
        return self._extract_embedding(resp.data[0])
