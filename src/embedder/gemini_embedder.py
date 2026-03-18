"""Gemini Embedding 适配器。

使用 google-genai SDK 调用 Gemini Embedding API。
支持 gemini-embedding-001 (纯文本) 和 gemini-embedding-2-preview (多模态)。
"""

from __future__ import annotations

from google import genai
from google.genai import types

from src.embedder.base import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):
    """Gemini Embedding 适配器。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
        query_task_type: str = "RETRIEVAL_QUERY",
        output_dimensionality: int | None = None,
    ):
        self.model = model
        self.task_type = task_type
        self.query_task_type = query_task_type
        self.output_dimensionality = output_dimensionality
        self._client = genai.Client(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding（文档侧，使用 RETRIEVAL_DOCUMENT task_type）。"""
        result = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=self.task_type,
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> list[float]:
        """单条查询 embedding（查询侧，使用 RETRIEVAL_QUERY task_type）。"""
        result = self._client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=self.query_task_type,
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return result.embeddings[0].values
