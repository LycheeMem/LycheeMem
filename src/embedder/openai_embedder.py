"""OpenAI Embedding 适配器。"""

from __future__ import annotations

from openai import OpenAI

from src.embedder.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI / OpenAI-compatible Embedding 适配器。"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]
