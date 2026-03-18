"""Embedder 统一抽象基类。"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """所有 Embedding 适配器的统一接口。"""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding。"""

    def embed_query(self, text: str) -> list[float]:
        """单条查询 embedding。"""
        return self.embed([text])[0]
