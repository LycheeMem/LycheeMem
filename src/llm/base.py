"""LLM 统一抽象基类。"""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """所有 LLM 适配器的统一接口。"""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """同步生成。返回纯文本。"""

    @abstractmethod
    async def agenerate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """异步生成。"""
