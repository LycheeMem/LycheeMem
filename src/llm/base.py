"""LLM 统一抽象基类。"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
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

    async def astream_generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式异步生成，逐 token yield 字符串。

        默认实现：完整生成后作为单个 token 返回（降级兼容）。
        子类可 override 以实现真实 token 流。
        """
        text = await self.agenerate(messages, temperature=temperature, max_tokens=max_tokens)
        yield text
