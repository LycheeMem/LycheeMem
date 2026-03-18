"""OpenAI 兼容 LLM 适配器（也适用于 DeepSeek 等 OpenAI-compatible API）。"""

from __future__ import annotations

from typing import Any

from openai import OpenAI, AsyncOpenAI

from src.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI / OpenAI-compatible LLM 适配器。"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        resp = await self._aclient.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
