"""Google Gemini LLM 适配器。"""

from __future__ import annotations

from typing import Any

from google import genai

from a_frame.llm.base import BaseLLM


class GeminiLLM(BaseLLM):
    """Google Gemini LLM 适配器。"""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ):
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def _convert_messages(self, messages: list[dict[str, str]]) -> tuple[str | None, list[dict]]:
        """将 OpenAI 格式的 messages 转换为 Gemini 格式。

        Returns:
            (system_instruction, contents)
        """
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_instruction = text
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
        return system_instruction, contents

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        system_instruction, contents = self._convert_messages(messages)

        config = genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return resp.text or ""

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        system_instruction, contents = self._convert_messages(messages)

        config = genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        resp = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return resp.text or ""
