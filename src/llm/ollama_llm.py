"""Ollama 本地 LLM 适配器（通过 OpenAI-compatible API）。"""

from __future__ import annotations

from src.llm.openai_llm import OpenAILLM


class OllamaLLM(OpenAILLM):
    """Ollama 适配器，复用 OpenAI-compatible 接口。"""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
    ):
        super().__init__(model=model, api_key="ollama", base_url=base_url)
