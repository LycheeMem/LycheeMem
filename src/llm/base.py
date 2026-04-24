"""LLM 统一抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

# ── Per-turn token 累计器 ──────────────────────────────────────────────────────
# 每轮 pipeline 调用开始时，由 LycheePipeline 设置一个可变 dict 到此 ContextVar；
# asyncio.to_thread 会复制 Context（包含对同一 dict 对象的引用），因此子线程中的
# 累加操作会直接修改该 dict，调用方能看到最新值（不需要线程锁，dict 字段 += 受 GIL 保护）。
# 当 ContextVar 为 None（默认）时，_accumulate_usage 为空操作。
_token_accumulator: ContextVar[dict[str, int] | None] = ContextVar(
    "_token_accumulator", default=None
)


# ── 多模态消息类型 ───────────────────────────────────────────────────────────
# 支持文本和多模态内容（图片等）
# 格式参考：https://docs.litellm.ai/docs/
TextContent = dict[str, str]  # {"type": "text", "text": "..."}
ImageContent = dict[str, dict[str, str]]  # {"type": "image_url", "image_url": {"url": "..."}}
ContentItem = TextContent | ImageContent
MessageContent = str | list[ContentItem]
Message = dict[str, MessageContent]


class BaseLLM(ABC):
    """所有 LLM 适配器的统一接口。"""

    @staticmethod
    def _accumulate_usage(input_tokens: int, output_tokens: int) -> None:
        """将本次 LLM 调用的 token 计入当前 turn 的累计器（累计器未激活时为空操作）。"""
        acc = _token_accumulator.get()
        if acc is not None:
            acc["input"] += input_tokens
            acc["output"] += output_tokens

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """同步生成。返回纯文本。"""

    @abstractmethod
    async def agenerate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """异步生成。支持多模态消息（文本 + 图片）。"""

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
