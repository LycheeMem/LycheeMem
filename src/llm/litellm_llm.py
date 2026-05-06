"""统一 LiteLLM LLM 适配器——所有 provider 调用的最终实现层。

model 格式遵循 litellm 约定（参考 https://docs.litellm.ai/docs/）：
  - "openai/gpt-4o-mini"
  - "gemini/gemini-2.0-flash"
  - "ollama_chat/qwen2.5"
  - "anthropic/claude-3-5-sonnet-20241022"
  等等。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import litellm

from src.llm.base import BaseLLM, Message

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_RETRY_ATTEMPTS = 10
_DEFAULT_RETRY_BACKOFF_SECONDS = 1.0

# ── LiteLLM 全局性能优化（模块首次导入时执行一次）──────────────────────────────
# 1. telemetry=False：禁用 LiteLLM 在每次 API 调用后向其服务器发送遥测 HTTP 请求。
#    这是迁移到 LiteLLM 后延迟显著上升的主要原因。
litellm.telemetry = False
# 2. suppress_debug_info / set_verbose：清除内部 print/logging 判断分支的开销。
litellm.suppress_debug_info = True
litellm.set_verbose = False
# 3. 清空回调列表：即使无回调注册，LiteLLM 内部仍会 dispatch 空循环；
#    显式置空可彻底跳过该路径。
litellm.success_callback = []
litellm.failure_callback = []
litellm._async_success_callback = []
litellm._async_failure_callback = []


class LiteLLMLLM(BaseLLM):
    """通过 LiteLLM 统一调用任意 provider 的 LLM。
    
    支持多模态消息（文本 + 图片），用于 VLM（视觉语言模型）调用。
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        drop_params: bool = True,
        **extra_kwargs: Any,
    ) -> None:
        self.model = model
        self._api_key = api_key or None
        self._api_base = api_base or None
        self._drop_params = drop_params
        self._timeout = _DEFAULT_TIMEOUT_SECONDS
        self._retry_attempts = _DEFAULT_RETRY_ATTEMPTS
        self._retry_backoff_seconds = _DEFAULT_RETRY_BACKOFF_SECONDS
        self._extra = extra_kwargs

    def _build_kwargs(
        self,
        *,
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {**self._extra, "drop_params": self._drop_params}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._timeout is not None and self._timeout > 0:
            kwargs["timeout"] = self._timeout
        kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        """Return False for client/config errors that retrying will not fix."""
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            return True
        return code not in {400, 401, 403, 404, 422}

    def _sleep_before_retry(self, attempt: int) -> None:
        if self._retry_backoff_seconds <= 0:
            return
        time.sleep(self._retry_backoff_seconds * (2 ** max(0, attempt - 1)))

    async def _asleep_before_retry(self, attempt: int) -> None:
        if self._retry_backoff_seconds <= 0:
            return
        await asyncio.sleep(self._retry_backoff_seconds * (2 ** max(0, attempt - 1)))

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """同步生成文本。支持多模态消息。"""
        kwargs = self._build_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        t0 = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                resp = litellm.completion(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= self._retry_attempts or not self._is_retryable_exception(exc):
                    raise
                logger.warning(
                    "LLM completion failed; retrying model=%s attempt=%d/%d error=%s",
                    self.model,
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                self._sleep_before_retry(attempt)
        else:
            raise last_exc or RuntimeError("LLM completion failed")
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        if usage:
            self._accumulate_usage(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
                latency_ms,
            )
        return resp.choices[0].message.content or ""

    async def agenerate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """异步生成文本。支持多模态消息。"""
        kwargs = self._build_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        t0 = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                resp = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= self._retry_attempts or not self._is_retryable_exception(exc):
                    raise
                logger.warning(
                    "Async LLM completion failed; retrying model=%s attempt=%d/%d error=%s",
                    self.model,
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                await self._asleep_before_retry(attempt)
        else:
            raise last_exc or RuntimeError("Async LLM completion failed")
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        if usage:
            self._accumulate_usage(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
                latency_ms,
            )
        return resp.choices[0].message.content or ""

    async def astream_generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """真实 token 流式生成，逐 chunk yield 字符串。"""
        kwargs = self._build_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
        )
        # stream_options 让 LiteLLM 在最后一个 chunk 中附带 usage 信息。
        # 部分不支持该参数的 provider 会通过 drop_params=True 自动忽略。
        kwargs["stream_options"] = {"include_usage": True}
        t0 = time.perf_counter()
        emitted_any = False
        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    **kwargs,
                )
                async for chunk in response:
                    # 最后一个含 usage 的 chunk（content 可能为空）
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        pt = getattr(usage, "prompt_tokens", 0) or 0
                        ct = getattr(usage, "completion_tokens", 0) or 0
                        if pt or ct:
                            latency_ms = (time.perf_counter() - t0) * 1000
                            self._accumulate_usage(pt, ct, latency_ms)
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        emitted_any = True
                        yield delta.content
                return
            except Exception as exc:
                if emitted_any or attempt >= self._retry_attempts or not self._is_retryable_exception(exc):
                    raise
                logger.warning(
                    "Streaming LLM completion failed before first token; retrying model=%s attempt=%d/%d error=%s",
                    self.model,
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                await self._asleep_before_retry(attempt)
