"""LLM 统一抽象基类。"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Per-turn token 累计器 ──────────────────────────────────────────────────────
# 每轮 pipeline 调用开始时，由 LycheePipeline 设置一个可变 dict 到此 ContextVar；
# asyncio.to_thread 会复制 Context（包含对同一 dict 对象的引用），因此子线程中的
# 累加操作会直接修改该 dict，调用方能看到最新值（不需要线程锁，dict 字段 += 受 GIL 保护）。
# 当 ContextVar 为 None（默认）时，_accumulate_usage 为空操作。
_token_accumulator: ContextVar[dict[str, int] | None] = ContextVar(
    "_token_accumulator", default=None
)

# ── LLM 调用来源标签 ──────────────────────────────────────────────────────────
# 各 LLM 调用点在 generate/agenerate 前设置此标签，用于分来源统计 token 消耗。
_llm_call_source: ContextVar[str] = ContextVar("_llm_call_source", default="unknown")


def set_llm_call_source(source: str) -> None:
    """设置下一次 LLM 调用的来源标签（用于分来源 token 统计）。"""
    _llm_call_source.set(source)

# ── 使用轨迹捕获（Trace Store hook）───────────────────────────────────────
# factory.py 创建 EvolveLoop 后注入，此后每次 generate/agenerate 完成
# 都自动向 PromptStore 写入一条 UsageTrace，供 Optimizer 进行轨迹分析。
_trace_store: Any | None = None


def set_trace_store(store: Any) -> None:
    """注入 PromptStore 实例以启用使用轨迹自动捕获。

    应在 factory.py 中创建 EvolveLoop 后简即调用。
    """
    global _trace_store
    _trace_store = store

# ── 全局 token 统计（进程级单例）────────────────────────────────────────────────
# 统计文件路径：<项目根>/data/token_stats.json
_STATS_FILE = Path(__file__).parent.parent.parent / "data" / "token_stats.json"


class _GlobalTokenStats:
    """进程级累计 token 统计，线程安全，每次更新后原子写入文件。

    写入格式（JSON）::

        {
          "total_input_tokens": 12345,
          "total_output_tokens": 6789,
          "total_tokens": 19134,
          "last_updated": "2024-01-01T12:00:00+00:00"
        }

    启动时若文件已存在则自动恢复历史累计值；写入通过临时文件 + rename 保证原子性，
    最大程度避免并发写入导致的文件损坏。
    """

    def __init__(self, stats_file: Path) -> None:
        self._file = stats_file
        self._lock = threading.Lock()
        self._input: int = 0
        self._output: int = 0
        self._by_source: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        """从已有文件恢复历史累计值（失败时静默忽略）。"""
        try:
            if self._file.exists():
                data = json.loads(self._file.read_text(encoding="utf-8"))
                self._input = int(data.get("total_input_tokens", 0))
                self._output = int(data.get("total_output_tokens", 0))
                for src, val in (data.get("by_source") or {}).items():
                    if isinstance(val, dict):
                        self._by_source[src] = {
                            "input_tokens": int(val.get("input_tokens", 0)),
                            "output_tokens": int(val.get("output_tokens", 0)),
                            "calls": int(val.get("calls", 0)),
                            "total_latency_ms": float(val.get("total_latency_ms", 0.0)),
                        }
        except Exception:
            pass

    def add(self, input_tokens: int, output_tokens: int, source: str = "unknown", latency_ms: float = 0.0) -> None:
        """原子地累加 token 数并刷新文件。"""
        with self._lock:
            self._input += input_tokens
            self._output += output_tokens
            entry = self._by_source.get(source)
            if entry is None:
                entry = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "total_latency_ms": 0.0}
                self._by_source[source] = entry
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens
            entry["calls"] += 1
            entry["total_latency_ms"] += latency_ms
            self._flush_locked()

    def _flush_locked(self) -> None:
        """持有锁时将统计写入文件（临时文件 + rename 原子替换）。"""
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_input_tokens": self._input,
                "total_output_tokens": self._output,
                "total_tokens": self._input + self._output,
                "last_updated": datetime.now(tz=timezone.utc).isoformat(),
                "by_source": {
                    src: {
                        "calls": v["calls"],
                        "input_tokens": v["input_tokens"],
                        "output_tokens": v["output_tokens"],
                        "total_latency_ms": round(v["total_latency_ms"], 1),
                        "avg_latency_ms": round(v["total_latency_ms"] / v["calls"], 1) if v["calls"] else 0.0,
                    }
                    for src, v in sorted(self._by_source.items())
                },
            }
            tmp = self._file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._file)
        except Exception:
            pass  # 统计写入失败不影响主流程


_global_token_stats = _GlobalTokenStats(_STATS_FILE)


class BaseLLM(ABC):
    """所有 LLM 适配器的统一接口。"""

    @staticmethod
    def _accumulate_usage(input_tokens: int, output_tokens: int, latency_ms: float = 0.0) -> None:
        """将本次 LLM 调用的 token 计入当前 turn 的累计器，并同步更新全局统计文件。"""
        # per-turn 累计器
        acc = _token_accumulator.get()
        if acc is not None:
            acc["input"] += input_tokens
            acc["output"] += output_tokens
        # 进程级全局累计（按来源分类统计）
        source = _llm_call_source.get()
        _global_token_stats.add(input_tokens, output_tokens, source, latency_ms)

    def _post_generate_hook(self, messages: list[dict[str, str]], response_text: str) -> None:
        """在 generate/agenerate 完成后自动记录使用轨迹（若已注入 trace_store）。

        仅记录业务 prompt 调用，跳过 evolve 子系统的元调用（evolve_diagnosis 等）。
        """
        if _trace_store is None:
            return
        source = _llm_call_source.get()
        if source.startswith("evolve_") or source == "unknown":
            return
        try:
            from src.evolve.prompt_store import UsageTrace  # 延迟导入，避免循环依赖
            input_parts: list[str] = []
            for msg in messages:
                role = msg.get("role", "")
                if role == "system":
                    input_parts.append(f"[SYSTEM]\n{msg.get('content', '')[:800]}")
                elif role == "user":
                    input_parts.append(f"[USER]\n{msg.get('content', '')[:1200]}")
            _trace_store.record_trace(UsageTrace(
                prompt_name=source,
                input_text="\n\n".join(input_parts)[:3000],
                output_text=response_text[:1000],
            ))
        except Exception:
            pass  # 轨迹写入失败不阻断主流程

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
