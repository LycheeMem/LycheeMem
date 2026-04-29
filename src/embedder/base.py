"""Embedder 统一抽象基类。"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

# ── Embedding 调用来源标签 ─────────────────────────────────────────────────────
# 各 Embedding 调用点在 embed/embed_query 前设置此标签，用于分来源统计调用消耗。
_embedding_call_source: ContextVar[str] = ContextVar("_embedding_call_source", default="unknown")


def set_embedding_call_source(source: str) -> None:
    """设置下一次 Embedding 调用的来源标签（用于分来源统计）。"""
    _embedding_call_source.set(source)


# ── 全局 Embedding 统计（进程级单例）────────────────────────────────────────────
# 统计文件路径：<项目根>/data/embedding_stats.json
_STATS_FILE = Path(__file__).parent.parent.parent / "data" / "embedding_stats.json"


class _GlobalEmbeddingStats:
    """进程级累计 Embedding 统计，线程安全，每次更新后原子写入文件。

    写入格式（JSON）::

        {
          "total_texts": 1234,
          "total_calls": 567,
          "total_tokens": 8901,
          "last_updated": "2024-01-01T12:00:00+00:00",
          "by_source": {
            "record_ingest": {
              "calls": 10,
              "texts": 20,
              "tokens": 500,
              "total_latency_ms": 1234.5,
              "avg_latency_ms": 123.4
            }
          }
        }

    启动时若文件已存在则自动恢复历史累计值；写入通过临时文件 + rename 保证原子性。
    """

    def __init__(self, stats_file: Path) -> None:
        self._file = stats_file
        self._lock = threading.Lock()
        self._texts: int = 0
        self._calls: int = 0
        self._tokens: int = 0
        self._by_source: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """从已有文件恢复历史累计值（失败时静默忽略）。"""
        try:
            if self._file.exists():
                data = json.loads(self._file.read_text(encoding="utf-8"))
                self._texts = int(data.get("total_texts", 0))
                self._calls = int(data.get("total_calls", 0))
                self._tokens = int(data.get("total_tokens", 0))
                for src, val in (data.get("by_source") or {}).items():
                    if isinstance(val, dict):
                        self._by_source[src] = {
                            "calls": int(val.get("calls", 0)),
                            "texts": int(val.get("texts", 0)),
                            "tokens": int(val.get("tokens", 0)),
                            "total_latency_ms": float(val.get("total_latency_ms", 0.0)),
                        }
        except Exception:
            pass

    def add(
        self,
        texts_count: int,
        tokens: int,
        source: str = "unknown",
        latency_ms: float = 0.0,
    ) -> None:
        """原子地累加统计数并刷新文件。"""
        with self._lock:
            self._texts += texts_count
            self._calls += 1
            self._tokens += tokens
            entry = self._by_source.get(source)
            if entry is None:
                entry = {"calls": 0, "texts": 0, "tokens": 0, "total_latency_ms": 0.0}
                self._by_source[source] = entry
            entry["calls"] += 1
            entry["texts"] += texts_count
            entry["tokens"] += tokens
            entry["total_latency_ms"] += latency_ms
            self._flush_locked()

    def _flush_locked(self) -> None:
        """持有锁时将统计写入文件（临时文件 + rename 原子替换）。"""
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_texts": self._texts,
                "total_calls": self._calls,
                "total_tokens": self._tokens,
                "last_updated": datetime.now(tz=timezone.utc).isoformat(),
                "by_source": {
                    src: {
                        "calls": v["calls"],
                        "texts": v["texts"],
                        "tokens": v["tokens"],
                        "total_latency_ms": round(v["total_latency_ms"], 1),
                        "avg_latency_ms": (
                            round(v["total_latency_ms"] / v["calls"], 1) if v["calls"] else 0.0
                        ),
                    }
                    for src, v in sorted(self._by_source.items())
                },
            }
            tmp = self._file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._file)
        except Exception:
            pass  # 统计写入失败不影响主流程


_global_embedding_stats = _GlobalEmbeddingStats(_STATS_FILE)


class BaseEmbedder(ABC):
    """所有 Embedding 适配器的统一接口。"""

    @staticmethod
    def _accumulate_usage(texts_count: int, tokens: int, latency_ms: float = 0.0) -> None:
        """将本次 Embedding 调用的统计写入全局统计文件。"""
        source = _embedding_call_source.get()
        _global_embedding_stats.add(texts_count, tokens, source, latency_ms)

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding。"""

    def embed_query(self, text: str) -> list[float]:
        """单条查询 embedding。"""
        return self.embed([text])[0]
