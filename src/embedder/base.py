"""Embedder 统一抽象基类。"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

# ── Embedding 调用来源标签 ────────────────────────────────────────────────────
# 各调用点在 embed / embed_query 前设置此标签，用于分来源统计 embedding 消耗。
_embed_call_source: ContextVar[str] = ContextVar("_embed_call_source", default="unknown")


def set_embed_call_source(source: str) -> None:
    """设置下一次 Embedding 调用的来源标签（用于分来源统计）。"""
    _embed_call_source.set(source)


# ── 全局 Embedding 统计（进程级单例）─────────────────────────────────────────
_EMBED_STATS_FILE = Path(__file__).parent.parent.parent / "data" / "embedding_stats.json"


class _GlobalEmbedStats:
    """进程级累计 Embedding 统计，线程安全，每次更新后原子写入文件。

    写入格式（JSON）::

        {
          "total_texts": 234,
          "total_calls": 89,
          "last_updated": "2024-01-01T12:00:00+00:00",
          "by_source": {
            "memory_search": {
              "calls": 45,
              "texts": 45,
              "total_latency_ms": 1234.0,
              "avg_latency_ms": 27.4
            }
          }
        }
    """

    def __init__(self, stats_file: Path) -> None:
        self._file = stats_file
        self._lock = threading.Lock()
        self._total_texts: int = 0
        self._total_calls: int = 0
        self._by_source: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self._file.exists():
                data = json.loads(self._file.read_text(encoding="utf-8"))
                self._total_texts = int(data.get("total_texts", 0))
                self._total_calls = int(data.get("total_calls", 0))
                for src, val in (data.get("by_source") or {}).items():
                    if isinstance(val, dict):
                        self._by_source[src] = {
                            "calls": int(val.get("calls", 0)),
                            "texts": int(val.get("texts", 0)),
                            "total_latency_ms": float(val.get("total_latency_ms", 0.0)),
                        }
        except Exception:
            pass

    def add(self, texts_count: int, source: str = "unknown", latency_ms: float = 0.0) -> None:
        """原子地累加调用统计并刷新文件。"""
        with self._lock:
            self._total_texts += texts_count
            self._total_calls += 1
            entry = self._by_source.get(source)
            if entry is None:
                entry = {"calls": 0, "texts": 0, "total_latency_ms": 0.0}
                self._by_source[source] = entry
            entry["calls"] += 1
            entry["texts"] += texts_count
            entry["total_latency_ms"] += latency_ms
            self._flush_locked()

    def _flush_locked(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_texts": self._total_texts,
                "total_calls": self._total_calls,
                "last_updated": datetime.now(tz=timezone.utc).isoformat(),
                "by_source": {
                    src: {
                        "calls": v["calls"],
                        "texts": v["texts"],
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
            pass


_global_embed_stats = _GlobalEmbedStats(_EMBED_STATS_FILE)


class BaseEmbedder(ABC):
    """所有 Embedding 适配器的统一接口。"""

    @staticmethod
    def _accumulate_embed_usage(texts_count: int, latency_ms: float = 0.0) -> None:
        """将本次 Embedding 调用的统计计入全局统计文件（按来源分类）。"""
        source = _embed_call_source.get()
        _global_embed_stats.add(texts_count, source, latency_ms)

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding。"""

    def embed_query(self, text: str) -> list[float]:
        """单条查询 embedding。"""
        return self.embed([text])[0]

