"""
感觉记忆缓冲区。

FIFO 队列，缓存最近 N 条原始输入的特征。
MVP 阶段先做文本 buffer，多模态后续扩展。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SensoryItem:
    """感觉记忆条目。"""
    content: Any
    modality: str = "text"  # text / image / audio
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SensoryBuffer:
    """FIFO 感觉记忆缓冲区。"""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._buffer: deque[SensoryItem] = deque(maxlen=max_size)

    def push(self, content: Any, modality: str = "text") -> None:
        self._buffer.append(SensoryItem(content=content, modality=modality))

    def get_recent(self, n: int | None = None) -> list[SensoryItem]:
        if n is None:
            return list(self._buffer)
        return list(self._buffer)[-n:]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
