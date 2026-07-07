"""Structured debug tracing for semantic retrieval.

Tracing is disabled by default. Enable it with either:
- LYCHEE_SEMANTIC_TRACE=1
- LYCHEE_RETRIEVAL_DEBUG=1

The default output path is scripts/log.txt and can be overridden with
LYCHEE_SEMANTIC_TRACE_PATH or LYCHEE_RETRIEVAL_DEBUG_PATH.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import uuid
from typing import Any, Mapping


_TRUE_VALUES = {"1", "true", "yes", "on", "debug", "full"}
_TRACE_LOCK = threading.Lock()


def semantic_trace_enabled() -> bool:
    return True


def semantic_trace_path() -> Path:
    raw = (
        os.getenv("LYCHEE_SEMANTIC_TRACE_PATH")
        or os.getenv("LYCHEE_RETRIEVAL_DEBUG_PATH")
        or "scripts/log.txt"
    )
    return Path(raw)


def semantic_trace_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def make_trace_id(prefix: str = "ret") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def truncate_trace_text(value: Any, max_chars: int | None = None) -> str:
    text = str(value or "")
    if max_chars is None:
        max_chars = semantic_trace_int("LYCHEE_SEMANTIC_TRACE_MAX_TEXT", 800, minimum=0)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...<truncated {len(text) - max_chars} chars>"


def semantic_trace_event(event: str, payload: Mapping[str, Any]) -> None:
    if not semantic_trace_enabled():
        return

    path = semantic_trace_path()
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": str(event or "semantic_trace"),
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
    }
    row.update(dict(payload or {}))

    line = json.dumps(row, ensure_ascii=False, default=str)
    with _TRACE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
