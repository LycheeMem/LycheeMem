"""Graphiti cross-encoder reranker using the workspace LLM adapter.

This component is intentionally lightweight: it only provides pairwise
relevance scoring for (query, passage) and returns scores in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from src.llm.base import BaseLLM


_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    match = _JSON_ARRAY_RE.search(text or "")
    if not match:
        raise ValueError("Cross-encoder did not return a JSON array")
    payload = match.group(0)
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("Cross-encoder JSON is not a list")
    out: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


@dataclass(slots=True)
class LLMCrossEncoderReranker:
    llm: BaseLLM

    _response_format: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.llm is None or not hasattr(self.llm, "generate"):
            raise ValueError("LLMCrossEncoderReranker requires an llm with generate()")

        # OpenAI-compatible providers can honor this directly; others ignore it safely.
        self._response_format = {"type": "json_object"}

    def score(self, *, query: str, passages: list[dict[str, str]]) -> dict[str, float]:
        """Return a mapping id -> score in [0, 1].

        passages: [{"id": str, "text": str}, ...]
        """

        query = str(query or "").strip()
        if not query:
            return {}

        cleaned: list[dict[str, str]] = []
        for p in passages or []:
            pid = str(p.get("id") or "").strip()
            txt = str(p.get("text") or "").strip()
            if pid and txt:
                cleaned.append({"id": pid, "text": txt})

        if not cleaned:
            return {}

        system = """You are a precise cross-encoder reranker.
Given a query and multiple passages, output ONLY a JSON array.
Each array item must have fields:
- id: string
- score: number in [0,1]
Do not include explanation, markdown, or extra fields.
"""
        user = json.dumps(
            {
                "query": query,
                "passages": cleaned,
                "scoring": {
                    "range": "[0,1]",
                    "interpretation": "1=highly relevant, 0=irrelevant",
                },
                "output": "json_array_only",
            },
            ensure_ascii=False,
        )
        text = self.llm.generate(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=2048,
            response_format=self._response_format,
        )

        items = _extract_json_array(text)
        scores: dict[str, float] = {}
        for it in items:
            pid = str(it.get("id") or it.get("fact_id") or "").strip()
            if not pid:
                continue
            try:
                s = float(it.get("score"))
            except Exception:
                continue
            if s < 0.0:
                s = 0.0
            if s > 1.0:
                s = 1.0
            scores[pid] = s

        # Ensure all requested ids exist (default 0.0)
        for p in cleaned:
            pid = p["id"]
            scores.setdefault(pid, 0.0)
        return scores


# Backward-compatible alias for existing imports.
GeminiCrossEncoderReranker = LLMCrossEncoderReranker
