"""Graphiti cross-encoder reranker using Gemini.

This is intentionally lightweight: it only provides pairwise relevance scoring
for (query, passage) and returns scores in [0, 1].

Strict mode should fail-fast at wiring time if enabled but missing API key.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    match = _JSON_ARRAY_RE.search(text or "")
    if not match:
        raise ValueError("Gemini cross-encoder did not return a JSON array")
    payload = match.group(0)
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("Gemini cross-encoder JSON is not a list")
    out: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


@dataclass(slots=True)
class GeminiCrossEncoderReranker:
    api_key: str
    model: str = "gemini-3.1-flash-lite-preview"

    def __post_init__(self) -> None:
        if not str(self.api_key or "").strip():
            raise ValueError("GeminiCrossEncoderReranker requires a non-empty api_key")

        # Lazy import to avoid forcing dependency unless used.
        from google import genai  # noqa: PLC0415

        self._genai = genai
        self._client = genai.Client(api_key=self.api_key)

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

        system = (
            "You are a precise cross-encoder reranker. "
            "Given a query and multiple passages, output a JSON array only. "
            "Each item must have: id (string) and score (number in [0,1]). "
            "Score is the semantic relevance of the passage to the query. "
            "Do not include any explanation."
        )

        user = {
            "query": query,
            "passages": cleaned,
            "scoring": {
                "range": "[0,1]",
                "interpretation": "1=highly relevant, 0=irrelevant",
            },
            "output": "JSON array only",
        }

        config = self._genai.types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2048,
            system_instruction=system,
        )

        resp = self._client.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": [{"text": json.dumps(user, ensure_ascii=False)}]}],
            config=config,
        )
        text = resp.text or ""

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
