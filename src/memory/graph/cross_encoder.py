"""Graphiti cross-encoder reranker using the workspace LLM adapter.

This component is intentionally lightweight: it only provides pairwise
relevance scoring for (query, passage) and returns scores in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any

from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# 匹配 JSON 数组，允许前后有任意内容（含 markdown 代码块）
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*?\](?=\s*(?:```|$|,|\}))", re.DOTALL)
_JSON_ARRAY_RE_GREEDY = re.compile(r"\[[\s\S]*\]", re.DOTALL)


def _try_parse_list(text: str) -> list[Any] | None:
    """尝试从字符串中提取 JSON 数组，支持多种包装形式。"""
    text = text.strip()

    # 1. 整体直接是合法 JSON array
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 2. markdown 代码块：```json [...] ``` 或 ``` [...] ```
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        try:
            data = json.loads(code_block.group(1).strip())
            if isinstance(data, list):
                return data
        except Exception:
            pass

    # 3. 整体是 JSON object，其中某个 value 是 list（如 {"scores":[...]}）
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass

    # 4. 从 object 内部提取嵌套数组（先贪婪找 object，再找其中的 array）
    obj_match = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
    if obj_match:
        try:
            obj = json.loads(obj_match.group(0))
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, list):
                        return v
        except Exception:
            pass

    # 5. 正则提取裸数组（贪婪）
    for pattern in (_JSON_ARRAY_RE, _JSON_ARRAY_RE_GREEDY):
        for m in pattern.finditer(text):
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    return data
            except Exception:
                continue

    return None


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    result = _try_parse_list(text or "")
    if result is None:
        raise ValueError("Cross-encoder did not return a JSON array")
    return [item for item in result if isinstance(item, dict)]


@dataclass(slots=True)
class LLMCrossEncoderReranker:
    llm: BaseLLM

    def __post_init__(self) -> None:
        if self.llm is None or not hasattr(self.llm, "generate"):
            raise ValueError("LLMCrossEncoderReranker requires an llm with generate()")

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
            "You are a cross-encoder reranker. "
            "Output ONLY a raw JSON array (no markdown, no wrapping object, no explanation). "
            "Each element: {\"id\": <string>, \"score\": <float 0-1>}. "
            "Example: [{\"id\": \"abc\", \"score\": 0.85}]"
        )
        user = json.dumps(
            {
                "query": query,
                "passages": cleaned,
            },
            ensure_ascii=False,
        )
        try:
            text = self.llm.generate(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=2048,
            )
            items = _extract_json_array(text)
        except Exception as exc:
            logger.warning("Cross-encoder scoring failed (%s), falling back to 0.0 scores", exc)
            return {p["id"]: 0.0 for p in cleaned}

        scores: dict[str, float] = {}
        for it in items:
            pid = str(it.get("id") or it.get("fact_id") or "").strip()
            if not pid:
                continue
            try:
                s = float(it.get("score"))
            except Exception:
                continue
            scores[pid] = max(0.0, min(1.0, s))

        # Ensure all requested ids exist (default 0.0)
        for p in cleaned:
            scores.setdefault(p["id"], 0.0)
        return scores


# Backward-compatible alias for existing imports.
GeminiCrossEncoderReranker = LLMCrossEncoderReranker
