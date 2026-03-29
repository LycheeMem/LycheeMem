"""Action-Aware Search Planner。

分析用户查询 + 上下文，输出结构化 SearchPlan，
指导下游多通道召回和 scorer 打分。
"""

from __future__ import annotations

import json
from typing import Any

from src.llm.base import BaseLLM
from src.memory.semantic.models import SearchPlan
from src.memory.semantic.prompts import RETRIEVAL_PLANNING_SYSTEM


class ActionAwareSearchPlanner:
    """行动感知检索规划器。"""

    def __init__(self, llm: BaseLLM):
        self._llm = llm

    def plan(
        self,
        user_query: str,
        *,
        recent_context: str = "",
    ) -> SearchPlan:
        """生成检索计划。

        Args:
            user_query: 用户当前查询
            recent_context: 最近几轮对话（可选）

        Returns:
            结构化 SearchPlan
        """
        user_content = f"<USER_QUERY>\n{user_query}\n</USER_QUERY>"
        if recent_context:
            user_content += f"\n\n<RECENT_CONTEXT>\n{recent_context}\n</RECENT_CONTEXT>"

        response = self._llm.generate([
            {"role": "system", "content": RETRIEVAL_PLANNING_SYSTEM},
            {"role": "user", "content": user_content},
        ])

        try:
            parsed = self._parse_json(response)
            return self._dict_to_plan(parsed)
        except (ValueError, json.JSONDecodeError):
            # 兜底：默认的简单检索计划
            return SearchPlan(
                mode="answer",
                semantic_queries=[user_query],
                pragmatic_queries=[],
                depth=5,
                reasoning="plan_parse_failed, fallback to simple query",
            )

    def _dict_to_plan(self, d: dict[str, Any]) -> SearchPlan:
        mode = d.get("mode", "answer")
        if mode not in ("answer", "action", "mixed"):
            mode = "answer"

        temporal_filter = d.get("temporal_filter")
        if temporal_filter and not isinstance(temporal_filter, dict):
            temporal_filter = None

        return SearchPlan(
            mode=mode,
            semantic_queries=d.get("semantic_queries", []),
            pragmatic_queries=d.get("pragmatic_queries", []),
            temporal_filter=temporal_filter,
            tool_hints=d.get("tool_hints", []),
            required_constraints=d.get("required_constraints", []),
            required_affordances=d.get("required_affordances", []),
            missing_slots=d.get("missing_slots", []),
            depth=int(d.get("depth", 5)),
            reasoning=d.get("reasoning", ""),
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
