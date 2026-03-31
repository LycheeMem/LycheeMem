"""Action-Aware Search Planner。

分析用户查询 + 上下文，输出结构化 SearchPlan，
指导下游多通道召回和 scorer 打分。
"""

from __future__ import annotations

import json
from typing import Any

from src.llm.base import BaseLLM
from src.memory.semantic.models import ActionState, SearchPlan
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
        action_state: ActionState | None = None,
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
        if action_state is not None:
            user_content += (
                "\n\n<ACTION_STATE>\n"
                f"{json.dumps(self._action_state_to_dict(action_state), ensure_ascii=False)}\n"
                "</ACTION_STATE>"
            )

        response = self._llm.generate([
            {"role": "system", "content": RETRIEVAL_PLANNING_SYSTEM},
            {"role": "user", "content": user_content},
        ])

        try:
            parsed = self._parse_json(response)
            plan = self._dict_to_plan(parsed)
            return self._merge_plan_with_action_state(plan, action_state)
        except (ValueError, json.JSONDecodeError):
            # 兜底：默认的简单检索计划
            fallback = SearchPlan(
                mode="answer",
                semantic_queries=[user_query],
                pragmatic_queries=[],
                depth=5,
                reasoning="plan_parse_failed, fallback to simple query",
            )
            return self._merge_plan_with_action_state(fallback, action_state)

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

    def _merge_plan_with_action_state(
        self,
        plan: SearchPlan,
        action_state: ActionState | None,
    ) -> SearchPlan:
        if action_state is None:
            return plan

        plan.tool_hints = self._merge_unique(plan.tool_hints, action_state.available_tools)
        plan.required_constraints = self._merge_unique(
            plan.required_constraints,
            action_state.known_constraints,
        )
        plan.missing_slots = self._merge_unique(plan.missing_slots, action_state.missing_slots)

        if action_state.tentative_action and plan.mode == "answer":
            if plan.pragmatic_queries or plan.tool_hints or plan.missing_slots or plan.required_constraints:
                plan.mode = "mixed"

        if action_state.tentative_action and action_state.tentative_action not in plan.pragmatic_queries:
            if plan.mode in {"action", "mixed"}:
                plan.pragmatic_queries = [action_state.tentative_action, *plan.pragmatic_queries]

        return plan

    @staticmethod
    def _merge_unique(primary: list[str], secondary: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for value in list(primary or []) + list(secondary or []):
            item = str(value or "").strip()
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    @staticmethod
    def _action_state_to_dict(action_state: ActionState) -> dict[str, Any]:
        return {
            "current_subgoal": action_state.current_subgoal,
            "tentative_action": action_state.tentative_action,
            "last_tool_name": action_state.last_tool_name,
            "last_tool_result": action_state.last_tool_result,
            "missing_slots": list(action_state.missing_slots),
            "known_constraints": list(action_state.known_constraints),
            "available_tools": list(action_state.available_tools),
            "failure_signal": action_state.failure_signal,
            "token_budget": action_state.token_budget,
            "recent_context_excerpt": action_state.recent_context_excerpt,
        }

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
