"""
检索协调器 (Memory Search Coordinator)。

语义记忆检索直接调用 CompactSemanticEngine.search()，并在进入检索前构造
ActionState，让 planner 看到“当前想做什么、最近发生了什么、受什么约束”。
技能库检索则按 answer / action / mixed 三种模式自适应启用。
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.prompts import SEARCH_COORDINATOR_SYSTEM_PROMPT
from src.embedder.base import BaseEmbedder
from src.llm.base import BaseLLM, set_llm_call_source
from src.memory.procedural.sqlite_skill_store import SQLiteSkillStore
from src.memory.semantic.base import BaseSemanticMemoryEngine
from src.memory.semantic.models import ActionState
from src.memory.semantic.prompts import RETRIEVAL_PLANNING_SYSTEM


class SearchCoordinator(BaseAgent):
    """检索协调器：每次请求均同时检索语义记忆和技能库。"""

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        skill_store: SQLiteSkillStore,
        semantic_engine: BaseSemanticMemoryEngine,
        skill_top_k: int = 3,
        skill_reuse_threshold: float = 0.85,
    ):
        super().__init__(llm=llm, prompt_template=SEARCH_COORDINATOR_SYSTEM_PROMPT)
        self.embedder = embedder
        self.skill_store = skill_store
        self.semantic_engine = semantic_engine
        self.skill_top_k = skill_top_k
        self.skill_reuse_threshold = skill_reuse_threshold

    def run(
        self,
        user_query: str,
        **kwargs,
    ) -> dict[str, Any]:
        """同时检索语义记忆和技能库。"""
        session_id = kwargs.get("session_id")
        if session_id is not None:
            session_id = str(session_id)
        top_k = kwargs.get("top_k")
        include_skills = bool(kwargs.get("include_skills", True))

        recent_context = self._build_recent_context(
            raw_recent_turns=kwargs.get("raw_recent_turns") or [],
            compressed_history=kwargs.get("compressed_history") or [],
        )
        
        analysis = self._analyze_query_and_context(user_query, recent_context)
        
        action_state = self._build_action_state(
            user_query=user_query,
            recent_context=recent_context,
            wm_token_usage=int(kwargs.get("wm_token_usage", 0) or 0),
            tool_calls=kwargs.get("tool_calls") or [],
            analysis=analysis,
        )

        feedback_update: dict[str, Any] = {}
        if session_id:
            try:
                feedback_update = self.semantic_engine.apply_feedback_from_user_turn(
                    session_id=session_id,
                    user_turn=user_query,
                )
            except Exception:
                feedback_update = {}

        result = self._run_compact(
            user_query,
            session_id=session_id,
            recent_context=recent_context,
            action_state=action_state,
            top_k=int(top_k) if top_k is not None else None,
            include_skills=include_skills,
            analysis=analysis,
            retrieval_plan=self._build_retrieval_plan(
                user_query, recent_context, action_state,
            ),
        )
        result["feedback_update"] = feedback_update
        return result

    def _run_compact(
        self,
        user_query: str,
        *,
        session_id: str | None = None,
        recent_context: str = "",
        action_state: ActionState | None = None,
        top_k: int | None = None,
        include_skills: bool = True,
        analysis: dict[str, Any] | None = None,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compact 后端路径：semantic_engine.search() + mode-aware 技能检索。"""
        result = self.semantic_engine.search(
            query=user_query,
            session_id=session_id,
            top_k=int(top_k or 0),
            recent_context=recent_context,
            action_state=self._action_state_to_dict(action_state),
            retrieval_plan=retrieval_plan,
        )

        graph_memories = []
        if result.context.strip():
            graph_memories = [
                {
                    "anchor": {
                        "node_id": "compact_context",
                        "name": "CompactSemanticMemory",
                        "label": "SemanticContext",
                        "score": 1.0,
                    },
                    "subgraph": {"nodes": [], "edges": []},
                    "constructed_context": result.context,
                    "provenance": result.provenance,
                }
            ]

        skill_results: list[dict[str, Any]] = []
        if include_skills:
            skill_results = self._search_skills(
                user_query,
                plan=result.retrieval_plan,
                action_state=result.action_state,
                top_k=top_k,
                analysis=analysis,
            )

        return {
            "retrieved_graph_memories": graph_memories,
            "retrieved_skills": skill_results,
            "retrieval_plan": result.retrieval_plan,
            "action_state": result.action_state,
            "search_mode": result.mode,
            "semantic_usage_log_id": result.usage_log_id,
        }

    def _search_skills(
        self,
        query: str,
        *,
        plan: dict[str, Any] | None = None,
        action_state: dict[str, Any] | None = None,
        top_k: int | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """按 mode / decision state 自适应检索技能库。"""
        plan = plan or {}
        action_state = action_state or {}
        analysis = analysis or {}
        mode = str(plan.get("mode") or "answer").strip().lower() or "answer"
        looks_procedural = bool(analysis.get("looks_procedural", False))

        if mode == "answer" and not looks_procedural:
            return []

        hyde_doc = str(analysis.get("hyde_doc", "")).strip()
        if not hyde_doc:
            hyde_doc = self._build_skill_query(query, plan=plan, action_state=action_state)

        hyde_embedding = self.embedder.embed_query(hyde_doc)

        default_top_k = self.skill_top_k
        if mode == "mixed":
            default_top_k = max(1, self.skill_top_k - 1)
        elif mode == "answer":
            default_top_k = 1
        skill_top_k = top_k if top_k is not None else default_top_k

        skill_query_fallback = self._build_skill_query(query, plan=plan, action_state=action_state)

        results = self.skill_store.search(
            query=skill_query_fallback,
            top_k=skill_top_k,
            query_embedding=hyde_embedding,
        )

        reuse_threshold = self.skill_reuse_threshold
        if mode == "action":
            reuse_threshold = max(0.0, self.skill_reuse_threshold - 0.05)

        for skill in results:
            skill["reusable"] = skill.get("score", 0) >= reuse_threshold
            skill["retrieval_mode"] = mode

        return results

    def _build_recent_context(
        self,
        *,
        raw_recent_turns: list[dict[str, Any]],
        compressed_history: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []

        summaries = [m for m in (compressed_history or []) if m.get("role") == "system"]
        for msg in summaries[-2:]:
            content = str(msg.get("content") or "").strip()
            if content:
                parts.append(f"Summary: {content}")

        recent_turns = raw_recent_turns or [m for m in (compressed_history or []) if m.get("role") != "system"]
        for turn in recent_turns[-6:]:
            role = str(turn.get("role") or "").strip() or "unknown"
            content = str(turn.get("content") or "").strip()
            if content:
                parts.append(f"{role}: {content}")

        context = "\n".join(parts).strip()
        if len(context) > 2000:
            context = context[-2000:]
        return context

    def _build_action_state(
        self,
        *,
        user_query: str,
        recent_context: str,
        wm_token_usage: int,
        tool_calls: list[dict[str, Any]],
        analysis: dict[str, Any],
    ) -> ActionState:
        last_tool_name = ""
        last_tool_result = ""
        if tool_calls:
            last_tool = tool_calls[-1] or {}
            last_tool_name = str(
                last_tool.get("name") or last_tool.get("tool_name") or last_tool.get("id") or ""
            ).strip()
            last_tool_result = str(
                last_tool.get("result") or last_tool.get("output") or last_tool.get("content") or ""
            ).strip()

        available_tools = list(analysis.get("available_tools") or [])
        if last_tool_name and last_tool_name not in available_tools:
            available_tools.append(last_tool_name)

        return ActionState(
            current_subgoal=user_query.strip(),
            tentative_action=str(analysis.get("tentative_action", "")).strip(),
            last_tool_name=last_tool_name,
            last_tool_result=last_tool_result[:500],
            missing_slots=[],
            known_constraints=list(analysis.get("known_constraints") or [])[:6],
            available_tools=available_tools[:8],
            failure_signal=str(analysis.get("failure_signal", "")).strip()[:160],
            token_budget=max(0, wm_token_usage),
            recent_context_excerpt=recent_context[:1000],
        )

    def _analyze_query_and_context(self, user_query: str, recent_context: str) -> dict[str, Any]:
        user_content = f"<USER_QUERY>\n{user_query}\n</USER_QUERY>"
        if recent_context:
            user_content += f"\n\n<RECENT_CONTEXT>\n{recent_context}\n</RECENT_CONTEXT>"
        
        set_llm_call_source("query_analysis_and_hyde")
        response = self._call_llm(
            user_content,
            system_content=self.prompt_template,
            add_time_basis=True,
        )
        
        try:
            raw = response.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) >= 3 else parts[-1].strip()
            return json.loads(raw)
        except Exception:
            return {}

    def _build_retrieval_plan(
        self,
        user_query: str,
        recent_context: str,
        action_state: ActionState | None,
    ) -> dict[str, Any] | None:
        """调用 RETRIEVAL_PLANNING_SYSTEM LLM 生成结构化 SearchPlan。"""
        user_content = f"<USER_QUERY>\n{user_query}\n</USER_QUERY>"
        if recent_context:
            user_content += f"\n\n<RECENT_CONTEXT>\n{recent_context}\n</RECENT_CONTEXT>"
        if action_state:
            state_dict = self._action_state_to_dict(action_state)
            user_content += f"\n\n<ACTION_STATE>\n{json.dumps(state_dict, ensure_ascii=False)}\n</ACTION_STATE>"

        try:
            set_llm_call_source("retrieval_planning")
            response = self._call_llm(
                user_content,
                system_content=RETRIEVAL_PLANNING_SYSTEM,
                add_time_basis=True,
            )
            raw = response.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) >= 3 else parts[-1].strip()
            plan = json.loads(raw)
            plan.pop("reasoning", None)
            return plan
        except Exception:
            return None

    def _build_skill_query(
        self,
        query: str,
        *,
        plan: dict[str, Any],
        action_state: dict[str, Any],
    ) -> str:
        mode = str(plan.get("mode") or "answer").strip().lower() or "answer"
        parts: list[str] = [str(query or "").strip()]

        tentative_action = str(action_state.get("tentative_action") or "").strip()
        if tentative_action and tentative_action not in parts:
            parts.append(tentative_action)

        if mode in {"action", "mixed"}:
            for key in (
                "pragmatic_queries",
                "tool_hints",
                "required_constraints",
                "required_affordances",
                "missing_slots",
            ):
                for value in plan.get(key, []) or []:
                    item = str(value or "").strip()
                    if item and item not in parts:
                        parts.append(item)

        return "；".join(parts)

    @staticmethod
    def _action_state_to_dict(action_state: ActionState | None) -> dict[str, Any] | None:
        if action_state is None:
            return None
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
