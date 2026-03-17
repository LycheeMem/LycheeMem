"""
LangGraph Pipeline 构建。

定义节点和边，编译为可执行的状态图。

流水线拓扑：
  __start__ → wm_manager → search → synthesize → reason → __end__

固化 Agent 不在主图中，在 reason 节点完成后通过 asyncio.create_task 异步触发。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from langgraph.graph import StateGraph, START, END

from a_frame.agents.consolidator_agent import ConsolidatorAgent
from a_frame.agents.reasoning_agent import ReasoningAgent
from a_frame.agents.search_coordinator import SearchCoordinator
from a_frame.agents.synthesizer_agent import SynthesizerAgent
from a_frame.agents.wm_manager import WMManager
from a_frame.core.state import PipelineState

logger = logging.getLogger("a_frame.pipeline")


class AFramePipeline:
    """A-Frame 认知记忆 Pipeline。

    封装 LangGraph StateGraph 的构建与运行。
    所有组件通过构造函数注入，Pipeline 本身不持有任何配置。
    """

    def __init__(
        self,
        wm_manager: WMManager,
        search_coordinator: SearchCoordinator,
        synthesizer: SynthesizerAgent,
        reasoner: ReasoningAgent,
        consolidator: ConsolidatorAgent,
    ):
        self.wm_manager = wm_manager
        self.search_coordinator = search_coordinator
        self.synthesizer = synthesizer
        self.reasoner = reasoner
        self.consolidator = consolidator

        self._graph = self._build_graph()
        self._last_consolidation: dict[str, Any] | None = None

    # ──────────────────────────────────────
    # Node functions
    # ──────────────────────────────────────

    def _wm_manager_node(self, state: PipelineState) -> dict[str, Any]:
        """工作记忆管理节点：追加对话、token 预算检查、按需压缩、渲染上下文。"""
        result = self.wm_manager.run(
            session_id=state["session_id"],
            user_query=state["user_query"],
        )
        return {
            "compressed_history": result["compressed_history"],
            "raw_recent_turns": result["raw_recent_turns"],
            "wm_token_usage": result["wm_token_usage"],
        }

    def _search_node(self, state: PipelineState) -> dict[str, Any]:
        """检索协调节点：从图谱和技能库检索相关记忆。"""
        result = self.search_coordinator.run(
            user_query=state["user_query"],
        )
        return {
            "retrieved_graph_memories": result["retrieved_graph_memories"],
            "retrieved_skills": result["retrieved_skills"],
        }

    def _synthesize_node(self, state: PipelineState) -> dict[str, Any]:
        """整合排序节点：融合多源检索结果为 background_context + skill_reuse_plan。"""
        result = self.synthesizer.run(
            user_query=state["user_query"],
            retrieved_graph_memories=state.get("retrieved_graph_memories", []),
            retrieved_skills=state.get("retrieved_skills", []),
        )
        return {
            "background_context": result["background_context"],
            "skill_reuse_plan": result.get("skill_reuse_plan", []),
            "provenance": result.get("provenance", []),
        }

    def _reason_node(self, state: PipelineState) -> dict[str, Any]:
        """推理节点：生成最终回答（含技能复用计划）。"""
        result = self.reasoner.run(
            user_query=state["user_query"],
            compressed_history=state.get("compressed_history", []),
            background_context=state.get("background_context", ""),
            skill_reuse_plan=state.get("skill_reuse_plan", []),
        )

        # 将 assistant 回复写回会话日志
        self.wm_manager.append_assistant_turn(
            state["session_id"], result["final_response"]
        )

        # 标记固化待处理
        return {
            "final_response": result["final_response"],
            "consolidation_pending": True,
        }

    # ──────────────────────────────────────
    # Graph building
    # ──────────────────────────────────────

    def _build_graph(self):
        """构建并编译 LangGraph StateGraph。"""
        g = StateGraph(PipelineState)

        # 注册节点
        g.add_node("wm_manager", self._wm_manager_node)
        g.add_node("search", self._search_node)
        g.add_node("synthesize", self._synthesize_node)
        g.add_node("reason", self._reason_node)

        # 线性连接
        g.add_edge(START, "wm_manager")
        g.add_edge("wm_manager", "search")
        g.add_edge("search", "synthesize")
        g.add_edge("synthesize", "reason")
        g.add_edge("reason", END)

        return g.compile()

    # ──────────────────────────────────────
    # Public API
    # ──────────────────────────────────────

    def run(self, user_query: str, session_id: str) -> dict[str, Any]:
        """同步运行 Pipeline。

        Args:
            user_query: 用户输入。
            session_id: 会话 ID。

        Returns:
            完整的 PipelineState（包含 final_response 等所有字段）。
        """
        initial_state: dict[str, Any] = {
            "user_query": user_query,
            "session_id": session_id,
        }
        result = self._graph.invoke(initial_state)

        # 后台线程触发固化（fire-and-forget，不阻塞响应返回）
        if result.get("consolidation_pending"):
            self._trigger_consolidation_bg(session_id)

        return result

    async def arun(self, user_query: str, session_id: str) -> dict[str, Any]:
        """异步运行 Pipeline。"""
        initial_state: dict[str, Any] = {
            "user_query": user_query,
            "session_id": session_id,
        }
        result = await self._graph.ainvoke(initial_state)

        # 异步触发固化
        if result.get("consolidation_pending"):
            asyncio.create_task(self._aconsolidate(session_id))

        return result

    def consolidate(self, session_id: str) -> dict[str, Any]:
        """手动触发固化（公共方法，可由 API BackgroundTasks 调用）。

        Returns:
            dict 包含：entities_added, skills_added
        """
        turns = self.wm_manager.session_store.get_turns(session_id)
        if turns:
            return self.consolidator.run(turns=turns, session_id=session_id)
        return {"entities_added": 0, "skills_added": 0}

    def _trigger_consolidation_bg(self, session_id: str) -> None:
        """在守护线程中触发固化（fire-and-forget）。"""
        thread = threading.Thread(
            target=self._safe_consolidate,
            args=(session_id,),
            daemon=True,
        )
        thread.start()

    def _safe_consolidate(self, session_id: str) -> None:
        """安全执行固化，异常不影响主流程。"""
        try:
            result = self.consolidate(session_id)
            self._last_consolidation = {
                "session_id": session_id,
                "entities_added": result.get("entities_added", 0),
                "skills_added": result.get("skills_added", 0),
            }
        except Exception:
            logger.warning("固化失败 session=%s", session_id, exc_info=True)
            self._last_consolidation = {
                "session_id": session_id,
                "entities_added": 0,
                "skills_added": 0,
            }

    async def _aconsolidate(self, session_id: str) -> None:
        """异步场景下的后台固化。"""
        turns = self.wm_manager.session_store.get_turns(session_id)
        if turns:
            # 在线程池中运行（因为 consolidator.run 是同步的）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.consolidator.run, turns, session_id)

    @property
    def graph(self):
        """暴露底层 compiled graph 供调试/可视化。"""
        return self._graph
