"""
LangGraph Pipeline 构建。

定义节点和边，编译为可执行的状态图。

流水线拓扑：
  __start__ → wm_manager → router → (条件分支)
                                       ├─ need_retrieval → search → synthesize → reason → __end__
                                       └─ direct_answer  →                        reason → __end__

固化 Agent 不在主图中，在 reason 节点完成后通过 asyncio.create_task 异步触发。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.graph import StateGraph, START, END

from a_frame.agents.consolidator_agent import ConsolidatorAgent
from a_frame.agents.reasoning_agent import ReasoningAgent
from a_frame.agents.router_agent import RouterAgent
from a_frame.agents.search_coordinator import SearchCoordinator
from a_frame.agents.synthesizer_agent import SynthesizerAgent
from a_frame.agents.wm_manager import WMManager
from a_frame.core.state import PipelineState
from a_frame.memory.sensory.buffer import SensoryBuffer


class AFramePipeline:
    """A-Frame 认知记忆 Pipeline。

    封装 LangGraph StateGraph 的构建与运行。
    所有组件通过构造函数注入，Pipeline 本身不持有任何配置。
    """

    def __init__(
        self,
        wm_manager: WMManager,
        router: RouterAgent,
        search_coordinator: SearchCoordinator,
        synthesizer: SynthesizerAgent,
        reasoner: ReasoningAgent,
        consolidator: ConsolidatorAgent,
        sensory_buffer: SensoryBuffer,
    ):
        self.wm_manager = wm_manager
        self.router = router
        self.search_coordinator = search_coordinator
        self.synthesizer = synthesizer
        self.reasoner = reasoner
        self.consolidator = consolidator
        self.sensory_buffer = sensory_buffer

        self._graph = self._build_graph()

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

    def _router_node(self, state: PipelineState) -> dict[str, Any]:
        """路由节点：分析意图，决定检索哪些记忆。"""
        decision = self.router.run(
            user_query=state["user_query"],
            recent_turns=state.get("raw_recent_turns", []),
        )
        return {"route": decision}

    def _search_node(self, state: PipelineState) -> dict[str, Any]:
        """检索协调节点：按路由决策从各记忆基质检索。"""
        result = self.search_coordinator.run(
            user_query=state["user_query"],
            route=state["route"],
        )
        return {
            "retrieved_graph_memories": result["retrieved_graph_memories"],
            "retrieved_skills": result["retrieved_skills"],
            "retrieved_sensory": result["retrieved_sensory"],
        }

    def _synthesize_node(self, state: PipelineState) -> dict[str, Any]:
        """整合排序节点：融合多源检索结果为 background_context。"""
        result = self.synthesizer.run(
            user_query=state["user_query"],
            retrieved_graph_memories=state.get("retrieved_graph_memories", []),
            retrieved_skills=state.get("retrieved_skills", []),
            retrieved_sensory=state.get("retrieved_sensory", []),
        )
        return {"background_context": result["background_context"]}

    def _reason_node(self, state: PipelineState) -> dict[str, Any]:
        """推理节点：生成最终回答。"""
        result = self.reasoner.run(
            user_query=state["user_query"],
            compressed_history=state.get("compressed_history", []),
            background_context=state.get("background_context", ""),
        )

        # 将 assistant 回复写回会话日志
        self.wm_manager.append_assistant_turn(
            state["session_id"], result["final_response"]
        )

        # 将用户输入推入感觉缓冲
        self.sensory_buffer.push(state["user_query"])

        # 标记固化待处理
        return {
            "final_response": result["final_response"],
            "consolidation_pending": True,
        }

    # ──────────────────────────────────────
    # Routing logic
    # ──────────────────────────────────────

    @staticmethod
    def _route_decision(state: PipelineState) -> str:
        """条件边：根据路由结果决定走检索分支还是直接回答。"""
        route = state.get("route", {})
        if route.get("need_graph") or route.get("need_skills") or route.get("need_sensory"):
            return "need_retrieval"
        return "direct_answer"

    # ──────────────────────────────────────
    # Graph building
    # ──────────────────────────────────────

    def _build_graph(self):
        """构建并编译 LangGraph StateGraph。"""
        g = StateGraph(PipelineState)

        # 注册节点
        g.add_node("wm_manager", self._wm_manager_node)
        g.add_node("router", self._router_node)
        g.add_node("search", self._search_node)
        g.add_node("synthesize", self._synthesize_node)
        g.add_node("reason", self._reason_node)

        # 连接边
        g.add_edge(START, "wm_manager")
        g.add_edge("wm_manager", "router")

        # 条件分支
        g.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "need_retrieval": "search",
                "direct_answer": "reason",
            },
        )

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

        # 异步触发固化（如果 event loop 可用则异步，否则同步）
        if result.get("consolidation_pending"):
            self._trigger_consolidation(session_id)

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

    def _trigger_consolidation(self, session_id: str) -> None:
        """同步场景下触发固化（直接执行）。"""
        turns = self.wm_manager.session_store.get_turns(session_id)
        if turns:
            self.consolidator.run(turns=turns)

    async def _aconsolidate(self, session_id: str) -> None:
        """异步场景下的后台固化。"""
        turns = self.wm_manager.session_store.get_turns(session_id)
        if turns:
            # 在线程池中运行（因为 consolidator.run 是同步的）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.consolidator.run, turns)

    @property
    def graph(self):
        """暴露底层 compiled graph 供调试/可视化。"""
        return self._graph
