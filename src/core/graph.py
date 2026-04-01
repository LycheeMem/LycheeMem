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
from typing import Any, AsyncIterator

from langgraph.graph import StateGraph, START, END

from src.agents.consolidator_agent import ConsolidatorAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.search_coordinator import SearchCoordinator
from src.agents.synthesizer_agent import SynthesizerAgent
from src.agents.wm_manager import WMManager
from src.core.state import PipelineState
from src.llm.base import _token_accumulator

logger = logging.getLogger("src.pipeline")


class LycheePipeline:
    """LycheeMem 认知记忆 Pipeline。

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
        self._consolidation_results: dict[str, dict[str, Any]] = {}
        self._consolidation_state_lock = threading.Lock()
        self._consolidation_job_seq = 0

    @staticmethod
    def _consolidation_key(session_id: str, user_id: str = "") -> str:
        return f"{user_id}::{session_id}"

    def _begin_consolidation(self, session_id: str, user_id: str = "") -> int:
        key = self._consolidation_key(session_id, user_id)
        with self._consolidation_state_lock:
            self._consolidation_job_seq += 1
            job_id = self._consolidation_job_seq
            payload = {
                "session_id": session_id,
                "status": "pending",
                "entities_added": 0,
                "skills_added": 0,
                "facts_added": 0,
                "has_novelty": None,
                "skipped_reason": None,
                "steps": [],
                "job_id": job_id,
            }
            self._consolidation_results[key] = payload
            self._last_consolidation = dict(payload)
            return job_id

    def _finish_consolidation(
        self,
        *,
        session_id: str,
        user_id: str = "",
        job_id: int,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        key = self._consolidation_key(session_id, user_id)
        with self._consolidation_state_lock:
            current = self._consolidation_results.get(key)
            if current is not None and int(current.get("job_id") or 0) != job_id:
                return

            if error is not None:
                payload = {
                    "session_id": session_id,
                    "status": "done",
                    "entities_added": 0,
                    "skills_added": 0,
                    "facts_added": 0,
                    "has_novelty": None,
                    "skipped_reason": None,
                    "error": error,
                    "steps": [],
                    "job_id": job_id,
                }
            else:
                result = result or {}
                payload = {
                    "session_id": session_id,
                    "status": "skipped" if result.get("skipped_reason") else "done",
                    "entities_added": result.get("entities_added", 0),
                    "skills_added": result.get("skills_added", 0),
                    "facts_added": result.get("facts_added", 0),
                    "has_novelty": result.get("has_novelty"),
                    "skipped_reason": result.get("skipped_reason"),
                    "steps": result.get("steps", []),
                    "job_id": job_id,
                }

            self._consolidation_results[key] = payload
            self._last_consolidation = dict(payload)

    def get_last_consolidation(
        self,
        *,
        session_id: str | None = None,
        user_id: str = "",
    ) -> dict[str, Any] | None:
        if not session_id:
            return dict(self._last_consolidation) if self._last_consolidation is not None else None

        key = self._consolidation_key(session_id, user_id)
        with self._consolidation_state_lock:
            result = self._consolidation_results.get(key)
            return dict(result) if result is not None else None

    # ──────────────────────────────────────
    # Node functions
    # ──────────────────────────────────────

    def _wm_manager_node(self, state: PipelineState) -> dict[str, Any]:
        """工作记忆管理节点：追加对话、token 预算检查、按需压缩、渲染上下文。"""
        result = self.wm_manager.run(
            session_id=state["session_id"],
            user_query=state["user_query"],
            user_id=state.get("user_id", ""),
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
            session_id=state.get("session_id"),
            user_id=state.get("user_id", ""),
            compressed_history=state.get("compressed_history", []),
            raw_recent_turns=state.get("raw_recent_turns", []),
            wm_token_usage=state.get("wm_token_usage", 0),
            tool_calls=state.get("tool_calls", []),
        )
        return {
            "retrieved_graph_memories": result["retrieved_graph_memories"],
            "retrieved_skills": result["retrieved_skills"],
            "retrieval_plan": result.get("retrieval_plan", {}),
            "action_state": result.get("action_state", {}),
            "search_mode": result.get("search_mode", "answer"),
            "semantic_usage_log_id": result.get("semantic_usage_log_id", ""),
            "feedback_update": result.get("feedback_update", {}),
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
            retrieved_skills=state.get("retrieved_skills", []),
        )

        # 将 assistant 回复写回会话日志
        self.wm_manager.append_assistant_turn(
            state["session_id"], result["final_response"],
            user_id=state.get("user_id", ""),
        )

        semantic_engine = getattr(self.search_coordinator, "semantic_engine", None)
        usage_log_id = str(state.get("semantic_usage_log_id") or "")
        if semantic_engine is not None and usage_log_id:
            try:
                semantic_engine.finalize_usage_log(
                    log_id=usage_log_id,
                    final_response_excerpt=result["final_response"],
                )
            except Exception:
                logger.warning("finalize_usage_log failed session=%s", state.get("session_id"), exc_info=True)

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

    def run(self, user_query: str, session_id: str, user_id: str = "") -> dict[str, Any]:
        """同步运行 Pipeline。

        Args:
            user_query: 用户输入。
            session_id: 会话 ID。
            user_id: 用户 ID（用于多用户隔离）。

        Returns:
            完整的 PipelineState（包含 final_response 等所有字段）。
        """
        counter: dict[str, int] = {"input": 0, "output": 0}
        tok = _token_accumulator.set(counter)
        try:
            initial_state: dict[str, Any] = {
                "user_query": user_query,
                "session_id": session_id,
                "user_id": user_id,
            }
            result = self._graph.invoke(initial_state)
        finally:
            # 先读取计数，再复位——确保后台固化线程启动后的 LLM 调用不计入本轮
            in_tok, out_tok = counter["input"], counter["output"]
            _token_accumulator.reset(tok)

        result["turn_input_tokens"] = in_tok
        result["turn_output_tokens"] = out_tok

        # 后台线程触发固化（fire-and-forget，不阻塞响应返回）
        if result.get("consolidation_pending"):
            job_id = self._begin_consolidation(session_id, user_id=user_id)
            self._trigger_consolidation_bg(
                session_id,
                retrieved_context=str(result.get("background_context") or ""),
                user_id=user_id,
                job_id=job_id,
            )

        return result

    async def arun(self, user_query: str, session_id: str, user_id: str = "") -> dict[str, Any]:
        """异步运行 Pipeline。"""
        initial_state: dict[str, Any] = {
            "user_query": user_query,
            "session_id": session_id,
            "user_id": user_id,
        }
        result = await self._graph.ainvoke(initial_state)

        # 异步触发固化
        if result.get("consolidation_pending"):
            job_id = self._begin_consolidation(session_id, user_id=user_id)
            asyncio.create_task(
                self._aconsolidate(
                    session_id,
                    retrieved_context=str(result.get("background_context") or ""),
                    user_id=user_id,
                    job_id=job_id,
                )
            )

        return result

    async def astream_steps(
        self, user_query: str, session_id: str, user_id: str = ""
    ) -> AsyncIterator[dict[str, Any]]:
        """逐节点执行 Pipeline，每步完成后 yield 进度事件。

        事件格式：
          {"type": "step", "step": <node_name>, "status": "done", ...extra}
          {"type": "done", "result": <full_state>}
        """
        counter: dict[str, int] = {"input": 0, "output": 0}
        tok = _token_accumulator.set(counter)
        _tok_reset = False

        def _reset_once() -> None:
            nonlocal _tok_reset
            if not _tok_reset:
                _tok_reset = True
                _token_accumulator.reset(tok)

        try:
            state: dict[str, Any] = {"user_query": user_query, "session_id": session_id, "user_id": user_id}

            patch = await asyncio.to_thread(self._wm_manager_node, state)
            state.update(patch)
            yield {
                "type": "step",
                "step": "wm_manager",
                "status": "done",
                "wm_token_usage": patch.get("wm_token_usage", 0),
                "patch": patch,
            }

            patch = await asyncio.to_thread(self._search_node, state)
            state.update(patch)
            yield {"type": "step", "step": "search", "status": "done", "patch": patch}

            patch = await asyncio.to_thread(self._synthesize_node, state)
            state.update(patch)
            yield {"type": "step", "step": "synthesize", "status": "done", "patch": patch}

            # reason 阶段：流式生成，逐 token yield，最后再发 step:reason 完成事件
            streaming_response = ""
            async for token in self.reasoner.astream(
                user_query=state["user_query"],
                compressed_history=state.get("compressed_history", []),
                background_context=state.get("background_context", ""),
                skill_reuse_plan=state.get("skill_reuse_plan", []),
                retrieved_skills=state.get("retrieved_skills", []),
            ):
                streaming_response += token
                yield {"type": "token", "content": token}

            # 写回 assistant turn（保持与同步路径一致）
            await asyncio.to_thread(
                self.wm_manager.append_assistant_turn,
                state["session_id"],
                streaming_response,
                user_id,
            )
            semantic_engine = getattr(self.search_coordinator, "semantic_engine", None)
            usage_log_id = str(state.get("semantic_usage_log_id") or "")
            if semantic_engine is not None and usage_log_id:
                try:
                    await asyncio.to_thread(
                        semantic_engine.finalize_usage_log,
                        log_id=usage_log_id,
                        final_response_excerpt=streaming_response,
                    )
                except Exception:
                    logger.warning("finalize_usage_log failed session=%s", state.get("session_id"), exc_info=True)
            patch = {"final_response": streaming_response, "consolidation_pending": True}
            state.update(patch)
            yield {"type": "step", "step": "reason", "status": "done", "patch": patch}

            # 读取 token 计数并复位——在后台固化任务创建之前完成，确保固化的 LLM 调用不计入本轮
            state["turn_input_tokens"] = counter["input"]
            state["turn_output_tokens"] = counter["output"]
            _reset_once()

            if state.get("consolidation_pending"):
                job_id = self._begin_consolidation(session_id, user_id=user_id)
                asyncio.create_task(
                    self._aconsolidate(
                        session_id,
                        retrieved_context=str(state.get("background_context") or ""),
                        user_id=user_id,
                        job_id=job_id,
                    )
                )

            yield {"type": "done", "result": dict(state)}
        finally:
            _reset_once()

    def consolidate(
        self, session_id: str, retrieved_context: str = "", user_id: str = ""
    ) -> dict[str, Any]:
        """手动触发固化（公共方法，可由 API BackgroundTasks 调用）。

        只处理自上次固化以来新增的 turns（水位线机制），成功后更新水位线，
        彻底消除跨轮重复固化问题。

        Args:
            session_id: 会话 ID。
            retrieved_context: Pipeline 检索阶段合成的已有记忆上下文，
                用于新颖性判断，避免重复固化纯查询型对话。

        Returns:
            dict 包含：entities_added, skills_added
        """
        store = self.wm_manager.session_store
        log = store.get_or_create(session_id, user_id=user_id)
        watermark = log.last_consolidated_turn_index
        raw_total = len(log.turns)
        new_turns = [
            t for t in log.turns[watermark:] if not t.get("deleted", False)
        ]
        if not new_turns:
            return {"entities_added": 0, "skills_added": 0, "skipped_reason": "no_new_turns"}
        result = self.consolidator.run(
            turns=new_turns, session_id=session_id, retrieved_context=retrieved_context,
            user_id=user_id,
        )
        # 固化成功后推进水位线
        store.set_last_consolidated_turn_index(session_id, raw_total)
        return result

    def _trigger_consolidation_bg(
        self,
        session_id: str,
        retrieved_context: str = "",
        user_id: str = "",
        job_id: int = 0,
    ) -> None:
        """在守护线程中触发固化（fire-and-forget）。"""
        thread = threading.Thread(
            target=self._safe_consolidate,
            args=(session_id, retrieved_context, user_id, job_id),
            daemon=True,
        )
        thread.start()

    def _safe_consolidate(
        self,
        session_id: str,
        retrieved_context: str = "",
        user_id: str = "",
        job_id: int = 0,
    ) -> None:
        """安全执行固化，异常不影响主流程。"""
        graphiti = getattr(self.consolidator, "graphiti_engine", None)
        strict = bool(getattr(graphiti, "strict", False))
        try:
            result = self.consolidate(session_id, retrieved_context=retrieved_context, user_id=user_id)
            self._finish_consolidation(
                session_id=session_id,
                user_id=user_id,
                job_id=job_id,
                result=result,
            )
        except Exception as exc:
            logger.exception("固化失败 session=%s", session_id)
            self._finish_consolidation(
                session_id=session_id,
                user_id=user_id,
                job_id=job_id,
                error=str(exc),
            )
            if strict:
                raise

    async def _aconsolidate(
        self,
        session_id: str,
        retrieved_context: str = "",
        user_id: str = "",
        job_id: int = 0,
    ) -> None:
        """异步场景下的后台固化（使用水位线，只处理新增 turns）。"""
        graphiti = getattr(self.consolidator, "graphiti_engine", None)
        strict = bool(getattr(graphiti, "strict", False))
        store = self.wm_manager.session_store
        log = store.get_or_create(session_id, user_id=user_id)
        watermark = log.last_consolidated_turn_index
        raw_total = len(log.turns)
        new_turns = [t for t in log.turns[watermark:] if not t.get("deleted", False)]
        if not new_turns:
            self._finish_consolidation(
                session_id=session_id,
                user_id=user_id,
                job_id=job_id,
                result={
                    "entities_added": 0,
                    "skills_added": 0,
                    "facts_added": 0,
                    "has_novelty": False,
                    "skipped_reason": "no_new_turns",
                    "steps": [],
                },
            )
            return
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.consolidator.run(
                    turns=new_turns,
                    session_id=session_id,
                    retrieved_context=retrieved_context,
                    user_id=user_id,
                ),
            )
            # 固化成功后推进水位线
            store.set_last_consolidated_turn_index(session_id, raw_total)
            self._finish_consolidation(
                session_id=session_id,
                user_id=user_id,
                job_id=job_id,
                result=result,
            )
        except Exception as exc:
            logger.exception("固化失败 session=%s", session_id)
            self._finish_consolidation(
                session_id=session_id,
                user_id=user_id,
                job_id=job_id,
                error=str(exc),
            )
            if strict:
                raise

    @property
    def graph(self):
        """暴露底层 compiled graph 供调试/可视化。"""
        return self._graph
