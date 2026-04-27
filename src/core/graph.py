"""
LangGraph Pipeline 构建。

定义节点和边，编译为可执行的状态图。

流水线拓扑：
  __start__ → wm_manager → search → synthesize → reason → __end__

固化 Agent 不在主图中，在 reason 节点完成后通过 asyncio.create_task 异步触发。
"""

from __future__ import annotations

import asyncio
import json
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
from src.evolve.prompt_registry import get_active_versions_snapshot, select_prompt_versions
from src.llm.base import _token_accumulator

logger = logging.getLogger("src.pipeline")

_CHAT_API_PROMPTS: frozenset[str] = frozenset({
    "search_coordinator",
    "retrieval_planning",
    "composite_filter",
    "retrieval_adequacy_check",
    "synthesis",
    "reasoning",
})


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
        evolve_loop: Any | None = None,
    ):
        self.wm_manager = wm_manager
        self.search_coordinator = search_coordinator
        self.synthesizer = synthesizer
        self.reasoner = reasoner
        self.consolidator = consolidator
        self.evolve_loop = evolve_loop

        self._graph = self._build_graph()
        self._last_consolidation: dict[str, Any] | None = None
        self._consolidation_results: dict[str, dict[str, Any]] = {}
        self._consolidation_state_lock = threading.Lock()
        self._consolidation_job_seq = 0

    @staticmethod
    def _consolidation_key(session_id: str) -> str:
        return session_id

    def _begin_consolidation(self, session_id: str) -> int:
        key = self._consolidation_key(session_id)
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
        job_id: int,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        key = self._consolidation_key(session_id)
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
    ) -> dict[str, Any] | None:
        if not session_id:
            return dict(self._last_consolidation) if self._last_consolidation is not None else None

        key = self._consolidation_key(session_id)
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
            compressed_history=state.get("compressed_history", []),
            raw_recent_turns=state.get("raw_recent_turns", []),
            wm_token_usage=state.get("wm_token_usage", 0),
            tool_calls=state.get("tool_calls", []),
        )
        retrieved_graph_memories = result["retrieved_graph_memories"]
        feedback_update = result.get("feedback_update") or {}

        if self.evolve_loop is not None:
            diagnostics = result.get("retrieval_diagnostics") or {}
            if isinstance(diagnostics, dict):
                try:
                    prompt_versions_used = result.get("prompt_versions_used") or {}
                    final_ok = bool(diagnostics.get("final_is_sufficient", True))
                    rounds = int(diagnostics.get("reflection_rounds", 0) or 0)
                    if rounds > 0:
                        self.evolve_loop.collect_retrieval_adequacy(
                            is_sufficient=final_ok,
                            reflection_round=max(0, rounds - 1),
                            prompt_versions_used=(
                                dict(prompt_versions_used)
                                if isinstance(prompt_versions_used, dict)
                                else None
                            ),
                        )

                    max_rounds = int(diagnostics.get("max_reflection_rounds", 0) or 0)
                    if (not final_ok) and max_rounds and rounds >= max_rounds:
                        versions = prompt_versions_used
                        planning_ver = 0
                        if isinstance(versions, dict):
                            try:
                                planning_ver = int(versions.get("retrieval_planning", 0) or 0)
                            except Exception:
                                planning_ver = 0

                        missing = ""
                        hist = diagnostics.get("adequacy_history") or []
                        if isinstance(hist, list) and hist:
                            last = hist[-1] if isinstance(hist[-1], dict) else {}
                            missing = str(last.get("missing_info") or "").strip()

                        plan_summary = json.dumps(
                            {
                                "missing_info": missing[:200],
                                "retrieval_plan": result.get("retrieval_plan", {}),
                            },
                            ensure_ascii=False,
                        )
                        self.evolve_loop.signal_collector.collect_retrieval_miss(
                            query=str(state.get("user_query") or ""),
                            plan_summary=plan_summary,
                            planning_version=planning_ver,
                        )
                except Exception:
                    logger.warning("Evolve adequacy ingestion failed", exc_info=True)

        return {
            "retrieved_graph_memories": retrieved_graph_memories,
            "retrieved_skills": result["retrieved_skills"],
            "novelty_retrieved_context": self._build_novelty_retrieved_context(
                retrieved_graph_memories,
            ),
            "retrieval_plan": result.get("retrieval_plan", {}),
            "action_state": result.get("action_state", {}),
            "search_mode": result.get("search_mode", "answer"),
            "semantic_usage_log_id": result.get("semantic_usage_log_id", ""),
            "feedback_update": feedback_update,
            "prompt_versions_used": result.get("prompt_versions_used", {}),
            "retrieval_diagnostics": result.get("retrieval_diagnostics", {}),
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
            "synthesis_kept_count": result.get("kept_count", 0),
            "synthesis_dropped_count": result.get("dropped_count", 0),
            "synthesis_input_count": result.get("input_fragment_count", 0),
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
            state["session_id"], result["final_response"],
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

    @staticmethod
    def _build_novelty_retrieved_context(
        retrieved_graph_memories: list[dict[str, Any]] | None,
    ) -> str:
        """将 search 阶段召回的原始语义记忆片段格式化为 novelty check 上下文。

        这里刻意使用 pre-synthesis 的 provenance/raw fragments，而不是回答阶段的
        background_context，避免 LLM 融合改写后的文本与当前对话表述过度重叠，误判“无新信息”。
        """
        if not retrieved_graph_memories:
            return ""

        parts: list[str] = []
        seen_keys: set[str] = set()
        total_chars = 0
        max_items = 16
        max_chars = 6000

        def _append_part(part: str, *, dedupe_key: str) -> None:
            nonlocal total_chars
            text = str(part or "").strip()
            if not text or dedupe_key in seen_keys:
                return
            candidate_size = len(text) + (2 if parts else 0)
            if len(parts) >= max_items or total_chars + candidate_size > max_chars:
                return
            seen_keys.add(dedupe_key)
            parts.append(text)
            total_chars += candidate_size

        for wrapper in retrieved_graph_memories:
            provenance = wrapper.get("provenance")
            if isinstance(provenance, list) and provenance:
                for idx, item in enumerate(provenance, 1):
                    if not isinstance(item, dict):
                        continue
                    semantic_text = str(
                        item.get("semantic_text")
                        or item.get("summary")
                        or item.get("fact_text")
                        or ""
                    ).strip()
                    if not semantic_text:
                        continue
                    entities = [
                        str(entity or "").strip()
                        for entity in (item.get("entities") or [])
                        if str(entity or "").strip()
                    ]
                    header = f"[{idx}] ({item.get('memory_type') or 'unknown'}, source={item.get('source') or 'semantic'})"
                    if entities:
                        header += f" entities=[{', '.join(entities[:8])}]"
                    dedupe_key = str(
                        item.get("record_id")
                        or item.get("fact_id")
                        or item.get("skill_id")
                        or semantic_text
                    )
                    _append_part(f"{header}\n{semantic_text}", dedupe_key=dedupe_key)

            if parts:
                continue

            constructed = str(wrapper.get("constructed_context") or "").strip()
            if constructed:
                anchor = wrapper.get("anchor", {}) or {}
                dedupe_key = str(anchor.get("node_id") or constructed)
                _append_part(constructed, dedupe_key=dedupe_key)

        return "\n\n".join(parts)

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
        counter: dict[str, int] = {"input": 0, "output": 0}
        tok = _token_accumulator.set(counter)
        try:
            initial_state: dict[str, Any] = {
                "user_query": user_query,
                "session_id": session_id,
            }
            result = self._graph.invoke(initial_state)
        finally:
            in_tok, out_tok = counter["input"], counter["output"]
            _token_accumulator.reset(tok)

        result["turn_input_tokens"] = in_tok
        result["turn_output_tokens"] = out_tok

        # 后台线程触发固化（fire-and-forget，不阻塞响应返回）
        if result.get("consolidation_pending"):
            job_id = self._begin_consolidation(session_id)
            self._trigger_consolidation_bg(
                session_id,
                retrieved_context=str(result.get("novelty_retrieved_context") or ""),
                job_id=job_id,
                synthesis_kept=result.get("synthesis_kept_count", 0),
                synthesis_dropped=result.get("synthesis_dropped_count", 0),
                synthesis_input=result.get("synthesis_input_count", 0),
                prompt_versions_snapshot=result.get("prompt_versions_used"),
            )

        self._record_request_completion(result)
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
            job_id = self._begin_consolidation(session_id)
            asyncio.create_task(
                self._aconsolidate(
                    session_id,
                    retrieved_context=str(result.get("novelty_retrieved_context") or ""),
                    job_id=job_id,
                    synthesis_kept=result.get("synthesis_kept_count", 0),
                    synthesis_dropped=result.get("synthesis_dropped_count", 0),
                    synthesis_input=result.get("synthesis_input_count", 0),
                    prompt_versions_snapshot=result.get("prompt_versions_used"),
                )
            )

        self._record_request_completion(result)
        return result

    async def astream_steps(
        self, user_query: str, session_id: str
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
            state: dict[str, Any] = {"user_query": user_query, "session_id": session_id}

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
            ):
                streaming_response += token
                yield {"type": "token", "content": token}

            # 写回 assistant turn（保持与同步路径一致）
            await asyncio.to_thread(
                self.wm_manager.append_assistant_turn,
                state["session_id"],
                streaming_response,
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
                job_id = self._begin_consolidation(session_id)
                asyncio.create_task(
                    self._aconsolidate(
                        session_id,
                        retrieved_context=str(state.get("novelty_retrieved_context") or ""),
                        job_id=job_id,
                        synthesis_kept=state.get("synthesis_kept_count", 0),
                        synthesis_dropped=state.get("synthesis_dropped_count", 0),
                        synthesis_input=state.get("synthesis_input_count", 0),
                        prompt_versions_snapshot=state.get("prompt_versions_used"),
                    )
                )

            self._record_request_completion(state)
            yield {"type": "done", "result": dict(state)}
        finally:
            _reset_once()

    def consolidate(
        self, session_id: str, retrieved_context: str = ""
    ) -> dict[str, Any]:
        """手动触发固化（公共方法，可由 API BackgroundTasks 调用）。

        只处理自上次固化以来新增的 turns（水位线机制），成功后更新水位线，
        彻底消除跨轮重复固化问题。

        Args:
            session_id: 会话 ID。
            retrieved_context: search 阶段召回的原始已有语义记忆片段，
                用于新颖性判断；应优先传 pre-synthesis raw context，
                而不是回答期的 background_context。

        Returns:
            dict 包含：entities_added, skills_added
        """
        store = self.wm_manager.session_store
        log = store.get_or_create(session_id)
        watermark = log.last_consolidated_turn_index
        raw_total = len(log.turns)
        new_turns = [
            t for t in log.turns[watermark:] if not t.get("deleted", False)
        ]
        if not new_turns:
            return {"entities_added": 0, "skills_added": 0, "skipped_reason": "no_new_turns"}
        result = self.consolidator.run(
            turns=new_turns, session_id=session_id, retrieved_context=retrieved_context,
            turn_index_offset=watermark,
        )
        # 固化成功后推进水位线
        store.set_last_consolidated_turn_index(session_id, raw_total)
        return result

    def _trigger_consolidation_bg(
        self,
        session_id: str,
        retrieved_context: str = "",
        job_id: int = 0,
        synthesis_kept: int = 0,
        synthesis_dropped: int = 0,
        synthesis_input: int = 0,
        prompt_versions_snapshot: dict[str, int] | None = None,
    ) -> None:
        """在守护线程中触发固化（fire-and-forget）。"""
        thread = threading.Thread(
            target=self._safe_consolidate,
            args=(session_id, retrieved_context, job_id),
            kwargs={
                "synthesis_kept": synthesis_kept,
                "synthesis_dropped": synthesis_dropped,
                "synthesis_input": synthesis_input,
                "prompt_versions_snapshot": prompt_versions_snapshot,
            },
            daemon=True,
        )
        thread.start()

    def _safe_consolidate(
        self,
        session_id: str,
        retrieved_context: str = "",
        job_id: int = 0,
        *,
        synthesis_kept: int = 0,
        synthesis_dropped: int = 0,
        synthesis_input: int = 0,
        prompt_versions_snapshot: dict[str, int] | None = None,
    ) -> None:
        """安全执行固化，异常不影响主流程。"""
        graphiti = getattr(self.consolidator, "graphiti_engine", None)
        strict = bool(getattr(graphiti, "strict", False))
        try:
            result = self.consolidate(session_id, retrieved_context=retrieved_context)
            self._finish_consolidation(
                session_id=session_id,
                job_id=job_id,
                result=result,
            )
            self._collect_evolve_signals(
                result=result,
                synthesis_kept=synthesis_kept,
                synthesis_dropped=synthesis_dropped,
                synthesis_input=synthesis_input,
                prompt_versions_snapshot=prompt_versions_snapshot,
            )
        except Exception as exc:
            logger.exception("固化失败 session=%s", session_id)
            self._finish_consolidation(
                session_id=session_id,
                job_id=job_id,
                error=str(exc),
            )
            if strict:
                raise

    async def _aconsolidate(
        self,
        session_id: str,
        retrieved_context: str = "",
        job_id: int = 0,
        synthesis_kept: int = 0,
        synthesis_dropped: int = 0,
        synthesis_input: int = 0,
        prompt_versions_snapshot: dict[str, int] | None = None,
    ) -> None:
        """异步场景下的后台固化（使用水位线，只处理新增 turns）。"""
        graphiti = getattr(self.consolidator, "graphiti_engine", None)
        strict = bool(getattr(graphiti, "strict", False))
        store = self.wm_manager.session_store
        log = store.get_or_create(session_id)
        watermark = log.last_consolidated_turn_index
        raw_total = len(log.turns)
        new_turns = [t for t in log.turns[watermark:] if not t.get("deleted", False)]
        if not new_turns:
            self._finish_consolidation(
                session_id=session_id,
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
                    turn_index_offset=watermark,
                ),
            )
            # 固化成功后推进水位线
            store.set_last_consolidated_turn_index(session_id, raw_total)
            self._finish_consolidation(
                session_id=session_id,
                job_id=job_id,
                result=result,
            )
            self._collect_evolve_signals(
                result=result,
                synthesis_kept=synthesis_kept,
                synthesis_dropped=synthesis_dropped,
                synthesis_input=synthesis_input,
                prompt_versions_snapshot=prompt_versions_snapshot,
            )
        except Exception as exc:
            logger.exception("固化失败 session=%s", session_id)
            self._finish_consolidation(
                session_id=session_id,
                job_id=job_id,
                error=str(exc),
            )
            if strict:
                raise

    def _collect_evolve_signals(
        self,
        result: dict[str, Any],
        *,
        synthesis_kept: int = 0,
        synthesis_dropped: int = 0,
        synthesis_input: int = 0,
        prompt_versions_snapshot: dict[str, int] | None = None,
    ) -> None:
        """从固化结果中收集 evolve 信号。

        prompt_versions_snapshot 是在主线程请求期间拍的快照，
        确保异步固化完成时的指标归因到正确的 prompt 版本。
        """
        if self.evolve_loop is None:
            return
        try:
            self.evolve_loop.after_run(
                prompt_versions_used=prompt_versions_snapshot,
                synthesis_kept=synthesis_kept,
                synthesis_dropped=synthesis_dropped,
                synthesis_input=synthesis_input,
                consolidation_records_added=result.get("entities_added", 0),
                consolidation_records_merged=result.get("facts_added", 0),
                consolidation_records_expired=result.get("records_expired", 0),
                consolidation_has_novelty=bool(result.get("has_novelty")),
                consolidation_skills_added=result.get("skills_added", 0),
            )
        except Exception:
            logger.warning("Evolve signal collection failed", exc_info=True)

    def _record_request_completion(self, result: dict[str, Any]) -> None:
        """按 chat API 实际参与的 prompt 记录一次使用。"""
        versions = select_prompt_versions(
            _CHAT_API_PROMPTS,
            snapshot=result.get("prompt_versions_used") if isinstance(result, dict) else None,
        )
        self.record_api_usage(
            api_name="chat",
            prompt_versions_used=versions,
        )

    def record_api_usage(
        self,
        *,
        api_name: str,
        prompt_versions_used: dict[str, int] | None = None,
    ) -> None:
        """对外暴露：记录一次 API 调用涉及的 prompt 使用。"""
        if self.evolve_loop is None:
            return
        try:
            self.evolve_loop.record_api_call(
                api_name=api_name,
                prompt_versions_used=prompt_versions_used,
            )
        except Exception:
            logger.warning("Evolve API usage recording failed api=%s", api_name, exc_info=True)

    @staticmethod
    def _snapshot_prompt_versions() -> dict[str, int]:
        """保留统一的 prompt 版本快照入口。"""
        return get_active_versions_snapshot()

    @property
    def graph(self):
        """暴露底层 compiled graph 供调试/可视化。"""
        return self._graph
