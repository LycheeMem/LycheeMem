"""
LangGraph Pipeline 构建。

定义节点和边，编译为可执行的状态图。

流水线拓扑：
  __start__ → wm_manager → visual_memory → search → reason → __end__

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
        reasoner: ReasoningAgent,
        consolidator: ConsolidatorAgent,
    ):
        self.wm_manager = wm_manager
        self.search_coordinator = search_coordinator
        self.reasoner = reasoner
        self.consolidator = consolidator

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

    def _visual_memory_node(self, state: PipelineState) -> dict[str, Any]:
        """视觉记忆处理节点：提取、存储输入图片，检索相关视觉记忆。"""
        input_images = state.get("input_images") or []
        session_id = state.get("session_id", "")
        user_query = state.get("user_query", "")

        retrieved_visual = []
        visual_context = ""

        if input_images and hasattr(self, "visual_extractor"):
            import asyncio
            import hashlib
            from datetime import datetime, timezone
            from src.memory.visual.models import VisualMemoryRecord

            stored_records = []

            async def _process_images():
                stored = []
                for img_b64 in input_images:
                    try:
                        extraction = await self.visual_extractor.extract_from_base64(
                            image_b64=img_b64,
                            mime_type="image/jpeg",
                            session_id=session_id,
                        )
                        image_path = self.visual_store.save_image_file(
                            image_b64=img_b64,
                            image_hash=extraction["image_hash"],
                            mime_type=extraction.get("image_mime_type", "image/jpeg"),
                            session_id=session_id,
                        )
                        record_id = hashlib.sha256(
                            f"{extraction['image_hash']}:{session_id}:{datetime.now(timezone.utc).isoformat()}".encode()
                        ).hexdigest()[:32]

                        record = VisualMemoryRecord(
                            record_id=record_id,
                            session_id=session_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            image_path=image_path,
                            image_hash=extraction["image_hash"],
                            image_size=extraction.get("image_size", 0),
                            image_mime_type=extraction.get("image_mime_type", "image/jpeg"),
                            caption=extraction.get("caption", ""),
                            entities=extraction.get("entities", []),
                            scene_type=extraction.get("scene_type", "other"),
                            structured_data=extraction.get("structured_data", {}),
                            importance_score=extraction.get("importance_score", 0.5),
                            source_role="user",
                        )
                        
                        if self.visual_store.embedder:
                            embedding = self.visual_store.embedder.embed_query(record.caption)
                            if embedding:
                                record.caption_embedding = embedding
                        
                        if hasattr(self.visual_store, 'multimodal_embedder') and self.visual_store.multimodal_embedder:
                            try:
                                import base64
                                from pathlib import Path
                                import tempfile
                                
                                image_bytes = base64.b64decode(img_b64)
                                
                                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                                    tmp.write(image_bytes)
                                    tmp_path = tmp.name
                                
                                try:
                                    visual_embedding = self.visual_store.multimodal_embedder.embed_image(tmp_path)
                                    if visual_embedding:
                                        record.visual_embedding = visual_embedding
                                        logger.info("Visual embedding generated: dim=%d", len(visual_embedding))
                                finally:
                                    Path(tmp_path).unlink(missing_ok=True)
                                    
                            except Exception as e:
                                logger.warning("Failed to generate visual embedding: %s", e)

                        stored_id = self.visual_store.store(record)
                        if stored_id:
                            stored.append(stored_id)
                            logger.info("Visual memory stored: id=%s, caption=%s", stored_id, record.caption[:60])
                    except Exception as e:
                        logger.error("Failed to process image: %s", e)
                return stored

            try:
                loop = asyncio.new_event_loop()
                try:
                    stored_records = loop.run_until_complete(_process_images())
                finally:
                    loop.close()
            except Exception as e:
                logger.error("Visual processing failed: %s", e)

        if hasattr(self, "visual_retriever"):
            try:
                if input_images:
                    if stored_records:
                        for rec_id in stored_records:
                            rec = self.visual_store.get_by_id(rec_id)
                            if rec and not rec.expired:
                                retrieved_visual.append({
                                    "record_id": rec.record_id,
                                    "caption": rec.caption,
                                    "image_path": rec.image_path,
                                    "image_hash": rec.image_hash,
                                    "scene_type": rec.scene_type,
                                    "timestamp": rec.timestamp,
                                    "importance_score": rec.importance_score,
                                    "session_id": rec.session_id,
                                    "entities": rec.entities,
                                    "structured_data": rec.structured_data,
                                    "source_role": rec.source_role,
                                })
                        logger.info("Using %d newly stored visual memories", len(retrieved_visual))
                    else:
                        retrieved_visual = self.visual_retriever.retrieve_by_session(
                            session_id=session_id, top_k=10
                        )
                elif user_query:
                    retrieved_visual = self.visual_retriever.retrieve_for_context(
                        query=user_query, top_k=3
                    )

                if retrieved_visual:
                    parts = []
                    for i, vm in enumerate(retrieved_visual, 1):
                        parts.append(
                            f"[视觉记忆{i}] {vm.get('caption', '')} "
                            f"(类型: {vm.get('scene_type', '')}, "
                            f"重要性: {vm.get('importance_score', 0):.2f})"
                        )
                    visual_context = "\n".join(parts)
                    logger.info("Retrieved %d visual memories", len(retrieved_visual))
            except Exception as e:
                logger.warning("Visual retrieval failed: %s", e)

        return {
            "retrieved_visual_memories": retrieved_visual,
            "visual_context": visual_context,
        }

    @staticmethod
    def _memory_context_from_hit(mem: dict[str, Any]) -> str:
        context = str(mem.get("constructed_context") or "").strip()
        if context:
            return context

        provenance = mem.get("provenance")
        if isinstance(provenance, list):
            lines: list[str] = []
            for item in provenance[:20]:
                if not isinstance(item, dict):
                    continue
                text = str(
                    item.get("semantic_text")
                    or item.get("fact_text")
                    or item.get("summary")
                    or ""
                ).strip()
                if text:
                    lines.append(text)
            if lines:
                return "\n".join(f"- {line}" for line in lines)

        anchor = mem.get("anchor") if isinstance(mem.get("anchor"), dict) else mem
        props = anchor.get("properties", {}) if isinstance(anchor, dict) else {}
        text = str(
            props.get("name")
            or anchor.get("name", "")
            or anchor.get("label", "")
            or ""
        ).strip()
        return text

    def _build_background_context(self, state: PipelineState, graph_memories: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        visual_context = str(state.get("visual_context") or "").strip()
        if visual_context:
            parts.append(visual_context)

        seen: set[str] = set()
        for mem in graph_memories:
            text = self._memory_context_from_hit(mem)
            if not text or text in seen:
                continue
            seen.add(text)
            parts.append(text)

        return "\n\n".join(parts)

    def _search_node(self, state: PipelineState) -> dict[str, Any]:
        """检索协调节点：从图谱和技能库检索相关记忆。"""
        result = self.search_coordinator.run(
            user_query=state["user_query"],
            session_id=state.get("session_id"),
            compressed_history=state.get("compressed_history", []),
            raw_recent_turns=state.get("raw_recent_turns", []),
            wm_token_usage=state.get("wm_token_usage", 0),
            tool_calls=state.get("tool_calls", []),
            reference_time=state.get("reference_time"),
        )
        retrieved_graph_memories = result["retrieved_graph_memories"]
        return {
            "retrieved_graph_memories": retrieved_graph_memories,
            "retrieved_skills": result["retrieved_skills"],
            "background_context": self._build_background_context(state, retrieved_graph_memories),
            "retrieval_plan": result.get("retrieval_plan", {}),
            "action_state": result.get("action_state", {}),
            "search_mode": result.get("search_mode", "answer"),
            "semantic_usage_log_id": result.get("semantic_usage_log_id", ""),
            "feedback_update": result.get("feedback_update", {}),
        }

    def _reason_node(self, state: PipelineState) -> dict[str, Any]:
        """推理节点：生成最终回答（含技能复用计划）。"""
        result = self.reasoner.run(
            user_query=state["user_query"],
            compressed_history=state.get("compressed_history", []),
            background_context=state.get("background_context", ""),
            skill_reuse_plan=state.get("skill_reuse_plan", []),
            retrieved_skills=state.get("retrieved_skills", []),
            reference_time=state.get("reference_time"),
        )

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

        return {
            "final_response": result["final_response"],
            "consolidation_pending": bool(state.get("auto_consolidate", True)),
        }

    def _build_graph(self):
        """构建并编译 LangGraph StateGraph。"""
        g = StateGraph(PipelineState)

        g.add_node("wm_manager", self._wm_manager_node)
        g.add_node("visual_memory", self._visual_memory_node)
        g.add_node("search", self._search_node)
        g.add_node("reason", self._reason_node)

        g.add_edge(START, "wm_manager")
        g.add_edge("wm_manager", "visual_memory")
        g.add_edge("visual_memory", "search")
        g.add_edge("search", "reason")
        g.add_edge("reason", END)

        return g.compile()

    def run(
        self,
        user_query: str,
        session_id: str,
        input_images: list[str] | None = None,
        reference_time: str | None = None,
        auto_consolidate: bool = True,
    ) -> dict[str, Any]:
        """同步运行 Pipeline。

        Args:
            user_query: 用户输入。
            session_id: 会话 ID。
            input_images: Base64 编码的图片列表（可选）。

        Returns:
            完整的 PipelineState（包含 final_response 等所有字段）。
        """
        counter: dict[str, int] = {"input": 0, "output": 0}
        tok = _token_accumulator.set(counter)
        try:
            initial_state: dict[str, Any] = {
                "user_query": user_query,
                "session_id": session_id,
                "input_images": input_images or [],
                "reference_time": reference_time or "",
                "auto_consolidate": auto_consolidate,
            }
            result = self._graph.invoke(initial_state)
        finally:
            in_tok, out_tok = counter["input"], counter["output"]
            _token_accumulator.reset(tok)

        result["turn_input_tokens"] = in_tok
        result["turn_output_tokens"] = out_tok

        if result.get("consolidation_pending"):
            job_id = self._begin_consolidation(session_id)
            self._trigger_consolidation_bg(
                session_id,
                job_id=job_id,
                session_date=reference_time,
            )

        return result

    async def arun(
        self,
        user_query: str,
        session_id: str,
        input_images: list[str] | None = None,
        reference_time: str | None = None,
        auto_consolidate: bool = True,
    ) -> dict[str, Any]:
        """异步运行 Pipeline。"""
        initial_state: dict[str, Any] = {
            "user_query": user_query,
            "session_id": session_id,
            "input_images": input_images or [],
            "reference_time": reference_time or "",
            "auto_consolidate": auto_consolidate,
        }
        result = await self._graph.ainvoke(initial_state)

        if result.get("consolidation_pending"):
            job_id = self._begin_consolidation(session_id)
            asyncio.create_task(
                self._aconsolidate(
                    session_id,
                    job_id=job_id,
                    session_date=reference_time,
                )
            )

        return result

    async def astream_steps(
        self,
        user_query: str,
        session_id: str,
        input_images: list[str] | None = None,
        reference_time: str | None = None,
        auto_consolidate: bool = True,
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
            state: dict[str, Any] = {
                "user_query": user_query,
                "session_id": session_id,
                "input_images": input_images or [],
                "reference_time": reference_time or "",
                "auto_consolidate": auto_consolidate,
            }

            patch = await asyncio.to_thread(self._wm_manager_node, state)
            state.update(patch)
            yield {
                "type": "step",
                "step": "wm_manager",
                "status": "done",
                "wm_token_usage": patch.get("wm_token_usage", 0),
                "patch": patch,
            }

            patch = await asyncio.to_thread(self._visual_memory_node, state)
            state.update(patch)
            yield {"type": "step", "step": "visual_memory", "status": "done", "patch": patch}

            patch = await asyncio.to_thread(self._search_node, state)
            state.update(patch)
            yield {"type": "step", "step": "search", "status": "done", "patch": patch}

            # reason 阶段：流式生成，逐 token yield，最后再发 step:reason 完成事件
            streaming_response = ""
            async for token in self.reasoner.astream(
                user_query=state["user_query"],
                compressed_history=state.get("compressed_history", []),
                background_context=state.get("background_context", ""),
                skill_reuse_plan=state.get("skill_reuse_plan", []),
                retrieved_skills=state.get("retrieved_skills", []),
                reference_time=state.get("reference_time"),
            ):
                streaming_response += token
                yield {"type": "token", "content": token}

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
            patch = {
                "final_response": streaming_response,
                "consolidation_pending": bool(state.get("auto_consolidate", True)),
            }
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
                        job_id=job_id,
                        session_date=reference_time,
                    )
                )

            yield {"type": "done", "result": dict(state)}
        finally:
            _reset_once()

    def consolidate(
        self,
        session_id: str,
        session_date: str | None = None,
        flush_session: bool = True,
    ) -> dict[str, Any]:
        """手动触发固化（公共方法，可由 API BackgroundTasks 调用）。

        只处理自上次固化以来新增的 turns（水位线机制），成功后更新水位线，
        彻底消除跨轮重复固化问题。

        Args:
            session_id: 会话 ID。

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
        if not new_turns and not flush_session:
            return {"entities_added": 0, "skills_added": 0, "skipped_reason": "no_new_turns"}
        result = self.consolidator.run(
            turns=new_turns, session_id=session_id,
            turn_index_offset=watermark,
            flush_session=flush_session,
            session_date=session_date,
        )
        raw_consumed = result.get("turns_consumed")
        consumed = int(len(new_turns) if raw_consumed is None else raw_consumed)
        store.set_last_consolidated_turn_index(
            session_id,
            min(raw_total, watermark + max(0, consumed)),
        )
        return result

    def _trigger_consolidation_bg(
        self,
        session_id: str,
        job_id: int = 0,
        session_date: str | None = None,
        flush_session: bool = True,
    ) -> None:
        """在守护线程中触发固化（fire-and-forget）。"""
        thread = threading.Thread(
            target=self._safe_consolidate,
            args=(session_id, job_id, session_date, flush_session),
            daemon=True,
        )
        thread.start()

    def _safe_consolidate(
        self,
        session_id: str,
        job_id: int = 0,
        session_date: str | None = None,
        flush_session: bool = True,
    ) -> None:
        """安全执行固化，异常不影响主流程。"""
        graphiti = getattr(self.consolidator, "graphiti_engine", None)
        strict = bool(getattr(graphiti, "strict", False))
        try:
            result = self.consolidate(
                session_id,
                session_date=session_date,
                flush_session=flush_session,
            )
            self._finish_consolidation(
                session_id=session_id,
                job_id=job_id,
                result=result,
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
        job_id: int = 0,
        session_date: str | None = None,
        flush_session: bool = True,
    ) -> None:
        """异步场景下的后台固化（使用水位线，只处理新增 turns）。"""
        graphiti = getattr(self.consolidator, "graphiti_engine", None)
        strict = bool(getattr(graphiti, "strict", False))
        store = self.wm_manager.session_store
        log = store.get_or_create(session_id)
        watermark = log.last_consolidated_turn_index
        raw_total = len(log.turns)
        new_turns = [t for t in log.turns[watermark:] if not t.get("deleted", False)]
        if not new_turns and not flush_session:
            self._finish_consolidation(
                session_id=session_id,
                job_id=job_id,
                result={
                    "entities_added": 0,
                    "skills_added": 0,
                    "facts_added": 0,
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
                    turn_index_offset=watermark,
                    flush_session=flush_session,
                    session_date=session_date,
                ),
            )
            raw_consumed = result.get("turns_consumed")
            consumed = int(len(new_turns) if raw_consumed is None else raw_consumed)
            store.set_last_consolidated_turn_index(
                session_id,
                min(raw_total, watermark + max(0, consumed)),
            )
            self._finish_consolidation(
                session_id=session_id,
                job_id=job_id,
                result=result,
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

    @property
    def graph(self):
        """暴露底层 compiled graph 供调试/可视化。"""
        return self._graph
