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

    def _visual_memory_node(self, state: PipelineState) -> dict[str, Any]:
        """视觉记忆处理节点：提取、存储输入图片，检索相关视觉记忆。"""
        input_images = state.get("input_images") or []
        session_id = state.get("session_id", "")
        user_query = state.get("user_query", "")

        retrieved_visual = []
        visual_context = ""

        # 如果有图片输入，处理存储
        if input_images and hasattr(self, "visual_extractor"):
            import asyncio
            import hashlib
            from datetime import datetime, timezone
            from src.memory.visual.models import VisualMemoryRecord

            stored_records = []  # 记录本次存储的 record_id

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
                        
                        # 生成双嵌入：caption embedding + visual embedding
                        # 1. Caption embedding（文本嵌入，向后兼容）
                        if self.visual_store.embedder:
                            embedding = self.visual_store.embedder.embed_query(record.caption)
                            if embedding:
                                record.caption_embedding = embedding
                        
                        # 2. Visual embedding（CLIP 视觉嵌入，新增，支持跨模态检索）
                        if hasattr(self.visual_store, 'multimodal_embedder') and self.visual_store.multimodal_embedder:
                            try:
                                # 临时保存图像以生成嵌入
                                import base64
                                from pathlib import Path
                                import tempfile
                                
                                # 解码图像
                                image_bytes = base64.b64decode(img_b64)
                                
                                # 使用临时文件生成视觉嵌入
                                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                                    tmp.write(image_bytes)
                                    tmp_path = tmp.name
                                
                                try:
                                    # 生成 CLIP 视觉嵌入（与文本同一空间）
                                    visual_embedding = self.visual_store.multimodal_embedder.embed_image(tmp_path)
                                    if visual_embedding:
                                        record.visual_embedding = visual_embedding
                                        logger.info("Visual embedding generated: dim=%d", len(visual_embedding))
                                finally:
                                    # 清理临时文件
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

        # 检索视觉记忆
        if hasattr(self, "visual_retriever"):
            try:
                if input_images:
                    # 有图片输入时：优先使用本次存储的记录，否则获取会话的所有视觉记忆
                    if stored_records:
                        # 直接获取本次存储的记录
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
                        # 没有新存储的记录，获取会话的所有视觉记忆
                        retrieved_visual = self.visual_retriever.retrieve_by_session(
                            session_id=session_id, top_k=10
                        )
                elif user_query:
                    # 纯文本查询时，检索相关视觉记忆
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
            "feedback_update": result.get("feedback_update", {}),
        }

    def _synthesize_node(self, state: PipelineState) -> dict[str, Any]:
        """整合排序节点：融合多源检索结果为 background_context + skill_reuse_plan。"""
        visual_context = state.get("visual_context", "")
        input_images = state.get("input_images") or []

        # 如果用户上传了新图片，视觉记忆是唯一权威来源
        # 直接忽略图谱记忆中的旧视觉描述
        if visual_context and input_images:
            # 新图片上传时，视觉记忆直接作为 background_context
            # 不使用 Synthesizer 过滤，避免旧记忆干扰
            logger.info("New image uploaded - using visual memory as authoritative source")
            return {
                "background_context": visual_context,
                "skill_reuse_plan": [],
                "provenance": [{"source": "visual_memory", "index": 0, "relevance": 1.0, "summary": "Newly uploaded image"}],
            }

        # 没有新图片但有视觉记忆（后续追问）
        if visual_context:
            graph_memories = state.get("retrieved_graph_memories", [])[:]
            # 将视觉记忆放在最前面
            graph_memories.insert(0, {
                "constructed_context": visual_context,
                "anchor": {"node_id": "visual_memory", "type": "visual"},
            })
            logger.info("Visual context injected as first memory source")

            result = self.synthesizer.run(
                user_query=state["user_query"],
                retrieved_graph_memories=graph_memories,
                retrieved_skills=state.get("retrieved_skills", []),
            )

            bg_context = result.get("background_context", "")
            # 确保视觉记忆不被 Synthesizer 过滤掉
            visual_key_info = visual_context[:80] if visual_context else ""
            if visual_key_info and visual_key_info not in bg_context[:300]:
                bg_context = visual_context + "\n\n" + bg_context
                logger.info("Visual context was filtered out, prepending manually")

            return {
                "background_context": bg_context,
                "skill_reuse_plan": result.get("skill_reuse_plan", []),
                "provenance": result.get("provenance", []),
            }
        else:
            # 没有视觉记忆，正常合成
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
        g.add_node("visual_memory", self._visual_memory_node)
        g.add_node("search", self._search_node)
        g.add_node("synthesize", self._synthesize_node)
        g.add_node("reason", self._reason_node)

        # 线性连接: START → wm_manager → visual_memory → search → synthesize → reason → END
        g.add_edge(START, "wm_manager")
        g.add_edge("wm_manager", "visual_memory")
        g.add_edge("visual_memory", "search")
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

    def run(self, user_query: str, session_id: str, input_images: list[str] | None = None) -> dict[str, Any]:
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
            )

        return result

    async def arun(self, user_query: str, session_id: str, input_images: list[str] | None = None) -> dict[str, Any]:
        """异步运行 Pipeline。"""
        initial_state: dict[str, Any] = {
            "user_query": user_query,
            "session_id": session_id,
            "input_images": input_images or [],
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
                )
            )

        return result

    async def astream_steps(
        self, user_query: str, session_id: str, input_images: list[str] | None = None
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
                    )
                )

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
    ) -> None:
        """在守护线程中触发固化（fire-and-forget）。"""
        thread = threading.Thread(
            target=self._safe_consolidate,
            args=(session_id, retrieved_context, job_id),
            daemon=True,
        )
        thread.start()

    def _safe_consolidate(
        self,
        session_id: str,
        retrieved_context: str = "",
        job_id: int = 0,
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
