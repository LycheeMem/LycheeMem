"""
工作记忆管理器 (Working Memory Manager)。

混合逻辑 Agent：
- Token 预算监控（双阈值）
- 预警阈值 (warn) → 后台线程异步预压缩，不阻塞主流程
- 阻塞阈值 (block) → 优先复用已就绪的后台摘要，否则同步压缩
- 上下文渲染（摘要锚定 + 原始近期轮次）
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Any

from src.memory.working.compressor import WorkingMemoryCompressor
from src.memory.working.session_store import InMemorySessionStore

logger = logging.getLogger("src.wm_manager")


class WMManager:
    """工作记忆管理器。

    不继承 BaseAgent —— 这是一个混合逻辑节点，
    使用 token 计数 + 规则决策 + 压缩器协作，不直接发 prompt。

    双阈值压缩机制：
    - warn_threshold (默认 70%): 后台异步预压缩，不阻塞当前请求
    - block_threshold (默认 90%): 优先复用后台已就绪摘要，否则同步压缩
    """

    def __init__(
        self,
        session_store: InMemorySessionStore,
        compressor: WorkingMemoryCompressor,
    ):
        self.session_store = session_store
        self.compressor = compressor
        # 后台预压缩: session_id → Future[(summary_text, boundary)]
        self._pending: dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="wm_compress",
        )

    def run(
        self,
        session_id: str,
        user_query: str,
        **kwargs,
    ) -> dict[str, Any]:
        """管理工作记忆并返回渲染后的上下文。

        双阈值压缩流程：
        1. 追加用户消息 → 计算 token 用量
        2. none  → 不压缩
        3. async → 后台线程预压缩（不阻塞本次返回）
        4. sync  → 优先复用后台已就绪摘要，否则同步压缩
        5. 渲染最终上下文（摘要 + 原始近期轮次）

        Returns:
            dict 包含：compressed_history, raw_recent_turns, wm_token_usage
        """
        # 1. 追加用户消息
        self.session_store.append_turn(session_id, "user", user_query)
        log = self.session_store.get_or_create(session_id)
        turns = log.turns

        # 2. 计算 token 用量
        current_tokens = self.compressor.count_tokens(turns)

        # 3. 分流处理
        action = self.compressor.should_compress(current_tokens)

        if action == "sync":
            self._apply_or_sync_compress(session_id, turns)
        elif action == "async":
            self._start_background_compression(session_id, turns)

        # 刷新 log（可能已追加 summary）
        log = self.session_store.get_or_create(session_id)

        # 4. 渲染最终上下文
        rendered = self.compressor.render_context(log.turns, log.summaries)

        # 分离出原始近期轮次（boundary 之后的）
        if log.summaries:
            latest_summary = max(log.summaries, key=lambda s: s["boundary_index"])
            boundary = latest_summary["boundary_index"]
            raw_recent = log.turns[boundary:]
        else:
            raw_recent = log.turns

        return {
            "compressed_history": rendered,
            "raw_recent_turns": raw_recent,
            "wm_token_usage": self.compressor.count_tokens(rendered),
        }

    # ── 双阈值压缩核心 ──

    def _apply_or_sync_compress(self, session_id: str, turns: list[dict]) -> None:
        """阻塞阈值：优先复用后台已就绪摘要，否则同步压缩。"""
        with self._lock:
            future = self._pending.pop(session_id, None)

        if future is not None:
            if future.done():
                try:
                    summary_text, boundary = future.result()
                    if summary_text and boundary > 0:
                        self.session_store.add_summary(session_id, boundary, summary_text)
                        logger.info(
                            "session=%s 复用后台预压缩摘要 (boundary=%d)", session_id, boundary
                        )
                        return
                except Exception:
                    logger.warning(
                        "session=%s 后台预压缩结果异常，回退同步压缩", session_id, exc_info=True
                    )
            else:
                future.cancel()

        # 同步压缩
        boundary = self.compressor.find_compression_boundary(turns)
        if boundary > 0:
            summary_text = self.compressor.compress(turns, boundary)
            self.session_store.add_summary(session_id, boundary, summary_text)
            logger.info("session=%s 同步压缩完成 (boundary=%d)", session_id, boundary)

    def _start_background_compression(self, session_id: str, turns: list[dict]) -> None:
        """预警阈值：提交后台预压缩任务（不阻塞主流程）。"""
        with self._lock:
            if session_id in self._pending:
                return  # 已有任务在跑

        boundary = self.compressor.find_compression_boundary(turns)
        if boundary <= 0:
            return

        # 快照对话列表，避免线程安全问题
        turns_snapshot = list(turns)
        future = self._executor.submit(
            self._do_background_compress,
            turns_snapshot,
            boundary,
        )
        with self._lock:
            self._pending[session_id] = future
        logger.info("session=%s 启动后台预压缩 (boundary=%d)", session_id, boundary)

    def _do_background_compress(
        self,
        turns: list[dict],
        boundary: int,
    ) -> tuple[str, int]:
        """在后台线程中执行压缩。"""
        summary_text = self.compressor.compress(turns, boundary)
        return (summary_text, boundary)

    # ── 公共方法 ──

    def append_assistant_turn(self, session_id: str, content: str) -> None:
        """在推理器回答后，将 assistant 回复追加到会话日志。"""
        self.session_store.append_turn(session_id, "assistant", content)
