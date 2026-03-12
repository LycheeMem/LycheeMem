"""
工作记忆管理器 (Working Memory Manager)。

混合逻辑 Agent：
- Token 预算监控（双阈值）
- 压缩器调用（async / sync）
- 上下文渲染（摘要锚定 + 原始近期轮次）
"""

from __future__ import annotations

from typing import Any

from a_frame.memory.working.compressor import WorkingMemoryCompressor
from a_frame.memory.working.session_store import InMemorySessionStore


class WMManager:
    """工作记忆管理器。

    不继承 BaseAgent —— 这是一个混合逻辑节点，
    使用 token 计数 + 规则决策 + 压缩器协作，不直接发 prompt。
    """

    def __init__(
        self,
        session_store: InMemorySessionStore,
        compressor: WorkingMemoryCompressor,
    ):
        self.session_store = session_store
        self.compressor = compressor

    def run(
        self,
        session_id: str,
        user_query: str,
        **kwargs,
    ) -> dict[str, Any]:
        """管理工作记忆并返回渲染后的上下文。

        流程：
        1. 将新的用户消息追加到会话日志
        2. 计算当前 token 用量
        3. 判断是否需要压缩
        4. 如需要，执行压缩并存储摘要
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

        # 3. 判断是否需要压缩
        action = self.compressor.should_compress(current_tokens)

        if action in ("sync", "async"):
            # 4. 找到压缩边界并执行压缩
            boundary = self.compressor.find_compression_boundary(turns)
            if boundary > 0:
                summary_text = self.compressor.compress(turns, boundary)
                self.session_store.add_summary(session_id, boundary, summary_text)

        # 5. 渲染最终上下文
        rendered = self.compressor.render_context(turns, log.summaries)

        # 分离出原始近期轮次（boundary 之后的）
        if log.summaries:
            latest_summary = max(log.summaries, key=lambda s: s["boundary_index"])
            boundary = latest_summary["boundary_index"]
            raw_recent = turns[boundary:]
        else:
            raw_recent = turns

        return {
            "compressed_history": rendered,
            "raw_recent_turns": raw_recent,
            "wm_token_usage": self.compressor.count_tokens(rendered),
        }

    def append_assistant_turn(self, session_id: str, content: str) -> None:
        """在推理器回答后，将 assistant 回复追加到会话日志。"""
        self.session_store.append_turn(session_id, "assistant", content)
