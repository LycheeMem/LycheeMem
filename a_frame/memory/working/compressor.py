"""
预算驱动的上下文压缩器。

实现双阈值机制 (warn / block) + 结构化状态交接文档。
这是整个系统最核心的组件之一。
"""

from __future__ import annotations

from typing import Any

import tiktoken

from a_frame.llm.base import BaseLLM


class WorkingMemoryCompressor:
    """工作记忆压缩器。

    借鉴 VSCode Copilot Agent 的历史折叠机制：
    - 不丢弃历史，而是折叠成结构化摘要
    - 摘要锚定在历史边界上
    - 最新几轮始终保留原始细节
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_tokens: int = 128_000,
        warn_threshold: float = 0.7,
        block_threshold: float = 0.9,
        min_recent_turns: int = 4,
        compression_prompt_template: str | None = None,
    ):
        self.llm = llm
        self.max_tokens = max_tokens
        self.warn_threshold = warn_threshold
        self.block_threshold = block_threshold
        self.min_recent_turns = min_recent_turns
        self._encoder = tiktoken.get_encoding("cl100k_base")

        if compression_prompt_template is None:
            self.compression_prompt = self._default_compression_prompt()
        else:
            self.compression_prompt = compression_prompt_template

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """估算消息列表的 token 数。"""
        total = 0
        for msg in messages:
            total += len(self._encoder.encode(msg.get("content", "")))
            total += 4  # role + formatting overhead
        return total

    def should_compress(self, current_tokens: int) -> str:
        """判断是否需要压缩。

        Returns:
            "sync"  - 超过阻塞阈值，必须同步压缩
            "async" - 超过预警阈值，后台预压缩
            "none"  - 无需压缩
        """
        if current_tokens > self.max_tokens * self.block_threshold:
            return "sync"
        if current_tokens > self.max_tokens * self.warn_threshold:
            return "async"
        return "none"

    def find_compression_boundary(self, turns: list[dict[str, str]]) -> int:
        """找到压缩边界：保留最近 min_recent_turns 轮不压缩。

        返回 boundary_index，该索引之前的内容将被压缩。
        """
        # 一轮 = user + assistant，所以保留的消息数 = min_recent_turns * 2
        keep_count = self.min_recent_turns * 2
        boundary = max(0, len(turns) - keep_count)
        return boundary

    def compress(self, turns: list[dict[str, str]], boundary_index: int) -> str:
        """将 boundary_index 之前的历史压缩为结构化状态交接文档。

        输出包含三个固定字段：
        - Intent Mapping：用户原始目标约束
        - Progress Assessment：已完成 vs 待完成
        - Recent Commands Analysis：边界前最后几次工具调用
        """
        history_to_compress = turns[:boundary_index]
        if not history_to_compress:
            return ""

        history_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in history_to_compress)
        prompt = self.compression_prompt.format(history=history_text)

        summary = self.llm.generate([{"role": "user", "content": prompt}])
        return summary

    def render_context(
        self,
        turns: list[dict[str, str]],
        summaries: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """渲染最终上下文：摘要 + 原始近期轮次。

        从新到旧遍历，碰到摘要锚点就停止展开更早历史。
        """
        if not summaries:
            return turns

        # 取最新的摘要（可能经过多次压缩）
        latest_summary = max(summaries, key=lambda s: s["boundary_index"])
        boundary = latest_summary["boundary_index"]

        context = [
            {"role": "system", "content": f"[历史摘要]\n{latest_summary['content']}"},
        ]
        # 保留 boundary 之后的原始对话
        context.extend(turns[boundary:])
        return context

    @staticmethod
    def _default_compression_prompt() -> str:
        return """Your task is to create a comprehensive, detailed summary of the provided conversation history. This summary will be used to compact the conversation while preserving critical technical details, decisions, tool executions, and progress for seamless continuation.

    ## Recent Context Analysis
    Pay special attention to the most recent agent commands and tool executions. Include:
    - Last Agent Commands: What specific actions/tools were just executed
    - Tool Results: Key outcomes from recent tool calls (truncate if very long, but preserve essential information)
    - Immediate State: What was the system doing right before summarization

    ## Analysis Process
    Before providing your final summary, wrap your analysis in `<analysis>` tags to organize your thoughts systematically:
    1. Chronological Review
    2. Intent Mapping
    3. Technical Inventory
    4. Progress Assessment
    5. Recent Commands Analysis

    ### Input History to Compress
    <history_to_compress>
    {history}
    </history_to_compress>

    ### Output Format (MUST follow this EXACT structure)
    <analysis>
    [Your step-by-step thinking process here]
    </analysis>

    <summary>
    1. Conversation Overview: [Primary Objectives and Session Context]
    2. Technical Foundation / Environment: [Core tech, paths, setups]
    3. Problem Resolution: [Issues encountered and solutions implemented]
    4. Progress Tracking: [Completed vs. Pending]
    5. Recent Operations: [Last tool calls and results summary]
    6. Continuation Plan: [Immediate next steps]
    </summary>"""
