"""
核心推理器 (Main Reasoning Agent)。

接收工作记忆管理器和整合器准备好的高浓度上下文，
执行最终的思考、工具调用和回答生成。
这是唯一面向用户生成最终回复的 Agent。
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.llm.base import BaseLLM

REASONING_SYSTEM_PROMPT = """\
你是一个智能助手，拥有丰富的背景知识和记忆能力。

{history_section}

{background_section}

{skill_plan_section}

请根据以上背景知识和对话历史，为用户提供准确、有帮助的回答。

规则：
- 优先使用记忆中的事实信息回答
- 如果有可复用的技能文档（Markdown），优先参考其中的步骤、命令与注意事项
- 如果记忆信息不足以回答，可以基于通用知识回答，但需说明
- 保持回答简洁聚焦
- 不要虚构不存在的事实"""


class ReasoningAgent(BaseAgent):
    """核心推理器：生成最终用户回复。"""

    def __init__(self, llm: BaseLLM):
        super().__init__(llm=llm, prompt_template=REASONING_SYSTEM_PROMPT)

    def run(
        self,
        user_query: str,
        compressed_history: list[dict[str, str]] | None = None,
        background_context: str = "",
        skill_reuse_plan: list[dict] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """执行推理，生成最终回答。

        Args:
            user_query: 用户当前查询。
            compressed_history: 压缩后的对话历史（含摘要）。
            background_context: 整合器输出的背景知识。
            skill_reuse_plan: 可复用技能执行计划。

        Returns:
            dict 包含：final_response (str)
        """
        # ── 从 compressed_history 中分离 system 摘要与对话轮次 ──
        # 压缩后 render_context 会将历史摘要作为 role="system" 消息写入
        # compressed_history。如果将其作为独立的 system message 发送，
        # 会被模型优先关注而「挤掉」background_context（Lost-in-the-Middle）。
        # 因此这里提取摘要文本合并进统一的 system prompt，
        # 仅保留 user/assistant 轮次作为对话历史。
        history_summary_parts: list[str] = []
        conversation_turns: list[dict[str, str]] = []
        if compressed_history:
            for msg in compressed_history:
                if msg["role"] == "system":
                    history_summary_parts.append(msg["content"])
                else:
                    conversation_turns.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

        # ── 构建统一 system prompt ──
        if history_summary_parts:
            history_section = (
                "以下是之前对话的压缩摘要：\n\n" + "\n\n".join(history_summary_parts)
            )
        else:
            history_section = ""

        if background_context:
            background_section = (
                "以下是从你的记忆中检索到的相关背景知识：\n\n" + background_context
            )
        else:
            background_section = "当前没有检索到相关的背景记忆。"

        skill_plan_section = self._format_skill_plan(skill_reuse_plan)

        system_prompt = self.prompt_template.format(
            history_section=history_section,
            background_section=background_section,
            skill_plan_section=skill_plan_section,
        )

        system_prompt = self._append_time_basis(system_prompt)

        # ── 构建完整消息列表（单一 system + 对话轮次）──
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        for msg in conversation_turns:
            messages.append(msg)

        # 用户当前查询（确保在末尾）
        if not messages or messages[-1].get("content") != user_query:
            messages.append({"role": "user", "content": user_query})

        response = self.llm.generate(messages)
        return {"final_response": response}

    @staticmethod
    def _format_skill_plan(plan: list[dict] | None) -> str:
        """将可复用技能文档列表格式化为 system prompt 片段。"""
        if not plan:
            return ""
        lines = ["以下是可复用的技能文档（Markdown），可作为操作指南："]
        for i, step in enumerate(plan, 1):
            intent = step.get("intent", "?")
            skill_id = step.get("skill_id", "")
            score = step.get("score", 0)
            conditions = step.get("conditions", "")
            doc = step.get("doc_markdown", "")
            header = f"\n---\n技能{i}: {intent}"
            if skill_id:
                header += f" (id={skill_id})"
            if score:
                header += f" | score={score:.2f}" if isinstance(score, (int, float)) else ""
            lines.append(header)
            if conditions:
                lines.append(f"适用条件：{conditions}")
            if doc:
                lines.append(doc)
        return "\n".join(lines)
