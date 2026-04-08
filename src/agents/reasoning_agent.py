"""
核心推理器 (Main Reasoning Agent)。

接收工作记忆管理器和整合器准备好的高浓度上下文，
执行最终的思考、工具调用和回答生成。
这是唯一面向用户生成最终回复的 Agent。
"""

from __future__ import annotations

import datetime
from collections.abc import AsyncIterator
from typing import Any

from src.agents.base_agent import BaseAgent
from src.llm.base import BaseLLM

REASONING_SYSTEM_PROMPT = """\
You are an intelligent assistant with access to background knowledge and memory.

{history_section}

{background_section}

{skill_plan_section}

Based on the background knowledge and conversation history above, provide an accurate and helpful answer to the user.

Rules:
- Prefer factual information from memory when answering
- If reusable skill documents (Markdown) are available, prioritize their steps, commands, and cautions
- If memory is insufficient, you may answer from general knowledge, but state that clearly
- Keep the answer concise and focused
- Do not fabricate facts that are not present"""


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
        retrieved_skills: list[dict] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        messages = self._build_messages(user_query, compressed_history, background_context, skill_reuse_plan, retrieved_skills)
        response = self.llm.generate(messages)
        return {"final_response": response}

    async def astream(
        self,
        user_query: str,
        compressed_history: list[dict[str, str]] | None = None,
        background_context: str = "",
        skill_reuse_plan: list[dict] | None = None,
        retrieved_skills: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """流式生成最终回复，逐 token yield。"""
        messages = self._build_messages(user_query, compressed_history, background_context, skill_reuse_plan, retrieved_skills)
        async for token in self.llm.astream_generate(messages):
            yield token

    def _build_messages(
        self,
        user_query: str,
        compressed_history: list[dict[str, str]] | None = None,
        background_context: str = "",
        skill_reuse_plan: list[dict] | None = None,
        retrieved_skills: list[dict] | None = None,
    ) -> list[dict[str, str]]:
        """构建发送给 LLM 的消息列表（供 run 和 astream 共用）。"""
        # ── 从 compressed_history 中分离 system 摘要与对话轮次 ──
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

        if history_summary_parts:
            history_section = (
                "Compressed summary of previous conversation:\n\n" + "\n\n".join(history_summary_parts)
            )
        else:
            history_section = ""

        if background_context:
            background_section = (
                "Relevant background knowledge retrieved from memory:\n\n" + background_context
            )
        else:
            background_section = "No relevant background memory was retrieved."

        # 优先使用原始检索技能（保留完整原文），否则退回到经 Synthesizer 过滤的计划
        if retrieved_skills:
            skill_plan_section = self._format_retrieved_skills(retrieved_skills)
        else:
            skill_plan_section = self._format_skill_plan(skill_reuse_plan)

        system_prompt = self.prompt_template.format(
            history_section=history_section,
            background_section=background_section,
            skill_plan_section=skill_plan_section,
        )

        system_prompt = self._append_time_basis(system_prompt)

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        for msg in conversation_turns:
            messages.append(msg)

        if not messages or messages[-1].get("content") != user_query:
            messages.append({"role": "user", "content": user_query})

        return messages

    @staticmethod
    def _format_retrieved_skills(skills: list[dict] | None) -> str:
        """将原始检索技能列表格式化为 system prompt 片段（保留完整 doc_markdown 原文）。"""
        if not skills:
            return ""
        lines = ["Retrieved skill documents (original text), which may be used as operational guidance:"]
        for i, skill in enumerate(skills, 1):
            intent = skill.get("intent", "?")
            skill_id = skill.get("id", "")
            score = skill.get("score", 0)
            doc = skill.get("doc_markdown", "")
            header = f"\n---\nSkill {i}: {intent}"
            if skill_id:
                header += f" (id={skill_id})"
            if score:
                header += f" | score={score:.2f}" if isinstance(score, (int, float)) else ""
            lines.append(header)
            if doc:
                lines.append(doc)
        return "\n".join(lines)

    @staticmethod
    def _format_skill_plan(plan: list[dict] | None) -> str:
        """将可复用技能文档列表格式化为 system prompt 片段。"""
        if not plan:
            return ""
        lines = ["Reusable skill documents (Markdown), which may be used as operational guidance:"]
        for i, step in enumerate(plan, 1):
            intent = step.get("intent", "?")
            skill_id = step.get("skill_id", "")
            score = step.get("score", 0)
            conditions = step.get("conditions", "")
            doc = step.get("doc_markdown", "")
            header = f"\n---\nSkill {i}: {intent}"
            if skill_id:
                header += f" (id={skill_id})"
            if score:
                header += f" | score={score:.2f}" if isinstance(score, (int, float)) else ""
            lines.append(header)
            if conditions:
                lines.append(f"Applicable conditions: {conditions}")
            if doc:
                lines.append(doc)
        return "\n".join(lines)
