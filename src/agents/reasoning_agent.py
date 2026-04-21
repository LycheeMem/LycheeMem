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
from src.agents.prompts import REASONING_SYSTEM_PROMPT
from src.evolve.prompt_registry import get_prompt
from src.llm.base import BaseLLM, set_llm_call_source


class ReasoningAgent(BaseAgent):
    """核心推理器：生成最终用户回复。"""

    _MAX_SKILL_DOCS = 2

    def __init__(self, llm: BaseLLM):
        super().__init__(llm=llm, prompt_template=REASONING_SYSTEM_PROMPT)

    def run(
        self,
        user_query: str,
        compressed_history: list[dict[str, str]] | None = None,
        background_context: str = "",
        skill_reuse_plan: list[dict] | None = None,
        retrieved_skills: list[dict] | None = None,
        reference_time: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        messages = self._build_messages(user_query, compressed_history, background_context, skill_reuse_plan, retrieved_skills, reference_time=reference_time)
        set_llm_call_source("reasoning")
        response = self.llm.generate(messages)
        return {"final_response": response}

    async def astream(
        self,
        user_query: str,
        compressed_history: list[dict[str, str]] | None = None,
        background_context: str = "",
        skill_reuse_plan: list[dict] | None = None,
        retrieved_skills: list[dict] | None = None,
        reference_time: str | None = None,
    ) -> AsyncIterator[str]:
        """流式生成最终回复，逐 token yield。"""
        messages = self._build_messages(user_query, compressed_history, background_context, skill_reuse_plan, retrieved_skills, reference_time=reference_time)
        set_llm_call_source("reasoning")
        async for token in self.llm.astream_generate(messages):
            yield token

    def _build_messages(
        self,
        user_query: str,
        compressed_history: list[dict[str, str]] | None = None,
        background_context: str = "",
        skill_reuse_plan: list[dict] | None = None,
        retrieved_skills: list[dict] | None = None,
        *,
        reference_time: str | None = None,
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

        selected_skills = self._select_skill_documents(
            skill_reuse_plan=skill_reuse_plan,
            retrieved_skills=retrieved_skills,
        )
        skill_plan_section = self._format_skill_plan(selected_skills)

        system_prompt = get_prompt("reasoning", self.prompt_template).format(
            history_section=history_section,
            background_section=background_section,
            skill_plan_section=skill_plan_section,
        )

        system_prompt = self._append_time_basis(system_prompt, now=self._parse_reference_time(reference_time))

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        for msg in conversation_turns:
            messages.append(msg)

        if not messages or messages[-1].get("content") != user_query:
            messages.append({"role": "user", "content": user_query})

        return messages

    @staticmethod
    def _parse_reference_time(reference_time: str | None) -> datetime.datetime | None:
        """将 reference_time 字符串解析为 datetime 对象。

        接受 ISO 8601 格式，例如 "2023-12-31" 或 "2023-12-31T23:59:59Z"。
        解析失败时返回 None（上层回退到系统时间）。
        """
        if not reference_time:
            return None
        raw = str(reference_time).strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.datetime.strptime(raw[:len(fmt) + 5], fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return dt
            except ValueError:
                continue
        return None

    @classmethod
    def _select_skill_documents(
        cls,
        *,
        skill_reuse_plan: list[dict] | None,
        retrieved_skills: list[dict] | None,
    ) -> list[dict[str, Any]]:
        """选择真正注入推理提示词的技能文档。

        优先使用 Synthesizer 已筛选过的 skill_reuse_plan；
        仅在外部直接调用 /memory/reason 且未提供该计划时，
        才回退到原始 retrieved_skills。
        """
        if skill_reuse_plan:
            normalized = [
                {
                    "skill_id": item.get("skill_id", ""),
                    "intent": item.get("intent", ""),
                    "doc_markdown": item.get("doc_markdown", ""),
                    "score": item.get("score", 0),
                    "conditions": item.get("conditions", ""),
                }
                for item in skill_reuse_plan
                if str(item.get("doc_markdown") or item.get("intent") or "").strip()
            ]
            normalized.sort(
                key=lambda item: float(item.get("score") or 0.0),
                reverse=True,
            )
            return normalized[:cls._MAX_SKILL_DOCS]

        fallback_skills = list(retrieved_skills or [])
        if not fallback_skills:
            return []

        reusable_skills = [
            item for item in fallback_skills
            if bool(item.get("reusable"))
        ]
        selected = reusable_skills or fallback_skills
        selected.sort(
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )
        return [
            {
                "skill_id": item.get("id", "") or item.get("skill_id", ""),
                "intent": item.get("intent", ""),
                "doc_markdown": item.get("doc_markdown", ""),
                "score": item.get("score", 0),
                "conditions": item.get("conditions", ""),
            }
            for item in selected[:cls._MAX_SKILL_DOCS]
            if str(item.get("doc_markdown") or item.get("intent") or "").strip()
        ]

    @staticmethod
    def _format_skill_plan(plan: list[dict[str, Any]] | None) -> str:
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
