"""
Agent 基类。

所有认知 Agent 共享：
- LLM 调用接口
- Prompt 模板加载
- 结构化输出解析
"""

from __future__ import annotations

import datetime
import json
from abc import ABC, abstractmethod
from typing import Any

from src.llm.base import BaseLLM


class BaseAgent(ABC):
    """所有认知 Agent 的抽象基类。"""

    def __init__(self, llm: BaseLLM, prompt_template: str):
        self.llm = llm
        self.prompt_template = prompt_template

    @abstractmethod
    def run(self, **kwargs) -> dict[str, Any]:
        """执行 Agent 逻辑，返回状态更新 patch。"""

    def _call_llm(
        self,
        user_content: str,
        system_content: str | None = None,
        *,
        add_time_basis: bool = False,
        now: datetime.datetime | None = None,
    ) -> str:
        messages = []
        if system_content:
            if add_time_basis:
                system_content = self._append_time_basis(system_content, now=now)
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        return self.llm.generate(messages)

    @staticmethod
    def _append_time_basis(system_prompt: str, now: datetime.datetime | None = None) -> str:
        """Append the current time basis to the system prompt to help resolve relative time expressions."""
        if not system_prompt:
            return system_prompt
        if now is None:
            now = datetime.datetime.now().astimezone()
        now_utc = now.astimezone(datetime.timezone.utc)
        today_local = now.date().isoformat()
        return (
            system_prompt
            + "\n\nCurrent time basis (for resolving relative time expressions):\n"
            + f"- Current local date: {today_local}\n"
            + f"- Current UTC time: {now_utc.isoformat()}\n"
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """从 LLM 输出中提取 JSON。"""
        # 尝试直接解析
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
