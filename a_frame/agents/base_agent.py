"""
Agent 基类。

所有认知 Agent 共享：
- LLM 调用接口
- Prompt 模板加载
- 结构化输出解析
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from a_frame.llm.base import BaseLLM


class BaseAgent(ABC):
    """所有认知 Agent 的抽象基类。"""

    def __init__(self, llm: BaseLLM, prompt_template: str):
        self.llm = llm
        self.prompt_template = prompt_template

    @abstractmethod
    def run(self, **kwargs) -> dict[str, Any]:
        """执行 Agent 逻辑，返回状态更新 patch。"""

    def _call_llm(self, user_content: str, system_content: str | None = None) -> str:
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        return self.llm.generate(messages)

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
