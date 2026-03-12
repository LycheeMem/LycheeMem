"""
核心推理器 (Main Reasoning Agent)。

接收工作记忆管理器和整合器准备好的高浓度上下文，
执行最终的思考、工具调用和回答生成。
这是唯一面向用户生成最终回复的 Agent。
"""

from __future__ import annotations

from typing import Any

from a_frame.agents.base_agent import BaseAgent
from a_frame.llm.base import BaseLLM

REASONING_SYSTEM_PROMPT = """\
你是一个智能助手，拥有丰富的背景知识和记忆能力。

{background_section}

请根据以上背景知识和对话历史，为用户提供准确、有帮助的回答。

规则：
- 优先使用记忆中的事实信息回答
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
        **kwargs,
    ) -> dict[str, Any]:
        """执行推理，生成最终回答。

        Args:
            user_query: 用户当前查询。
            compressed_history: 压缩后的对话历史（含摘要）。
            background_context: 整合器输出的背景知识。

        Returns:
            dict 包含：final_response (str)
        """
        # 构建 system prompt
        if background_context:
            background_section = f"以下是从你的记忆中检索到的相关背景知识：\n\n{background_context}"
        else:
            background_section = "当前没有检索到相关的背景记忆。"

        system_prompt = self.prompt_template.format(background_section=background_section)

        # 构建完整消息列表
        messages = [{"role": "system", "content": system_prompt}]

        # 加入对话历史（已经压缩过的）
        if compressed_history:
            for msg in compressed_history:
                # 跳过系统消息（摘要已在 system prompt 之外）
                if msg["role"] != "system":
                    messages.append(msg)
                else:
                    # 历史摘要作为额外的 system 信息
                    messages.append({"role": "system", "content": msg["content"]})

        # 用户当前查询（确保在末尾）
        # 检查最后一条是否就是用户当前查询（避免重复）
        if not messages or messages[-1].get("content") != user_query:
            messages.append({"role": "user", "content": user_query})

        response = self.llm.generate(messages)
        return {"final_response": response}
