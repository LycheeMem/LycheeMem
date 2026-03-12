"""
认知路由器 (Cognitive Router)。

分析用户输入和短期对话历史，决定需要调用哪些记忆模块。
输出结构化的路由决策 JSON。
"""

from __future__ import annotations

from typing import Any

from a_frame.agents.base_agent import BaseAgent
from a_frame.llm.base import BaseLLM

ROUTER_SYSTEM_PROMPT = """\
你是一个智能路由分析器。根据用户查询和当前上下文，判断该查询需要哪些记忆检索支持。

分析维度：
1. need_graph — 是否需要查询知识图谱？（涉及实体关系、事实、历史事件时为 true）
2. need_skills — 是否需要查询技能库？（涉及操作步骤、工具使用、流程时为 true）
3. need_sensory — 是否需要查看最近的感觉记忆？（需要最近几条原始输入上下文时为 true）
4. reasoning — 你的判断理由（一句话）

以 **纯 JSON** 回复，无其它文字：
{
  "need_graph": true/false,
  "need_skills": true/false,
  "need_sensory": true/false,
  "reasoning": "..."
}

示例：
- "张三在哪里工作？" → need_graph=true（实体关系查询）
- "帮我写一个爬虫" → need_skills=true（工具调用步骤）
- "你好" → 全部 false（简单对话不需要检索）
- "刚才你说的那个方案再说一遍" → need_sensory=true（引用最近输入）"""


class RouterAgent(BaseAgent):
    """认知路由器：决定激活哪些记忆检索。"""

    def __init__(self, llm: BaseLLM):
        super().__init__(llm=llm, prompt_template=ROUTER_SYSTEM_PROMPT)

    def run(self, user_query: str, recent_turns: list[dict[str, str]] | None = None, **kwargs) -> dict[str, Any]:
        """分析查询意图，返回路由决策。

        Args:
            user_query: 用户当前查询。
            recent_turns: 最近几轮对话（可选，提供上下文）。

        Returns:
            RouteDecision 字典：{need_graph, need_skills, need_sensory, reasoning}
        """
        context_lines = []
        if recent_turns:
            context_lines.append("最近对话：")
            for turn in recent_turns[-6:]:  # 最多 3 轮 = 6 条消息
                context_lines.append(f"  {turn['role']}: {turn['content']}")
            context_lines.append("")

        context_lines.append(f"当前查询：{user_query}")
        user_content = "\n".join(context_lines)

        response = self._call_llm(user_content, system_content=self.prompt_template)

        try:
            decision = self._parse_json(response)
        except (ValueError, KeyError):
            # 解析失败时的安全默认值：都不检索
            decision = {
                "need_graph": False,
                "need_skills": False,
                "need_sensory": False,
                "reasoning": "路由解析失败，使用默认值",
            }

        return {
            "need_graph": bool(decision.get("need_graph", False)),
            "need_skills": bool(decision.get("need_skills", False)),
            "need_sensory": bool(decision.get("need_sensory", False)),
            "reasoning": decision.get("reasoning", ""),
        }
