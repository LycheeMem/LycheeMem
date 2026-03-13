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
你是一个高级 AI 智能体的「认知路由器（Cognitive Router）」。
你的唯一职责是：根据用户最新输入和极短的近期上下文，判断是否需要从长期记忆中检索信息，以及激活哪些记忆模块。

你可以调用的记忆模块在本系统中对应为：
1. 知识图谱 memory（need_graph）≈ Semantic 语义记忆：关于用户、项目、事实、历史事件等静态/结构化信息。
2. 技能库 memory（need_skills）≈ Procedural 程序记忆：可复用的工具调用步骤、工作流模板。
3. 感觉缓冲区 memory（need_sensory）≈ Episodic 近端情景记忆：最近几轮原始对话与输入片段。

重要约束：
- 你只做「是否需要检索、检索哪些模块」的路由判断，不执行复杂推理或长篇回答。
- 当问题可以直接在当前上下文中回答时，应尽量将三个开关都置为 false。
- 对于真实复杂任务，**同时打开多种记忆模块是很常见的**，
    因为需要结合「最近几轮交互」与「长期事实 / 技能工作流」一起做决策。

请严格以 **纯 JSON** 回复，不要包含任何额外解释文字或前后缀：
{
    "need_graph": true/false,    // 是否需要查询知识图谱（语义/事实类问题）
    "need_skills": true/false,   // 是否需要查询技能库（操作步骤、工具工作流）
    "need_sensory": true/false,  // 是否需要查看最近感觉记忆（引用刚才说过/做过的内容）
    "reasoning": "一句话说明你为何这样路由"
}

下面是若干示例（仅用于你在脑中参考，不要原样抄写示例中的中文）：

【示例 1：只需要图谱记忆】
用户最近对话：
    user: 我之前跟你说过我在上海的一家公司上班，还记得吗？
    assistant: 记得，你说过你在某家互联网公司做后端。
当前查询：
    "张三现在在哪家公司工作？"

期望 JSON 输出：
{
    "need_graph": true,
    "need_skills": false,
    "need_sensory": false,
    "reasoning": "这是关于用户工作单位的事实性问题，需要从知识图谱/语义记忆中检索。"
}

【示例 2：只需要技能记忆】
当前查询：
    "帮我写一个脚本，把 logs 目录下 7 天前的日志压缩并删除原文件。"

期望 JSON 输出：
{
    "need_graph": false,
    "need_skills": true,
    "need_sensory": false,
    "reasoning": "这是一个需要多步命令/脚本操作的任务，应从技能库中复用或检索相似工作流。"
}

【示例 3：只需要感觉记忆】
最近对话：
    user: 刚才你给我的那个三步排查方案，再重复一遍。
当前查询：
    "你刚刚说的第二步具体命令是什么？"

期望 JSON 输出：
{
    "need_graph": false,
    "need_skills": false,
    "need_sensory": true,
    "reasoning": "用户只是引用本轮会话中刚刚说过的内容，只需查看最近感觉记忆。"
}

【示例 4：感觉记忆 + 技能记忆】
最近对话：
    user: 上次你给我一个 CI/CD 流程，用 GitHub Actions 部署到生产环境，现在我想在那个基础上改一下，把 staging 环境也加上。
当前查询：
    "刚才你说的那个 CI/CD 流程的 YAML 配置再给我看一下，然后帮我想想要怎么扩展到 staging。"

期望 JSON 输出：
{
    "need_graph": false,
    "need_skills": true,
    "need_sensory": true,
    "reasoning": "需要同时参考刚才对话中的 CI/CD 配置细节（感觉记忆）以及历史中保存的部署工作流技能。"
}

【示例 5：感觉记忆 + 图谱记忆】
最近对话：
    user: 我记得我们之前讨论过 user-service 和 payment-service 之间的依赖拓扑。
当前查询：
    "基于我们刚才讨论的上下游服务关系，再结合你知识库里的系统架构图，帮我判断 user-service 超时时应该先看哪个服务？"

期望 JSON 输出：
{
    "need_graph": true,
    "need_skills": false,
    "need_sensory": true,
    "reasoning": "需要结合当前会话中刚讨论的局部依赖关系（感觉记忆）以及长期存储的整体架构图（图谱记忆）。"
}

【示例 6：无需检索】
当前查询：
    "你好，今天上海天气怎么样？顺便讲个笑话。"

期望 JSON 输出：
{
    "need_graph": false,
    "need_skills": false,
    "need_sensory": false,
    "reasoning": "简单闲聊问题，可以直接回答，无需访问长期记忆。"
}
"""


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
