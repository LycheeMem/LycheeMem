"""
记忆固化 Agent (Memory Consolidation Agent)。

异步后台进程，在每次交互结束后：
1. 分析完整对话记录
2. 提取新的事实/偏好变化 → 更新图谱
3. 提取成功的工具调用链 → 存入技能库
"""

from __future__ import annotations

from typing import Any

from a_frame.agents.base_agent import BaseAgent
from a_frame.embedder.base import BaseEmbedder
from a_frame.llm.base import BaseLLM
from a_frame.memory.graph.entity_extractor import EntityExtractor
from a_frame.memory.graph.graph_store import NetworkXGraphStore
from a_frame.memory.procedural.skill_store import InMemorySkillStore

CONSOLIDATION_SYSTEM_PROMPT = """\
你是一个「记忆固化专家（Memory Consolidator）」。
你需要审查刚刚结束的完整对话日志，从中判断是否有值得沉淀为长期记忆的内容。

需要关注两类信息：
1. 图谱事实 (Graph Facts)：
     - 用户偏好、项目属性、稳定的客观事实等，可以表示为 [主体, 关系, 客体] 的三元组；
     - 本系统会在后续步骤中调用专门的实体抽取器来生成具体三元组，
         因此你只需判断「是否存在值得抽取的事实」，不必直接输出三元组。
2. 程序技能 (Procedural Skills)：
     - 如果在本次对话中出现了 **成功的多步工具调用/操作流程**，
         请将其提炼为可复用的“工作流模板”。

请以 JSON 格式回复，结构如下（字段名必须保持一致）：
{
    "new_skills": [
        {
            "intent": "任务意图的一句话描述",
            "tool_chain": [
                {"step": 1, "action": "做了什么操作", "details": "涉及到的关键参数/工具/文件等"}
            ]
        }
    ],
    "should_extract_entities": true/false
}

说明：
- 如果对话没有值得保存的复杂操作模式，`new_skills` 应为一个空数组；
- 当你认为对话中包含稳定的用户偏好/事实/关系，适合写入知识图谱时，
    请将 `should_extract_entities` 设为 true；否则为 false；
- 忽略闲聊、重复说法、明显错误尝试等不值得长期保存的内容。

下面是几个示例（只用于帮助你理解格式与抽取标准，不要原样抄写示例中的中文内容）：

【示例 1：既有图谱事实，也有新技能】
<session_log>
user: 我想在这个项目里统一用 Python 3.10，并且所有新服务都部署到 k8s 集群 prod-a 上。
assistant: 好的，我会记住：语言用 Python 3.10，部署目标是 prod-a 集群。
user: 这次帮我把 user-service 做一个蓝绿发布的流程，我想先在 prod-a 的一半节点上灰度。
assistant: 我们可以这么做：
    1）更新 Helm values，把 user-service 的新版本镜像打上 v2 标签；
    2）使用 kubectl apply 应用新的 Deployment；
    3）观察 prometheus 的告警和日志，如果无异常，再将所有副本切到新版本。
</session_log>

期望 JSON 输出示例：
{
    "new_skills": [
        {
            "intent": "对 user-service 执行蓝绿发布到 prod-a 集群",
            "tool_chain": [
                {"step": 1, "action": "准备镜像与 Helm 配置", "details": "更新 Helm values，将 user-service 镜像标记为 v2"},
                {"step": 2, "action": "应用新版本部署", "details": "使用 kubectl apply 部署到 prod-a 集群的部分节点"},
                {"step": 3, "action": "观测并切流量", "details": "通过 Prometheus 和日志确认稳定后，将全部副本切到 v2"}
            ]
        }
    ],
    "should_extract_entities": true
}

【示例 2：只有图谱事实，没有可复用技能】
<session_log>
user: 以后在这个项目里，所有文档一律用中文撰写，不要再给我英文模版了。
assistant: 明白了，这个项目的文档统一使用中文。
</session_log>

期望 JSON 输出示例：
{
    "new_skills": [],
    "should_extract_entities": true
}

【示例 3：纯闲聊，不需要固化】
<session_log>
user: 哈哈，今天心情不错，随便聊聊八卦吧。
assistant: 好的，我们可以聊点轻松的话题～
</session_log>

期望 JSON 输出示例：
{
    "new_skills": [],
    "should_extract_entities": false
}
"""


class ConsolidatorAgent(BaseAgent):
    """记忆固化 Agent：异步分析对话并更新长期记忆。"""

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        graph_store: NetworkXGraphStore,
        skill_store: InMemorySkillStore,
        entity_extractor: EntityExtractor,
    ):
        super().__init__(llm=llm, prompt_template=CONSOLIDATION_SYSTEM_PROMPT)
        self.embedder = embedder
        self.graph_store = graph_store
        self.skill_store = skill_store
        self.entity_extractor = entity_extractor

    def run(
        self,
        turns: list[dict[str, str]],
        **kwargs,
    ) -> dict[str, Any]:
        """分析对话并固化到长期记忆。

        Args:
            turns: 完整的对话轮次列表。

        Returns:
            dict 包含：entities_added (int), skills_added (int)
        """
        if not turns:
            return {"entities_added": 0, "skills_added": 0}

        # 格式化对话用于分析
        conversation_text = "\n".join(
            f"{t['role']}: {t['content']}" for t in turns
        )

        # 1. 用 LLM 分析对话，判断是否有新技能和是否需要实体抽取
        response = self._call_llm(conversation_text, system_content=self.prompt_template)
        analysis = self._safe_parse(response)

        entities_added = 0
        skills_added = 0

        # 2. 实体抽取 → 更新图谱
        if analysis.get("should_extract_entities", False):
            triples = self.entity_extractor.extract_from_turns(turns)
            if triples:
                self.graph_store.add(triples)
                entities_added = len(triples)

        # 3. 新技能 → 存入技能库
        new_skills = analysis.get("new_skills", [])
        for skill in new_skills:
            intent = skill.get("intent", "")
            tool_chain = skill.get("tool_chain", [])
            if intent and tool_chain:
                embedding = self.embedder.embed_query(intent)
                self.skill_store.add([{
                    "intent": intent,
                    "embedding": embedding,
                    "tool_chain": tool_chain,
                }])
                skills_added += 1

        return {
            "entities_added": entities_added,
            "skills_added": skills_added,
        }

    def _safe_parse(self, response: str) -> dict[str, Any]:
        """安全解析 LLM 输出，失败时返回安全默认值。"""
        try:
            return self._parse_json(response)
        except (ValueError, KeyError):
            return {"new_skills": [], "should_extract_entities": False}
