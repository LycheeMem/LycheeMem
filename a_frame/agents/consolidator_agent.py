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
你是一个记忆固化分析器。分析以下完成的对话，判断是否有值得长期保存的信息。

需要识别：
1. **新技能**：对话中是否出现了成功的多步操作模式？
   如果有，提取 intent（意图描述）和 tool_chain（操作步骤列表）。
2. **需要忽略的内容**：闲聊、重复、错误尝试等不值得保存的内容。

以 JSON 格式回复：
{
  "new_skills": [
    {
      "intent": "任务意图的一句话描述",
      "tool_chain": [{"step": 1, "action": "...", "details": "..."}]
    }
  ],
  "should_extract_entities": true/false
}

如果对话没有值得保存的操作模式，new_skills 为空数组。
should_extract_entities 表示对话中是否包含值得提取的实体关系事实。"""


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
