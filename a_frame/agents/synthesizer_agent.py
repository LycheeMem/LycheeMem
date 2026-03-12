"""
整合排序器 (Memory Synthesizer & Ranker)。

对多源召回的记忆片段进行：
- LLM-as-judge 二元有效性打分
- 去重与聚类融合
- 输出精炼的 Background Context
"""

from __future__ import annotations

import json
from typing import Any

from a_frame.agents.base_agent import BaseAgent
from a_frame.llm.base import BaseLLM

SYNTHESIS_SYSTEM_PROMPT = """\
你是一个信息整合专家。根据用户查询和检索到的多源记忆片段，完成以下任务：

1. **打分**：为每个记忆片段评估与查询的相关性 (0.0-1.0)
2. **排序去重**：按相关性降序排列，合并语义重复的片段
3. **融合**：将保留的片段组织成一段连贯的 Background Context

以 JSON 格式回复：
{
  "scored_fragments": [
    {"source": "graph|skill|sensory", "index": 0, "relevance": 0.95, "summary": "..."}
  ],
  "kept_count": 保留的片段数,
  "dropped_count": 丢弃的片段数,
  "background_context": "整合后的背景知识文本（直接可用于注入 system prompt）"
}

规则：
- 如果全部片段都不相关，background_context 为空字符串
- background_context 应简洁聚焦，不要原样搬运原文，要做信息压缩
- 保持事实准确，不添加检索结果中没有的信息
- scored_fragments 按 relevance 降序排列"""


class SynthesizerAgent(BaseAgent):
    """整合排序器：将多源检索结果融合为 Background Context。"""

    def __init__(self, llm: BaseLLM):
        super().__init__(llm=llm, prompt_template=SYNTHESIS_SYSTEM_PROMPT)

    def run(
        self,
        user_query: str,
        retrieved_graph_memories: list[dict[str, Any]] | None = None,
        retrieved_skills: list[dict[str, Any]] | None = None,
        retrieved_sensory: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """将多源检索结果整合为 background_context + 技能复用计划。

        三步流程：score → rank → fuse
        对标记了 reusable=True 的技能，输出结构化执行计划。

        Returns:
            dict 包含：background_context, skill_reuse_plan, provenance
        """
        skills = retrieved_skills or []
        fragments = self._format_fragments(
            retrieved_graph_memories or [],
            skills,
            retrieved_sensory or [],
        )

        # 构建可复用技能执行计划
        skill_reuse_plan = self._build_reuse_plan(skills)

        if not fragments:
            return {
                "background_context": "",
                "skill_reuse_plan": skill_reuse_plan,
                "provenance": [],
            }

        user_content = f"用户查询：{user_query}\n\n检索到的记忆片段：\n{fragments}"
        response = self._call_llm(user_content, system_content=self.prompt_template)

        try:
            parsed = self._parse_json(response)
            provenance = parsed.get("scored_fragments", [])
            return {
                "background_context": parsed.get("background_context", ""),
                "skill_reuse_plan": skill_reuse_plan,
                "provenance": provenance,
            }
        except (ValueError, KeyError):
            return {
                "background_context": response,
                "skill_reuse_plan": skill_reuse_plan,
                "provenance": [],
            }

    @staticmethod
    def _build_reuse_plan(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从标记了 reusable=True 的技能构建执行计划。"""
        plan = []
        for skill in skills:
            if skill.get("reusable"):
                plan.append({
                    "skill_id": skill.get("id", ""),
                    "intent": skill.get("intent", ""),
                    "tool_chain": skill.get("tool_chain", []),
                    "score": skill.get("score", 0),
                    "conditions": skill.get("conditions", ""),
                })
        return plan

    @staticmethod
    def _format_fragments(
        graph_memories: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        sensory: list[dict[str, Any]],
    ) -> str:
        """将不同来源的检索结果格式化为统一文本。"""
        sections = []

        if graph_memories:
            lines = ["[知识图谱]"]
            for i, mem in enumerate(graph_memories, 1):
                anchor = mem.get("anchor", {})
                subgraph = mem.get("subgraph", {})
                nodes = subgraph.get("nodes", [])
                edges = subgraph.get("edges", [])
                lines.append(f"  片段{i}: 锚点={json.dumps(anchor, ensure_ascii=False)}")
                if edges:
                    for edge in edges[:5]:  # 限制数量防止过长
                        lines.append(
                            f"    {edge.get('source')} --[{edge.get('relation', '?')}]--> {edge.get('target')}"
                        )
            sections.append("\n".join(lines))

        if skills:
            lines = ["[技能库]"]
            for i, skill in enumerate(skills, 1):
                lines.append(
                    f"  技能{i}: 意图={skill.get('intent', '?')}, "
                    f"命令链={json.dumps(skill.get('tool_chain', []), ensure_ascii=False)[:200]}"
                )
            sections.append("\n".join(lines))

        if sensory:
            lines = ["[感觉记忆]"]
            for i, item in enumerate(sensory, 1):
                content = str(item.get("content", ""))[:200]
                lines.append(f"  条目{i}: [{item.get('modality', 'text')}] {content}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)
