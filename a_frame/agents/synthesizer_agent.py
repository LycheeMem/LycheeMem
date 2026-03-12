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

1. **过滤**：判断每个记忆片段是否与当前查询相关（相关=keep，不相关=drop）
2. **去重**：合并语义重复的片段
3. **融合**：将保留的片段组织成一段连贯的 Background Context

以 JSON 格式回复：
{
  "kept_count": 保留的片段数,
  "dropped_count": 丢弃的片段数,
  "background_context": "整合后的背景知识文本（直接可用于注入 system prompt）"
}

规则：
- 如果全部片段都不相关，background_context 为空字符串
- background_context 应简洁聚焦，不要原样搬运原文，要做信息压缩
- 保持事实准确，不添加检索结果中没有的信息"""


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
        """将多源检索结果整合为 background_context。

        Returns:
            dict 包含：background_context (str)
        """
        fragments = self._format_fragments(
            retrieved_graph_memories or [],
            retrieved_skills or [],
            retrieved_sensory or [],
        )

        if not fragments:
            return {"background_context": ""}

        user_content = f"用户查询：{user_query}\n\n检索到的记忆片段：\n{fragments}"
        response = self._call_llm(user_content, system_content=self.prompt_template)

        try:
            parsed = self._parse_json(response)
            return {"background_context": parsed.get("background_context", "")}
        except (ValueError, KeyError):
            # 解析失败则直接使用 LLM 原始输出作为 context
            return {"background_context": response}

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
