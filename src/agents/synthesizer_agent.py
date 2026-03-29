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

from src.agents.base_agent import BaseAgent
from src.llm.base import BaseLLM

SYNTHESIS_SYSTEM_PROMPT = """\
你是严苛的记忆整合与判断器（Memory Synthesizer & Judge）。
你将收到用户当前的任务需求，以及从不同记忆源（graph / skill）召回的若干原始记忆片段。

你的任务：
1. 为每一段记忆评估其对当前任务的绝对贡献度；
2. 将明显无关或价值很低的记忆丢弃；
3. 将保留下来的高价值记忆去重、融合为一段高密度的 Background Context。

打分与阈值策略：
- 你可以先在脑中按 0-10 分评估贡献度，然后归一化为 0.0-1.0 写入 `relevance` 字段；
- 约等价：6/10 ≈ 0.6；
- 请尽量丢弃 relevance < 0.6 的片段，只保留真正有用的内容。

请严格以 JSON 格式回复，结构如下（字段名必须保持一致）：
{
    "scored_fragments": [
        {"source": "graph|skill", "index": 0, "relevance": 0.95, "summary": "精炼后的该片段要点"}
    ],
    "kept_count": 保留的片段数,
    "dropped_count": 丢弃的片段数,
    "background_context": "整合后的背景知识文本（直接可用于注入系统的上文）"
}

规则：
- 如果全部片段都不相关，background_context 必须是空字符串；
- background_context 应是高密度的融合文本，不要简单拼接原文，要进行信息压缩和改写；
- 保持事实准确，不虚构检索结果中不存在的信息；
- scored_fragments 按 relevance 降序排列，summary 用简短中文概括该片段的核心信息。

## 示例（仅供参考，不要原样抄写）

用户查询：
    "帮我回顾一下这个项目里和 'user-service 超时' 相关的历史问题，避免我这次排查踩坑。"

来自不同记忆源的片段（由系统预先整理好给你）：
- [graph] 片段 0：图谱中存在三元组 [user-service, HAS_INCIDENT, 2024-01-15-timeout]，备注为 "上次因为下游 payment-service 慢导致整体超时"。
- [skill] 片段 1：技能库中有一条技能，其 intent 为 "排查 user-service 超时问题"，包含一份 Markdown 技能文档（步骤、命令、注意事项等）。

期望的 JSON 输出示例：
{
    "scored_fragments": [
        {"source": "skill", "index": 1, "relevance": 0.95, "summary": "历史上专门用于排查 user-service 超时问题的多步排查技能，可直接复用其检查顺序。"},
        {"source": "graph", "index": 0, "relevance": 0.85, "summary": "图谱记录表明 2024-01-15 的超时主要由下游 payment-service 变慢引起。"}
    ],
    "kept_count": 2,
    "dropped_count": 0,
    "background_context": "历史信息显示：user-service 的超时曾与 payment-service 性能问题高度相关，并且你之前已经有一套成熟的排查技能（包含查看网关 QPS、user-service 错误率以及下游依赖状态等步骤）。建议本次排查优先复用该技能的步骤顺序，并重点关注 payment-service 与网关流量情况，以避免重复踩入相同故障模式。"
}
"""


class SynthesizerAgent(BaseAgent):
    """整合排序器：将多源检索结果融合为 Background Context。"""

    def __init__(self, llm: BaseLLM):
        super().__init__(llm=llm, prompt_template=SYNTHESIS_SYSTEM_PROMPT)

    def run(
        self,
        user_query: str,
        retrieved_graph_memories: list[dict[str, Any]] | None = None,
        retrieved_skills: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """将多源检索结果整合为 background_context + 技能复用计划。

        三步流程：score → rank → fuse
        对标记了 reusable=True 的技能，输出结构化执行计划。

        Returns:
            dict 包含：background_context, skill_reuse_plan, provenance
        """
        skills = retrieved_skills or []

        # 收集检索阶段 provenance（Graphiti constructor/rerank 信号等），用于 pipeline 端到端溯源。
        retrieval_provenance: list[dict[str, Any]] = []
        for mem in retrieved_graph_memories or []:
            p = mem.get("provenance")
            if isinstance(p, list):
                for item in p:
                    if isinstance(item, dict):
                        retrieval_provenance.append(item)
        if len(retrieval_provenance) > 50:
            retrieval_provenance = retrieval_provenance[:50]

        fragments = self._format_fragments(
            retrieved_graph_memories or [],
            skills,
        )

        # 构建可复用技能执行计划
        skill_reuse_plan = self._build_reuse_plan(skills)

        if not fragments:
            return {
                "background_context": "",
                "skill_reuse_plan": skill_reuse_plan,
                "provenance": (
                    [{"source": "graphiti_retrieval", "items": retrieval_provenance}]
                    if retrieval_provenance
                    else []
                ),
            }

        user_content = f"用户查询：{user_query}\n\n检索到的记忆片段：\n{fragments}"
        response = self._call_llm(
            user_content,
            system_content=self.prompt_template,
            add_time_basis=True,
        )

        try:
            parsed = self._parse_json(response)
            provenance = parsed.get("scored_fragments", [])
            if not isinstance(provenance, list):
                provenance = []
            if retrieval_provenance:
                provenance.append({"source": "graphiti_retrieval", "items": retrieval_provenance})
            return {
                "background_context": parsed.get("background_context", ""),
                "skill_reuse_plan": skill_reuse_plan,
                "provenance": provenance,
            }
        except (ValueError, KeyError):
            return {
                "background_context": response,
                "skill_reuse_plan": skill_reuse_plan,
                "provenance": (
                    [{"source": "graphiti_retrieval", "items": retrieval_provenance}]
                    if retrieval_provenance
                    else []
                ),
            }

    @staticmethod
    def _build_reuse_plan(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从标记了 reusable=True 的技能构建"可复用技能文档列表"。"""
        plan = []
        for skill in skills:
            if skill.get("reusable"):
                plan.append(
                    {
                        "skill_id": skill.get("id", ""),
                        "intent": skill.get("intent", ""),
                        "doc_markdown": skill.get("doc_markdown", ""),
                        "score": skill.get("score", 0),
                        "conditions": skill.get("conditions", ""),
                    }
                )
        return plan

    @staticmethod
    def _format_fragments(
        graph_memories: list[dict[str, Any]],
        skills: list[dict[str, Any]],
    ) -> str:
        """将不同来源的检索结果格式化为统一文本。"""
        sections = []

        if graph_memories:
            lines = ["[知识图谱]"]
            for i, mem in enumerate(graph_memories, 1):
                anchor = mem.get("anchor", {})
                subgraph = mem.get("subgraph", {})
                edges = subgraph.get("edges", [])
                lines.append(f"  片段{i}: 锚点={json.dumps(anchor, ensure_ascii=False)}")

                constructed = str(mem.get("constructed_context", "")).strip()
                if constructed:
                    preview = constructed.strip().replace("\r\n", "\n")
                    # 控制长度，避免 prompt 过长
                    if len(preview) > 1500:
                        preview = preview[:1500] + "…"
                    lines.append(f"    构造上下文: {preview}")

                if edges:
                    for edge in edges[:5]:  # 限制数量防止过长
                        fact = str(edge.get("fact", "")).strip()
                        if fact:
                            lines.append(f"    事实: {fact[:300]}")
                        else:
                            lines.append(
                                f"    {edge.get('source')} --[{edge.get('relation', '?')}]--> {edge.get('target')}"
                            )
                        evidence = str(edge.get("evidence", "")).strip()
                        if evidence:
                            lines.append(f"      证据: {evidence[:200]}")
            sections.append("\n".join(lines))

        if skills:
            lines = ["[技能库]"]
            for i, skill in enumerate(skills, 1):
                doc = str(skill.get("doc_markdown", ""))
                doc_preview = doc.replace("\n", " ").strip()[:200]
                lines.append(f"  技能{i}: 意图={skill.get('intent', '?')}, 文档={doc_preview}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)
