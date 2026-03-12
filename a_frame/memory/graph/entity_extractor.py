"""
LLM 驱动的实体/关系三元组抽取。

从对话文本中提取 (subject, predicate, object) 三元组，
用于更新语义与情景记忆图谱。
"""

from __future__ import annotations

import json
from typing import Any

from a_frame.llm.base import BaseLLM

EXTRACTION_SYSTEM_PROMPT = """\
你是一个知识图谱实体抽取器。从以下文本中提取所有实体和它们之间的关系。

输出格式为 JSON 数组 triples：
[
  {
    "subject": {"name": "...", "label": "Person|Place|Event|Concept|Organization|Tool|Skill"},
    "predicate": "关系描述（英文动词短语，如 works_at, prefers, lives_in）",
    "object": {"name": "...", "label": "..."},
    "confidence": 0.0-1.0
  }
]

规则：
- 只提取明确表述的事实，不要推测
- label 必须是以下之一：Person, Place, Event, Concept, Organization, Tool, Skill
- predicate 使用 snake_case 英文
- confidence < 0.5 的三元组不要输出
- 如果文本中没有可提取的实体关系，返回空数组 []"""


class EntityExtractor:
    """从对话文本中提取知识图谱三元组。"""

    def __init__(self, llm: BaseLLM, confidence_threshold: float = 0.6):
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    def extract(self, text: str) -> list[dict[str, Any]]:
        """从文本中提取三元组列表。

        Args:
            text: 要分析的对话或文本内容。

        Returns:
            过滤后的三元组列表，每个三元组包含 subject, predicate, object, confidence。
        """
        if not text.strip():
            return []

        response = self.llm.generate(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,  # 低温度保证一致性
        )

        triples = self._parse_triples(response)
        # 按置信度过滤
        return [t for t in triples if t.get("confidence", 0) >= self.confidence_threshold]

    def extract_from_turns(self, turns: list[dict[str, str]]) -> list[dict[str, Any]]:
        """从对话轮次中提取三元组。"""
        text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
        return self.extract(text)

    @staticmethod
    def _parse_triples(response: str) -> list[dict[str, Any]]:
        """从 LLM 输出中解析三元组 JSON 数组。"""
        text = response.strip()
        # 处理 markdown code block 包裹
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            return []
        except json.JSONDecodeError:
            # 尝试找到 JSON 数组部分
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return []
            return []
