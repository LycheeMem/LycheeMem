"""
LLM 驱动的实体/关系三元组抽取。

从对话文本中提取 (subject, predicate, object) 三元组，
用于更新语义与情景记忆图谱。
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from a_frame.llm.base import BaseLLM

EXTRACTION_SYSTEM_PROMPT = """\
你是一个知识图谱实体抽取器。任务：从用户提供的文本中抽取“明确表述”的事实三元组 (subject, predicate, object)。

你必须只输出一个 JSON 数组（不要输出解释、不要输出 markdown code block、不要输出多余文本）。

每个三元组对象格式：
[
    {
        "subject": {"name": "实体名", "label": "Person|Place|Event|Concept|Organization|Tool|Skill"},
        "predicate": "snake_case 英文动词短语（例如 works_at, lives_in, prefers, uses, created, plans_to）",
        "object": {"name": "实体名", "label": "Person|Place|Event|Concept|Organization|Tool|Skill"},
        "confidence": 0.0,
        "fact": "一句完整事实（中文优先，能直接给人读懂）",
        "evidence": "支持该事实的原文片段（尽量短、尽量原样摘录）"
    }
]

抽取规则：
- 只提取文本中明确出现的事实，不要推测/脑补（例如‘可能’‘大概’‘也许’不抽）
- label 必须且只能是：Person, Place, Event, Concept, Organization, Tool, Skill
- predicate 必须是 snake_case 英文；不要用中文 predicate
- 只输出 confidence >= 0.5 的三元组
- evidence 必须来自原文的直接片段（不要自己改写）
- 没有可抽取的实体关系时，返回空数组 []

示例 1（单条明确事实）：
输入：
“我叫张三，在字节跳动工作。我更喜欢用 Python 做原型。”
输出：
[
    {
        "subject": {"name": "张三", "label": "Person"},
        "predicate": "works_at",
        "object": {"name": "字节跳动", "label": "Organization"},
        "confidence": 0.86,
        "fact": "张三在字节跳动工作。",
        "evidence": "在字节跳动工作"
    },
    {
        "subject": {"name": "张三", "label": "Person"},
        "predicate": "prefers",
        "object": {"name": "Python", "label": "Tool"},
        "confidence": 0.74,
        "fact": "张三更喜欢用 Python 做原型。",
        "evidence": "更喜欢用 Python 做原型"
    }
]

示例 2（多条事实 + 过滤推测/低置信度）：
输入：
“下周三我可能会去上海出差。我们已经确定 4 月 8 日在北京开产品评审会。评审会上要用 Notion 做记录。”
输出：
[
    {
        "subject": {"name": "产品评审会", "label": "Event"},
        "predicate": "happens_on",
        "object": {"name": "4月8日", "label": "Event"},
        "confidence": 0.78,
        "fact": "产品评审会定在 4 月 8 日。",
        "evidence": "确定 4 月 8 日"
    },
    {
        "subject": {"name": "产品评审会", "label": "Event"},
        "predicate": "takes_place_in",
        "object": {"name": "北京", "label": "Place"},
        "confidence": 0.8,
        "fact": "产品评审会在北京召开。",
        "evidence": "在北京开产品评审会"
    },
    {
        "subject": {"name": "产品评审会", "label": "Event"},
        "predicate": "uses",
        "object": {"name": "Notion", "label": "Tool"},
        "confidence": 0.7,
        "fact": "产品评审会要用 Notion 做记录。",
        "evidence": "用 Notion 做记录"
    }
]

示例 3（没有明确关系）：
输入：
“今天有点累，想休息一下。”
输出：
[]"""


def _build_extraction_system_prompt(now: datetime.datetime) -> str:
    # 将“当前日期/时间”作为抽取基准注入，便于把“今天/明天/下周三”等相对时间解析成明确日期。
    now_utc = now.astimezone(datetime.timezone.utc)
    today_local = now.date().isoformat()
    return (
        EXTRACTION_SYSTEM_PROMPT
        + "\n\n当前时间基准（用于解析相对时间表达）：\n"
        + f"- 当前本地日期：{today_local}\n"
        + f"- 当前 UTC 时间：{now_utc.isoformat()}\n"
        + "\n相对时间解析规则：\n"
        + "- 如果文本里出现 今天/明天/后天/下周X/本周X 等相对时间，请基于‘当前本地日期’推断为具体日期（YYYY-MM-DD）并写入 fact；必要时也可把该日期作为 object.name 输出（label 用 Concept）。\n"
        + "- 不要虚构具体日期：只有当相对时间表达明确（例如‘明天’‘下周三’）时才解析；模糊表达（例如‘最近’‘过几天’）不要强行给日期。\n"
    )


def _build_default_fact(triple: dict[str, Any]) -> str:
    subj = triple.get("subject", {}) or {}
    obj = triple.get("object", {}) or {}
    subj_name = subj.get("name") or subj.get("id") or ""
    obj_name = obj.get("name") or obj.get("id") or ""
    pred = triple.get("predicate", "")
    if subj_name and pred and obj_name:
        return f"{subj_name} {pred} {obj_name}"
    return ""


class EntityExtractor:
    """从对话文本中提取知识图谱三元组。"""

    def __init__(self, llm: BaseLLM, confidence_threshold: float = 0.6):
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    def extract(self, text: str, source_session: str = "") -> list[dict[str, Any]]:
        """从文本中提取三元组列表。

        Args:
            text: 要分析的对话或文本内容。
            source_session: 来源会话 ID，附加到每个三元组。

        Returns:
            过滤后的三元组列表，每个三元组包含 subject, predicate, object, confidence。
        """
        if not text.strip():
            return []

        system_prompt = _build_extraction_system_prompt(datetime.datetime.now().astimezone())
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.1,  # 低温度保证一致性
        )

        triples = self._parse_triples(response)
        # 按置信度过滤 + 附加元数据 + 补全可读事实
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result = []
        for t in triples:
            if t.get("confidence", 0) >= self.confidence_threshold:
                t.setdefault("source_session", source_session)
                t.setdefault("timestamp", now)
                if not t.get("fact"):
                    t["fact"] = _build_default_fact(t)
                if not t.get("evidence"):
                    t["evidence"] = text.strip().replace("\n", " ")[:200]
                result.append(t)
        return result

    def extract_from_turns(
        self, turns: list[dict[str, str]], source_session: str = ""
    ) -> list[dict[str, Any]]:
        """从对话轮次中提取三元组。"""
        text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
        return self.extract(text, source_session=source_session)

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
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return []
            return []
