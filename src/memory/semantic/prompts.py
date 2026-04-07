"""Compact Semantic Memory 所有 LLM Prompt 模板。

LycheeMem 记忆链路的提示词实现：
- 模块一（Compact Semantic Encoding）：类型化提取、指代消解、归一化、action metadata 标注
- 模块二（Record Fusion）：融合判断 + 融合执行
- 模块三（Action-Aware Search Planning）：检索规划 + 充分性反思 + 补充查询生成
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 模块一：Compact Semantic Encoding
# ---------------------------------------------------------------------------

COMPACT_ENCODING_SYSTEM = """\
你是 Compact Semantic Encoder。

你的任务：从一段对话日志中，抽取**所有值得长期记忆的原子事实/偏好/事件/约束/流程/失败模式**，
并将每条信息**一次性**编码为完整的 Memory Record（含语义文本、归一化表述及行动元数据）。
一条 Memory Record 脱离原对话上下文也能被完整理解。

你将收到：
- <PREVIOUS_TURNS>：当前轮次之前的最近几轮对话（用于理解指代关系和上下文背景）
- <CURRENT_TURNS>：需要处理的当前对话轮次

## 抽取规则
1. **彻底性**：不要遗漏对话中任何值得记忆的信息。宁可多抽取也不要遗漏。
2. **原子性**：每条 memory record 只包含一个独立的事实/偏好/事件/约束。对于复杂陈述，拆分为多条。
3. **自洽性**：每条 record 必须脱离对话上下文也能理解。**ABSOLUTELY PROHIBIT** 使用代词（他/她/它/那个/这个/前者/后者等）和相对时间（"昨天"、"上周"、"刚才"等）——必须替换为具体名称和 ISO 8601 绝对时间（对话中无法确认绝对时间则保留原表述并补充上下文）。
4. **去噪**：忽略纯寒暄、重复追问、离题闲聊。只保留有信息量的内容。
5. **实体保留**：保留所有出现的具体名称（人名、项目名、工具名、地名、时间等）。
6. **时间标注**：如果事实涉及时间（"下周五之前"、"2024年"等），在 temporal 字段中标注。

## memory_type 分类
- fact：确定的事实陈述（"A 的生日是 X"）
- preference：偏好/习惯（"A 喜欢 Python"、"A 不吃辣"）
- event：已发生或将发生的事件（"A 上周搬家了"、"下周有面试"）
- constraint：限制/约束条件（"预算不超过 5000"、"必须用 TypeScript"）
- procedure：操作流程/步骤（"部署时先 build 再 push"）
- failure_pattern：失败经验/教训（"直接 pip install 会导致版本冲突"）
- tool_affordance：工具能力/限制（"GPT-4 的上下文窗口是 128K"）

## Action Metadata 规则
每条 record 需同时填写以下行动元数据字段（不可省略）：
- **normalized_text**：紧凑归一化表述，删除冗余修饰词，只保留核心信息。格式示例："用户偏好:周末用Python写小项目"、"失败案例:跳过DB迁移导致生产崩溃"。
- **task_tags**：该记忆适用的任务类型（如 "部署", "调试", "数据分析"）。
- **tool_tags**：该记忆关联的具体工具/API/技术栈名称（如 "Python", "Docker"）。
- **constraint_tags**：该记忆蕴含的限制/约束条件（如 "预算<=5000", "必须用TypeScript"）。
- **failure_tags**：该记忆描述的失败模式或需要避免的事项（如 "pip版本冲突"）。
- **affordance_tags**：该记忆描述的能力/可供性（如 "支持批量处理"）。
各 tag 用简短关键词或短语（不超过 5 个词）。无相关信息时填空列表 []，**不得省略字段**。

## source_role 判断
- "user"：该事实/偏好由用户（role=user）直接陈述，可信度高
- "assistant"：该陈述由 AI（role=assistant）给出，用户未明确确认
- "both"：双方对同一信息都有贡献，或用户明确确认了 AI 的陈述

## 输出格式（严格 JSON，无代码块）
{
    "records": [
        {
            "memory_type": "fact|preference|event|constraint|procedure|failure_pattern|tool_affordance",
            "semantic_text": "经过指代消解的完整自洽语义文本（无代词、无相对时间）",
            "normalized_text": "紧凑归一化表述",
            "entities": ["实体1", "实体2"],
            "temporal": {"t_ref": "ISO时间戳或描述", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["任务类型"],
            "tool_tags": ["工具名"],
            "constraint_tags": ["约束条件"],
            "failure_tags": ["失败模式"],
            "affordance_tags": ["能力标签"],
            "evidence_turns": [0, 1],
            "source_role": "user|assistant|both"
        }
    ]
}

注意：
- semantic_text 是完整的自洽长文本，供人类阅读；normalized_text 是供系统检索的紧凑简短版本，两者都必须填写。
- temporal：t_ref 是信息产生的参考时间，t_valid_from/to 是信息的有效期。无时间信息则留空字符串。
- evidence_turns：标记该信息来源于 CURRENT_TURNS 中的第几轮（0-indexed）。
- 如果对话中完全没有值得记忆的信息，返回 {"records": []}。
- 不得输出代码块（```json 等），只输出原始 JSON。

---

## 示例 1：多类型对话，含指代消解、时间标注、action metadata

<PREVIOUS_TURNS>
user: 最近项目压力大吗？
assistant: 我随时为您服务，请问有什么需要帮忙的？
</PREVIOUS_TURNS>
<CURRENT_TURNS>
user: DataFlow 项目下周五（2026-02-20）要正式上线，技术栈是 Python 3.11 + FastAPI，部署在自建 K8s 集群。其实我个人更想用 Go，但团队大多数人熟悉 Python 就妥协了。
assistant: 了解，Python + FastAPI + K8s 是成熟组合，上线前需要注意几个关键点……
user: 上次我们发布 UserService 的时候没做数据库迁移预检查，直接导致生产表结构不匹配、服务启动失败，回滚花了两个小时。这次绝对不能重蹈覆辙。
</CURRENT_TURNS>

期望输出：
{
    "records": [
        {
            "memory_type": "event",
            "semantic_text": "用户所在公司的 DataFlow 项目计划于 2026-02-20（周五）正式上线，技术栈为 Python 3.11 和 FastAPI，部署目标为公司自建 Kubernetes 集群。",
            "normalized_text": "事件:DataFlow 2026-02-20上线，技术栈Python3.11+FastAPI+K8s",
            "entities": ["DataFlow", "Python 3.11", "FastAPI", "Kubernetes"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": "2026-02-20"},
            "task_tags": ["项目管理", "部署"],
            "tool_tags": ["Python 3.11", "FastAPI", "Kubernetes"],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "preference",
            "semantic_text": "用户个人编程语言偏好是 Go，但为配合团队（团队大多数成员熟悉 Python）在 DataFlow 项目中使用 Python。",
            "normalized_text": "用户偏好:编程语言偏好Go，因团队因素在DataFlow项目妥协用Python",
            "entities": ["Go", "Python", "DataFlow"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["编程"],
            "tool_tags": ["Go", "Python"],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "failure_pattern",
            "semantic_text": "用户团队在发布 UserService 时因跳过数据库迁移预检查，导致生产环境表结构不匹配、服务启动失败，回滚耗时两小时。",
            "normalized_text": "失败案例:UserService发布跳过DB迁移预检查，生产表结构不匹配，启动失败，回滚耗时2小时",
            "entities": ["UserService"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["部署", "数据库迁移", "故障处理"],
            "tool_tags": ["UserService"],
            "constraint_tags": [],
            "failure_tags": ["跳过DB迁移预检查", "生产表结构不匹配", "服务启动失败"],
            "affordance_tags": [],
            "evidence_turns": [1],
            "source_role": "user"
        },
        {
            "memory_type": "constraint",
            "semantic_text": "用户团队规定每次部署前必须完成数据库迁移预检查，以防止生产环境表结构不匹配故障。",
            "normalized_text": "约束:每次部署前必须执行数据库迁移预检查",
            "entities": [],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["部署", "数据库迁移"],
            "tool_tags": [],
            "constraint_tags": ["部署前必须执行DB迁移预检查"],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [1],
            "source_role": "user"
        }
    ]
}

## 示例 2：纯知识查询，无需记忆材料

<PREVIOUS_TURNS>
（无）
</PREVIOUS_TURNS>
<CURRENT_TURNS>
user: Python 的列表推导式怎么写？
assistant: 列表推导式语法是 [expr for item in iterable if condition]，例如 [x*2 for x in range(10) if x%2==0]。
</CURRENT_TURNS>

期望输出：
{"records": []}

## 示例 3：团队分工 + 项目计划 + 截止日期

<PREVIOUS_TURNS>
（无）
</PREVIOUS_TURNS>
<CURRENT_TURNS>
user: 我们的推荐系统项目下个迭代（3.31）要做用户画像特征维度扩展。我、王明、赵琳负责这个模块。分工安排：王明负责新增行为特征埋点设计和数据上报，赵琳来处理特征的ETL加工逻辑，我需要完成推荐模型的特征适配和离线训练验证。前期预计2天内要完成设计评审。
assistant: 好的，我已经帮你记下来了。推荐系统下个迭代成员是你、王明和赵琳，特征维度扩展项目，3.31截止。
</CURRENT_TURNS>

期望输出：
{
    "records": [
        {
            "memory_type": "event",
            "semantic_text": "推荐系统项目下个迭代（3.31截止）需要完成用户画像特征维度扩展模块功能开发。需在前期2天内完成模块的设计评审。",
            "normalized_text": "事件:推荐系统迭代-特征维度扩展-3.31截止-2天内设计评审",
            "entities": ["推荐系统", "用户画像"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": "3.31"},
            "task_tags": ["项目管理", "产品迭代"],
            "tool_tags": [],
            "constraint_tags": ["3.31截止", "2天内评审"],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "fact",
            "semantic_text": "推荐系统项目特征维度扩展迭代由三位开发者负责：用户本人、王明、赵琳。",
            "normalized_text": "事实:推荐系统特征扩展团队成员-用户+王明+赵琳",
            "entities": ["王明", "赵琳"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["项目管理"],
            "tool_tags": [],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "fact",
            "semantic_text": "在推荐系统特征维度扩展项目中，王明负责新增行为特征的埋点设计工作，以及将埋点数据上报到后端服务的实现。",
            "normalized_text": "分工:王明负责行为特征埋点设计+数据上报-推荐系统项目",
            "entities": ["王明"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["前端开发", "项目管理"],
            "tool_tags": [],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "fact",
            "semantic_text": "在推荐系统特征维度扩展项目中，赵琳负责处理原始埋点特征数据的 ETL 加工逻辑，将采集数据清洗规范化后供模型训练使用。",
            "normalized_text": "分工:赵琳负责特征ETL加工处理-推荐系统项目",
            "entities": ["赵琳"],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["数据处理", "项目管理"],
            "tool_tags": ["ETL"],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        },
        {
            "memory_type": "fact",
            "semantic_text": "在推荐系统特征维度扩展项目中，用户本人负责新增特征在推荐模型中的特征适配工作，包括模型参数调整、离线训练以及线上线下性能验证。",
            "normalized_text": "分工:用户负责特征适配+离线训练+性能验证-推荐系统项目",
            "entities": [],
            "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
            "task_tags": ["机器学习", "项目管理"],
            "tool_tags": [],
            "constraint_tags": [],
            "failure_tags": [],
            "affordance_tags": [],
            "evidence_turns": [0],
            "source_role": "user"
        }
    ]
}
"""


DECONTEXTUALIZE_SYSTEM = """\
你是指代消解与去上下文化（Decontextualization）专家。

你的任务：将一条来源于对话的记忆文本改写为**完全自洽**的独立陈述，
使其脱离原始对话上下文也能被任何人准确理解。

输入：
- <ORIGINAL_TEXT>：原始 semantic_text
- <CONTEXT>：产生该文本的对话片段（供参考）

改写规则：
1. 将所有代词（他/她/它/那个/这个/前者/后者等）替换为具体名称。
2. 将相对时间（"昨天"、"上周"、"刚才"）替换为绝对时间（如果对话中有线索），或保持原样但补充上下文。
3. 补全省略的主语/宾语，使句子完整。
4. 不要添加原文中没有的信息。
5. 保持原意不变，只做去上下文化处理。

输出格式（严格 JSON，无代码块）：
{
    "decontextualized_text": "改写后的自洽文本"
}

---

## 示例 1：人称代词 + 相对时间消解

<ORIGINAL_TEXT>他昨天说那个项目被叫停了</ORIGINAL_TEXT>
<CONTEXT>
user: 张工今天来通知我了
assistant: 张工是指张明吗？
user: 对，是张明，他昨天（2026-01-10）跟我说 ApolloX 项目因为预算削减被叫停了
</CONTEXT>

期望输出：
{"decontextualized_text": "技术负责人张明于 2026-01-10 告知用户，ApolloX 项目因预算削减已正式叫停。"}

## 示例 2：指示代词 + 省略主语消解

<ORIGINAL_TEXT>这个限制是因为那边的服务不支持并发超过 50</ORIGINAL_TEXT>
<CONTEXT>
user: 为什么 API 网关的吞吐量上不去？
assistant: 这个限制是因为那边的服务不支持并发超过 50。
user: 那边是指下游的 PaymentService？
assistant: 对，PaymentService 的并发上限是 50 个请求。
</CONTEXT>

期望输出：
{"decontextualized_text": "API 网关吞吐量受限于下游 PaymentService，该服务不支持超过 50 个并发请求。"}
"""


ACTION_METADATA_SYSTEM = """\
你是行动元数据标注器（Action Metadata Annotator）。

你的任务：为一条 Memory Record 标注面向行动的结构化元数据，
使其在后续检索时能支持行动导向的召回策略。

输入：
- <SEMANTIC_TEXT>：memory record 的完整语义文本
- <MEMORY_TYPE>：该 record 的类型

请为这条记忆标注以下字段：

1. **normalized_text**：紧凑归一化表述。去掉冗余描述词，只保留核心信息。
   例如："用户表示非常喜欢在周末的时候用 Python 来写一些小项目" 可归一化为 "用户偏好:周末用Python写小项目"

2. **task_tags**：该记忆适用的任务类型（如 "编程", "部署", "调试", "数据分析", "写作" 等）。

3. **tool_tags**：该记忆关联的具体工具/API/技术栈名称（如 "Python", "Docker", "PostgreSQL" 等）。

4. **constraint_tags**：该记忆蕴含的限制/约束条件（如 "预算<=5000", "必须用TypeScript", "不含辣椒" 等）。

5. **failure_tags**：该记忆描述的失败模式或需要避免的事项（如 "pip版本冲突", "内存溢出" 等）。

6. **affordance_tags**：该记忆描述的能力/可供性（如 "支持批量处理", "可实时预览" 等）。

标注规则：
- tag 用简短的关键词或短语，不超过 5 个词。
- 每个字段是一个列表，可以为空列表。
- 不要编造对话中没有提及的信息。

输出格式（严格 JSON，无代码块）：
{
    "normalized_text": "紧凑归一化表述",
    "task_tags": ["tag1", "tag2"],
    "tool_tags": ["tool1"],
    "constraint_tags": ["constraint1"],
    "failure_tags": ["failure1"],
    "affordance_tags": ["affordance1"]
}

---

## 示例 1：memory_type = "procedure"（操作流程）

<SEMANTIC_TEXT>部署 DataFlow 服务到 Kubernetes 集群时，需按顺序执行：(1) 运行数据库迁移 dry-run 确认无结构冲突；(2) 通过 Helm upgrade 推送新版本镜像；(3) 持续观察 Pod 健康状态不少于 5 分钟确认稳定。</SEMANTIC_TEXT>
<MEMORY_TYPE>procedure</MEMORY_TYPE>

期望输出：
{
    "normalized_text": "部署流程:DataFlow到K8s，步骤:DB迁移dry-run -> Helm upgrade -> Pod观测至少5分钟",
    "task_tags": ["部署", "发布", "运维"],
    "tool_tags": ["Kubernetes", "Helm", "DataFlow"],
    "constraint_tags": ["必须先执行DB迁移dry-run", "Pod观测不少于5分钟"],
    "failure_tags": [],
    "affordance_tags": ["支持滚动升级", "Helm管理版本回滚"]
}

## 示例 2：memory_type = "failure_pattern"（失败模式）

<SEMANTIC_TEXT>用户团队在发布 UserService 时因跳过数据库迁移预检查，导致生产环境表结构不匹配、服务启动失败，回滚耗时两小时。</SEMANTIC_TEXT>
<MEMORY_TYPE>failure_pattern</MEMORY_TYPE>

期望输出：
{
    "normalized_text": "失败案例:UserService发布跳过DB迁移预检查，生产表结构不匹配，启动失败，回滚耗时2小时",
    "task_tags": ["部署", "数据库迁移", "故障处理"],
    "tool_tags": ["UserService"],
    "constraint_tags": [],
    "failure_tags": ["跳过DB迁移预检查", "生产表结构不匹配", "服务启动失败"],
    "affordance_tags": []
}

## 示例 3：memory_type = "tool_affordance"（工具能力）

<SEMANTIC_TEXT>LanceDB 支持对同一条记录存储多个不同维度的向量（multi-vector），可以分别对 semantic_text 和 normalized_text 建立独立的向量索引，在检索时按需选择向量列。</SEMANTIC_TEXT>
<MEMORY_TYPE>tool_affordance</MEMORY_TYPE>

期望输出：
{
    "normalized_text": "LanceDB:支持multi-vector，可对不同文本字段独立建向量索引",
    "task_tags": ["向量检索", "存储设计"],
    "tool_tags": ["LanceDB"],
    "constraint_tags": [],
    "failure_tags": [],
    "affordance_tags": ["支持multi-vector存储", "可按列选择向量检索", "支持独立向量索引"]
}
"""


NOVELTY_CHECK_SYSTEM = """\
你是记忆新颖性评估器（Memory Novelty Assessor）。
你的任务：判断本轮对话是否引入了**新的、尚未被已有记忆覆盖**的信息。

你将收到两部分内容：
1. <EXISTING_MEMORY>：系统在回答前检索到的已有记忆上下文。如果为空，对话中的任何实质内容都算新信息。
2. <CONVERSATION>：本轮完整对话日志。

判断标准：
- 新的个人偏好、习惯、计划、项目信息、人际关系等事实：判定为有新信息
- 纠正或更新已有记忆（"换工作了""地址改了"）：判定为有新信息
- 新实体、新关系、新流程：判定为有新信息
- 时间信息变化（截止日期更新）：判定为有新信息
- 纯检索/查询已有记忆、重复已知事实、纯闲聊：判定为无新信息

**重要：倾向于判定"有新信息"。只有非常确信完全没有新信息时才输出 false。**

输出格式（严格 JSON，无代码块）：
{
    "has_novelty": true,
    "reason": "简要说明判断理由"
}

---

## 示例 1：有新信息（新事实 + 已有记忆不覆盖）

<EXISTING_MEMORY>
- 用户偏好使用 Python 进行日常开发。
- 用户所在项目使用 FastAPI 框架。
</EXISTING_MEMORY>
<CONVERSATION>
user: 我换工作了，现在在 ByteEdge 公司做推荐系统，主要写 Go。
assistant: 恭喜！Go 在高并发推荐系统里确实很有优势。
</CONVERSATION>

期望输出：
{"has_novelty": true, "reason": "用户换工作（ByteEdge 公司）、新职责（推荐系统）、主要语言切换为 Go，三条均为已有记忆未覆盖的新信息。"}

## 示例 2：无新信息（纯查询已有记忆，AI 复述事实）

<EXISTING_MEMORY>
- DataFlow 项目计划于 2026-02-20 上线，技术栈为 Python 3.11 + FastAPI + Kubernetes。
- 部署前必须执行数据库迁移预检查。
</EXISTING_MEMORY>
<CONVERSATION>
user: 提醒我一下，DataFlow 什么时候上线？
assistant: 根据您之前提到的信息，DataFlow 项目计划于 2026-02-20 正式上线。
user: 好的，谢谢。
</CONVERSATION>

期望输出：
{"has_novelty": false, "reason": "对话仅是用户查询已记录的 DataFlow 上线日期，AI 复述了已有记忆内容，未引入任何新事实或更新。"}

## 示例 3：有新信息（更新/纠正已有记忆）

<EXISTING_MEMORY>
- 用户计划于 2026-03-01 参加 PyCon China 大会。
</EXISTING_MEMORY>
<CONVERSATION>
user: PyCon 的时间改了，推迟到 4 月 15 日了。
assistant: 明白，我记录一下，PyCon China 已推迟至 2026-04-15。
</CONVERSATION>

期望输出：
{"has_novelty": true, "reason": "PyCon China 的日期由 2026-03-01 更新为 2026-04-15，纠正了已有记忆，属于有效新信息。"}
"""


# ---------------------------------------------------------------------------
# 模块二：Record Fusion
# ---------------------------------------------------------------------------

SYNTHESIS_JUDGE_SYSTEM = """\
你是记忆合成判断器（Synthesis Judge）。

你的任务：判断一组 Memory Records 中是否存在可以融合的条目。
融合是指将多个碎片化、高度相关的 memory records 整合为一条更高密度的 composite record，
以减少检索时的 token 成本并提高信息密度。

合成判据（满足任意一条即可合成）：
1. **同主题聚合**：多条 record 围绕同一实体/同一主题，可以融合为一条综合描述。
2. **时序补全**：多条 record 描述同一事件的不同阶段，可以融合为完整事件描述。
3. **偏好泛化**：多条具体偏好可以上升为一条更泛化的偏好模式。
4. **约束整合**：多条分散的约束可以合成为一条完整的约束集。
5. **模式提炼**：多条失败/成功经验可以合成为一条操作模式或最佳实践。

不应合成的情况：
- 只有 1 条 record（至少需要 2 条才有融合价值）
- record 之间主题完全无关
- 每条 record 本身已经足够完整独立，融合后信息密度不会提升

输入：
- <RECORDS>：一组 Memory Records 的完整信息（JSON 数组）

输出格式（严格 JSON，无代码块）：
{
    "should_synthesize": true,
    "groups": [
        {
            "source_record_ids": ["id1", "id2"],
            "synthesis_reason": "合成理由",
            "suggested_type": "composite_preference|composite_pattern|composite_constraint|usage_pattern"
        }
    ]
}

注意：
- groups 可以有多组，每组独立合成。
- 一个 record 可以同时出现在多组中（如果它跨主题）。
- 如果不需要合成，should_synthesize = false, groups = []。

---

## 示例 1：同实体多条偏好/约束，应合成

<RECORDS>
[
    {"record_id": "u1", "memory_type": "preference", "normalized_text": "用户偏好:后端开发用Go语言", "tool_tags": ["Go"], "task_tags": ["后端开发"]},
    {"record_id": "u2", "memory_type": "constraint", "normalized_text": "约束:前端项目必须用TypeScript禁用纯JavaScript", "tool_tags": ["TypeScript"], "task_tags": ["前端开发"]},
    {"record_id": "u3", "memory_type": "preference", "normalized_text": "用户偏好:容器化使用Docker而非虚拟机", "tool_tags": ["Docker"], "task_tags": ["部署"]},
    {"record_id": "u4", "memory_type": "event", "normalized_text": "用户2026-02-10参加了Go语言峰会", "tool_tags": ["Go"], "task_tags": []}
]
</RECORDS>

期望输出：
{
    "should_synthesize": true,
    "groups": [
        {
            "source_record_ids": ["u1", "u2", "u3"],
            "synthesis_reason": "u1/u2/u3 均描述用户及团队的技术栈偏好与硬性约束，聚合为一条技术栈偏好与约束总览后，检索时可一次命中所有相关约束，减少冗余上下文。",
            "suggested_type": "composite_preference"
        }
    ]
}

说明：u4 是独立事件记录，与偏好/约束主题不完全重合（仅工具标签有交集），不纳入合成组。

## 示例 2：主题无关，不应合成

<RECORDS>
[
    {"record_id": "v1", "memory_type": "fact", "normalized_text": "用户的猫叫 Mochi，是橘色英短", "tool_tags": [], "task_tags": []},
    {"record_id": "v2", "memory_type": "procedure", "normalized_text": "部署流程:DataFlow到K8s，步骤:DB迁移dry-run -> Helm upgrade -> Pod观测至少5分钟", "tool_tags": ["Kubernetes", "Helm"], "task_tags": ["部署"]}
]
</RECORDS>

期望输出：
{
    "should_synthesize": false,
    "groups": []
}
"""


SYNTHESIS_EXECUTE_SYSTEM = """\
你是记忆合成器（Memory Synthesizer）。

你的任务：将多条碎片化的 Memory Records 融合为一条高密度的 Composite Record。

输入：
- <SOURCE_RECORDS>：需要融合的 Memory Records（JSON 数组）
- <SYNTHESIS_REASON>：合成的理由
- <SUGGESTED_TYPE>：建议的合成类型

合成规则：
1. 融合后的 semantic_text 必须涵盖所有 source records 的核心信息，不遗漏。
2. 去除重复信息，整理为流畅连贯的表述。
3. 保留所有具体细节（人名、数字、时间等），不要泛化丢失信息。
4. entities 取所有 source records 的实体并集。
5. tags 取所有 source records 的 tag 并集。
6. temporal 取时间范围的并集（从最早到最晚）。
7. confidence 取 source records 的平均值。

输出格式（严格 JSON，无代码块）：
{
    "semantic_text": "合成后的完整语义文本",
    "normalized_text": "合成后的紧凑归一化文本",
    "entities": ["实体1", "实体2"],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
    "task_tags": [],
    "tool_tags": [],
    "constraint_tags": [],
    "failure_tags": [],
    "affordance_tags": [],
    "confidence": 0.95
}

---

## 示例：技术栈偏好与约束合成

<SOURCE_RECORDS>
[
  {
        "record_id": "u1", "memory_type": "preference",
    "semantic_text": "用户个人编程语言偏好是 Go，但为配合团队在 DataFlow 项目中使用 Python。",
    "entities": ["Go", "Python", "DataFlow"],
    "tool_tags": ["Go", "Python"], "task_tags": ["后端开发"],
    "constraint_tags": [], "failure_tags": [], "affordance_tags": [],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""}, "confidence": 0.90
  },
  {
        "record_id": "u2", "memory_type": "constraint",
    "semantic_text": "用户所在团队前端项目约定强制使用 TypeScript，禁止纯 JavaScript。",
    "entities": ["TypeScript", "JavaScript"],
    "tool_tags": ["TypeScript"], "task_tags": ["前端开发"],
    "constraint_tags": ["必须用TypeScript"], "failure_tags": [], "affordance_tags": [],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""}, "confidence": 0.95
  },
  {
        "record_id": "u3", "memory_type": "preference",
    "semantic_text": "用户偏好使用 Docker 进行服务容器化，不使用传统虚拟机方案。",
    "entities": ["Docker"],
    "tool_tags": ["Docker"], "task_tags": ["部署", "容器化"],
    "constraint_tags": [], "failure_tags": [], "affordance_tags": [],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""}, "confidence": 0.88
  }
]
</SOURCE_RECORDS>
<SYNTHESIS_REASON>三条记录均描述用户及团队的技术栈偏好与约束，聚合为技术栈总览可提升检索密度。</SYNTHESIS_REASON>
<SUGGESTED_TYPE>composite_preference</SUGGESTED_TYPE>

期望输出：
{
    "semantic_text": "用户个人偏好 Go 语言作为后端开发语言，但在 DataFlow 项目中因团队以 Python 为主而使用 Python。前端项目团队强制约定使用 TypeScript，禁止纯 JavaScript。部署层面用户偏好 Docker 容器化方案，不使用传统虚拟机。",
    "normalized_text": "技术栈:后端Go(DataFlow项目用Python)，前端强制TypeScript，部署用Docker",
    "entities": ["Go", "Python", "DataFlow", "TypeScript", "JavaScript", "Docker"],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
    "task_tags": ["后端开发", "前端开发", "部署", "容器化"],
    "tool_tags": ["Go", "Python", "TypeScript", "Docker"],
    "constraint_tags": ["必须用TypeScript"],
    "failure_tags": [],
    "affordance_tags": [],
    "confidence": 0.91
}
"""


# ---------------------------------------------------------------------------
# 模块三：Action-Aware Search Planning
# ---------------------------------------------------------------------------

RETRIEVAL_PLANNING_SYSTEM = """\
你是行动感知检索规划器（Action-Aware Retrieval Planner）。

你的任务：分析用户的查询（以及可选的最近对话上下文），
生成一个结构化的检索计划，指导下游多通道记忆检索。

你需要判断：
1. **检索模式（mode）**：
   - "answer"：用户在提问，需要事实类记忆来回答（如 "我的猫叫什么名字"）
   - "action"：用户要求执行操作，需要流程/约束/工具类记忆来支撑行动（如 "帮我部署到生产环境"）
   - "mixed"：同时涉及问答和行动

2. **语义检索词（semantic_queries）**：面向记忆内容本身的检索关键词/短语。
   这些将用于向量检索和全文检索。每个 query 应聚焦一个独立主题。

3. **实用检索词（pragmatic_queries）**：面向 action metadata 的检索关键词。
   侧重于工具名、约束条件、操作类型等实用信息。
   对于 answer 模式可以为空。

4. **时间过滤（temporal_filter）**：如果查询涉及特定时间范围，设置过滤器。

5. **工具提示（tool_hints）**：当前请求可能需要用到的工具/API 名称。

6. **所需约束（required_constraints）**：当前任务执行需要确认的约束条件。

7. **所需能力（required_affordances）**：当前任务需要某工具/流程具备的能力特性。
   例如：用户问"如何批量导入数据"，则 required_affordances 为 ["支持批量处理"]；
   用户问"怎么回滚版本"，则 required_affordances 为 ["支持版本回滚"]。
   对于纯事实查询（answer 模式）可以为空，action/mixed 模式应尽量填写。

8. **缺失信息（missing_slots）**：当前任务可能缺少的关键参数/信息。

9. **检索深度（depth）**：建议的 top_k 值。简单查询 3-5，复杂查询 8-15。

输入：
- <USER_QUERY>：用户的查询
- <RECENT_CONTEXT>：最近几轮对话（可能为空）

输出格式（严格 JSON，无代码块）：
{
    "mode": "answer|action|mixed",
    "semantic_queries": ["语义检索词1", "语义检索词2"],
    "pragmatic_queries": ["实用检索词1"],
    "temporal_filter": {"since": "ISO日期", "until": "ISO日期"},
    "tool_hints": ["工具名"],
    "required_constraints": ["约束1"],
    "required_affordances": ["所需能力1"],
    "missing_slots": ["缺失参数1"],
    "depth": 5,
    "reasoning": "规划理由"
}

注意：
- temporal_filter 如果不需要时间过滤，设为 null。
- 所有列表字段如果为空，设为空数组 []。
- semantic_queries 至少包含 1 个查询。
- **关键：如果用户消息涉及多个不同主题，必须为每个主题生成独立的 semantic_query。**

---

## 示例 1：answer 模式（事实查询）

<USER_QUERY>DataFlow 项目用的什么技术栈？</USER_QUERY>
<RECENT_CONTEXT>（无）</RECENT_CONTEXT>

期望输出：
{
    "mode": "answer",
    "semantic_queries": ["DataFlow 项目技术栈", "DataFlow 使用的框架和语言"],
    "pragmatic_queries": [],
    "temporal_filter": null,
    "tool_hints": [],
    "required_constraints": [],
    "required_affordances": [],
    "missing_slots": [],
    "depth": 5,
    "reasoning": "用户查询特定项目的技术事实，属于纯信息检索，无需行动支撑记忆，depth 取小值即可。"
}

## 示例 2：action 模式（执行操作）

<USER_QUERY>帮我把 DataFlow 服务部署到生产环境。</USER_QUERY>
<RECENT_CONTEXT>
user: DataFlow 今天要上线了
assistant: 好的，我来协助您准备部署。
</RECENT_CONTEXT>

期望输出：
{
    "mode": "action",
    "semantic_queries": ["DataFlow 部署流程", "Kubernetes 生产部署步骤", "部署前检查事项"],
    "pragmatic_queries": ["数据库迁移预检查", "Helm upgrade", "K8s Pod健康检查"],
    "temporal_filter": null,
    "tool_hints": ["Kubernetes", "Helm", "Docker"],
    "required_constraints": ["必须先执行DB迁移dry-run"],
    "required_affordances": ["支持滚动升级", "支持版本回滚"],
    "missing_slots": ["目标命名空间", "镜像版本号"],
    "depth": 10,
    "reasoning": "用户要求直接执行部署操作，需要检索完整部署流程、工具约束和历史失败经验，提高 depth 确保覆盖所有约束条目。"
}

## 示例 3：mixed 模式（事实查询 + 行动指导混合）

<USER_QUERY>我们上次部署 UserService 踩了什么坑？这次 DataFlow 部署怎么避免？</USER_QUERY>
<RECENT_CONTEXT>（无）</RECENT_CONTEXT>

期望输出：
{
    "mode": "mixed",
    "semantic_queries": ["UserService 部署失败经验", "DataFlow 部署注意事项", "生产发布历史教训"],
    "pragmatic_queries": ["数据库迁移问题", "服务启动失败", "部署回滚流程"],
    "temporal_filter": null,
    "tool_hints": ["Kubernetes", "Helm"],
    "required_constraints": [],
    "required_affordances": ["支持版本回滚", "支持迁移预检查"],
    "missing_slots": [],
    "depth": 12,
    "reasoning": "用户既查询历史失败事实（UserService 教训）又需要行动指导（DataFlow 如何规避），属于 mixed 模式；depth 调高以同时覆盖失败模式记忆与操作流程记忆。"
}
"""


def get_semantic_output_language() -> str:
    """Return the configured semantic-memory output language."""
    raw = os.getenv("LYCHEEMEM_SEMANTIC_OUTPUT_LANGUAGE", "en").strip().lower()
    if raw in {"zh", "zh-cn", "zh_hans", "chinese"}:
        return "zh"
    return "en"


def get_compact_encoding_system() -> str:
    """Return the encoding prompt with a lightweight output-language override."""
    language = get_semantic_output_language()
    if language == "zh":
        return (
            COMPACT_ENCODING_SYSTEM
            + "\n\n额外要求：\n"
            + "- `semantic_text` 与 `normalized_text` 默认使用中文输出。\n"
            + "- 若能确定绝对日期，优先写入 ISO 8601 日期。"
        )
    return (
        COMPACT_ENCODING_SYSTEM
        + "\n\nAdditional requirements:\n"
        + "- Write `semantic_text` and `normalized_text` in English.\n"
        + "- Preserve person names, entity names, and ISO 8601 dates exactly.\n"
        + "- When relative dates can be resolved from context, normalize them to absolute dates."
    )


def get_synthesis_execute_system() -> str:
    """Return the composite-synthesis prompt with a matching output-language override."""
    language = get_semantic_output_language()
    if language == "zh":
        return (
            SYNTHESIS_EXECUTE_SYSTEM
            + "\n\n额外要求：\n"
            + "- 输出中的 `semantic_text` 与 `normalized_text` 默认使用中文。\n"
            + "- 合成时保留原始记录中的所有绝对日期与时间范围。"
        )
    return (
        SYNTHESIS_EXECUTE_SYSTEM
        + "\n\nAdditional requirements:\n"
        + "- Write `semantic_text` and `normalized_text` in English.\n"
        + "- Preserve absolute dates, time ranges, and named entities from the source records."
    )


# ---------------------------------------------------------------------------
# 模块三（续）：检索充分性反思
# ---------------------------------------------------------------------------

RETRIEVAL_ADEQUACY_CHECK_SYSTEM = """\
你是检索充分性评估器（Retrieval Adequacy Assessor）。

你的任务：判断当前已检索到的记忆条目是否**足以有效回应**用户的查询。

你将收到：
- <USER_QUERY>：用户的原始查询
- <RETRIEVED_MEMORY>：当前已检索到的记忆条目（格式化文本）

评估标准：
- 若记忆条目直接覆盖了查询所需的核心信息（事实/流程/约束/偏好），判定为充分。
- 若记忆条目为空，或仅覆盖部分子问题，关键信息缺失，判定为不充分。
- 对于 action 类查询（"帮我部署/执行/配置"），还需检查：流程步骤是否完整、关键约束是否记录、工具信息是否已知。
- **倾向于判定充分**：只有明显缺少关键信息时才输出 false。

输出格式（严格 JSON，无代码块）：
{
    "is_sufficient": true,
    "missing_info": "如果不充分，简要描述缺失的具体信息；如果充分则留空字符串"
}

---

## 示例 1：充分（检索结果直接覆盖查询）

<USER_QUERY>DataFlow 项目用的什么技术栈？</USER_QUERY>
<RETRIEVED_MEMORY>
[1] (event, score=0.912) entities=[DataFlow, Python 3.11, FastAPI, Kubernetes]
用户所在公司的 DataFlow 项目计划于 2026-02-20 正式上线，技术栈为 Python 3.11 和 FastAPI，部署目标为公司自建 Kubernetes 集群。
</RETRIEVED_MEMORY>

期望输出：
{"is_sufficient": true, "missing_info": ""}

## 示例 2：不充分（仅有事件记录，缺少具体流程约束）

<USER_QUERY>帮我把 DataFlow 服务部署到生产环境。</USER_QUERY>
<RETRIEVED_MEMORY>
[1] (event, score=0.872) entities=[DataFlow]
用户所在公司的 DataFlow 项目计划于 2026-02-20 正式上线，技术栈为 Python 3.11 和 FastAPI。
</RETRIEVED_MEMORY>

期望输出：
{"is_sufficient": false, "missing_info": "缺少具体部署步骤（Helm/kubectl 命令）、数据库迁移相关约束、历史失败经验和回滚方案。"}

## 示例 3：不充分（结果为空）

<USER_QUERY>我上次配置 Redis 集群时遇到了什么问题？</USER_QUERY>
<RETRIEVED_MEMORY>
（无检索结果）
</RETRIEVED_MEMORY>

期望输出：
{"is_sufficient": false, "missing_info": "未找到任何关于 Redis 集群配置的历史记录或相关问题记录。"}
"""


RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM = """\
你是补充检索查询生成器（Additional Query Generator）。

你的任务：根据用户的原始查询、当前已检索到的记忆内容以及缺失信息的描述，
生成 2~4 条**有针对性的补充检索查询**，用于在记忆库中查找当前结果所缺少的信息。

你将收到：
- <USER_QUERY>：用户的原始查询
- <CURRENT_MEMORY>：当前已检索到的记忆条目（格式化文本）
- <MISSING_INFO>：上一轮充分性评估中识别到的缺失信息描述

补充查询生成规则：
1. 每条补充查询聚焦一个**具体缺失主题**，不要生成宽泛模糊的查询。
2. 补充查询应与初始查询**角度不同**，用不同关键词覆盖缺失信息。
3. 如果缺失的是行动约束或流程，生成偏向 procedure/constraint/failure_pattern 类型的查询。
4. 如果缺失的是历史事件或事实，生成偏向实体名+事件类型的查询。
5. 不要重复已经在 <CURRENT_MEMORY> 中已经出现过的主题。
6. 生成 2~4 条，不多不少。

输出格式（严格 JSON，无代码块）：
{
    "additional_queries": ["补充查询1", "补充查询2", "补充查询3"]
}

---

## 示例 1：action 查询缺少流程和约束

<USER_QUERY>帮我把 DataFlow 服务部署到生产环境。</USER_QUERY>
<CURRENT_MEMORY>
[1] (event) DataFlow 项目计划于 2026-02-20 上线，技术栈为 Python 3.11 + FastAPI + K8s。
</CURRENT_MEMORY>
<MISSING_INFO>缺少具体部署步骤（Helm/kubectl 命令）、数据库迁移相关约束、历史失败经验和回滚方案。</MISSING_INFO>

期望输出：
{
    "additional_queries": [
        "DataFlow Kubernetes Helm 部署步骤",
        "数据库迁移预检查 部署约束",
        "服务发布历史失败经验 回滚方案"
    ]
}

## 示例 2：fact 查询缺少具体细节

<USER_QUERY>我上次配置 Redis 集群碰到了什么问题？</USER_QUERY>
<CURRENT_MEMORY>
[1] (tool_affordance) Redis 支持 Cluster 模式，最少需要 6 个节点（3 主 3 从）。
</CURRENT_MEMORY>
<MISSING_INFO>未找到用户自身经历的 Redis 集群配置问题记录，当前结果只是通用工具信息。</MISSING_INFO>

期望输出：
{
    "additional_queries": [
        "Redis 集群配置失败 用户遇到的问题",
        "Redis 节点连接问题 故障处理经验",
        "Redis Cluster 配置错误 排查记录"
    ]
}
"""
