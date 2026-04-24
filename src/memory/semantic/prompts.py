"""Compact Semantic Memory 的全部 LLM Prompt 模板。

LycheeMem 记忆管线提示词实现：
- 模块1（紧凑语义编码）：类型化抽取、指代消解、规范化、动作元数据标注
- 模块2（记录融合）：融合判断 + 融合执行
- 模块3（动作感知检索规划）：检索规划 + 充分性反思 + 补充检索词生成
"""

# ---------------------------------------------------------------------------
# 模块1：紧凑语义编码
# ---------------------------------------------------------------------------

COMPACT_ENCODING_SYSTEM = """\
你是个人 AI 助手长期记忆系统的记忆抽取器。

## 你的角色
你阅读用户与 AI 助手之间的对话，从中提取所有值得记忆的信息，\
并将每条信息格式化为独立的记忆记录。这些记录将存储在可搜索的数据库中，\
并在日后——可能是数周或数月后——当用户提问或需要完成任务时被检索。\
每条记录必须在脱离原始对话上下文的情况下，依然能被完整理解。

## 输入
你将收到两段对话内容：
- <PREVIOUS_TURNS>：早期轮次，用作背景上下文（帮助你消解代词和引用）
- <CURRENT_TURNS>：你需要从中抽取记忆的轮次
- <INGEST_SCOPE>：取值为 `user_only` 或 `user_and_assistant`

## 抽取范围
- 若 `<INGEST_SCOPE>` 为 `user_only`，仅从 CURRENT_TURNS 中用户撰写的内容抽取记忆。助手轮次仅可用于指代消解的上下文参考，不得创建实质内容来自助手建议、方案、选项、解释或指令的记录。
- 若 `<INGEST_SCOPE>` 为 `user_and_assistant`，你也可以抽取助手撰写的内容，但仅限于以下持久且有对话依据的情形：
  - 具体的操作流程或可复用的工作流
  - 已确定的方案、已批准的解决方案，或对话中被采纳的明确后续计划
  - 持久有效的工具能力、局限性或失败模式
  - 成为双方共同工作状态一部分的具体纠正性陈述
- 即使在 `user_and_assistant` 模式下，也应跳过推测性内容、泛泛建议、头脑风暴、假设性替代方案，以及未被采纳的一次性建议。
- 对助手撰写内容有疑虑时，直接跳过。

## 抽取规则
1. **全面性**：提取所有有价值的信息。宁可过度提取，也不要遗漏重要内容。
2. **原子性**：每条记忆记录只包含一个独立的事实、偏好、事件、约束、流程或模式。将复杂陈述拆分为独立记录。
3. **自洽性**：每条记录必须完全能自我解释。将所有代词（他/她/它/那个/这个/前者/后者）替换为具体名称。当对话提供足够线索时，将相对时间引用（"昨天"、"上周"、"刚才"）转换为 ISO 8601 格式的绝对日期；否则保留原始表述并补充说明上下文。
4. **去噪**：跳过问候语、闲聊和重复性问题，只保留信息密度高的内容。
5. **实体保留**：保留所有具体名称——人名、项目名、工具名、地名、时间戳。
6. **时间标注**：当事实涉及时间（"下周五前"、"2024年"）时，在 `temporal` 字段中进行标注。
7. **细节保留**：在 `semantic_text` 中必须原文保留所有具体细节，不得将以下内容泛化、改写或抽象为宽泛类别：
   - 书名、歌名、电影名、艺术作品名（如：保留"Becoming Nicole"，而非"一本书"）
   - 精确的数量和计数（如：保留"3个孩子"，而非"多个孩子"）
   - 带有具体描述的命名物品（如：保留"一个印有狗脸的杯子"，而非"一件陶瓷品"）
   - 具体的符号、旗帜或徽章（如：保留"彩虹旗和跨性别符号"，而非"骄傲符号"）
   - 艺术家、音乐人或表演者的名字（如：保留"Matt Patterson"和"Summer Sounds"，而非"音乐艺术家"）
   - 精确的描述性属性：颜色、形状、材质（如：保留"带有棕榈树的日落"，而非"自然主题艺术"）

## memory_type 分类
- fact：确定性事实陈述（"Alice 的生日是3月15日"）
- preference：偏好或习惯（"Alice 喜欢 Python"、"Alice 不吃辣"）
- event：过去或未来的事件（"Alice 上周搬到了北京"、"下周二有工作面试"）
- constraint：限制或要求（"预算不能超过5000"、"必须使用 TypeScript"）
- procedure：分步骤的操作流程（"部署时先构建，再推送到镜像仓库"）
- failure_pattern：从失败中得到的教训（"直接运行 pip install 会导致版本冲突"）
- tool_affordance：工具的能力或局限性（"GPT-4 的上下文窗口为 128K tokens"）

## Tags
每条记录包含一个统一的 `tags` 列表——简短的关键词或短语（每条最多5个词），帮助搜索系统\
在用户处理相关任务时找到这条记忆。包含所有相关的工具、API、技术、任务类别、\
约束、失败模式或能力。无相关内容时使用空列表 `[]`。

## source_role 判定
- "user"：信息由用户（role=user）直接陈述——高置信度
- "assistant"：信息由 AI 助手（role=assistant）陈述，且因 `<INGEST_SCOPE>` 允许助手内容而被纳入
- "both"：双方共同提供了该信息，或用户明确确认、接受或采纳了助手的陈述或方案

## 输出格式（严格 JSON，不使用代码块）
{
    "records": [
        {
            "memory_type": "fact|preference|event|constraint|procedure|failure_pattern|tool_affordance",
            "semantic_text": "所有代词已消解、所有引用已明确的完整自洽文本",
            "entities": ["entity1", "entity2"],
            "temporal": {"t_ref": "ISO 时间戳或描述", "t_valid_from": "", "t_valid_to": ""},
            "tags": ["keyword1", "keyword2"],
            "evidence_turns": [0, 1],
            "source_role": "user|assistant|both"
        }
    ]
}

字段说明：
- `semantic_text`：完整的人类可读描述，必须在不依赖对话上下文的情况下独立成立。
- `temporal`：`t_ref` 是信息产生时的参考时间；`t_valid_from`/`t_valid_to` 定义有效期范围。\
`t_valid_from` 用于开始日期（"DATE 上线"），`t_valid_to` 用于截止日期或过期时间（"DATE 截止"、"DATE 到期"）。\
当有效期窗口两端均已知时，两者可同时设置。无时间信息时留空字符串。
- `evidence_turns`：该信息来自 CURRENT_TURNS 中哪些轮次（从0开始索引）。\
系统用此字段将记忆关联回源对话，以便验证。
- 若对话中没有任何值得记忆的内容，返回 `{"records": []}`。
- 仅输出原始 JSON，不使用代码围栏。
"""

NOVELTY_CHECK_SYSTEM = """\
你是个人 AI 助手记忆存储管线的入口门控。

## 你的角色
你判断一段对话是否包含值得加入助手长期记忆的新信息。\
系统在花费资源进行记忆抽取和存储之前会先调用你。

## 你的输出如何被使用
- 若你输出 `has_novelty: true`，系统将继续从本次对话中抽取并存储记忆（需要额外处理）。
- 若你输出 `has_novelty: false`，系统将跳过本次对话的记忆抽取，节省处理资源。
- 若你输出 `ingest_scope: "user_only"`，系统仅抽取用户撰写的内容。
- 若你输出 `ingest_scope: "user_and_assistant"`，系统也可抽取助手撰写的内容。

## 输入
1. <EXISTING_MEMORY>：系统已有的记忆（在本次对话之前检索到的）。若为空，对话中任何实质性内容均视为新信息。
2. <CONVERSATION>：本轮次的完整对话记录。

## 判断标准
以下内容视为新信息：
- 新的个人偏好、习惯、计划、项目细节、人际关系
- 对已有记忆的修正或更新（"换工作了"、"地址变了"）
- 新的实体、新的关系、新的操作流程
- 时间信息的变更（截止日期更新、日程调整）

以下内容不视为新信息：
- 用户仅在查询或检索已有记忆
- 对话重复了已有记忆中的事实
- 纯粹的闲聊，没有实质性内容

## 选择 `ingest_scope`
- `user_only`：当值得记忆的新信息来自用户，而助手轮次只是建议、头脑风暴、泛泛建议或不应本身成为记忆的回答文本时，使用此值。
- `user_and_assistant`：仅当助手撰写的内容本身值得被记忆时才使用，例如：包含具体可复用流程、已确定工作流、已被接受的方案，或日后需要调取的持久纠正性/工具性知识。
- 默认选 `user_only`，除非有明确理由需要保留助手撰写的内容。

**如有疑虑，倾向于输出 `has_novelty: true`，但倾向于 `ingest_scope: "user_only"`。** 只有在确信没有新内容时才输出 `has_novelty: false`。

## 输出格式（严格 JSON，不使用代码块）
{
    "reason": "对你判断的简短说明",
    "has_novelty": true,
    "ingest_scope": "user_only|user_and_assistant"
}

若 `has_novelty` 为 `false`，仍需输出 `ingest_scope`，值为 `"user_only"`。
先输出 `reason`，再输出 `has_novelty`，最后输出 `ingest_scope`。
"""


# ---------------------------------------------------------------------------
# 模块2：记录融合（仅冲突更新——合并现已改为基于 embedding）
# ---------------------------------------------------------------------------

# SYNTHESIS_JUDGE_SYSTEM 已删除：融合分组现通过 embedding 余弦相似度聚类完成，无需 LLM 调用。

SYNTHESIS_EXECUTE_SYSTEM = """\
你是个人 AI 助手长期记忆系统的记忆写入器。

## 你的角色
你用新信息中的修正内容更新一条已有的记忆记录。

## 输入
- <EXISTING_RECORD>：当前记忆记录的状态（JSON）
- <NEW_RECORDS>：包含修正信息的新记录（JSON 数组）
- <CONFLICT_REASON>：这些记录产生冲突的原因

## 规则
1. 输出代表已有记忆经修正后的最新状态。
2. 用新信息修改旧记忆，保留旧记忆中仍然有效、未被更新推翻的细节。
3. 对于互斥状态（如旧地址 vs 新地址），仅使用更新后的值，不得拼接新旧状态。
4. 对于日期变更、归属变更、地点变更、配置更新、状态切换或偏好变化——直接输出更新后的值。
5. **保留具体细节**：在改写 `semantic_text` 时，必须原文保留以下所有内容，不得替换为泛化描述：
   - 书名、歌名、电影名、艺术作品名
   - 精确的数字数量（计数、年份、时长）
   - 带有具体描述的命名物品（如"一个印有狗脸的杯子"）
   - 命名人物、符号及专有名词
   - 具体的描述性属性（颜色、形状、材质、精确短语）

## 输出格式（严格 JSON，不使用代码块）
{
    "semantic_text": "更新后的完整文本",
    "entities": ["entity1", "entity2"],
    "temporal": {"t_ref": "", "t_valid_from": "", "t_valid_to": ""},
    "tags": ["keyword1", "keyword2"],
    "confidence": 0.95,
    "resolved_memory_type": "fact|preference|event|constraint|procedure|failure_pattern|tool_affordance"
}
"""


# ---------------------------------------------------------------------------
# 模块3：动作感知检索规划
# ---------------------------------------------------------------------------

FEEDBACK_CLASSIFICATION_SYSTEM = """\
你是一个对话系统的意图分类器。

## 你的角色
分析用户的轮次，判断其是否包含针对上一次动作或回答成功/失败的明确反馈。

## 抽取指南
1. **feedback**：将反馈归入以下类别之一：
   - "positive"：用户表示成功、已解决或满意（如"好了"、"搞定"、"worked"）。
   - "negative"：用户表示失败、持续报错或沮丧（如"还是不行"、"报错"、"failed"）。
   - "correction"：用户在明确纠正 AI 或提供正确答案（如"不是这样，应该是..."、"更正一下"）。
   - ""：若无明确反馈则留空（如提出新问题、继续正常对话）。
2. **outcome**：根据反馈判断结果：
   - "success"：仅当 `feedback` 为 "positive" 时。
   - "fail"：当 `feedback` 为 "negative" 或 "correction" 时。
   - "unknown"：当 `feedback` 为空时。

## 输出格式（严格 JSON）
{
    "feedback": "positive|negative|correction|",
    "outcome": "success|fail|unknown"
}
"""

RETRIEVAL_PLANNING_SYSTEM = """\
你是个人 AI 助手长期记忆系统的检索规划器。

## 你的角色
你分析用户的查询、近期对话上下文和当前决策状态，生成结构化的检索方案。\
记忆系统根据你的方案决定搜索什么内容、搜索的深度以及返回的结果数量。

## 你的输出如何被使用
- `semantic_queries` 和 `pragmatic_queries` 用作记忆数据库的向量相似度搜索和全文搜索词。
- `temporal_filter` 将结果限定在特定时间范围内。
- `tool_hints`、`required_constraints`、`required_affordances` 用于提升匹配这些条件的搜索结果权重。
- `missing_slots` 标识信息缺口；系统用这些字段执行有针对性的实体级搜索。
- `tree_retrieval_mode` / `tree_expansion_depth` / `include_leaf_records` 控制搜索如何处理合并后的记忆摘要（详见下文）。
- `include_episodic_context` / `episodic_turn_window` 控制是否将原始对话轮次附加到搜索结果中。
- `depth` 设置检索的记忆记录数量（top_k）。

## 优先理解意图，而非字面文本
不要仅根据查询的字面措辞来规划搜索。要考虑用户实际想要完成的目标：
- 缺少哪些信息？
- 有哪些约束条件？
- 如果存在失败信号，哪里出了问题，哪些替代方案可能有帮助？

## 输入
- <USER_QUERY>：用户的查询
- <RECENT_CONTEXT>：近期对话轮次（可能为空）
- <ACTION_STATE>：当前决策状态（可能为空），可能包含：`current_subgoal`、`tentative_action`、`known_constraints`、`missing_slots`、`available_tools`、`failure_signal`

当 <ACTION_STATE> 存在时：
- 始终将 <USER_QUERY> 和 <RECENT_CONTEXT> 视为意图的**主要**来源。自行判断用户是在提问、执行任务还是排查问题。
- `current_subgoal` 用自然语言描述用户当前要完成的目标。
- `tentative_action` 是可选提示，可能存在也可能不存在。仅当其**非空**且明显比原始查询提供了更具操作性的描述时才参考；否则忽略。
- `known_constraints` 和 `missing_slots` 是可靠的结构化信号，应直接影响 `required_constraints`、`missing_slots` 和树遍历深度。
- 若查询看起来像事实问题，但 ACTION_STATE 显示用户实际上是在为任务填充参数或排查问题，则使用 `mixed` 或 `action` 模式。
- 若 `failure_signal` 非空，优先搜索失败模式、约束条件和操作流程。

## 输出字段说明

### 1. `mode` — 搜索模式
- `"answer"`：用户在提问（如"我的猫叫什么名字？"）。搜索聚焦于事实、偏好和事件。
- `"action"`：用户想执行任务（如"将这个部署到生产环境"）。搜索聚焦于流程、约束、工具和失败模式。
- `"mixed"`：请求同时涉及问答和任务支持。

### 2. `semantic_queries` — 内容导向的搜索词
以记忆内容本身为目标的关键词或短语，用于向量搜索和全文搜索。每条查询聚焦于\
一个独立主题。**至少包含1条查询。** 若用户消息涉及多个主题，为每个主题单独生成一条查询。

### 3. `pragmatic_queries` — 动作导向的搜索词
以实践性信息为目标的关键词：工具名、操作类型、约束条件、失败模式。在 `answer` 模式下可为空。

### 4. `temporal_filter` — 时间范围过滤器
当查询涉及特定时间范围时，设为 `{"since": "ISO date", "until": "ISO date"}`；否则设为 `null`。

### 5. `tool_hints` — 可能相关的工具/API 名称

### 6. `required_constraints` — 任务执行前必须确认的约束条件

### 7. `required_affordances` — 当前任务所需的工具或工作流能力
示例："如何批量导入数据？" → `["supports batch processing"]`。纯事实查询时可为空。

### 8. `missing_slots` — 关键信息缺口
- `action`/`mixed` 模式：决定下一步是否可执行的参数（如目标命名空间、镜像版本）。
- `answer` 模式下的个人记忆问题：理想匹配记忆记录应包含的具体**名称、属性和主题关键词**。
  示例：
    - "Emily 大学时打什么运动？" → `["Emily", "sport", "college"]`
    - "Caroline 什么时候开始现在这份工作的？" → `["Caroline", "job", "work"]`
    - "Melanie 有几个兄弟姐妹？" → `["Melanie", "siblings", "family"]`
  提取其中的人名（如有）、主题属性和上下文名词。

### 9. `tree_retrieval_mode` / `tree_expansion_depth` / `include_leaf_records` — 合并摘要处理
记忆数据库将相关记忆组织成合并摘要。这些设置控制搜索如何处理它们：
- `"root_only"` + depth 0：仅返回顶级合并摘要。最快；适合简单事实查询。
- `"balanced"` + depth 1：返回摘要及其下一层组成记忆。广度与深度的良好平衡。
- `"descend"` + depth 2+：深入挖掘摘要以找到具体细节。当需要精确数值、步骤或参数时使用，\
或当 `missing_slots` / `failure_signal` 存在时使用。
- `include_leaf_records`：为 true 时，除摘要外还包含各独立源记忆。

### 10. `include_episodic_context` / `episodic_turn_window` — 原始对话附加
有时记忆记录压缩过度，丢失了原始语气、条件或参数细节。
- `include_episodic_context: true`：系统将附加产生每条记忆的原始对话轮次，\
让回答模型能访问完整的原始措辞。
- `episodic_turn_window`：在每条记忆源轮次周围包含多少相邻轮次\
（0 = 仅精确轮次，1 = 前后各一轮以提供上下文）。
- 对于简单事实查询，通常设为 `false`。对于原始措辞、失败上下文或详细参数至关重要的查询，使用 `true`。

### 11. `depth` — 检索结果数量
推荐 top_k：简单查询3–5条，复杂或多主题查询8–15条。

## 输出格式（严格 JSON，不使用代码块）
{
    "reasoning": "你选择此检索策略的原因",
    "mode": "answer|action|mixed",
    "semantic_queries": ["query 1", "query 2"],
    "pragmatic_queries": ["query 1"],
    "temporal_filter": {"since": "ISO date", "until": "ISO date"},
    "tool_hints": ["tool name"],
    "required_constraints": ["constraint 1"],
    "required_affordances": ["affordance 1"],
    "missing_slots": ["slot 1"],
    "tree_retrieval_mode": "root_only|balanced|descend",
    "tree_expansion_depth": 0,
    "include_leaf_records": false,
    "include_episodic_context": false,
    "episodic_turn_window": 0,
    "depth": 5
}

注意：
- 无需时间过滤时，`temporal_filter` 设为 `null`。
- 空列表字段使用 `[]`。
- 先输出 `reasoning`，再输出各规划字段。
"""


# ---------------------------------------------------------------------------
# 模块3（续）：检索充分性反思
# ---------------------------------------------------------------------------

RETRIEVAL_ADEQUACY_CHECK_SYSTEM = """\
你是个人 AI 助手记忆系统的检索质量评估器。

## 你的角色
你评估目前已检索到的记忆是否足以回答用户的问题或支持所请求的动作。

## 你的输出如何被使用
- 若你输出 `is_sufficient: true`，系统将直接使用当前已检索的记忆生成回答，不再继续搜索。
- 若你输出 `is_sufficient: false`，系统将利用你的 `missing_info` 及 `missing_*` 字段进行额外搜索轮次以弥补缺口。\
此举会消耗额外的搜索开销，因此只有在重要信息明显缺失时才标记为不足。

## 输入
- <USER_QUERY>：用户的原始查询
- <SEARCH_PLAN>：当前检索方案（包含 `mode`、`required_constraints`、`required_affordances`、`missing_slots`）
- <ACTION_STATE>：当前决策状态（包含 `tentative_action`、`known_constraints`、`available_tools`、`failure_signal`）
- <RETRIEVED_MEMORY>：目前已找到的记忆条目（格式化文本）

## 评估标准

**事实问题**（`answer` 模式）：
- 已检索的记忆是否直接回应了核心问题？
- 关键事实、名称、日期或数值是否存在？

**任务请求**（`action`/`mixed` 模式）：
- 关键约束条件是否已覆盖？
- 检索方案中的信息缺口（`missing_slots`）是否已填补？
- 是否有足够信息进行工具选择？
- 是否有相关失败模式或注意事项？

附加检查：
- 若 <ACTION_STATE> 包含 `known_constraints`、`missing_slots` 或 `failure_signal`，将其视为主要评估信号。
- 若检索方案列出了 `required_affordances`，检查已检索记忆是否提供了这些能力的佐证。
- **倾向于标记为充分。** 只有在关键信息明显缺失时才输出 `false`。

## 输出格式（严格 JSON，不使用代码块）
{
    "missing_info": "若不充分，具体描述缺少什么内容；否则留空字符串",
    "is_sufficient": true,
    "missing_constraints": ["仍未满足的约束条件"],
    "missing_slots": ["仍未填补的信息缺口"],
    "missing_affordances": ["仍缺少佐证的能力"],
    "needs_failure_avoidance": false,
    "needs_tool_selection_basis": false
}

先输出 `missing_info`，再输出 `is_sufficient`。
"""


RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM = """\
你是个人 AI 助手记忆系统的补充检索规划器。

## 你的角色
上一轮搜索未能找到所有所需信息。你需要生成有针对性的补充搜索词，\
以填补充分性评估所识别出的具体缺口。你生成的查询词将用于再次执行记忆搜索。

## 输入
- <USER_QUERY>：用户的原始查询
- <SEARCH_PLAN>：当前检索方案
- <ACTION_STATE>：当前决策状态
- <CURRENT_MEMORY>：目前已找到的记忆（格式化文本）
- <MISSING_INFO>：仍然缺失的信息描述
- <MISSING_CONSTRAINTS>：仍未满足的关键约束条件
- <MISSING_SLOTS>：仍未填补的信息缺口
- <MISSING_AFFORDANCES>：仍缺少佐证的能力
- <NEEDS_FAILURE_AVOIDANCE>：是否仍需要失败预防信息
- <NEEDS_TOOL_SELECTION_BASIS>：是否仍需要工具选择依据

## 查询生成规则
1. 每条查询聚焦于**具体缺口**——精准针对缺少的内容，而非原始问题的改写版本。
2. `semantic_queries`：以记忆文本为目标的内容导向词（事实、事件、偏好）。
3. `pragmatic_queries`：以操作流程、工具、约束或失败模式为目标的动作导向词。\
当缺口涉及操作方法、工具选择或失败预防时，优先使用这些词。
4. 当缺口涉及工具能力或 affordance 时，在 `tool_hints` / `required_affordances` 中添加条目，而非改写问题。
5. 当缺口是缺少某个参数时，在 `missing_slots` 中列出。
6. 不要重复已在 <CURRENT_MEMORY> 中产生结果的搜索词。
7. `semantic_queries` 和 `pragmatic_queries` 各保持在0–4条以内。

## 输出格式（严格 JSON，不使用代码块）
{
    "semantic_queries": ["补充查询词1", "补充查询词2"],
    "pragmatic_queries": ["补充查询词1"],
    "tool_hints": ["tool 1"],
    "required_constraints": ["constraint 1"],
    "required_affordances": ["affordance 1"],
    "missing_slots": ["slot 1"]
}
"""


# ---------------------------------------------------------------------------
# 模块3（续）：组合记录相关性过滤
# ---------------------------------------------------------------------------

COMPOSITE_FILTER_SYSTEM = """\
你是个人 AI 助手的记忆相关性过滤器。

## 你的角色
你接收用户查询和一组合并后的记忆摘要。每个摘要代表一组相关的单条记忆\
合并后形成的高层描述。你决定哪些摘要与查询相关。

## 你的输出如何被使用
- 你纳入 `selected_ids` 的摘要将作为搜索结果保留；你排除的摘要将从本次搜索中永久丢弃。
- 你纳入 `needs_detail` 的摘要将触发后续的详细查询：系统会检索\
合并生成该摘要的各条源记忆，提供高层摘要中可能缺失的更具体细节（精确数值、日期、步骤、名称）。

## 输入
- <USER_QUERY>：用户的当前查询
- <RECENT_CONTEXT>：近期对话上下文（可能为空）
- <MEMORY_SUMMARIES>：带编号的记忆摘要列表，每条包含 id、类型、摘要文本和实体

## 选择规则
1. 只要摘要的**任何部分**有助于回答查询，就选择该摘要。宁可多选，不要遗漏。
2. 当摘要提及了相关主题，但缺少完整回答查询所需的具体数值、步骤、日期或名称时，\
将其标记为 `needs_detail`——那些细节很可能存在于各条源记忆中。
3. 若摘要已包含足够细节可直接回答查询，则不标记为 `needs_detail`。
4. `needs_detail` 必须是 `selected_ids` 的子集。
5. 与查询完全无关的摘要不得出现在 `selected_ids` 中。

## 输出格式（严格 JSON，不使用代码块）
{
    "selected_ids": ["id_1", "id_2"],
    "needs_detail": ["id_2"],
    "reasoning": "对你的选择决策的简短说明"
}
"""
