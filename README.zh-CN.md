# LycheeMem

**无训练的智能代理认知记忆框架**

LycheeMem 是一套为长期推理AI代理设计的认知记忆系统，提供持久化、结构化、体系统一的时间感知记忆，无需任何微调。它按照人脑使用记忆的方式组织信息 —— 区分你**记得发生了什么**与你**已经知晓的内容** —— 并通过多阶段推理管道在推理时刻将这些记忆激活。

---

## 记忆架构

LycheeMem 将记忆组织为三个相辅相成的存储库，镜像了神经科学中的认知记忆分类法：

```
┌─────────────────────────────────────────────────────────────────────┐
│                       LycheeMem 记忆系统                            │
│                                                                     │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌───────────────┐  │
│  │   工作记忆       │  │    语义记忆         │  │  程序记忆     │  │
│  │  （情景记忆）    │  │  （知识图谱）       │  │  （技能库）   │  │
│  │                  │  │                     │  │               │  │
│  │  • 会话轮次      │  │  • 实体节点         │  │  • 技能条目   │  │
│  │  • 信息摘要      │  │  • 双时态事实       │  │  • HyDE 检索  │  │
│  │  • Token 预算    │  │  • 社区聚类         │  │  • 向量匹配   │  │
│  │  • 智能压缩      │  │  • 情景溯源         │  │               │  │
│  └──────────────────┘  └─────────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 工作记忆

工作记忆窗口保存会话期间的活跃对话上下文。它运作于一个**双阈值 Token 预算**机制下：

- **预警阈值（70%）** —— 触发异步后台预压缩；当前请求**不被阻塞**。
- **阻塞阈值（90%）** —— 管道暂停，将较早的轮次刷入压缩摘要后继续。

压缩产生两种形式的历史：*摘要锚点*（旧内容，凝聚化）+ *原始最近轮次*（最后 N 轮，逐字保留）。两者一起作为对话上下文传递给下游阶段。

### 语义记忆 —— Graphiti 知识图谱

知识图谱以**Graphiti 风格的双时态图**实现，后端存储在 Neo4j。它用四类节点存储世界状态：

| 节点 | 职责 |
|------|------|
| `Episode`（情景） | 单次对话轮次；所有事实都可溯源到源情景 |
| `Entity`（实体） | 命名实体（人、项目、地点、概念等） |
| `Fact`（事实） | 两个实体之间的类型化关系，带时间有效期和事务元数据 |
| `Community`（社区） | 强关联实体的聚类，携带定期刷新的摘要 |

#### 双时态模型

每条 `Fact` 都携带四个时间戳，分离**现实世界中的真实期间**与**系统何时获知**：

```
t_valid_from / t_valid_to   →  有效时间   （真实世界事实的有效区间）
t_tx_created / t_tx_expired →  事务时间   （系统端记录的生命周期）
```

这使图谱可以正确回答"用户上月的家庭住址是什么？"这样的查询，即使地址已更改；也能区分真正矛盾的事实与在不同时期为真的事实。

#### 图谱检索流程

检索结合三种互补的信号源：

1. **BM25 全文搜索** —— 针对 `Entity.name` 和 `Fact.fact_text` 的关键词级召回，经由 Neo4j 全文索引。
2. **BFS 图遍历** —— 从会话最近的情景节点开始，按可配置深度向外扩展，即使没有匹配关键词也能浮现语义关联的事实。
3. **向量 ANN 搜索** —— 对 `Entity.embedding` 向量索引的近似最近邻查询（维度和相似函数均可配置）。

检索后，候选结果通过三个列表的**倒数排名融合（RRF）**进行重排。可选地，一个**交叉编码器重排器**（驱动方式与管道内已有的 LLM 适配器相同，无额外厂商 SDK）精化前 N 个结果，随后进行**最大边际相关性（MMR）**多样化以避免在最终提示中出现近重复上下文。

#### 社区检测

后台扫描每隔 N 个情景（默认 50 个）全局运行一次 `refresh_all_communities()`。社区摘要被包含在图谱搜索结果中，即使没有具体事实匹配查询也能提供宽泛的语境框架。

### 程序记忆 —— 技能库

技能库保存可复用的**操作方法**知识，每个技能条目携带：

- **意图** —— 简短描述该技能的功能。
- **`doc_markdown`** —— 完整 Markdown 文档，描述步骤、命令、参数和注意事项。
- **向量** —— 意图文本的密集向量，用于相似度搜索。
- **元数据** —— 使用计数、最后使用时间戳、前置条件。

技能检索使用 **HyDE（假设性文档嵌入）**：查询首先被 LLM 展开成*假设的理想回答*，然后对该草稿文本嵌入以产生查询向量，该向量能很好地匹配存储的流程描述，即使用户的原始表述模糊。

---

## 管道架构

每个请求经过固定的五阶段序列。四个是 LangGraph 管道中的同步阶段；一个是后台后处理任务。

```
开始
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. WMManager         Token 预算检查 + 压缩/渲染                │
│     ↓                                                           │
│  2. SearchCoordinator  多查询 → 图谱 + 技能 检索                │
│     ↓                                                           │
│  3. SynthesizerAgent   LLM-as-Judge 评分 + 上下文融合           │
│     ↓                                                           │
│  4. ReasoningAgent     最终回答生成                             │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
结束  →  asyncio.create_task( ConsolidatorAgent )   [后台运行]
```

### 阶段 1 —— WMManager

规则型 Agent（无 LLM 提示词）。将用户轮次追加到会话日志，用 `tiktoken` 计算 Token 数，若任一阈值越过则触发压缩。生成 `compressed_history` 和 `raw_recent_turns` 供下游使用。

### 阶段 2 —— SearchCoordinator

使用 LLM 规划步骤将用户查询分解为多个**子查询**（每个源类型 `graph`/`skill` 一个或多个）。这一"多查询"策略确保涉及多个主题的复合问题能为各主题独立检索相关事实，而非混淆到单一向量搜索。

图谱子查询被发往 `GraphitiEngine.search()`；技能子查询使用 HyDE 嵌入再查询技能库。

### 阶段 3 —— SynthesizerAgent

作为 **LLM-as-Judge**：对每条检索的记忆片段进行 0–1 绝对相关度评分，丢弃低于阈值（默认 0.6）的片段，将存活者融合成单一密集的 `background_context` 字符串。它还识别能直接指导最终回答的 `skill_reuse_plan` 条目。此阶段输出 `provenance` —— 人工可读的引文列表，含完整检索元数据包括 RRF 分数、BFS/BM25 排名和源情景反向引用。

### 阶段 4 —— ReasoningAgent

接收 `compressed_history`、`background_context` 和 `skill_reuse_plan` 并生成最终助手回答。它将助手轮次追加回会话存储，完成反馈循环。

### 后台 —— ConsolidatorAgent

在 `ReasoningAgent` 完成后立即触发，在线程池中运行且**不阻塞响应**。它执行：

1. **新颖性检查** —— LLM 判断对话是否引入值得持久化的新信息。跳过纯检索交互的固化。
2. 调用 `GraphitiEngine.ingest_episode()` 从对话轮次提取实体和事实并提交到 Neo4j 图表（双时态时间戳由壁钟时间衍生）。
3. 从对话中成功的工具使用模式提取技能条目并添加到技能库。
4. 可选地触发周期性 `refresh_communities_for_session()`。

---

## 快速开始

### 前置要求

- Python 3.11+
- Neo4j 5.x，已安装 **GDS 插件**并创建**向量索引**
- LLM API 密钥（OpenAI、Gemini 或任何兼容 litellm 的供应商）

### 安装

```bash
git clone https://github.com/LycheeMem/LycheeMem.git
cd LycheeMem
pip install -e ".[dev]"
```

### 配置

将 `.env.example` 复制为 `.env` 并填入您的值：

```dotenv
# LLM —— litellm 格式：供应商/模型
LLM_MODEL=openai/gpt-4o-mini
LLM_API_KEY=sk-...
LLM_API_BASE=                     # 可选，用于代理

# 嵌入模型
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIM=1536

# Neo4j
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你的密码
```

> **支持的 LLM 供应商**（经 [litellm](https://github.com/BerriAI/litellm)）：  
> `openai/gpt-4o-mini` · `gemini/gemini-2.0-flash` · `ollama_chat/qwen2.5` · 任何 OpenAI 兼容端点

### 启动服务器

```bash
python main.py
# 带热重载：
python main.py --reload
```

API 服务于 `http://localhost:8000`。交互式文档于 `/docs`。

---

## API 参考

### `POST /memory/search` —— 统一记忆检索

在一次调用中同时查询知识图谱和技能库。

```json
// 请求
{
  "query": "我用什么工具做数据库备份",
  "top_k": 5,
  "include_graph": true,
  "include_skills": true
}

// 响应
{
  "query": "...",
  "graph_results": [ { "fact_id": "...", "summary": "...", "relevance": 0.91, ... } ],
  "skill_results": [ { "id": "...", "intent": "pg_dump 备份到 S3", "score": 0.87, ... } ],
  "total": 6
}
```

---

### `POST /memory/synthesize` —— 记忆融合

使用 LLM-as-Judge 将原始检索结果融合成精炼记忆上下文。

```json
// 请求
{
  "user_query": "我用什么工具做数据库备份",
  "graph_results": [...],   // 来自 /memory/search
  "skill_results": [...]
}

// 响应
{
  "background_context": "用户通常使用 pg_dump 配合 cron 任务...",
  "skill_reuse_plan": [ { "skill_id": "...", "intent": "...", "doc_markdown": "..." } ],
  "provenance": [ { "fact_id": "...", "relevance": 0.91, "rrf_score": 0.72, ... } ],
  "kept_count": 4,
  "dropped_count": 2
}
```

---

### `POST /memory/reason` —— 基础推理

给定预合成上下文运行 ReasoningAgent。可在 `/memory/synthesize` 之后链式调用以获得完整管道控制。

```json
// 请求
{
  "session_id": "my-session",
  "user_query": "我用什么工具做数据库备份",
  "background_context": "用户通常使用 pg_dump...",
  "skill_reuse_plan": [...],
  "append_to_session": true   // 将结果写入会话历史（默认：true）
}

// 响应
{
  "final_response": "你通常使用 pg_dump 通过 cron 调度...",
  "session_id": "my-session",
  "wm_token_usage": 3412
}
```

---

### `POST /memory/consolidate/{session_id}` —— 触发固化

手动为会话触发记忆固化（通常在每次对话后自动在后台运行）。

```bash
curl -X POST http://localhost:8000/memory/consolidate/my-session
```

```json
// 响应
{ "message": "Consolidation done: 5 entities, 2 skills extracted." }
```

---

## 前端演示

项目根目录下的 `web-demo/` 包含一个 React + Vite 前端。它提供对话界面加上知识图谱、技能库和工作记忆状态的实时视图。

```bash
cd web-demo
npm install
npm run dev      # 服务于 http://localhost:5173
```

主要面板：
- **对话** —— 完整对话含每阶段迹象展开（点击任何管道阶段查看检索和评分细节）。
- **图谱记忆** —— Neo4j 知识图谱的交互可视化；按实体名或关系类型搜索；按有效时间范围筛选。
- **技能库** —— 浏览和搜索程序技能库。
- **工作记忆** —— 检查当前会话的 Token 用量、压缩摘要和原始最近轮次。

> 确保后端运行在端口 8000（或在 `web-demo/vite.config.ts` 中更新代理设置）后再启动前端。

---

## 许可证

Apache 2.0
