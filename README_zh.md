<div align="center">
  <img src="assert/logo.png" alt="LycheeMem Logo" width="200">
  <h1>LycheeMem</h1>
  <p>
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
    <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/LangGraph-000?style=flat&logo=langchain" alt="LangGraph">
    <img src="https://img.shields.io/badge/litellm-000?style=flat&logo=python" alt="litellm">
  </p>
    <p>
      中文 | <a href="README.md">English</a>
    </p>
</div>


LycheeMem 是一个面向 LLM Agent 的紧凑型终身记忆框架。它以高效对话记忆为起点，通过结构化组织、轻量化固化和自适应检索，建立稳定且实用的记忆底座，并逐步扩展到面向行动、面向使用的记忆机制，以支撑更强的 Agent 能力。

---

<div align="center">
  <a href="#最新动态">最新动态</a>
  •
  <a href="#记忆架构">记忆架构</a>
  •
  <a href="#管道架构">管道架构</a>
  •
  <a href="#快速开始">快速开始</a>
  •
  <a href="#前端演示">前端演示</a>
  •
  <a href="#openclaw-插件">OpenClaw 插件</a>
  •
  <a href="#mcp">MCP</a>
  •
  <a href="#api-参考">API 参考</a>
</div>

---

<a id="最新动态"></a>

## 🔥 最新动态

- [03/30/2026] 我们在 PinchBench 上测评了 LycheeMem OpenClaw 插件：相比 OpenClaw 原生记忆，评分提升约 6%，同时 Token 消耗大幅下降约 71%，成本降低约 55%！
- [03/28/2026] 语义记忆已升级为 Compact Semantic Memory（SQLite + LanceDB），不再依赖 Neo4j，详见 [快速开始](#快速开始) !
- [03/27/2026] OpenClaw 插件正式上线，详见 [openclaw-插件](#openclaw-插件) ! [配置指南 →](openclaw-plugin/INSTALL_OPENCLAW_zh.md)
- [03/26/2026] MCP 服务已上线，详见 [/mcp](#mcp) !
- [03/23/2026] LycheeMem 正式开源：[GitHub 仓库](https://github.com/LycheeMem/LycheeMem)

---

<a id="记忆架构"></a>

## 📚 记忆架构

LycheeMem 将记忆组织为三个相辅相成的存储库：

<table>
  <thead>
    <tr>
      <th>工作记忆</th>
      <th>语义记忆</th>
      <th>程序记忆</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <p>(情景记忆)</p>
        <ul>
          <li>会话轮次管理</li>
          <li>自动摘要生成</li>
          <li>Token 预算管理</li>
        </ul>
      </td>
      <td>
        <p>(类型化行动存储)</p>
        <ul>
          <li>7 类 MemoryRecord</li>
          <li>Record 融合</li>
          <li>行动感知检索规划</li>
          <li>RL-ready 使用统计</li>
        </ul>
      </td>
      <td>
        <p>(技能库)</p>
        <ul>
          <li>技能条目持久化</li>
          <li>HyDE 假设性检索</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

### 💾 工作记忆

工作记忆窗口保存会话期间的活跃对话上下文。它运作于一个**双阈值 Token 预算**机制下：

- **预警阈值（70%）** —— 触发异步后台预压缩；当前请求不被阻塞。
- **阻塞阈值（90%）** —— 管道暂停，将较早的轮次刷入压缩摘要后继续。

压缩产生两种形式的历史：摘要锚点（旧内容，凝聚化）+ 原始最近轮次（最后 N 轮，逐字保留）。两者一起作为对话上下文传递给下游阶段。

### 🗺️ 语义记忆 —— Compact Semantic Memory

语义记忆以**类型化、行动标注的 MemoryRecord** 为核心组织形式，存储层使用 SQLite（FTS5 全文检索）+ LanceDB（向量索引）。

#### 记忆记录类型

每条记忆以 `MemoryRecord` 形式存储，`memory_type` 字段区分七种语义类型：

| 类型 | 描述 |
|------|------|
| `fact` | 关于用户、环境或世界的客观事实 |
| `preference` | 用户偏好（风格、习惯、喜恶） |
| `event` | 曾经发生的具体事件 |
| `constraint` | 必须遵守的限制条件 |
| `procedure` | 可复用的操作步骤/方法 |
| `failure_pattern` | 曾经失败的操作路径及原因 |
| `tool_affordance` | 工具/API 的能力与适用场景 |

每个 `MemoryRecord` 除语义文本外，还携带**行动导向元数据**（`tool_tags`、`constraint_tags`、`failure_tags`、`affordance_tags`），以及**使用统计字段**（`retrieval_count`、`action_success_count` 等），为后续强化学习阶段储备信号。

多个相关 `MemoryRecord` 可由 **Record Fusion Engine** 在线融合为高密度 `CompositeRecord`，融合后的条目在检索排序中优先于碎片记录。

#### 四模块流水线

##### 模块一：Compact Semantic Encoding（类型化记忆编码）

单次编码流水线，将对话轮次转换为 `MemoryRecord` 列表：

1. **类型化提取** —— LLM 从对话中提取自洽事实，并为每条记录分配语义类别。
2. **指代消解** —— 将代词和上下文依赖短语展开为完整表述，使每条 record 脱离原始对话也能被理解。
3. **行动元数据标注** —— LLM 为每条 record 打 `memory_type`、`tool_tags`、`constraint_tags`、`failure_tags`、`affordance_tags` 等结构化标签。

`record_id = SHA256(normalized_text)`，天然幂等，重复内容自动去重。

##### 模块二：Record Fusion（记录融合）

每次固化后在线触发：

1. FTS 检测与新 record 文本相似的已有条目（候选池）。
2. LLM 判断候选池是否值得合并（`synthesis_judge`）。
3. 若判断为是，LLM 执行合并并生成 `CompositeRecord`，写入 SQLite + LanceDB；原始 record 保留不删除。

##### 模块三：Action-Aware Search Planning（行动感知检索规划）

搜索前由 `ActionAwareSearchPlanner` 分析用户查询，输出 `SearchPlan`：

- `mode`：`answer`（事实回答）/ `action`（需要执行操作）/ `mixed`
- `semantic_queries`：面向内容的检索词列表
- `pragmatic_queries`：面向 action/tool/constraint 的检索词列表
- `tool_hints`：当前请求可能需要的工具
- `required_constraints`：缺失的约束条件
- `missing_slots`：缺失的参数/slot

再经五通道召回：

1. **FTS 全文通道** —— SQLite FTS5 关键词召回 `MemoryRecord` + `CompositeRecord`
2. **语义向量通道** —— LanceDB ANN 查询 `semantic_text` 嵌入
3. **归一化向量通道** —— LanceDB ANN 查询 `normalized_text` 嵌入（面向 pragmatic 查询）
4. **标签过滤通道** —— 按 `tool_hints`/`constraint_tags` 精确过滤
5. **时间通道** —— 按 `SearchPlan.temporal_filter` 过滤时间区间内的记忆

##### 模块四：Multi-Dimensional Scorer（多维度打分）

全通道候选汇聚后去重，由 `MemoryScorer` 按加权线性公式综合评分并排序：

$$\text{Score} = \alpha \cdot S_\text{sem} + \beta \cdot S_\text{action} + \gamma \cdot S_\text{temporal} + \delta \cdot S_\text{recency} + \eta \cdot S_\text{evidence} - \lambda \cdot C_\text{token}$$

| 系数 | 含义 | 默认值 |
|------|------|--------|
| α | SemanticRelevance（语义向量距离转相似度） | 0.30 |
| β | ActionUtility（tag 匹配度，mode-aware） | 0.25 |
| γ | TemporalFit（时间引用匹配度） | 0.15 |
| δ | Recency（记忆新鲜度） | 0.10 |
| η | EvidenceDensity（证据跨度密度） | 0.10 |
| λ | TokenCost penalty（文本长度惩罚） | 0.10 |

### 🛠️ 程序记忆 —— 技能库

技能库保存可复用的**操作方法**知识，每个技能条目携带：

- **意图** —— 简短描述该技能的功能。
- **`doc_markdown`** —— 完整 Markdown 文档，描述步骤、命令、参数和注意事项。
- **向量** —— 意图文本的密集向量，用于相似度搜索。
- **元数据** —— 使用计数、最后使用时间戳、前置条件。

技能检索使用 **HyDE（假设性文档嵌入）**：查询首先被 LLM 展开成假设的理想回答，然后对该草稿文本嵌入以产生查询向量，该向量能很好地匹配存储的流程描述，即使用户的原始表述模糊。

---

<a id="管道架构"></a>

## ⚙️ 管道架构

每个请求经过固定的五阶段序列。四个是管道中的同步阶段；一个是后台后处理任务。

<div align="center">
  <div>
    <div>开始</div>
    <div>▼</div>
    <div>
      <div>
        <div>
          <strong>1. WMManager</strong> — Token 预算检查 + 压缩/渲染
        </div>
        <div>↓</div>
        <div>
          <strong>2. SearchCoordinator</strong> — 规划器 → 语义记忆 + 技能检索
        </div>
        <div>↓</div>
        <div>
          <strong>3. SynthesizerAgent</strong> — LLM-as-Judge 评分 + 上下文融合
        </div>
        <div>↓</div>
        <div>
          <strong>4. ReasoningAgent</strong> — 最终回答生成
        </div>
      </div>
    </div>
    <div>▼</div>
    <div>结束</div>
    <div>
      <span>后台任务</span>
      <span>asyncio.create_task( <strong>ConsolidatorAgent</strong> )</span>
    </div>
  </div>
</div>

### 阶段 1 —— WMManager

规则型 Agent（无 LLM 提示词）。将用户轮次追加到会话日志，计算 Token 数，若任一阈值越过则触发压缩。生成 `compressed_history` 和 `raw_recent_turns` 供下游使用。

### 阶段 2 —— SearchCoordinator

首先由 `ActionAwareSearchPlanner` 分析用户查询，生成包含 `mode`、`semantic_queries`、`pragmatic_queries`、`tool_hints` 等字段的搜索计划。随后通过五通道并行召回（FTS 全文、语义向量、归一化向量、标签过滤、时间过滤）在 SQLite + LanceDB 中取出候选，经 Scorer 按六维公式排序后，合并为 `background_context` 供下游使用。技能子查询使用 HyDE 嵌入查询技能库。

### 阶段 3 —— SynthesizerAgent

作为 **LLM-as-Judge**：对每条检索的记忆片段进行 0-1 绝对相关度评分，丢弃低于阈值（默认 0.6）的片段，将存活者融合成单一密集的 `background_context` 字符串。它还识别能直接指导最终回答的 `skill_reuse_plan` 条目。此阶段输出 `provenance` —— 人工可读的引文列表，包含每条保留记忆的评分拆解与来源引用。

### 阶段 4 —— ReasoningAgent

接收 `compressed_history`、`background_context` 和 `skill_reuse_plan` 并生成最终助手回答。它将助手轮次追加回会话存储，完成反馈循环。

### 后台 —— ConsolidatorAgent

在 `ReasoningAgent` 完成后立即触发，在线程池中运行且**不阻塞响应**。它执行：

1. **新颖性检查** —— LLM 判断对话是否引入值得持久化的新信息。跳过纯检索交互的固化。
2. **Compact 编码固化** —— 调用 `CompactSemanticEngine.ingest_conversation()`，经单次编码（类型化提取 → 指代消解 → 行动元数据标注）将对话内容提取为 `MemoryRecord` 并写入 SQLite + LanceDB；随后触发 Record Fusion，融合相关条目生成 `CompositeRecord`。
3. **技能提取** —— 从对话中识别成功的工具使用模式，提取技能条目并添加到技能库；与 Compact 固化并行执行（ThreadPoolExecutor）。

---

<a id="快速开始"></a>

## ⚡ 快速开始

### 前置要求

- Python 3.11+
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

# 语义记忆存储路径（可选，默认 data/ 目录）
COMPACT_MEMORY_DB_PATH=data/compact_memory.db
COMPACT_VECTOR_DB_PATH=data/compact_vector
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

<a id="前端演示"></a>

## 🎨 前端演示

项目根目录下的 `web-demo/` 包含一个 React + Vite 前端。它提供对话界面加上语义记忆、技能库和工作记忆状态的实时视图。

```bash
cd web-demo
npm install
npm run dev      # 服务启动于 http://localhost:5173
```

> 确保后端运行在端口 8000（或在 `web-demo/vite.config.ts` 中更新代理设置）后再启动前端。

---

<a id="openclaw-插件"></a>

## 🦞 OpenClaw 插件

LycheeMem 提供原生 [OpenClaw](https://openclaw.ai) 插件，让任何 OpenClaw 会话无需手动配置即可获得持久化长期记忆。

**插件提供：**

- `lychee_memory_smart_search` — 默认的长期记忆检索入口
- **自动对话镜像**（通过 hook 实现）— 模型无需手动调用 `append_turn`
  - 用户消息自动追加
  - 助手消息自动追加
- `/new`、`/reset`、`/stop`、`session_end` 自动触发边界 consolidate
- 对明显长期知识信号，提前触发 consolidate

**正常使用时：**
- 模型只需在需要回忆长期上下文时调用 `lychee_memory_smart_search`
- 模型在必要时可手动调用 `lychee_memory_consolidate`
- 模型**无需**手动调用 `lychee_memory_append_turn`

### 快速安装

```bash
openclaw plugins install "/path/to/LycheeMem/openclaw-plugin"
openclaw gateway restart
```

完整配置说明：[openclaw-plugin/INSTALL_OPENCLAW_zh.md](openclaw-plugin/INSTALL_OPENCLAW_zh.md)

---

<a id="mcp"></a>

## 🔧 MCP

LycheeMem 还通过 `http://localhost:8000/mcp` 暴露了 MCP 端点。

- 可用工具：`lychee_memory_search`, `lychee_memory_synthesize`, `lychee_memory_consolidate`
- 如果需要每个用户的内存隔离，请使用 `Authorization: Bearer <token>`
- `lychee_memory_consolidate` 仅适用于已经通过 `/chat` 或 `/memory/reason` 写入的会话

### MCP 传输

- `POST /mcp` 处理 JSON-RPC 请求
- `GET /mcp` 暴露一些 MCP 客户端使用的 SSE 流
- 服务器在 `initialize` 期间返回 `Mcp-Session-Id`；在后续请求中重用该 header

### 身份验证

如果你需要每个用户的隔离记忆，请先从 `/auth/register` 或 `/auth/login` 获取 JWT token，然后发送：

```text
Authorization: Bearer <token>
```

如果没有 token，请求将以空的 `user_id` 运行，因此匿名流量共享相同的命名空间。

### 客户端配置

对于任何支持远程 HTTP 服务器的 MCP 客户端，将 MCP URL 配置为：

```text
http://localhost:8000/mcp
```

通用配置示例：

```json
{
  "mcpServers": {
    "lycheemem": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

### 手动 JSON-RPC 流程

1. 调用 `initialize`
2. 重用返回的 `Mcp-Session-Id`
3. 发送 `initialized`
4. 调用 `tools/list`
5. 调用 `tools/call`

Initialize 示例：

```bash
curl -i -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-03-26",
      "capabilities": {},
      "clientInfo": {
        "name": "debug-client",
        "version": "0.1.0"
      }
    }
  }'
```

工具调用示例：

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -H "Mcp-Session-Id: <session-id>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "lychee_memory_search",
      "arguments": {
        "query": "what tools do I use for database backups",
        "top_k": 5,
        "include_graph": true,
        "include_skills": true
      }
    }
  }'
```

### 推荐的 MCP 使用模式

1. 使用 `/chat` 或 `/memory/reason` 配合稳定的 `session_id` 来写入对话轮次。
2. 使用 `lychee_memory_search` 检索相关的长期记忆。
3. 使用 `lychee_memory_synthesize` 将检索结果合成为 `background_context`。
4. 对话结束后，使用相同的 `session_id` 调用 `lychee_memory_consolidate`。

---

<a id="api-参考"></a>

## 🔌 API 参考

### `POST /memory/search` —— 统一记忆检索

在一次调用中同时查询语义记忆通道和技能库。

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
  "graph_results": [
    {
      "anchor": {
        "node_id": "compact_context",
        "name": "CompactSemanticMemory",
        "label": "Context",
        "score": 1.0
      },
      "constructed_context": "...",
      "provenance": [ { "id": "...", "source": "semantic_memory", "relevance": 0.91, ... } ]
    }
  ],
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
  "provenance": [ { "id": "...", "source": "semantic_memory", "relevance": 0.91, ... } ],
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

### 使用示例

```bash
# 基础单轮演示（自动注册用户 demo_user）
python examples/api_pipeline_demo.py

# 多轮对话演示（3 轮连续对话，最后统一固化）
python examples/api_pipeline_demo.py --multi-turn

# 自定义查询和用户
python examples/api_pipeline_demo.py --username alice --password secret123 \
  --query "如何用 pg_dump 备份我的数据库？"

# 使用固定 session_id（方便多次运行复现会话历史累积效果）
python examples/api_pipeline_demo.py --session-id my-test-session
```
