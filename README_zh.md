<div align="center">
  <img src="assets/logo.png" alt="LycheeMem Logo" width="200">
  <h1>LycheeMem: Lightweight Long-Term Memory for LLM Agents</h1>
  <p>
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/LangGraph-000?style=flat&logo=langchain" alt="LangGraph">
    <img src="https://img.shields.io/badge/litellm-000?style=flat&logo=python" alt="litellm">
    <a href="https://lancedb.com/">
      <img src="https://img.shields.io/badge/LanceDB-vector%20database-0ea5e9?style=flat" alt="LanceDB">
    </a>
    <a href="https://lycheemem.github.io/">
      <img src="https://img.shields.io/badge/Homepage-lycheemem.github.io-2ea44f?style=flat&logo=github&logoColor=white" alt="项目主页">
    </a>
    <a href="https://pypi.org/project/lycheemem/">
      <img src="https://img.shields.io/pypi/v/lycheemem?style=flat&logo=pypi&logoColor=white&label=PyPI&color=3775A9" alt="PyPI">
    </a>
    <a href="https://github.com/HITsz-TMG">
      <img src="https://img.shields.io/badge/HITsz--TMG-24292f?style=flat&logo=github&logoColor=white" alt="HITsz-TMG">
    </a>
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
  <a href="#相关项目">相关项目</a>
  •
  <a href="#快速开始">快速开始</a>
  •
  <a href="#前端演示">前端演示</a>
  •
  <a href="#openclaw-插件">OpenClaw 插件</a>
  •
  <a href="#mcp">MCP</a>
  •
  <a href="#记忆架构">记忆架构</a>
  •
  <a href="#管道架构">管道架构</a>
  •
  <a href="#api-参考">API 参考</a>
</div>

---

<a id="最新动态"></a>

## 🔥 最新动态

- **[04/10/2026]** 语义记忆检索管线升级为 Action-Aware 层级化检索：以 CompositeRecord 为主要检索单元，结合当前查询与 ActionState 进行整体语义相关性判断，并按需沿记忆树展开至原子 MemoryRecord；辅以充分性驱动的反思补充召回，全面提升检索精度与上下文覆盖率。
- **[04/03/2026]** 项目已支持通过 `pip install lycheemem` 打包安装，现在你可以随时随地使用 `lycheemem-cli` 快速启动服务了！
- **[03/30/2026]** 我们在 PinchBench 上测评了 LycheeMem OpenClaw 插件：相比 OpenClaw 原生记忆，评分提升约 6%，同时 Token 消耗大幅下降约 71%，成本降低约 55%！
- **[03/28/2026]** 语义记忆已升级为 Compact Semantic Memory（SQLite + LanceDB），不再依赖 Neo4j，详见 [快速开始](#快速开始) !
- **[03/27/2026]** OpenClaw 插件正式上线，详见 [openclaw-插件](#openclaw-插件) ! [配置指南 →](openclaw-plugin/INSTALL_OPENCLAW_zh.md)
- **[03/26/2026]** MCP 服务已上线，详见 [/mcp](#mcp) !
- **[03/23/2026]** LycheeMem 正式开源：[GitHub 仓库](https://github.com/LycheeMem/LycheeMem)

---

<a id="相关工作"></a>

## 🔗 相关工作

LycheeMem 是 **第三代“立知”大模型系列** 的一部分，该系列专注于记忆智能、持续学习和长上下文推理。

欢迎探索我们的相关工作：

- **LycheeMemory (ACL 2026, CCF-A)**：实现隐式长期记忆与显式工作记忆协同的大模型统一框架
  [![arXiv](https://img.shields.io/badge/arXiv-2602.08382-B31B1B?logo=arxiv&logoColor=fff)](https://arxiv.org/abs/2602.08382) [![GitHub](https://img.shields.io/badge/GitHub-LycheeMemory-181717?logo=github&logoColor=fff)](https://github.com/owoakuma/LycheeMemory) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-LycheeMemory--7B-FFD21E?logo=huggingface)](https://huggingface.co/lerverson/LycheeMemory-7B)

- **LycheeMem (本项目)**：面向 LLM Agent 的长期记忆基础设施  
  [![Project Page](https://img.shields.io/badge/Project_Page-LycheeMem-blue?logo=google-chrome&logoColor=fff)](https://lycheemem.github.io) [![GitHub](https://img.shields.io/badge/GitHub-LycheeMem-181717?logo=github&logoColor=fff)](https://github.com/LycheeMem/LycheeMem)

- **LycheeDecode (ICLR 2026, CCF-A)**：从大规模 KV-cache 上下文记忆中进行选择性召回
  [![Project Page](https://img.shields.io/badge/Project_Page-lycheedecode-blue?logo=google-chrome&logoColor=fff)](https://lg9077.github.io/lycheedecode) [![arXiv](https://img.shields.io/badge/arXiv-2602.04541-B31B1B?logo=arxiv&logoColor=fff)](https://arxiv.org/abs/2602.04541) [![GitHub](https://img.shields.io/badge/GitHub-LycheeDecode-181717?logo=github&logoColor=fff)](https://github.com/HITsz-TMG/TMGNLP/tree/main/LycheeDecode)

- **LycheeCluster (ACL 2026, CCF-A)**：用于上下文记忆的结构化组织与层次化索引
  [![arXiv](https://img.shields.io/badge/arXiv-2603.08453-B31B1B?logo=arxiv&logoColor=fff)](https://arxiv.org/abs/2603.08453)

---

<a id="快速开始"></a>

## ⚡ 快速开始

### 前置要求

- Python 3.9+
- LLM API 密钥（OpenAI、Gemini 或任何兼容 litellm 的供应商）

### 安装

你可以直接通过 pip 安装 LycheeMem：

```bash
pip install lycheemem
```

安装完成后，你可以随时随地通过命令行直接启动后端服务：

```bash
lycheemem-cli
```

如果想体验最新源码或者做二次开发：

```bash
git clone https://github.com/LycheeMem/LycheeMem.git
cd LycheeMem
pip install -e .
```

### 配置

在你的工作目录下，创建一个 `.env` 文件并填入您的值：

```dotenv
# LLM —— litellm 格式：供应商/模型
LLM_MODEL=openai/gpt-4o-mini
LLM_API_KEY=sk-...
LLM_API_BASE=                     # 可选，用于代理

# 嵌入模型
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_API_KEY=                # 可选
EMBEDDING_API_BASE=               # 可选
```

> **支持的 LLM 供应商**（经 [litellm](https://github.com/BerriAI/litellm)）：  
> `openai/gpt-4o-mini` · `gemini/gemini-2.0-flash` · `ollama_chat/qwen2.5` · 任何 OpenAI 兼容端点

### 启动服务器

如果你是通过 pip 安装的包，直接在终端任意位置执行以下命令即可启动 LycheeMem 后台服务：

```bash
lycheemem-cli
```

*(如果你是使用拉取源码方式安装的，也可以执行 `python main.py` 启动)*
```
python main.py
```

API 服务于 `http://localhost:8000`。交互式文档于 `/docs`。

> 当前 `main.py` 启动 Uvicorn 时并不会开启热重载。开发时如果需要自动重载，请直接运行 Uvicorn，例如：
>
> ```bash
> uvicorn src.api.server:create_app --factory --reload
> ```

---

<a id="前端演示"></a>

## 🎨 前端演示

项目根目录下的 `web-demo/` 包含一个 React + Vite 前端。它提供对话界面加上**语义记忆树**、技能库和工作记忆状态的实时视图。

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

- 可用工具：`lychee_memory_smart_search`, `lychee_memory_search`, `lychee_memory_append_turn`, `lychee_memory_synthesize`, `lychee_memory_consolidate`
- `lychee_memory_consolidate` 适用于已经通过 `/chat`、`/memory/reason` 或 `lychee_memory_append_turn` 写入/镜像过轮次的会话

### MCP 传输

- `POST /mcp` 处理 JSON-RPC 请求
- `GET /mcp` 暴露一些 MCP 客户端使用的 SSE 流
- 服务器在 `initialize` 期间返回 `Mcp-Session-Id`；在后续请求中重用该 header

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
      "url": "http://localhost:8000/mcp"
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
  -H "Mcp-Session-Id: <session-id>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "lychee_memory_smart_search",
      "arguments": {
        "query": "what tools do I use for database backups",
        "top_k": 5,
        "mode": "compact",
        "include_graph": true,
        "include_skills": true
      }
    }
  }'
```

### 推荐的 MCP 使用模式

1. 使用 `/chat` 或 `/memory/reason` 配合稳定的 `session_id` 写入对话轮次；如果是外部宿主，也可以用 `lychee_memory_append_turn` 镜像轮次。
2. 默认推荐使用 `lychee_memory_smart_search` 的 `compact` 模式完成一体化检索。
3. 只有在你明确需要把检索和合成拆开控制时，再使用 `lychee_memory_search` + `lychee_memory_synthesize`。
4. 对话结束后，使用相同的 `session_id` 调用 `lychee_memory_consolidate`。

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
          <li>冲突感知的 Record 融合</li>
          <li>层级记忆树</li>
          <li>action-aware 层级化检索</li>
          <li>usage feedback 闭环 + RL-ready 使用统计</li>
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

语义记忆以**类型化 MemoryRecord + action-grounded 检索状态**为核心组织形式，存储层使用 SQLite（FTS5 全文检索）+ LanceDB（向量索引）；检索则显式受 recent context、tentative action、约束条件和 missing slots 共同驱动。

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

每个 `MemoryRecord` 除语义文本外，还携带**行动导向元数据**（`tool_tags`、`constraint_tags`、`failure_tags`、`affordance_tags`），以及**使用统计字段**（`retrieval_count`、`action_success_count` 等），为后续强化学习阶段储备信号。检索日志还会额外保存 `retrieval_plan`、`action_state`、回答摘要以及后续用户反馈，从而在不训练的前提下形成轻量的 action outcome 闭环。

多个相关 `MemoryRecord` 可由 **Record Fusion Engine** 在线融合为高密度 `CompositeRecord`。融合后的节点会持久化直接 `child_composite_ids`，因此长期语义记忆不再只是扁平摘要集合，而是组织成一棵**层级记忆树**。

#### 四模块流水线

##### 模块一：Compact Semantic Encoding（类型化记忆编码）

单次编码流水线，将对话轮次转换为 `MemoryRecord` 列表：

1. **类型化提取** —— LLM 从对话中提取自洽事实，并为每条记录分配语义类别。
2. **指代消解** —— 将代词和上下文依赖短语展开为完整表述，使每条 record 脱离原始对话也能被理解。
3. **行动元数据标注** —— LLM 为每条 record 打 `memory_type`、`tool_tags`、`constraint_tags`、`failure_tags`、`affordance_tags` 等结构化标签。

`record_id = SHA256(normalized_text)`，天然幂等，重复内容自动去重。

##### 模块二：记录融合、冲突更新与层级整合

每次固化后在线触发：

1. 先通过 FTS / 向量召回，为新 record 收集相关的**已有 atomic record** 候选池。
2. 复用原有 synthesis judge prompt，让 LLM 判断当前候选集是应该生成新的 `CompositeRecord`，还是对某条已有 atomic record 执行 `conflict_update`。
3. 若判定为 `conflict_update`，则原有 anchor record 原地更新，冲突的新 incoming records 被软过期，同时覆盖相关 source records 的 composite 会被失效。
4. 若判定为 synthesis，则写入新的 `CompositeRecord` 到 SQLite + LanceDB。
5. 随后继续执行 `record -> composite` 与 `composite -> composite` 的多轮层级融合，并持久化 `child_composite_ids`，让记忆树持续向上生长。

##### 模块三：Action-Aware 层级化检索

检索以层级记忆树为核心，以 CompositeRecord 为主要检索单元，结合当前查询、近期上下文与 ActionState 进行整体语义相关性判断，并按需沿记忆树向下展开至原子 MemoryRecord；最终由充分性驱动的反思循环覆盖残余信息盲区。

**Composite-Level 相关性判断**

检索首先在 CompositeRecord 层面运作。检索引擎将全量 CompositeRecord 的类型、摘要与实体信息连同当前查询提交 LLM，由 LLM 对各条 composite 进行整体语义相关性判断；同时标记出摘要层次过于抽象、需进一步展开至原子记录方能充分回答查询的候选项。与基于向量距离阈值的多通道召回不同，这一判断直接在语义层运作，能够捕捉向量空间中不易对齐的深层关联。

**层级记忆树展开**

对标记为需要展开的 composite，检索引擎沿记忆树向下递归遍历 `source_record_ids` 与 `child_composite_ids`，取回对应的原子 `MemoryRecord`。这一机制同时保留了高层 composite 提供的全局语义视野，以及在需要细粒度信息时深入树内节点的能力，兼顾检索效率与细节覆盖。

**充分性驱动的反思补充召回**

在初始候选集形成后，检索引擎对当前上下文进行充分性评估。若存在信息缺口，则激活多通道补充召回：FTS 全文通道与向量通道（`semantic_text` 及 `normalized_text` 双路）在 `MemoryRecord` 维度扩充候选；同时在原始对话 turns 向量索引中检索尚未被提炼为 `MemoryRecord` 的对话内容，覆盖编码层面的遗漏。反思循环最多执行数轮，每轮感知当前已召回内容，仅在仍存在信息缺口时继续搜索。

##### 模块四：候选汇聚与上下文增强

三阶段候选汇聚后按来源层级优先排序取 top-k：Composite-Level 相关性判断直接选中的 CompositeRecord 优先级最高，层级展开所得的原子 MemoryRecord 次之，反思补充召回的记录依次递后。全部候选最终附加情节上下文（episodic context）——从 session store 取回与对应证据关联的原始对话片段，拼接至候选的展示文本，为下游 SynthesizerAgent 提供可溯源的完整背景。

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

`SearchCoordinator` 先从压缩摘要与最近原始轮次构造 `recent_context`，并从当前查询、约束条件、近期失败信号、token 预算和近期工具使用中推导 `ActionState`。语义记忆检索基于 Action-Aware 层级化检索管线展开：首先在 CompositeRecord 层面对全量 composite 进行整体相关性判断，筛选出关联候选并标记需展开的条目；随后沿记忆树递归取出对应的原子 MemoryRecord；最后经充分性评估，在存在信息缺口时激活 FTS + 向量 + 原始对话 turns 的补充召回。此阶段输出原始语义片段、技能结果、检索 provenance，以及专供新颖性检查使用的 `novelty_retrieved_context`（**pre-synthesis** 的原始语义片段）；本阶段不构造最终的 `background_context`。技能检索 mode-aware，依 `answer / action / mixed` 模式按需启用 HyDE 查询技能库。

当新一轮用户输入到来时，`SearchCoordinator` 还会尝试把它解释为上一轮 action/mixed 检索的轻量 outcome feedback，用于将之前的记忆使用标记为 success / fail / correction。

### 阶段 3 —— SynthesizerAgent

作为 **LLM-as-Judge**：对每条检索的记忆片段进行 0-1 绝对相关度评分，丢弃低于阈值（默认 0.6）的片段，将存活者融合成单一密集的 `background_context` 字符串。它还识别能直接指导最终回答的 `skill_reuse_plan` 条目。最终用于回答阶段的融合上下文是在这里生成的。此阶段输出 `provenance` —— 人工可读的引文列表，包含每条保留记忆的评分拆解与来源引用。

### 阶段 4 —— ReasoningAgent

接收 `compressed_history`、`background_context` 和 `skill_reuse_plan` 并生成最终助手回答。它会将助手轮次追加回会话存储，随后由 pipeline 将回答摘要写入 semantic usage log，供下一轮用户输入补充 action outcome feedback。

### 后台 —— ConsolidatorAgent

在 `ReasoningAgent` 完成后立即触发，在线程池中运行且**不阻塞响应**。它执行：

1. **新颖性检查** —— LLM 判断对话是否引入值得持久化的新信息。跳过纯检索交互的固化。
2. **Compact 编码固化** —— 调用 `CompactSemanticEngine.ingest_conversation()`，经单次编码（类型化提取 → 指代消解 → 行动元数据标注）将对话内容提取为 `MemoryRecord` 并写入 SQLite + LanceDB；随后触发带冲突处理的 Record Fusion。这里的新颖性检查使用的是 search 阶段的 `novelty_retrieved_context`（原始语义片段），而不是回答期的 `background_context`，从而避免 query-conditioned 融合文本误伤真正的新记忆。
3. **技能提取** —— 从对话中识别成功的工具使用模式，提取技能条目并添加到技能库；与 Compact 固化并行执行（ThreadPoolExecutor）。

---

<a id="api-参考"></a>

## 🔌 API 参考

### `POST /memory/search` —— 统一记忆检索

在一次调用中同时查询语义记忆通道和技能库。新的接入方式应优先使用 `semantic_results`；`graph_results` 仅作为兼容别名保留。响应里还会返回 `novelty_retrieved_context`，用于后续 `/memory/consolidate` 的新颖性检查。

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
        "label": "SemanticContext",
        "score": 1.0
      },
      "constructed_context": "...",
      "provenance": [ { "record_id": "...", "source": "record", "semantic_source_type": "record", "score": 0.91, ... } ]
    }
  ],
  "semantic_results": [
    {
      "anchor": { "node_id": "compact_context", "name": "CompactSemanticMemory", "label": "SemanticContext", "score": 1.0 },
      "constructed_context": "...",
      "provenance": [ { "record_id": "...", "source": "record", "semantic_source_type": "record", "score": 0.91, ... } ]
    }
  ],
  "novelty_retrieved_context": "[1] (procedure, source=record) 使用 pg_dump 配合 cron ...",
  "skill_results": [ { "id": "...", "intent": "pg_dump 备份到 S3", "score": 0.87, ... } ],
  "total": 6
}
```

---

### `POST /memory/smart-search` —— 一体化检索

在一个 API 调用中完成 search，并可按需自动执行 synthesize。`mode=compact` 是默认推荐集成方式，适合直接拿到简洁的 `background_context`。即使在 compact 模式下，响应仍会返回 `novelty_retrieved_context`，便于宿主在固化时使用“原始召回记忆”而不是回答期融合上下文。

```json
// 请求
{
  "query": "我用什么工具做数据库备份",
  "top_k": 5,
  "synthesize": true,
  "mode": "compact"
}

// 响应
{
  "query": "...",
  "mode": "compact",
  "synthesized": true,
  "background_context": "用户通常使用 pg_dump 配合 cron 任务...",
  "skill_reuse_plan": [ { "skill_id": "...", "intent": "...", "doc_markdown": "..." } ],
  "provenance": [ { "record_id": "...", "source": "record", "score": 0.91, ... } ],
  "novelty_retrieved_context": "[1] (procedure, source=record) 使用 pg_dump 配合 cron ...",
  "kept_count": 4,
  "dropped_count": 2,
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
  "semantic_results": [...], // 优先使用 /memory/search 的该字段
  "graph_results": [...],    // 兼容别名也可接受
  "skill_results": [...]
}

// 响应
{
  "background_context": "用户通常使用 pg_dump 配合 cron 任务...",
  "skill_reuse_plan": [ { "skill_id": "...", "intent": "...", "doc_markdown": "..." } ],
  "provenance": [ { "record_id": "...", "source": "semantic", "semantic_source_type": "record", "score": 0.91, ... } ],
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
  "response": "你通常使用 pg_dump 通过 cron 调度...",
  "session_id": "my-session",
  "wm_token_usage": 3412
}
```

---

### `POST /memory/append-turn` —— 镜像外部宿主轮次

向 LycheeMem 的 session store 追加一条用户或助手轮次，供后续 consolidate 使用。

```json
// 请求
{
  "session_id": "my-session",
  "role": "user",
  "content": "我通常使用 pg_dump 把 PostgreSQL 备份到 S3。"
}

// 响应
{
  "status": "appended",
  "session_id": "my-session",
  "turn_count": 3
}
```

---

### `POST /memory/consolidate` —— 触发固化

手动为会话触发记忆固化。这是当前的主固化接口，支持后台异步和同步等待两种模式。

`retrieved_context` 建议直接使用 `/memory/search` 或 `/memory/smart-search` 返回的 `novelty_retrieved_context`，也就是 **search 阶段的原始语义记忆片段**，而不是 `/memory/synthesize` 的 `background_context`。

```json
// 请求
{
  "session_id": "my-session",
  "retrieved_context": "[1] (procedure, source=record) 使用 pg_dump 配合 cron ...",
  "background": true
}

// 响应（后台模式）
{
  "status": "started",
  "entities_added": 0,
  "skills_added": 0,
  "facts_added": 0
}
```

兼容旧路径：`POST /memory/consolidate/{session_id}`。

---

### `GET /memory/graph` —— 语义记忆树

返回当前语义记忆的层级结构。默认 `mode=cleaned` 会输出 `tree_roots` 和直接树边，供前端“记忆树”视图使用；`mode=debug` 则导出更底层的扁平关系，便于排查。

---

### `GET /pipeline/status` 和 `GET /pipeline/last-consolidation`

这两个接口适合做运行态检查和后台固化轮询：

- `GET /pipeline/status` 返回 session、语义记忆和技能库的聚合计数。
- `GET /pipeline/last-consolidation?session_id=<id>` 返回指定会话最近一次固化结果；如果后台任务尚未完成，则返回 `pending`。

### 使用示例

```bash
# 基础单轮演示（自动注册用户 demo_user）
python examples/api_pipeline_demo.py

# 多轮对话演示（3 轮连续对话，最后统一固化）
python examples/api_pipeline_demo.py --multi-turn

# 使用固定 session_id（方便多次运行复现会话历史累积效果）
python examples/api_pipeline_demo.py --session-id my-test-session
```

