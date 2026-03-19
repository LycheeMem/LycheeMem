# LycheeMemOS 工作区 Copilot 指南

你正在协助开发 **LycheeMemOS：Training-free Agentic Cognitive Memory Framework**（Python + FastAPI + LangGraph + 可选 Neo4j/LanceDB，另含一个 React/Vite Web Demo）。

目标：在不破坏既有架构的前提下，快速定位代码入口、按项目约定实现改动，并通过 Ruff 校验。

---

## 快速命令（Backend）

- 安装（开发依赖）：`pip install -e ".[dev]"`
- 启动服务（根目录的 `main.py`）：
  - `python main.py`
  - 开发热重载：`python main.py --reload`
- 代码风格：只格式化改动的文件，避免大范围 diff：
  - `ruff check src/path/to/changed.py`
  - `ruff format src/path/to/changed.py`

## 快速命令（Web Demo）

在 `web-demo/` 目录：`npm install`  `npm run dev`

---

## 实际 Pipeline 拓扑（以 `src/core/graph.py` 为准）

```
__start__  wm_manager  search  synthesize  reason  __end__
                                                    （reason 节点完成后）
                                             asyncio.create_task(consolidator)
```

**重要**：`ConsolidatorAgent` **不是 LangGraph 节点**，而是在 `reason` 节点结束后通过 `asyncio.create_task`（异步）或 `threading.Thread`（同步路径）触发的后台任务。它的错误不传播到主流程。

> CLAUDE.md 中描述的 Router 条件分支（`need_retrieval` vs `direct_answer`）是早期设计文档，**当前实现为线性拓扑，无条件分支**，Router 职能已内化到 `SearchCoordinator` 的多路子查询逻辑中。

### Agent 执行顺序与职责

| 顺序 | 类 | 文件 | 继承 | 说明 |
|------|----|------|------|------|
| 1 | `WMManager` | `agents/wm_manager.py` | 无（规则型） | token 双阈值压缩；70% 预压缩，90% 阻塞 |
| 2 | `SearchCoordinator` | `agents/search_coordinator.py` | `BaseAgent` | 多路子查询；skill 搜索用 HyDE embedding |
| 3 | `SynthesizerAgent` | `agents/synthesizer_agent.py` | `BaseAgent` | LLM-as-Judge（阈值 0.6）；融合 background_context |
| 4 | `ReasoningAgent` | `agents/reasoning_agent.py` | `BaseAgent` | 生成 final_response；追加 assistant turn |
| BG | `ConsolidatorAgent` | `agents/consolidator_agent.py` | `BaseAgent` | 图谱实体抽取 + 技能提取（后台，不阻塞） |

`BaseAgent` 提供：`_call_llm()`、`_parse_json()`、`_append_time_basis()`。

### PipelineState 关键字段（`src/core/state.py`）

```python
user_query, session_id                      # 输入
compressed_history, raw_recent_turns        # WMManager 输出
retrieved_graph_memories, retrieved_skills  # SearchCoordinator 输出
background_context, skill_reuse_plan        # SynthesizerAgent 输出
final_response                              # ReasoningAgent 输出
consolidation_pending                       # 触发固化的标志
```

---

## 配置与环境变量（必读：危险默认值）

所有配置由 `src/core/config.py` 的 `Settings`（pydantic-settings）读取，来源 `.env` 或环境变量。

### 启动即崩的默认值组合（本地无 Neo4j 时）

| 字段 | 硬编码默认 | 后果 |
|------|-----------|------|
| `graph_backend` | `"neo4j"` | 启动时连接失败 |
| `graphiti_enabled` | `True` | 要求 `graph_backend=neo4j` |
| `graphiti_strict` | `True` | `preflight()` 检测 GDS/向量索引缺失  `RuntimeError` |
| `graphiti_require_gds` | `True` | GDS 插件未安装  启动崩溃 |
| `graphiti_require_vector_index` | `True` | 向量索引缺失  启动崩溃 |
| `session_backend` | `"sqlite"` | 在当前目录创建 `.db` 文件 |
| `skill_backend` | `"file"` | 在当前目录创建 `.json` 文件 |

**本地纯内存开发配置（`.env`）**：

```dotenv
GRAPH_BACKEND=memory
GRAPHITI_ENABLED=false
SESSION_BACKEND=memory
SKILL_BACKEND=memory
```

### 常用配置分组

- LLM：`LLM_MODEL`（litellm 完整格式，如 `openai/gpt-4o-mini`）、`LLM_API_KEY`、`LLM_API_BASE`
- Embedder：`EMBEDDING_MODEL`（如 `openai/text-embedding-3-small`）、`EMBEDDING_DIM`（默认 1536；Gemini 等非 1536 维模型**必须**显式设置）
- Neo4j：`NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`
- 其他：`API_PORT`（默认 8000，端口冲突时覆盖）、`SQLITE_DB_PATH`、`SKILL_FILE_PATH`
- `.env` 已在 `.gitignore`；不要把真实密钥提交到仓库

---

## 代码入口与结构导航

| 要找什么 | 从哪里看 |
|----------|---------|
| CLI 启动 | 根目录 `main.py` |
| FastAPI 路由注册 | `src/api/server.py`（`create_app()`） |
| 组件依赖注入 | `src/core/factory.py`（`create_pipeline()`） |
| LangGraph 节点定义 | `src/core/graph.py`（`LycheePipeline`） |
| 共享状态结构 | `src/core/state.py`（`PipelineState`） |
| 所有请求/响应模型 | `src/api/models.py` |
| SSE trace 构建 | `src/api/trace_builders.py` |

核心分层：

- `src/agents/`：认知 Agent（多数继承 `base_agent.py`；`WMManager` 为规则型）
- `src/memory/`：
  - `working/`：会话存储 + `WorkingMemoryCompressor`（tiktoken 双阈值）
  - `graph/`：`NetworkXGraphStore`（memory）/ `Neo4jGraphStore`（legacy）/ `GraphitiEngine`+`GraphitiNeo4jStore`（生产）
  - `procedural/`：技能库（`InMemorySkillStore` / `FileSkillStore` / `LanceDBSkillStore`）
- `src/llm/` & `src/embedder/`：litellm 统一适配器（`BaseLLM` / `BaseEmbedder`）
- 生产依赖（Neo4j/LanceDB）：`pip install -e ".[production]"` 

---

## Graphiti 开发约定（严格对齐论文，不得违反）

1. **双时态字段命名**：统一使用 `t_valid_from`、`t_valid_to`、`t_tx_created`、`t_tx_expired`。旧名称（`t_valid`/`t_invalid`/`t_expired`）已废弃，Cypher 查询和 schema 索引必须用新名称。

2. **失效条件**：仅当两个 Fact 的有效区间存在**交叠**时，新 Fact 才能使旧 Fact 失效。有效区间不重叠的矛盾 Fact **不得**触发失效。

3. **GDS 结果映射**：从流式结果（stream）回映射 domain id 时，使用 `gds.util.asNode(nodeId).entity_id`，**不要**假设 `nodeId == id(node)`（跨 projection 不成立）。

4. **Cross-Encoder Reranker**：必须复用 pipeline 注入的 `BaseLLM` 适配器（`cross_encoder.py` 中的 `llm` 参数），**不要**在内部直接调用特定供应商的 SDK（如 Gemini SDK），否则将破坏后端选择的统一性。

5. **Episode-only 引擎**（无 `.store`）：`ConsolidatorAgent` 应在 `graphiti_engine` + `session_id` 均存在时即执行 episode ingest；仅当 `.store` 存在时才调用语义构建器（semantic builder）。

6. **Strict 模式**：`GRAPHITI_STRICT=true`（默认）时，任何能力缺失应 **fail-fast**，不得静默回退到 legacy 实现。

---

## 贡献约定

- **根因修复**优先，不加表面补丁。
- 改动小而聚焦，不引入无关重构，不新增不必要依赖。
- 不要编写或执行测试文件，由用户手动执行。
- FastAPI 接口：请求/响应模型集中在 `src/api/models.py`；SSE 与非流式端点行为必须保持一致。
- 路径：优先用 `pathlib`/相对路径，避免硬编码反斜杠（Windows 兼容）。

---

## 需要更多上下文时

- `README.md`：中文快速上手
- `CLAUDE.md`：工程约定（注意其 Pipeline 拓扑图为早期版本，Router 条件分支已不存在）
- 找实现位置：全局搜索符号  从 `src/core/factory.py` 追 DI 关系
