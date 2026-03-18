# LycheeMemOS 工作区 Copilot 指南

你正在协助开发 **LycheeMemOS：Training-free Agentic Cognitive Memory Framework**（Python + FastAPI + LangGraph + 可选 Neo4j/LanceDB，另含一个 React/Vite Web Demo）。

目标：在不破坏既有架构的前提下，快速定位代码入口、按项目约定实现改动，并用现有测试与 Ruff 校验。

## 快速命令（Backend）

- 安装（开发依赖）：
  - `pip install -e ".[dev]"`
- 启动服务：
  - `python -m lychee_memos --llm openai`
  - `python -m lychee_memos --llm gemini`
  - `python -m lychee_memos --llm ollama`
  - 开发热重载：`python -m lychee_memos --reload`
- 测试：
  - 全量：`pytest tests/ -v`
  - 单测：`pytest tests/test_graph_store.py -v`
  - 单用例：`pytest tests/test_graph_store.py::test_name -v`
- 代码风格：
  - `ruff check src/`
  - `ruff format src/`

## 快速命令（Web Demo）

在 `web-demo/` 目录：

- 安装：`npm install`
- 开发：`npm run dev`
- 构建：`npm run build`
- 预览：`npm run preview`

## 配置与环境变量（务必注意默认值陷阱）

- 所有配置由 `src/core/config.py` 的 Pydantic Settings 读取，来源：环境变量或 `.env`（参考 `.env.example`）。
- `.env` 已在 `.gitignore` 中忽略；不要把真实密钥提交到仓库。

常用配置项（以 `.env.example` 为准）：

- LLM：`OPENAI_API_KEY` / `GEMINI_API_KEY` / `OLLAMA_BASE_URL`，以及对应 `*_MODEL`
- Embedder：`EMBEDDING_BACKEND`（`openai` 或 `gemini`）
- 存储后端：
  - `SESSION_BACKEND=memory|sqlite`
  - `GRAPH_BACKEND=memory|neo4j`
  - `SKILL_BACKEND=memory|file|lancedb`

重要提醒：

- `src/core/config.py` 里存在“可运行但不适合默认使用”的硬编码默认值（例如第三方 API key、Neo4j 密码、以及默认启用的后端选择）。开发/部署时应通过 `.env` 明确覆盖这些值。
- 如果你本地没有 Neo4j，先在 `.env` 里设置 `GRAPH_BACKEND=memory`，否则启动时可能会尝试连接 Neo4j。

## 代码入口与结构导航

建议按以下顺序定位问题：

- CLI/启动入口：`src/__main__.py`
- FastAPI：`src/api/server.py`
- Pipeline DI 工厂：`src/core/factory.py`（`create_pipeline()`）
- LangGraph 封装：`src/core/graph.py`
- 共享状态结构：`src/core/state.py`（`PipelineState`）

核心分层：

- `src/agents/`：认知 Agent（多数继承 `base_agent.py`；`WMManager` 为规则型）
- `src/memory/`：四类记忆基质
  - `working/`：会话存储 + 压缩器
  - `graph/`：NetworkX/Neo4j 图谱 + 实体抽取
  - `procedural/`：技能库（memory/file/lancedb）
  - `sensory/`：感官缓冲
- `src/llm/` 与 `src/embedder/`：OpenAI/Gemini/Ollama 适配器

说明：README 中的“Router”概念在实现上主要体现在 `SearchCoordinator` 的检索编排逻辑里，不一定有独立的 `router_agent.py`。

## 贡献时的约定（请遵守）

- 优先做“根因修复”，避免只加表面补丁。
- 保持改动小而聚焦：不要引入与需求无关的重构、不要新增不必要依赖。
- 新增功能或修复 bug 时：
  - 优先补充/更新对应 `tests/` 用例（项目大量使用 `FakeLLM` / `FakeEmbedder` 来避免真实 API 调用）。
  - 运行 `ruff check src/`、`ruff format src/`、`pytest tests/ -v`。
- FastAPI 接口：保持请求/响应模型在 `src/api/models.py` 中集中定义，SSE 与非流式端点行为要一致。

## 常见坑位（尤其在 Windows）

- 端口冲突：默认 8000，被占用时使用 `--port`。
- 路径：优先用 `pathlib`/相对路径，避免硬编码反斜杠。
- 可选生产依赖：Neo4j/LanceDB 相关功能需要 `pip install -e ".[production]"`。

## 需要更多上下文时

- 先读：`README.md`（中文快速上手）与 `CLAUDE.md`（更偏工程约定/测试模式）。
- 需要找“在哪里实现的”：优先使用全局搜索定位符号，再从 `src/core/factory.py` 追 DI 关系。
