# LycheeMem Hermes 插件快速配置

把 LycheeMem 接到 Hermes Agent，并完成基础验证。

---

## 这版插件提供什么

- `lycheemem_smart_search` 作为手动长期记忆检索工具
- `pre_llm_call` 自动召回 LycheeMem 记忆，并把 `background_context` 注入当前轮用户消息
- `post_llm_call` 自动把用户消息和助手回答镜像到 LycheeMem session store
- `on_session_end` / `on_session_finalize` 自动触发长期记忆固化
- `/lycheemem` slash command，方便手动检查状态、健康检查、检索和固化

正常情况下：

- 模型不需要手动调用 `lycheemem_append_turn`
- 模型不需要手动调用 `lycheemem_consolidate`
- 模型也不一定需要显式调用 `lycheemem_smart_search`，因为插件会在 `pre_llm_call` 阶段自动 recall
- 如果你想调试检索结果，可以手动使用 `/lycheemem recall <query>`

---

## 前提

先确认以下条件成立：

- 已安装 Hermes Agent，且可执行 `hermes`
- 已知道 LycheeMem 仓库在本机上的路径
- LycheeMem 服务端可以启动
- Hermes 和 LycheeMem 能访问同一个 `127.0.0.1:<port>` 或同一个内网地址

当前插件是 Hermes standalone plugin，也就是 Path B 插件形态。它不是 Hermes memory provider，因此不会替换 Hermes 原生 memory slot，而是通过 hooks + tools 给 Hermes 增加 LycheeMem 能力。

---

## 1. 启动 LycheeMem 服务端

假设 LycheeMem 仓库位于：

```bash
export LYCHEEMEM_REPO="/path/to/LycheeMem"
```

启动服务端：

```bash
cd "$LYCHEEMEM_REPO"
conda activate lycheemem
python main.py
```

确认健康检查：

```bash
curl http://127.0.0.1:8000/health
```

如果你的 `.env` 里配置了其他端口，例如：

```env
API_PORT=8100
```

那后续 `LYCHEEMEM_BASE_URL` 也要对应改成：

```bash
export LYCHEEMEM_BASE_URL="http://127.0.0.1:8100"
```

---

## 2. 安装 Hermes 插件

Hermes 用户插件目录是 `~/.hermes/plugins/`。推荐用软链接安装，方便后续直接跟随仓库代码更新：

```bash
mkdir -p ~/.hermes/plugins
ln -sfn "$LYCHEEMEM_REPO/hermes-plugin/lycheemem" ~/.hermes/plugins/lycheemem
hermes plugins enable lycheemem
hermes plugins list
```

`hermes plugins enable lycheemem` 会启用插件；后续用 `hermes plugins list` 确认状态即可。

---

## 3. 配置连接参数

最小配置只需要指定 LycheeMem 地址：

```bash
export LYCHEEMEM_BASE_URL="http://127.0.0.1:8000"
export LYCHEEMEM_TRANSPORT="http"
```

如果你使用 MCP transport：

```bash
export LYCHEEMEM_BASE_URL="http://127.0.0.1:8000"
export LYCHEEMEM_TRANSPORT="mcp"
export LYCHEEMEM_MCP_URL="http://127.0.0.1:8000/mcp"
```

当前推荐先用 `http` 路径完成验证。HTTP 路径更直接，适合排查插件、服务端和检索逻辑是否跑通；MCP 路径确认 HTTP 正常后再切换。

常用配置项：

```bash
export LYCHEEMEM_ENABLE_PRE_LLM_CONTEXT=true
export LYCHEEMEM_ENABLE_AUTO_APPEND=true
export LYCHEEMEM_ENABLE_AUTO_CONSOLIDATE=true
export LYCHEEMEM_ENABLE_FINALIZE_CONSOLIDATION=true
export LYCHEEMEM_AUTO_CONSOLIDATE_BACKGROUND=true
export LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS=60
export LYCHEEMEM_SMART_SEARCH_MODE=compact
export LYCHEEMEM_RESPONSE_LEVEL=minimal
export LYCHEEMEM_SMART_SEARCH_TOP_K=5
```

调试时可以让固化同步执行，方便马上验证结果：

```bash
export LYCHEEMEM_AUTO_CONSOLIDATE_BACKGROUND=false
export LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS=0
```

生产或长会话使用时建议恢复后台固化：

```bash
export LYCHEEMEM_AUTO_CONSOLIDATE_BACKGROUND=true
export LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS=60
```

---

## 4. 启动 Hermes 并验证

启动 Hermes：

```bash
hermes
```

进入 Hermes 后，先确认插件可见：

```text
/plugins
```

测试插件状态：

```text
/lycheemem status
```

测试 LycheeMem 服务端连接：

```text
/lycheemem health
```

测试手动检索：

```text
/lycheemem recall favorite fruit
```

---

## 5. 端到端记忆测试

先让 Hermes 记住一条偏好：

```text
Please remember that my favorite fruit is lychee.
```

再问：

```text
What is my favorite fruit?
```

为了排除 Hermes 原生上下文或原生 memory slot 的影响，可以开启一个新的 Hermes session 后再问：

```text
What is my favorite fruit?
```

也可以手动检索 LycheeMem：

```text
/lycheemem recall favorite fruit
```

如果链路正常，`/lycheemem recall` 应该返回包含 `lychee` 的 `background_context`。

---

## 6. 自动 recall 和手动工具调用的区别

插件默认会通过 `pre_llm_call` 自动 recall。也就是说，当你输入：

```text
What is my favorite fruit?
```

插件会在模型回答前自动调用 LycheeMem smart search，并把结果注入当前用户消息。Hermes UI 通常不会把 hook 调用显示成显式 tool call。

手动工具和 slash command 主要用于调试：

```text
/lycheemem recall favorite fruit
```

如果你希望模型显式调用 `lycheemem_smart_search`，可以关闭自动 recall：

```bash
export LYCHEEMEM_ENABLE_PRE_LLM_CONTEXT=false
```

然后在 Hermes 中提示模型：

```text
Before answering memory-related questions, use the lycheemem_smart_search tool.
```

更推荐的默认模式仍然是自动 recall，因为它不依赖模型自己决定是否调用工具。

---

## 7. 可用工具和命令

Hermes tools：

- `lycheemem_health`
- `lycheemem_smart_search`
- `lycheemem_append_turn`
- `lycheemem_consolidate`

Slash command：

```text
/lycheemem status
/lycheemem health
/lycheemem recall <query>
/lycheemem consolidate
```

---

## 8. 常见问题

### `/plugins` 里看不到 `lycheemem`

检查软链接：

```bash
ls -la ~/.hermes/plugins/lycheemem
```

重新启用并检查插件列表：

```bash
hermes plugins enable lycheemem
hermes plugins list
```

然后重启 Hermes。

### `/lycheemem health` 失败

先确认 LycheeMem 服务端已经启动：

```bash
curl http://127.0.0.1:8000/health
```

如果你的服务端端口不是 `8000`，同步修改：

```bash
export LYCHEEMEM_BASE_URL="http://127.0.0.1:8100"
```

### `/lycheemem recall` 返回空

先确认是否已经发生过 append 和 consolidate。调试时建议使用：

```bash
export LYCHEEMEM_AUTO_CONSOLIDATE_BACKGROUND=false
export LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS=0
```

然后重新跑一轮端到端测试。

也可以直接查 LycheeMem 服务端日志，确认是否出现：

- `/memory/append-turn`
- `/memory/consolidate`
- `/memory/smart-search`

### `python main.py` 启动很慢

LycheeMem 启动时可能会补全历史 session 的 turn 向量索引。如果历史数据较多，且 embedding 走远端 API，启动会变慢。

排查方向：

- 检查 `.env` 里的 `EMBEDDING_API_BASE`
- 优先使用本地 embedding 服务
- 确认远端 embedding endpoint 可用
- 如果只是调试插件，可以先用空数据目录或临时数据目录启动

### Hermes 能答对，但 `/lycheemem recall` 空

这通常说明 Hermes 原生 memory slot 或当前上下文答对了，但 LycheeMem 没召回。应该以 `/lycheemem recall <query>` 和 LycheeMem 服务端日志为准来判断 LycheeMem 链路是否真的工作。

---

## 9. 更新插件

如果使用软链接安装，更新仓库代码后重启 Hermes 即可：

```bash
cd "$LYCHEEMEM_REPO"
git pull
```

然后重启 Hermes。

---

## 10. 卸载插件

删除软链接：

```bash
rm ~/.hermes/plugins/lycheemem
```

禁用插件：

```bash
hermes plugins disable lycheemem
```

然后重启 Hermes。
