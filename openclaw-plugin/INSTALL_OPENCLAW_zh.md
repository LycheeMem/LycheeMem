# LycheeMem OpenClaw 快速配置

5 分钟内把 LycheeMem 插件接到 OpenClaw，并完成基础验证。

---

## 这版插件提供什么

- `lychee_memory_smart_search` 作为默认长期记忆检索入口
- 宿主自动镜像对话 turn（通过 hook 实现，模型无需手动调用）
  - 用户消息自动 `append_turn`
  - 助手消息自动 `append_turn`
- `/new`、`/reset`、`/stop`、`session_end` 自动触发边界 `consolidate`
- 对明显长期知识信号触发提前 `consolidate`

**正常情况下：**
- 模型不需要手动调用 `lychee_memory_append_turn`
- 模型可以继续调用 `lychee_memory_smart_search`
- 模型在必要时可以手动调用 `lychee_memory_consolidate`

---

## 前提

先确认以下条件成立：

- 已安装 OpenClaw，且可执行 `openclaw`
- 已知道 LycheeMem 仓库在本机上的路径

> **当前认证模型：** 合并后的 LycheeMem 后端目前是无认证 / 单租户模式。OpenClaw 插件在本地正常使用时不再需要登录流程或 Bearer token。若你之后恢复带认证的多用户部署，`apiToken` 仍然保留为可选兼容字段。

---

## 1. 安装插件

假设 LycheeMem 仓库位于：
```bash
export LYCHEEMEM_REPO="/path/to/LycheeMem"
```

执行安装：
```bash
openclaw plugins install "$LYCHEEMEM_REPO/openclaw-plugin"
```

如果新版 OpenClaw 报 `HOOK.md missing`，通常不是你真的要装 hook，而是本地目录没有被识别成新版插件包。这个仓库现在已经补上了带 `openclaw.extensions` 的插件 `package.json`，用于兼容新版本地安装器。

另外，较新的 OpenClaw 版本还会拦截“读取环境变量并发起网络请求”的插件安装路径。这个插件现在改为以 OpenClaw 插件配置页里的参数为准，不再依赖安装时的环境变量兜底，这也是当前版本更稳妥的接入方式。

安装后检查：
```bash
openclaw plugins list
openclaw skills info lycheemem
openclaw skills check
```

---

## 2. 打开 agent 工具并确认 skill 已启用

即使插件已经装好了，如果当前 agent 没有放行 LycheeMem 工具，或者 skill 没有启用，模型仍然看不到它。

### 方式 A：OpenClaw 页面操作

在左侧栏中：

1. 打开 **代理**
2. 选择你当前使用的 agent，通常是 `main`
3. 打开 **Tools**
4. 勾选需要给模型开放的 LycheeMem 工具
5. 保存
6. 再打开 **Skills**
7. 确认 `lycheemem` 已启用
8. 如有需要再次保存

建议至少开启这些工具：
- `lychee_memory_smart_search`
- `lychee_memory_consolidate`
- `lychee_memory_append_turn`

可选但推荐：
- `lychee_memory_search`
- `lychee_memory_synthesize`

### 方式 B：直接修改 `~/.openclaw/openclaw.json`

```json
{
  "agents": {
    "list": [
      {
        "id": "main",
        "tools": {
          "alsoAllow": [
            "lychee_memory_smart_search",
            "lychee_memory_consolidate",
            "lychee_memory_append_turn",
            "lychee_memory_search",
            "lychee_memory_synthesize"
          ]
        }
      }
    ]
  },
  "skills": {
    "entries": {
      "lycheemem": {
        "enabled": true
      }
    }
  }
}
```

---

## 3. 在“自动化 -> Plugins”中配置插件

### 方式 A：OpenClaw 页面操作

在左侧栏中：

1. 打开 **自动化**
2. 打开 **Plugins**
3. 找到 LycheeMem 插件项：
   `Thin OpenClaw adapter for LycheeMem structured memory tools. (plugin: lycheemem-tools)`
4. 打开 **LycheeMem Tools Config**
5. 需要时展开 **advanced**
6. 填写下面这些连接配置和开关
7. 保存

至少填写：
- **LycheeMem Base URL** = `http://127.0.0.1:8000`
- **Transport** = `mcp`
- **API Token（可选）** = 当前无认证后端可留空

先确认这两个插件级开关已开启：
- **Enable LycheeMem Tools**
- **Plugin Hook Policy**

建议保持以下开关开启：
- **Enable Host Lifecycle** = `true`
- **Inject Prompt Presence** = `true`
- **Auto Append Turns** = `true`
- **Boundary Consolidation** = `true`
- **Proactive Consolidation** = `true`

推荐默认值：
- **Proactive Cooldown** = `180`

### 方式 B：直接修改 `~/.openclaw/openclaw.json`

```json
{
  "skills": {
    "entries": {
      "lycheemem": {
        "enabled": true
      }
    }
  },
  "plugins": {
    "entries": {
      "lycheemem-tools": {
        "enabled": true,
        "config": {
          "baseUrl": "http://127.0.0.1:8000",
          "transport": "mcp",
          "apiToken": "",
          "enableHostLifecycle": true,
          "enablePromptPresence": true,
          "enableAutoAppendTurns": true,
          "enableBoundaryConsolidation": true,
          "enableProactiveConsolidation": true,
          "proactiveConsolidationCooldownSeconds": 180
        }
      }
    }
  }
}
```

保存插件配置后，重启 gateway：
```bash
openclaw gateway restart
```
> **注意：** 不要假设插件安装或配置更新会被当前 gateway 热加载。如不确定，建议停止后手动重启。

---

## 4. 运行前确保 LycheeMem 服务端已启动

真正开始对话和测试前，先确认 LycheeMem 后端已经启动，并且 OpenClaw 能访问到它。

默认本地地址：
- `http://127.0.0.1:8000`

健康检查：
```bash
curl http://127.0.0.1:8000/health
```

预期：
- 服务正常返回
- **自动化 -> Plugins -> LycheeMem Tools Config** 中填写的 URL 和真实后端地址一致

如果你在 WSL 里运行服务，服务端可以监听 `0.0.0.0:8000`，但 OpenClaw 侧通常应填写 `http://127.0.0.1:8000` 或 `http://localhost:8000`，不要直接写 `http://0.0.0.0:8000`。

---

## 快速验证

### 验证 skill 已挂载
```bash
openclaw skills info lycheemem
openclaw skills check
```
**预期：** lycheemem 显示 `Ready`。

同时确认：
- 当前 agent 在 **代理 -> Skills** 中已启用 `lycheemem`
- 当前 agent 在 **代理 -> Tools** 中已经启用了 LycheeMem 相关工具
- `lycheemem-tools` 插件项已在 **自动化 -> Plugins** 中启用

### 验证插件配置页已正确保存

在 OpenClaw 页面里重新打开：
- **自动化**
- **Plugins**
- `Thin OpenClaw adapter for LycheeMem structured memory tools. (plugin: lycheemem-tools)`
- **LycheeMem Tools Config**

确认以下内容：
- **Enable LycheeMem Tools** 已开启
- **Plugin Hook Policy** 已开启
- Base URL / token / transport 已正确保存

### 验证长期记忆检索
在会话里提问：
- *"这个项目的长期背景是什么？"*
- *"上次这个项目怎么处理的？"*

**预期：** 模型会调用 `lychee_memory_smart_search`。

### 验证自动 append
发送一条普通消息，不手动调用任何 LycheeMem 工具。

**预期：**
- 后端出现一次用户 `append_turn`
- 后端出现一次助手 `append_turn`

### 验证边界 consolidate
执行 `/new` 或 `/reset`。

**预期：** 后端出现一次 `consolidate(background=true)`。

### 验证提前 consolidate
发送一条明显带长期记忆信号的话，例如：
- *"记住：这个项目文档默认中文"*
- *"以后部署前要跑 Ruff 和 Pytest"*

**预期：** 随后出现一次后台 `consolidate`。

---

## 常见问题

### 模型看不到 skill
先检查：
```bash
openclaw skills info lycheemem
openclaw skills check
```
如果 skill 已 `Ready` 但模型仍像没看到，通常是 gateway 还未重启或当前会话使用旧 prompt。

还要额外确认：
- 当前 agent 已在 **代理 -> Tools** 中启用 LycheeMem 工具
- 当前 agent 已在 **代理 -> Skills** 中启用 `lycheemem`
- `lycheemem-tools` 插件项已在 **自动化 -> Plugins** 中启用
- **Enable LycheeMem Tools** 已开启
- **Plugin Hook Policy** 已开启
- `http://127.0.0.1:8000` 对 OpenClaw 来说确实可达

**解决办法：** 
1. 重启 gateway
2. 新开一个会话再测

### 模型重复手动调用 append_turn
当前版本宿主已通过 hook 自动镜像 turn，模型无需手动调用。如果仍然重复，通常是 gateway 还未重启或当前会话仍在使用旧提示。

### `/new` 后没有看到 consolidate
优先检查：
1. `enableBoundaryConsolidation` 是否为 `true`
2. 是否真的跑在新的 gateway 进程上

---

## 相关文档
- `openclaw-plugin/SKILL.md`
