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
- LycheeMem 后端已启动（默认地址 `http://127.0.0.1:8000`）
- 已准备好 LycheeMem Bearer Token（见下方获取方式）

**检查后端健康状态：**
```bash
curl http://127.0.0.1:8000/health
```

### 获取 Bearer Token

LycheeMem 使用 JWT 认证。首次使用需注册账号，之后通过登录获取 token。

#### 注册（首次）
```bash
curl -s -X POST http://127.0.0.1:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}' \
  | python3 -m json.tool
```
*响应示例：*
```json
{
  "user_id": "...",
  "username": "your_username",
  "display_name": "your_username",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### 登录（已有账号）
```bash
curl -s -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}' \
  | python3 -m json.tool
```
*响应结构与注册相同，取 `token` 字段的值即可。*

> **注意：** Token 默认有效期 7 天。过期后需重新登录获取新 token。

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

安装后检查：
```bash
openclaw plugins list
openclaw skills info lycheemem
openclaw skills check
```

---

## 2. 配置插件

### 方式 A：Dashboard

至少填写以下配置：
- **LycheeMem Base URL** = `http://127.0.0.1:8000`
- **Transport** = `mcp`
- **API Token** = `你的 LycheeMem Bearer Token`

建议保持以下开关开启：
- **Enable Host Lifecycle** = `true`
- **Inject Prompt Presence** = `true`
- **Auto Append Turns** = `true`
- **Boundary Consolidation** = `true`
- **Proactive Consolidation** = `true`

推荐默认值：
- **Proactive Cooldown** = `180`

### 方式 B：`~/.openclaw/openclaw.json`

```json
{
  "plugins": {
    "entries": {
      "lycheemem-tools": {
        "enabled": true,
        "config": {
          "baseUrl": "http://127.0.0.1:8000",
          "transport": "mcp",
          "apiToken": "YOUR_LYCHEEMEM_BEARER_TOKEN",
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

---

## 3. 重启 gateway

配置完成后，需重启 gateway 使插件生效：
```bash
openclaw gateway restart
```
> **注意：** 不要假设插件安装或配置更新会被当前 gateway 热加载。如不确定，建议停止后手动重启。

---

## 4. 快速验证

### 验证 skill 已挂载
```bash
openclaw skills info lycheemem
openclaw skills check
```
**预期：** lycheemem 显示 `Ready`。

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
**解决办法：** 
1. 重启 gateway
2. 新开一个会话再测

### 模型重复手动调用 append_turn
当前版本宿主已通过 hook 自动镜像 turn，模型无需手动调用。如果仍然重复，通常是 gateway 还未重启或当前会话仍在使用旧提示。

### `/new` 后没有看到 consolidate
优先检查：
1. `enableBoundaryConsolidation` 是否为 `true`
2. token 是否有效
3. 是否真的跑在新的 gateway 进程上

---

## 相关文档
- `openclaw-plugin/SKILL.md`
