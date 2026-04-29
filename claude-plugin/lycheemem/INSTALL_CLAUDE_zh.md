# 安装 LycheeMem Claude Code 插件

这份文档用于本地开发测试 LycheeMem 的 Claude Code 插件。

## 1. 启动 LycheeMem

```bash
cd /path/to/LycheeMem
python main.py
```

默认情况下，LycheeMem 会监听 `8000` 端口。如果你的服务端启动命令不同，可以继续使用自己的命令，但要保持端口一致，或者在下面设置 `LYCHEEMEM_BASE_URL`。

## 2. 校验插件

```bash
export LYCHEEMEM_REPO="/path/to/LycheeMem"
export LYCHEEMEM_BASE_URL="http://127.0.0.1:8000"
claude plugin validate "$LYCHEEMEM_REPO/claude-plugin/lycheemem"
```

## 3. 带插件启动 Claude Code

```bash
claude --plugin-dir "$LYCHEEMEM_REPO/claude-plugin/lycheemem"
```

这个命令会在当前 Claude Code 会话中加载插件。插件会提供 LycheeMem MCP tools、`memory` skill，以及自动 recall/save hooks。

## 4. 在 Claude Code 中验证

进入 Claude Code 后运行：

```text
/plugin
/mcp
```

然后尝试：

```text
Please remember that my favorite fruit is lychee.
What is my favorite fruit?
Use LycheeMem to recall my favorite fruit.
```

如果自动 recall 生效，Claude 应该能基于 LycheeMem 上下文回答。如果 MCP 生效，Claude 也能调用 `lycheemem` MCP server 下的工具。

## 可选调试项

```bash
export LYCHEEMEM_CLAUDE_DEBUG=true
export LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS=0
export LYCHEEMEM_RESPONSE_LEVEL=minimal
```

修改插件或 MCP 配置后，需要重启 Claude Code。
