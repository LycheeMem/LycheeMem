# Install LycheeMem For Claude Code

This guide installs the local LycheeMem Claude Code plugin for development testing.

## 1. Start LycheeMem

```bash
cd /path/to/LycheeMem
python main.py
```

By default, LycheeMem listens on port `8000`. If your server command is different, keep the same port or set `LYCHEEMEM_BASE_URL` below.

## 2. Validate The Plugin

```bash
export LYCHEEMEM_REPO="/path/to/LycheeMem"
export LYCHEEMEM_BASE_URL="http://127.0.0.1:8000"
claude plugin validate "$LYCHEEMEM_REPO/claude-plugin/lycheemem"
```

## 3. Start Claude Code With The Plugin

```bash
claude --plugin-dir "$LYCHEEMEM_REPO/claude-plugin/lycheemem"
```

This loads the plugin for the current Claude Code session. The plugin exposes LycheeMem MCP tools, the `memory` skill, and automatic recall/save hooks.

## 4. Verify In Claude Code

Inside Claude Code, run:

```text
/plugin
/mcp
```

Then try:

```text
Please remember that my favorite fruit is lychee.
What is my favorite fruit?
Use LycheeMem to recall my favorite fruit.
```

If automatic recall is working, Claude should answer from LycheeMem context. If MCP is working, Claude can also call tools under the `lycheemem` MCP server.

## Optional Debug Settings

```bash
export LYCHEEMEM_CLAUDE_DEBUG=true
export LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS=0
export LYCHEEMEM_RESPONSE_LEVEL=minimal
```

Restart Claude Code after changing plugin or MCP configuration.
