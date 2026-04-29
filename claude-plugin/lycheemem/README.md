# LycheeMem Claude Code Plugin

This directory contains the Claude Code plugin prototype for LycheeMem.

Installation guides:

- [English](./INSTALL_CLAUDE.md)
- [中文](./INSTALL_CLAUDE_zh.md)

## What it does

- Registers LycheeMem as a Claude Code MCP server through `.mcp.json`.
- Adds a `memory` skill that teaches Claude when to use LycheeMem retrieval and consolidation.
- Uses a `UserPromptSubmit` hook to recall LycheeMem context before each user prompt is processed.
- Uses `Stop` and `SessionEnd` hooks to append assistant responses and consolidate durable memory.
- Keeps the hook implementation stdlib-only, so no extra Python dependencies are required.

## Environment Variables

- `LYCHEEMEM_BASE_URL` default: `http://127.0.0.1:8000`
- `LYCHEEMEM_API_TOKEN` optional bearer token for hook HTTP calls
- `LYCHEEMEM_TIMEOUT` default: `120`
- `LYCHEEMEM_SESSION_PREFIX` default: `claude`
- `LYCHEEMEM_ENABLE_PRE_LLM_CONTEXT` default: `true`
- `LYCHEEMEM_ENABLE_AUTO_APPEND` default: `true`
- `LYCHEEMEM_ENABLE_AUTO_CONSOLIDATE` default: `true`
- `LYCHEEMEM_AUTO_CONSOLIDATE_BACKGROUND` default: `true`
- `LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS` default: `60`
- `LYCHEEMEM_SMART_SEARCH_TOP_K` default: `5`
- `LYCHEEMEM_SMART_SEARCH_MODE` default: `compact`
- `LYCHEEMEM_RESPONSE_LEVEL` default: `minimal`
- `LYCHEEMEM_MAX_CONTEXT_CHARS` default: `6000`
- `LYCHEEMEM_CLAUDE_DEBUG` default: `false`

This first version is environment-variable driven because Claude Code 2.1.72 does not yet accept `userConfig` in plugin manifests.

## Local Development Load

```bash
export LYCHEEMEM_REPO="/path/to/LycheeMem"
export LYCHEEMEM_BASE_URL="http://127.0.0.1:8000"
claude plugin validate "$LYCHEEMEM_REPO/claude-plugin/lycheemem"
claude --plugin-dir "$LYCHEEMEM_REPO/claude-plugin/lycheemem"
```

Start the LycheeMem server before starting Claude Code.
