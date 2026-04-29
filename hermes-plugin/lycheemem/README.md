# LycheeMem Hermes Plugin

This directory contains the Hermes "Path B" standalone plugin for LycheeMem.

Installation guides:

- [English](./INSTALL_HERMES.md)
- [中文](./INSTALL_HERMES_zh.md)

## What it does

- Injects recalled LycheeMem context through the Hermes `pre_llm_call` hook.
- Mirrors completed user/assistant turns into LycheeMem through `post_llm_call`.
- Triggers best-effort consolidation on turn end and session finalize.
- Exposes manual Hermes tools:
  - `lycheemem_health`
  - `lycheemem_smart_search`
  - `lycheemem_append_turn`
  - `lycheemem_consolidate`
- Exposes a slash command:
  - `/lycheemem status`
  - `/lycheemem health`
  - `/lycheemem recall <query>`
  - `/lycheemem consolidate`

## Environment variables

- `LYCHEEMEM_BASE_URL` default: `http://127.0.0.1:8000`
- `LYCHEEMEM_TRANSPORT` default: `http`
- `LYCHEEMEM_MCP_URL` optional override for MCP transport
- `LYCHEEMEM_API_TOKEN` optional bearer token
- `LYCHEEMEM_TIMEOUT` default: `120`
- `LYCHEEMEM_ENABLE_PRE_LLM_CONTEXT` default: `true`
- `LYCHEEMEM_ENABLE_AUTO_APPEND` default: `true`
- `LYCHEEMEM_ENABLE_AUTO_CONSOLIDATE` default: `true`
- `LYCHEEMEM_ENABLE_FINALIZE_CONSOLIDATION` default: `true`
- `LYCHEEMEM_AUTO_CONSOLIDATE_BACKGROUND` default: `true`
- `LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS` default: `60`
- `LYCHEEMEM_SMART_SEARCH_TOP_K` default: `5`
- `LYCHEEMEM_SMART_SEARCH_MODE` default: `compact`
- `LYCHEEMEM_RESPONSE_LEVEL` default: `minimal`
- `LYCHEEMEM_SESSION_PREFIX` default: `hermes`

## Install as a Hermes user plugin

```bash
export LYCHEEMEM_REPO="/path/to/LycheeMem"
mkdir -p ~/.hermes/plugins
ln -sfn "$LYCHEEMEM_REPO/hermes-plugin/lycheemem" ~/.hermes/plugins/lycheemem
hermes plugins enable lycheemem
hermes plugins list
```

Start Hermes with the LycheeMem server already running.
