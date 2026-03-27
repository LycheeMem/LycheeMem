# LycheeMem OpenClaw Setup Guide

Get the LycheeMem plugin connected to OpenClaw in under 5 minutes.

---

## What this plugin provides

- `lychee_memory_smart_search` as the default long-term memory retrieval entry point
- Automatic turn mirroring via hooks — the model does not need to call `append_turn` manually
  - User messages are appended automatically
  - Assistant messages are appended automatically
- `/new`, `/reset`, `/stop`, and `session_end` automatically trigger boundary `consolidate`
- Proactive `consolidate` on strong long-term knowledge signals

**Under normal operation:**
- The model does **not** need to call `lychee_memory_append_turn` manually
- The model can call `lychee_memory_smart_search` as needed
- The model may call `lychee_memory_consolidate` manually when necessary

---

## Prerequisites

Confirm the following before proceeding:

- OpenClaw is installed and the `openclaw` command is available
- The LycheeMem backend is running (default: `http://127.0.0.1:8000`)
- You have a LycheeMem Bearer Token (see below)

**Check backend health:**
```bash
curl http://127.0.0.1:8000/health
```

### Obtaining a Bearer Token

LycheeMem uses JWT authentication. Register once, then log in to retrieve a token.

#### Register (first time)
```bash
curl -s -X POST http://127.0.0.1:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}' \
  | python3 -m json.tool
```
*Example response:*
```json
{
  "user_id": "...",
  "username": "your_username",
  "display_name": "your_username",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### Log in (existing account)
```bash
curl -s -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}' \
  | python3 -m json.tool
```
*The response structure is identical to registration. Copy the `token` field value.*

> **Note:** Tokens are valid for 7 days by default. Log in again when the token expires.

---

## 1. Install the plugin

Set the path to the LycheeMem repository:
```bash
export LYCHEEMEM_REPO="/path/to/LycheeMem"
```

Install:
```bash
openclaw plugins install "$LYCHEEMEM_REPO/openclaw-plugin"
```

Verify installation:
```bash
openclaw plugins list
openclaw skills info lycheemem
openclaw skills check
```

---

## 2. Configure the plugin

### Option A: Dashboard

Fill in at minimum:
- **LycheeMem Base URL** = `http://127.0.0.1:8000`
- **Transport** = `mcp`
- **API Token** = `your LycheeMem Bearer Token`

Recommended switches (all on):
- **Enable Host Lifecycle** = `true`
- **Inject Prompt Presence** = `true`
- **Auto Append Turns** = `true`
- **Boundary Consolidation** = `true`
- **Proactive Consolidation** = `true`

Recommended default:
- **Proactive Cooldown** = `180`

### Option B: `~/.openclaw/openclaw.json`

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

## 3. Restart the gateway

Restart the gateway to apply changes:
```bash
openclaw gateway restart
```
> **Note:** Do not assume that plugin installation or config updates are hot-reloaded by a running gateway. When in doubt, stop and restart manually.

---

## 4. Verification

### Verify the skill is mounted
```bash
openclaw skills info lycheemem
openclaw skills check
```
**Expected:** `lycheemem` shows `Ready`.

### Verify long-term memory retrieval
Ask in a session:
- *"What is the long-term context for this project?"*
- *"How did we handle this last time?"*

**Expected:** the model calls `lychee_memory_smart_search`.

### Verify automatic turn appending
Send a regular message without manually calling any LycheeMem tool.

**Expected:**
- One user `append_turn` appears in the backend
- One assistant `append_turn` appears in the backend

### Verify boundary consolidate
Run `/new` or `/reset`.

**Expected:** one `consolidate(background=true)` call appears in the backend.

### Verify proactive consolidate
Send a message with a clear long-term memory signal, for example:
- *"Remember: project docs default to Chinese"*
- *"Always run Ruff and Pytest before deploying"*

**Expected:** one background `consolidate` call follows shortly after.

---

## Troubleshooting

### Model cannot see the skill
Check:
```bash
openclaw skills info lycheemem
openclaw skills check
```
If the skill shows `Ready` but the model still does not use it, the gateway has likely not been restarted or the current session is using a stale prompt. 

**Solution:**
1. Restart the gateway
2. Open a new session and retry

### Model is calling `append_turn` manually
Turn mirroring is handled automatically by hooks in this version. If the model continues to call `append_turn` manually, the gateway has likely not been restarted or the session is still using an old prompt.

### No consolidate after `/new`
Check:
1. `enableBoundaryConsolidation` is `true`
2. The token is valid
3. The gateway process was actually restarted

---

## Related docs
- `openclaw-plugin/SKILL.md`
