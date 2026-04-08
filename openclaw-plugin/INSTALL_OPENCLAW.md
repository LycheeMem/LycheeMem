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
- You know the local path of the LycheeMem repository

> **Current auth model:** the merged LycheeMem backend is currently running in a no-auth / single-tenant mode. The OpenClaw plugin therefore does not need a login flow or Bearer token for normal local use. If you later restore authenticated multi-user deployments, the optional `apiToken` field remains available for backward compatibility.

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

If newer OpenClaw builds complain about `HOOK.md missing`, that usually means the
local directory was not recognized as a plugin package. This repository now
ships a plugin `package.json` with `openclaw.extensions`, which newer OpenClaw
expects during local installs.

Newer OpenClaw builds may also block plugin installation if the plugin reads
environment variables and also performs network requests. This plugin now uses
the OpenClaw plugin config as the canonical source of settings during install
and runtime, which is the recommended path for current OpenClaw releases.

Verify installation:
```bash
openclaw plugins list
openclaw skills info lycheemem
openclaw skills check
```

---

## 2. Enable agent tools and confirm the skill

The plugin can be installed correctly and still remain invisible to the model if the current agent is not allowed to use the tools or the skill is disabled.

### Option A: OpenClaw page

In the left sidebar:

1. Open **Agents**
2. Select the agent you are using, usually `main`
3. Open **Tools**
4. Enable the LycheeMem tools you want the model to use
5. Save
6. Open **Skills**
7. Confirm `lycheemem` is enabled for the current agent
8. Save again if needed

Recommended minimum tool set:
- `lychee_memory_smart_search`
- `lychee_memory_consolidate`
- `lychee_memory_append_turn`

Optional but useful:
- `lychee_memory_search`
- `lychee_memory_synthesize`

### Option B: `~/.openclaw/openclaw.json`

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

## 3. Configure the plugin in Automation -> Plugins

### Option A: OpenClaw page

In the left sidebar:

1. Open **Automation**
2. Open **Plugins**
3. Find the LycheeMem entry:
   `Thin OpenClaw adapter for LycheeMem structured memory tools. (plugin: lycheemem-tools)`
4. Open **LycheeMem Tools Config**
5. Expand **advanced** if needed
6. Fill in the connection fields and switches below
7. Save

Fill in at minimum:
- **LycheeMem Base URL** = `http://127.0.0.1:8000`
- **Transport** = `mcp`
- **API Token (Optional)** = leave empty for the current no-auth backend

Make sure these plugin-level switches are enabled:
- **Enable LycheeMem Tools**
- **Plugin Hook Policy**

Recommended switches:
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

Restart the gateway after saving plugin config:
```bash
openclaw gateway restart
```
> **Note:** Do not assume that plugin installation or config updates are hot-reloaded by a running gateway. When in doubt, stop and restart manually.

---

## 4. Ensure the LycheeMem backend is running before use

Before testing real conversations, make sure the LycheeMem server is already up and reachable from OpenClaw.

Default local endpoint:
- `http://127.0.0.1:8000`

Health check:
```bash
curl http://127.0.0.1:8000/health
```

Expected:
- the server responds successfully
- the URL configured in **Automation -> Plugins -> LycheeMem Tools Config** matches the real backend address

If you are on WSL, bind the server to `0.0.0.0:8000` if needed, but access it from OpenClaw with `http://127.0.0.1:8000` or `http://localhost:8000`, not `http://0.0.0.0:8000`.

---

## Verification

### Verify the skill is mounted
```bash
openclaw skills info lycheemem
openclaw skills check
```
**Expected:** `lycheemem` shows `Ready`.

Also confirm:
- the skill is enabled
- the current agent has LycheeMem tools enabled under **Agents -> Tools**
- the plugin entry `lycheemem-tools` is enabled under **Automation -> Plugins**

### Verify plugin config is loaded

In the OpenClaw page, reopen:
- **Automation**
- **Plugins**
- `Thin OpenClaw adapter for LycheeMem structured memory tools. (plugin: lycheemem-tools)`
- **LycheeMem Tools Config**

Confirm:
- **Enable LycheeMem Tools** is on
- **Plugin Hook Policy** is on
- Base URL / token / transport are saved correctly

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

Also verify:
- the current agent has LycheeMem tools enabled under **Agents -> Tools**
- `lycheemem` is enabled under **Agents -> Skills**
- the plugin entry `lycheemem-tools` is enabled under **Automation -> Plugins**
- **Enable LycheeMem Tools** is on
- **Plugin Hook Policy** is on
- the backend at `http://127.0.0.1:8000` is actually reachable

**Solution:**
1. Restart the gateway
2. Open a new session and retry

### Model is calling `append_turn` manually
Turn mirroring is handled automatically by hooks in this version. If the model continues to call `append_turn` manually, the gateway has likely not been restarted or the session is still using an old prompt.

### No consolidate after `/new`
Check:
1. `enableBoundaryConsolidation` is `true`
2. The gateway process was actually restarted

---

## Related docs
- `openclaw-plugin/SKILL.md`
