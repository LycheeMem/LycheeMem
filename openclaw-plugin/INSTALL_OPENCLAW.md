# LycheeMem OpenClaw Setup Guide

Get the LycheeMem plugin connected to OpenClaw in under 5 minutes.

---

## What this plugin provides

- `lychee_memory_smart_search` as the default long-term memory retrieval entry point
- Automatic turn mirroring via hooks — the model does not need to call `append_turn` manually
  - User messages are appended automatically
  - Assistant messages are appended automatically
- `/new`, `/reset`, `/stop`, and `session_end` automatically trigger boundary `consolidate`
- Optional proactive `consolidate` on strong long-term knowledge signals

**Under normal operation:**
- The model does **not** need to call `lychee_memory_append_turn` manually
- The model can call `lychee_memory_smart_search` as needed
- The model may call `lychee_memory_consolidate` manually when necessary

---

## Prerequisites

Confirm the following before proceeding:

- OpenClaw is installed and the `openclaw` command is available
- You know the local path of the LycheeMem repository
- Python 3.9+ is available if you want to use the helper setup and verify scripts

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

## 2. Configure agent tools and the skill

The latest setup script can now handle this automatically.  
`setup_openclaw_plugin.py` does not just write `plugins.entries.lycheemem-tools`; it also updates:

- `skills.entries.lycheemem.enabled = true`
- `agents.list[main].skills`, adding `lycheemem`
- `agents.list[main].tools.alsoAllow`, adding the core LycheeMem tools

In normal use, this means you no longer need to manually enable the agent tools and skill in the UI.

### Option A: helper script

Run:
```bash
python "$LYCHEEMEM_REPO/openclaw-plugin/scripts/setup_openclaw_plugin.py"
```

By default it wires the following core tools into the `main` agent:
- `lychee_memory_smart_search`
- `lychee_memory_append_turn`
- `lychee_memory_consolidate`

To target a different agent:
```bash
python "$LYCHEEMEM_REPO/openclaw-plugin/scripts/setup_openclaw_plugin.py" \
  --agent-id your-agent-id
```

To also allow extra tools:
```bash
python "$LYCHEEMEM_REPO/openclaw-plugin/scripts/setup_openclaw_plugin.py" \
  --tool lychee_memory_search \
  --tool lychee_memory_synthesize
```

### Option B: OpenClaw page

If you want to confirm the result after the script runs:

1. Open **Agents**
2. Select the current agent, usually `main`
3. Open **Tools**
4. Confirm these are allowed:
   - `lychee_memory_smart_search`
   - `lychee_memory_append_turn`
   - `lychee_memory_consolidate`
5. Open **Skills**
6. Confirm `lycheemem` is enabled

### Option C: `~/.openclaw/openclaw.json`

```json
{
  "agents": {
    "list": [
      {
        "id": "main",
        "skills": [
          "lycheemem"
        ],
        "tools": {
          "alsoAllow": [
            "lychee_memory_smart_search",
            "lychee_memory_consolidate",
            "lychee_memory_append_turn"
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

The recommended path is: write the plugin config, skill entry, and agent tools
with the helper script first, then open the OpenClaw page once to confirm it saved correctly.

### Option A: helper script

Run:
```bash
python "$LYCHEEMEM_REPO/openclaw-plugin/scripts/setup_openclaw_plugin.py"
```

What it does:
- creates or updates `plugins.entries.lycheemem-tools`
- enables the plugin entry if it does not exist yet
- creates or updates `skills.entries.lycheemem`
- ensures `agents.list[main].skills` contains `lycheemem`
- ensures `agents.list[main].tools.alsoAllow` contains the core LycheeMem tools
- fills only missing config fields by default
- preserves your existing custom values unless you pass explicit overrides or `--force`

Recommended defaults written by the script:
- `baseUrl` = `http://127.0.0.1:8000`
- `transport` = `mcp`
- `timeout` = `300`
- `apiToken` = `""`
- `enableHostLifecycle` = `true`
- `enablePromptPresence` = `true`
- `enableAutoAppendTurns` = `true`
- `enableBoundaryConsolidation` = `true`
- `enableProactiveConsolidation` = `false`
- `proactiveConsolidationCooldownSeconds` = `180`

Useful examples:
```bash
python "$LYCHEEMEM_REPO/openclaw-plugin/scripts/setup_openclaw_plugin.py" \
  --base-url http://127.0.0.1:8000 \
  --transport mcp \
  --plugin-enabled
```

```bash
python "$LYCHEEMEM_REPO/openclaw-plugin/scripts/setup_openclaw_plugin.py" --force
```

> This script now manages the plugin entry, the `lycheemem` skill entry, and the target agent tool allowlist. It still does **not** install OpenClaw or start the LycheeMem server.

### Option B: OpenClaw page

In the left sidebar:

1. Open **Automation**
2. Open **Plugins**
3. Find the LycheeMem entry:
   `Thin OpenClaw adapter for LycheeMem structured memory tools. (plugin: lycheemem-tools)`
4. Open **LycheeMem Tools Config**
5. Expand **advanced** if needed
6. Confirm or adjust the values below
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
- **Proactive Consolidation** = `false`

Recommended default:
- **Proactive Cooldown** = `180`

### Option C: `~/.openclaw/openclaw.json`

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
          "timeout": 300,
          "apiToken": "",
          "enableHostLifecycle": true,
          "enablePromptPresence": true,
          "enableAutoAppendTurns": true,
          "enableBoundaryConsolidation": true,
          "enableProactiveConsolidation": false,
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

### Verify plugin config with the helper script
```bash
python "$LYCHEEMEM_REPO/openclaw-plugin/scripts/verify_openclaw_plugin.py"
```
**Expected:** plugin, skill, and agent tool checks all pass, and the backend is reachable.

### Verify the skill is mounted
```bash
openclaw skills info lycheemem
openclaw skills check
```
**Expected:** `lycheemem` shows `Ready`.

If you already ran `verify_openclaw_plugin.py`, then:
- `skills.entries.lycheemem.enabled`
- `agents.list[main].skills`
- `agents.list[main].tools.alsoAllow`

have already been checked automatically. The UI is only needed here as a manual cross-check.

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
- `enableProactiveConsolidation` is intentionally `false` unless you changed it on purpose

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
- `verify_openclaw_plugin.py` already passes
- the current agent is the same one that the setup script updated
- the plugin entry `lycheemem-tools` is enabled under **Automation -> Plugins**
- **Enable LycheeMem Tools** is on
- **Plugin Hook Policy** is on
- the backend at `http://127.0.0.1:8000` is actually reachable

**Solution:**
1. Restart the gateway
2. Open a new session and retry
3. If you are not using `main`, rerun:
```bash
python "$LYCHEEMEM_REPO/openclaw-plugin/scripts/setup_openclaw_plugin.py" \
  --agent-id your-agent-id
```

### Model is calling `append_turn` manually
Turn mirroring is handled automatically by hooks in this version. If the model continues to call `append_turn` manually, the gateway has likely not been restarted or the session is still using an old prompt.

### No consolidate after `/new`
Check:
1. `enableBoundaryConsolidation` is `true`
2. The gateway process was actually restarted

---

## Related docs
- `openclaw-plugin/SKILL.md`
