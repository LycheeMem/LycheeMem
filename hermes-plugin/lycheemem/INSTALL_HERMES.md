# LycheeMem Hermes Plugin Quick Setup

Connect LycheeMem to Hermes Agent and verify the basic memory flow.

---

## What This Plugin Provides

- `lycheemem_smart_search`: manual long-term memory recall tool
- `pre_llm_call`: automatically recalls LycheeMem memory before each model call and injects `background_context`
- `post_llm_call`: mirrors user and assistant turns into LycheeMem
- `on_session_end` / `on_session_finalize`: triggers long-term memory consolidation
- `/lycheemem`: slash command for status, health check, recall, and consolidation

In normal use, the model does not need to call `lycheemem_append_turn` or `lycheemem_consolidate` manually. Use `/lycheemem recall <query>` when debugging retrieval.

---

## Prerequisites

- Hermes Agent is installed and `hermes` is available
- LycheeMem dependencies are installed and the LycheeMem server can start
- Hermes and LycheeMem can reach the same local or internal network address

This is a Hermes standalone plugin. It does not replace the Hermes native memory provider; it adds LycheeMem long-term memory through hooks and tools.

---

## 1. Start LycheeMem

```bash
export LYCHEEMEM_REPO="/path/to/LycheeMem"
cd "$LYCHEEMEM_REPO"
conda activate lycheemem
python main.py
```

Check the server:

```bash
curl http://127.0.0.1:8000/health
```

If your `.env` uses another port, such as `API_PORT=8100`, use that same port in `LYCHEEMEM_BASE_URL` below.

---

## 2. Install and Enable the Plugin with Commands

```bash
mkdir -p ~/.hermes/plugins
ln -sfn "$LYCHEEMEM_REPO/hermes-plugin/lycheemem" ~/.hermes/plugins/lycheemem
hermes plugins enable lycheemem
hermes plugins list
```

`hermes plugins enable lycheemem` enables the plugin. Use `hermes plugins list` to confirm the status.

---

## 3. Configure Connection Environment Variables

Start with HTTP transport for the first verification pass.

```bash
export LYCHEEMEM_BASE_URL="http://127.0.0.1:8000"
export LYCHEEMEM_TRANSPORT="http"
export LYCHEEMEM_ENABLE_PRE_LLM_CONTEXT=true
export LYCHEEMEM_ENABLE_AUTO_APPEND=true
export LYCHEEMEM_ENABLE_AUTO_CONSOLIDATE=true
export LYCHEEMEM_ENABLE_FINALIZE_CONSOLIDATION=true
export LYCHEEMEM_SMART_SEARCH_MODE=compact
export LYCHEEMEM_RESPONSE_LEVEL=minimal
export LYCHEEMEM_SMART_SEARCH_TOP_K=5
```

If your server uses port `8100`:

```bash
export LYCHEEMEM_BASE_URL="http://127.0.0.1:8100"
```

For debugging, run consolidation synchronously:

```bash
export LYCHEEMEM_AUTO_CONSOLIDATE_BACKGROUND=false
export LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS=0
```

For longer sessions, use background consolidation:

```bash
export LYCHEEMEM_AUTO_CONSOLIDATE_BACKGROUND=true
export LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS=60
```

To test MCP transport:

```bash
export LYCHEEMEM_TRANSPORT="mcp"
export LYCHEEMEM_MCP_URL="$LYCHEEMEM_BASE_URL/mcp"
```

---

## 4. Start Hermes and Verify

```bash
hermes
```

Inside Hermes, run:

```text
/plugins
/lycheemem status
/lycheemem health
/lycheemem recall favorite fruit
```

`/plugins` should show `lycheemem`. `/lycheemem health` should return `success: true`.

---

## 5. End-to-End Test

```text
Please remember that my favorite fruit is lychee.
```

```text
What is my favorite fruit?
```

```text
/lycheemem recall favorite fruit
```

If the flow is working, the result should include a `background_context` mentioning `lychee`.

---

## 6. Automatic Recall vs Manual Tool Calls

By default, the plugin recalls memory through `pre_llm_call`. Hermes usually does not display this as a visible tool call because the hook runs before the model call.

Manual recall is mainly for debugging:

```text
/lycheemem recall <query>
```

If you want the model to explicitly call `lycheemem_smart_search`, temporarily disable automatic recall:

```bash
export LYCHEEMEM_ENABLE_PRE_LLM_CONTEXT=false
```

Then tell Hermes:

```text
Before answering memory-related questions, use the lycheemem_smart_search tool.
```

The recommended default is still automatic recall because it does not depend on the model deciding to call a tool.

---

## 7. Troubleshooting

### `/plugins` Does Not Show `lycheemem`

```bash
ls -la ~/.hermes/plugins/lycheemem
hermes plugins enable lycheemem
hermes plugins list
```

Then restart Hermes.

### `/lycheemem health` Fails

```bash
curl "$LYCHEEMEM_BASE_URL/health"
```

If this fails, start LycheeMem or fix `LYCHEEMEM_BASE_URL`.

### `/lycheemem recall` Returns Empty

For debugging, use synchronous consolidation:

```bash
export LYCHEEMEM_AUTO_CONSOLIDATE_BACKGROUND=false
export LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS=0
```

Run the end-to-end test again, then check LycheeMem server logs for `/memory/append-turn`, `/memory/consolidate`, and `/memory/smart-search`.

### `python main.py` Starts Slowly

LycheeMem may backfill historical session turn vectors on startup. If you have a lot of history and embeddings go through a remote API, startup can be slow. Check `EMBEDDING_API_BASE` in `.env`, prefer a local embedding service, or use an empty data directory for plugin-only debugging.

---

## 8. Update and Uninstall

If you installed with a symlink, update the repository and restart Hermes:

```bash
cd "$LYCHEEMEM_REPO"
git pull
```

Disable the plugin:

```bash
hermes plugins disable lycheemem
```

Remove the local plugin link:

```bash
rm ~/.hermes/plugins/lycheemem
```
