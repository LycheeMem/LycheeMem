<div align="center">
  <img src="assets/logo.png" alt="LycheeMem Logo" width="200">
  <h1>LycheeMem</h1>
  <p>
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
    <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/LangGraph-000?style=flat&logo=langchain" alt="LangGraph">
    <img src="https://img.shields.io/badge/litellm-000?style=flat&logo=python" alt="litellm">
  </p>
  <p>
    <a href="README_zh.md">中文</a> | English
  </p>
</div>


LycheeMem is a compact memory framework for LLM agents. It starts from efficient conversational memory—through structured organization, lightweight consolidation, and adaptive retrieval—and gradually extends toward action-aware, usage-aware memory for more capable agentic systems.

[**Demo**: LycheeMem vs ChatGPT Free on long-horizon memory tasks, click to watch the video](https://ik.imagekit.io/wvs0koh4k/99043dc9-b8dd-4e78-ac48-c38da4e7405c.mp4)

---

<div align="center">
  <a href="#news">News</a>
  •
  <a href="#related-projects">Related Projects</a>
  •
  <a href="#quick-start">Quick Start</a>
  •
  <a href="#web-demo">Web Demo</a>
  •
  <a href="#openclaw-plugin">OpenClaw Plugin</a>
  •
  <a href="#mcp">MCP</a>
  •
  <a href="#memory-architecture">Memory Architecture</a>
  •
  <a href="#pipeline">Pipeline</a>
  •
  <a href="#api-reference">API Reference</a>
</div>

---

<a id="news"></a>

## 🔥 News
- [04/03/2026] The project now supports installation via `pip install lycheemem`. You can easily start the service from anywhere using `lycheemem-cli`!
- [03/30/2026] We evaluated LycheeMem on PinchBench with the OpenClaw plugin: compared to OpenClaw's native memory, it achieved an ~6% score improvement, while reducing token consumption by ~71% and cost by ~55%!
- [03/28/2026] Semantic memory has been upgraded to Compact Semantic Memory (SQLite + LanceDB), no Neo4j required. See [/quick-start](#quick-start) for details.
- [03/27/2026] OpenClaw Plugin is now available at [/openclaw-plugin](#openclaw-plugin) ! [Setup guide →](openclaw-plugin/INSTALL_OPENCLAW.md)
- [03/26/2026] MCP support is available at [/mcp](#mcp) !
- [03/23/2026] LycheeMem is now open source: [GitHub Repository →](https://github.com/LycheeMem/LycheeMem)

---

<a id="related-projects"></a>

## 🔗 Related Projects 

LycheeMem is part of the **3rd-generation Lychee (立知) large model series**, which focuses on memory intelligence, continual learning, and long-context reasoning.

We welcome you to explore our related works:

- **LycheeMemory**: a unified framework for implicit long-term memory and explicit working memory collaboration in large language models  
  [![arXiv](https://img.shields.io/badge/arXiv-2602.08382-B31B1B?logo=arxiv&logoColor=fff)](https://arxiv.org/abs/2602.08382) [![GitHub](https://img.shields.io/badge/GitHub-LycheeMemory-181717?logo=github&logoColor=fff)](https://github.com/owoakuma/LycheeMemory) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-LycheeMemory--7B-FFD21E?logo=huggingface)](https://huggingface.co/lerverson/LycheeMemory-7B)

- **LycheeMem (this project)**: long-term memory infrastructure for LLM-based agents  
  [![Project Page](https://img.shields.io/badge/Project_Page-LycheeMem-blue?logo=google-chrome&logoColor=fff)](https://lycheemem.github.io) [![GitHub](https://img.shields.io/badge/GitHub-LycheeMem-181717?logo=github&logoColor=fff)](https://github.com/LycheeMem/LycheeMem)

- **LycheeDecode**: selective recall from massive KV-cache context memory  
  [![Project Page](https://img.shields.io/badge/Project_Page-lycheedecode-blue?logo=google-chrome&logoColor=fff)](https://lg9077.github.io/lycheedecode) [![arXiv](https://img.shields.io/badge/arXiv-2602.04541-B31B1B?logo=arxiv&logoColor=fff)](https://arxiv.org/abs/2602.04541) [![GitHub](https://img.shields.io/badge/GitHub-LycheeDecode-181717?logo=github&logoColor=fff)](https://github.com/HITsz-TMG/TMGNLP/tree/main/LycheeDecode)

- **LycheeCluster**: structured organization and hierarchical indexing for context memory  
  [![arXiv](https://img.shields.io/badge/arXiv-2603.08453-B31B1B?logo=arxiv&logoColor=fff)](https://arxiv.org/abs/2603.08453)

---

<a id="quick-start"></a>

## ⚡ Quick Start

### Prerequisites

- Python 3.9+
- An LLM API key (OpenAI, Gemini, or any litellm-compatible provider)

### Installation

You can install LycheeMem directly via pip:

```bash
pip install lycheemem
```

Once installed, you can start the backend server instantly using the CLI:

```bash
lycheemem-cli
```

For development or if you prefer to run from source:

```bash
git clone https://github.com/LycheeMem/LycheeMem.git
cd LycheeMem
pip install -e .
```

### Configuration

Create a `.env` file in your working directory and fill in your values. The full template in `.env.example` also includes session/user DB paths, JWT settings, and working-memory thresholds; the snippet below shows the most important ones:

```dotenv
# LLM — litellm format: provider/model
LLM_MODEL=openai/gpt-4o-mini
LLM_API_KEY=sk-...
LLM_API_BASE=                     # optional

# Embedder
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_API_KEY=                # optional
EMBEDDING_API_BASE=               # optional

```

> **Supported LLM providers** (via [litellm](https://github.com/BerriAI/litellm)):  
> `openai/gpt-4o-mini` · `gemini/gemini-2.0-flash` · `ollama_chat/qwen2.5` · any OpenAI-compatible endpoint

### Start the Server

If you installed via pip, you can start the LycheeMem background service from anywhere using:

```bash
lycheemem-cli
```

*(If running from source, you can also use `python main.py` to start the server.)*

The API is served at `http://localhost:8000`. Interactive docs at `/docs`.

> `main.py` currently starts Uvicorn without enabling live reload. For development reload, run Uvicorn directly, for example:
>
> ```bash
> uvicorn src.api.server:create_app --factory --reload
> ```

---

<a id="web-demo"></a>

## 🎨 Web Demo

A frontend demo is included under `web-demo/`. It provides a chat interface alongside live views of the **semantic memory tree**, skill library, and working memory state.

```bash
cd web-demo
npm install
npm run dev      # served at http://localhost:5173
```

> Make sure the backend is running on port 8000 (or update proxy settings in `web-demo/vite.config.ts`) before starting the frontend.

---

<a id="openclaw-plugin"></a>

## 🦞 OpenClaw Plugin

LycheeMem ships a native [OpenClaw](https://openclaw.ai) plugin that gives any OpenClaw session persistent long-term memory with zero manual wiring.

**What the plugin provides:**

- `lychee_memory_smart_search` — default long-term memory retrieval entry point
- **Automatic turn mirroring** via hooks — the model does **not** need to call `append_turn` manually
  - User messages are appended automatically
  - Assistant messages are appended automatically
- `/new`, `/reset`, `/stop`, and `session_end` automatically trigger boundary consolidation
- Proactive consolidation on strong long-term knowledge signals

**Under normal operation:**
- The model only calls `lychee_memory_smart_search` when recalling long-term context
- The model may call `lychee_memory_consolidate` manually when an immediate persist is warranted
- The model does **not** need to call `lychee_memory_append_turn` at all

### Quick Install

```bash
openclaw plugins install "/path/to/LycheeMem/openclaw-plugin"
openclaw gateway restart
```

See the full setup guide: [openclaw-plugin/INSTALL_OPENCLAW.md](openclaw-plugin/INSTALL_OPENCLAW.md)

---

<a id="mcp"></a>

## 🔧 MCP

LycheeMem also exposes an HTTP MCP endpoint at `http://localhost:8000/mcp`.

- Available tools: `lychee_memory_smart_search`, `lychee_memory_search`, `lychee_memory_append_turn`, `lychee_memory_synthesize`, `lychee_memory_consolidate`
- Use `Authorization: Bearer <token>` if you want per-user memory isolation
- `lychee_memory_consolidate` works for sessions that already contain mirrored turns from `/chat`, `/memory/reason`, or `lychee_memory_append_turn`

### MCP Transport

- `POST /mcp` handles JSON-RPC requests
- `GET /mcp` exposes the SSE stream used by some MCP clients
- The server returns `Mcp-Session-Id` during `initialize`; reuse that header on later requests

### Authentication

If you want isolated memory per user, first obtain a JWT token from `/auth/register` or `/auth/login`, then send:

```text
Authorization: Bearer <token>
```

Without a token, requests run with an empty `user_id`, so anonymous traffic shares the same namespace.

### Client Configuration

For any MCP client that supports remote HTTP servers, configure the MCP URL as:

```text
http://localhost:8000/mcp
```

Generic config example:

```json
{
  "mcpServers": {
    "lycheemem": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

### Manual JSON-RPC Flow

1. Call `initialize`
2. Reuse the returned `Mcp-Session-Id`
3. Send `initialized`
4. Call `tools/list`
5. Call `tools/call`

Initialize example:

```bash
curl -i -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-03-26",
      "capabilities": {},
      "clientInfo": {
        "name": "debug-client",
        "version": "0.1.0"
      }
    }
  }'
```

Tool call example:

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -H "Mcp-Session-Id: <session-id>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "lychee_memory_smart_search",
      "arguments": {
        "query": "what tools do I use for database backups",
        "top_k": 5,
        "mode": "compact",
        "include_graph": true,
        "include_skills": true
      }
    }
  }'
```

### Recommended MCP Usage Pattern

1. Use `/chat` or `/memory/reason` with a stable `session_id` to write conversation turns, or mirror external host turns with `lychee_memory_append_turn`.
2. Use `lychee_memory_smart_search` in `compact` mode for the default one-shot recall path.
3. Use `lychee_memory_search` + `lychee_memory_synthesize` only when you explicitly want search and synthesis as separate stages.
4. After the conversation ends, call `lychee_memory_consolidate` with the same `session_id`.

---

<a id="memory-architecture"></a>

## 📚 Memory Architecture

LycheeMem organizes memory into three complementary stores:

<table>
  <thead>
    <tr>
      <th>Working Memory</th>
      <th>Semantic Memory</th>
      <th>Procedural Memory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <p>(Episodic)</p>
        <ul>
          <li>Session turns</li>
          <li>Summaries</li>
          <li>Token budget management</li>
        </ul>
      </td>
      <td>
        <p>(Typed Action Store)</p>
        <ul>
          <li>7 MemoryRecord types</li>
          <li>Conflict-aware Record Fusion</li>
          <li>Hierarchical memory tree</li>
          <li>Action-grounded retrieval planning</li>
          <li>Usage feedback loop + RL-ready statistics</li>
        </ul>
      </td>
      <td>
        <p>(Skills)</p>
        <ul>
          <li>Skill entries</li>
          <li>HyDE retrieval</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

### 💾 Working Memory

The working memory window holds the active conversation context for a session. It operates under a **dual-threshold token budget**:

- **Warn threshold (70%)** — triggers asynchronous background pre-compression; the current request is not blocked.
- **Block threshold (90%)** — the pipeline pauses and flushes older turns to a compressed summary before proceeding.

Compression produces *summary anchors* (past context, distilled) + *raw recent turns* (last N turns, verbatim). Both are passed downstream as the conversation history.

### 🗺️ Semantic Memory — Compact Semantic Memory

Semantic memory is organised around **typed MemoryRecords plus action-grounded retrieval state**. The storage layer is SQLite (FTS5 full-text search) + LanceDB (vector index), while retrieval is conditioned on recent context, tentative action, constraints, and missing slots.

#### Memory Record Types

Each memory entry is stored as a `MemoryRecord`. The `memory_type` field distinguishes seven semantic categories:

| Type | Description |
|------|-------------|
| `fact` | Objective facts about the user, environment, or world |
| `preference` | User preferences (style, habits, likes/dislikes) |
| `event` | Specific events that have occurred |
| `constraint` | Conditions that must be respected |
| `procedure` | Reusable step-by-step procedures / methods |
| `failure_pattern` | Previously failed action paths and their causes |
| `tool_affordance` | Capabilities and applicable scenarios of tools/APIs |

Beyond text, every `MemoryRecord` carries **action-facing metadata** (`tool_tags`, `constraint_tags`, `failure_tags`, `affordance_tags`) and **usage statistics** (`retrieval_count`, `action_success_count`, etc.) to seed future reinforcement-learning signals. Retrieval logs also persist `retrieval_plan`, `action_state`, response excerpts, and later user feedback so the system can close a lightweight action-outcome loop without training.

Related `MemoryRecord`s can be fused online by the **Record Fusion Engine** into denser `CompositeRecord`s. Composite entries persist direct `child_composite_ids`, so long-term semantic memory is organised as a **hierarchical memory tree** instead of a flat bag of summaries.

#### Four-Module Pipeline

##### Module 1: Compact Semantic Encoding

A single-pass pipeline that converts conversation turns into a list of `MemoryRecord`s:

1. **Typed extraction** — LLM extracts self-contained facts and assigns a semantic category to each record.
2. **Decontextualization** — Pronouns and context-dependent phrases are expanded into full expressions, so each record is understandable without the original dialogue.
3. **Action metadata annotation** — LLM annotates each record with `memory_type`, `tool_tags`, `constraint_tags`, `failure_tags`, `affordance_tags`, and other structured labels.

`record_id = SHA256(normalized_text)` — naturally idempotent; duplicate content is deduplicated automatically.

##### Module 2: Record Fusion, Conflict Update, and Hierarchical Consolidation

Triggered online after each consolidation:

1. FTS / vector recall gathers related **existing atomic records** around the new records (candidate pool).
2. The existing synthesis judge prompt decides whether each candidate set should produce a new `CompositeRecord` **or** perform a `conflict_update` against an existing atomic record.
3. On `conflict_update`, the existing anchor record is updated in place, conflicting incoming records are soft-expired, and composites covering affected source records are invalidated.
4. On synthesis, the engine writes a new `CompositeRecord` to SQLite + LanceDB.
5. Additional hierarchy rounds can synthesize `record -> composite` and `composite -> composite`, persisting `child_composite_ids` so the memory tree can keep growing upward.

##### Module 3: Action-Grounded Retrieval Planning

Before retrieval, `ActionAwareRetrievalPlanner` analyses the **user query + recent context + ActionState** and emits a `SearchPlan`:

- `mode`: `answer` (factual Q&A) / `action` (needs execution) / `mixed`
- `semantic_queries`: content-facing search terms
- `pragmatic_queries`: action/tool/constraint-facing search terms
- `tool_hints`: tools likely needed for this request
- `required_constraints`: constraints that must be respected
- `required_affordances`: capabilities the retrieved memory should provide
- `missing_slots`: parameters / slots that are absent
- `tree_retrieval_mode` / `tree_expansion_depth` / `include_leaf_records`: whether retrieval should stay at high-level composites (`root_only`) or descend into child composites / direct leaf records (`balanced` / `descend`)

`ActionState` can carry fields such as `current_subgoal`, `tentative_action`, `known_constraints`, `available_tools`, `failure_signal`, and a recent-context excerpt. The planner merges this state with the LLM-produced plan so retrieval is conditioned on the current decision state rather than the query alone.

The plan drives multi-channel recall:

1. **FTS channel** — SQLite FTS5 keyword recall over `MemoryRecord` + `CompositeRecord`
2. **Semantic vector channel** — LanceDB ANN over `semantic_text` embeddings
3. **Normalised vector channel** — LanceDB ANN over `normalized_text` embeddings (for pragmatic queries)
4. **Tag filter channel** — exact filter by `tool_hints` / `required_constraints` / `required_affordances`
5. **Temporal channel** — filter by `SearchPlan.temporal_filter` time window
6. **Slot-hint supplementation** — when `missing_slots` is non-empty, extra FTS/tag recall is triggered to find records that can fill missing parameters

After base recall, retrieval can also expand along the **memory tree**. `root_only` keeps high-level composite summaries, `balanced` descends one level when tree hints match, and `descend` pulls child composites plus direct leaf records when the current action needs finer-grained detail.

##### Module 4: Multi-Dimensional Scorer

Candidates from all channels are de-duplicated and ranked by `MemoryScorer` using a weighted linear combination. Final top-k selection is **composite-first**: covering parent composites are preferred, covered child records are folded away unless they add unique value, and near-duplicate fragments are suppressed.

$$\text{Score} = \alpha \cdot S_\text{sem} + \beta \cdot S_\text{action} + \kappa \cdot S_\text{slot} + \gamma \cdot S_\text{temporal} + \delta \cdot S_\text{recency} + \eta \cdot S_\text{evidence} - \lambda \cdot C_\text{token}$$

| Weight | Meaning | Default |
|--------|---------|---------|
| α | SemanticRelevance (vector distance -> similarity) | 0.25 |
| β | ActionUtility (tag match score, mode-aware) | 0.25 |
| κ | SlotUtility (whether the memory helps fill missing action slots) | 0.15 |
| γ | TemporalFit (temporal reference match) | 0.15 |
| δ | Recency (memory freshness) | 0.10 |
| η | EvidenceDensity (evidence span density) | 0.10 |
| λ | TokenCost penalty (text length penalty) | 0.10 |

### 🛠️ Procedural Memory — Skill Store

The skill store preserves reusable *how-to* knowledge as structured skill entries, each carrying:

- **Intent** — a short description of what the skill does.
- **`doc_markdown`** — a full Markdown document describing the procedure, commands, parameters, and caveats.
- **Embedding** — a dense vector of the intent text, used for similarity search.
- **Metadata** — usage counters, last-used timestamp, preconditions.

Skill retrieval uses **HyDE (Hypothetical Document Embeddings)**: the query is first expanded into a *hypothetical ideal answer* by the LLM, then that draft text is embedded to produce a query vector that matches well against stored procedure descriptions, even when the user's original phrasing is vague.

---

<a id="pipeline"></a>

## ⚙️ Pipeline

Every request passes through a fixed sequence of five agents. Four are synchronous stages in the LangGraph pipeline; one is a background post-processing task.

<div align="center">
  <div>
    <div>START</div>
    <div>▼</div>
    <div>
      <div>
        <div>
          <strong>1. WMManager</strong> — Token budget check + compress/render
        </div>
        <div>↓</div>
        <div>
          <strong>2. SearchCoordinator</strong> — Planner → Semantic + Skill retrieval
        </div>
        <div>↓</div>
        <div>
          <strong>3. SynthesizerAgent</strong> — LLM-as-Judge scoring + context fusion
        </div>
        <div>↓</div>
        <div>
          <strong>4. ReasoningAgent</strong> — Final response generation
        </div>
      </div>
    </div>
    <div>▼</div>
    <div>END</div>
    <div>
      <span>Background</span>
      <span>asyncio.create_task( <strong>ConsolidatorAgent</strong> )</span>
    </div>
  </div>
</div>

### Stage 1 — WMManager

Rule-based agent (no LLM prompt). Appends the user turn to the session log, counts tokens, and fires compression if either threshold is crossed. Produces `compressed_history` and `raw_recent_turns` for downstream stages.

### Stage 2 — SearchCoordinator

`SearchCoordinator` first builds `recent_context` from compressed summaries + raw recent turns, then derives an `ActionState` from the current query, constraints, recent failures, token budget, and recent tool use. `ActionAwareRetrievalPlanner` uses that state to produce a `SearchPlan` containing `mode`, `semantic_queries`, `pragmatic_queries`, `tool_hints`, `required_affordances`, `missing_slots`, tree-traversal strategy, and more. Multi-channel recall (FTS, semantic vector, normalised vector, tag/affordance filter, temporal filter, slot-hint supplementation, plus tree expansion when needed) then queries SQLite + LanceDB. This stage returns raw semantic fragments, skill hits, retrieval provenance, and a dedicated `novelty_retrieved_context` built from **pre-synthesis** semantic fragments for later novelty checking; it does **not** build the final `background_context` yet. Skill retrieval is mode-aware (`answer` / `action` / `mixed`) and uses HyDE against the skill store only when it is likely to help.

When a new user turn arrives, `SearchCoordinator` also tries to apply lightweight feedback to the most recent unresolved action/mixed retrieval log, so the next turn can mark the prior memory usage as success / fail / correction.

### Stage 3 — SynthesizerAgent

Acts as an **LLM-as-Judge**: scores every retrieved memory fragment on an absolute 0-1 relevance scale, discards fragments below the threshold (default 0.6), and fuses the survivors into a single dense `background_context` string. It also identifies `skill_reuse_plan` entries that can directly guide the final response. This stage is where the final answer-time context is built; it outputs `provenance` — a citation list containing scoring breakdown and source references for each kept memory item.

### Stage 4 — ReasoningAgent

Receives `compressed_history`, `background_context`, and `skill_reuse_plan` and generates the final assistant reply. It appends the assistant turn back to the session store, and the pipeline finalizes the semantic usage log with a response excerpt so the next user turn can provide outcome feedback.

### Background — ConsolidatorAgent

Triggered immediately after `ReasoningAgent` completes, runs in a thread pool and **does not block the response**. It:

1. Performs a **novelty check** — LLM judges whether the conversation introduced new information worth persisting. Skips consolidation for pure retrieval exchanges.
2. **Compact consolidation** — calls `CompactSemanticEngine.ingest_conversation()`, which runs a single-pass encoder (typed extraction → decontextualization → action metadata annotation), writes `MemoryRecord`s to SQLite + LanceDB, then triggers conflict-aware Record Fusion. Novelty check uses the search-stage `novelty_retrieved_context` (raw semantic fragments), not the answer-time `background_context`, so query-conditioned synthesis does not suppress valid new-memory ingestion.
3. **Skill extraction** — identifies successful tool-usage patterns in the conversation and adds skill entries to the skill store. Runs in parallel with compact consolidation (ThreadPoolExecutor).

---

<a id="api-reference"></a>

## 🔌 API Reference

### `POST /memory/search` — Unified Memory Retrieval

Query both the semantic memory channel and the skill store in a single call. New integrations should prefer `semantic_results`; `graph_results` is kept as a backward-compatible alias. The response also includes `novelty_retrieved_context`, which is the correct input for later `/memory/consolidate` calls.

```json
// Request
{
  "query": "what tools do I use for database backups",
  "top_k": 5,
  "include_graph": true,
  "include_skills": true
}

// Response
{
  "query": "...",
  "graph_results": [
    {
      "anchor": {
        "node_id": "compact_context",
        "name": "CompactSemanticMemory",
        "label": "SemanticContext",
        "score": 1.0
      },
      "constructed_context": "...",
      "provenance": [ { "record_id": "...", "source": "record", "semantic_source_type": "record", "score": 0.91, ... } ]
    }
  ],
  "semantic_results": [
    {
      "anchor": { "node_id": "compact_context", "name": "CompactSemanticMemory", "label": "SemanticContext", "score": 1.0 },
      "constructed_context": "...",
      "provenance": [ { "record_id": "...", "source": "record", "semantic_source_type": "record", "score": 0.91, ... } ]
    }
  ],
  "novelty_retrieved_context": "[1] (procedure, source=record) Use pg_dump with cron ...",
  "skill_results": [ { "id": "...", "intent": "pg_dump backup to S3", "score": 0.87, ... } ],
  "total": 6
}
```

---

### `POST /memory/smart-search` — One-Shot Recall

Runs search and, optionally, synthesis in one API call. `mode=compact` is the default integration path when you want a concise `background_context` without handling intermediate payloads yourself. Even in compact mode, the response still returns `novelty_retrieved_context` so a host can consolidate against raw retrieved memory instead of answer-time synthesis.

```json
// Request
{
  "query": "what tools do I use for database backups",
  "top_k": 5,
  "synthesize": true,
  "mode": "compact"
}

// Response
{
  "query": "...",
  "mode": "compact",
  "synthesized": true,
  "background_context": "User regularly uses pg_dump with a cron job...",
  "skill_reuse_plan": [ { "skill_id": "...", "intent": "...", "doc_markdown": "..." } ],
  "provenance": [ { "record_id": "...", "source": "record", "score": 0.91, ... } ],
  "novelty_retrieved_context": "[1] (procedure, source=record) Use pg_dump with cron ...",
  "kept_count": 4,
  "dropped_count": 2,
  "total": 6
}
```

---

### `POST /memory/synthesize` — Memory Fusion

Takes raw retrieval results and produces a fused memory context using LLM-as-Judge.

```json
// Request
{
  "user_query": "what tools do I use for database backups",
  "semantic_results": [...], // preferred from /memory/search
  "graph_results": [...],    // compatibility alias also accepted
  "skill_results": [...]
}

// Response
{
  "background_context": "User regularly uses pg_dump with a cron job...",
  "skill_reuse_plan": [ { "skill_id": "...", "intent": "...", "doc_markdown": "..." } ],
  "provenance": [ { "record_id": "...", "source": "semantic", "semantic_source_type": "record", "score": 0.91, ... } ],
  "kept_count": 4,
  "dropped_count": 2
}
```

---

### `POST /memory/reason` — Grounded Reasoning

Runs the ReasoningAgent given pre-synthesized context. Can be chained after `/memory/synthesize` for full pipeline control.

```json
// Request
{
  "session_id": "my-session",
  "user_query": "what tools do I use for database backups",
  "background_context": "User regularly uses pg_dump...",
  "skill_reuse_plan": [...],
  "append_to_session": true   // write result to session history (default: true)
}

// Response
{
  "response": "You typically use pg_dump scheduled via cron...",
  "session_id": "my-session",
  "wm_token_usage": 3412
}
```

---

### `POST /memory/append-turn` — Mirror External Host Turns

Appends one user or assistant turn into LycheeMem's session store so it can be consolidated later.

```json
// Request
{
  "session_id": "my-session",
  "role": "user",
  "content": "I usually back up PostgreSQL with pg_dump to S3."
}

// Response
{
  "status": "appended",
  "session_id": "my-session",
  "turn_count": 3
}
```

---

### `POST /memory/consolidate` — Trigger Consolidation

Manually trigger memory consolidation for a session. This is the primary consolidation endpoint and supports both background and synchronous modes.

`retrieved_context` should preferably be the `novelty_retrieved_context` returned by `/memory/search` or `/memory/smart-search`, i.e. the **search-stage raw semantic fragments**, not `/memory/synthesize`'s `background_context`.

```json
// Request
{
  "session_id": "my-session",
  "retrieved_context": "[1] (procedure, source=record) Use pg_dump with cron ...",
  "background": true
}

// Response (background mode)
{
  "status": "started",
  "entities_added": 0,
  "skills_added": 0,
  "facts_added": 0
}
```

Legacy compatibility endpoint: `POST /memory/consolidate/{session_id}`.

---

### `GET /memory/graph` — Semantic Memory Tree

Returns the current semantic memory as a hierarchy. `mode=cleaned` (default) emits `tree_roots` plus direct tree edges for the frontend memory-tree view; `mode=debug` exposes the lower-level flattened relations for inspection.

---

### `GET /pipeline/status` and `GET /pipeline/last-consolidation`

Use these endpoints for operational checks and background consolidation polling:

- `GET /pipeline/status` returns aggregate counts for sessions, semantic memory, and skills.
- `GET /pipeline/last-consolidation?session_id=<id>` returns the latest consolidation result for a session, or `pending` if the background task has not finished yet.

### Usage Examples

```bash
# Basic single-turn demo (automatically registers 'demo_user')
python examples/api_pipeline_demo.py

# Multi-turn chat demo (3 consecutive turns, followed by consolidation)
python examples/api_pipeline_demo.py --multi-turn

# Use a fixed session_id (useful for accumulating history across multiple runs)
python examples/api_pipeline_demo.py --session-id my-test-session
```

