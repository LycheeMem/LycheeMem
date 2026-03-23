<div align="center">
  <img src="assert/logo.png" alt="LycheeMem Logo" width="200">
  <h1>LycheeMem</h1>
  <p>
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
    <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
    <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI">
    <img src="https://img.shields.io/badge/LangGraph-000?style=flat&logo=langchain" alt="LangGraph">
  </p>
  <p>
    <a href="README_zh.md">中文</a> | English
  </p>
</div>


LycheeMem is a cognitive memory system for long-horizon AI agents, providing persistent, structured, and temporally-aware memory. It models memory the way humans use it — distinguishing what you remember *happening* from what you have come to *know* — and makes those memories available at inference time through a multi-stage reasoning pipeline.

---

## Memory Architecture

LycheeMem organizes memory into three complementary stores:

<div align="center">
  <div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; background-color: #f6f8fa; font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">
    <h3 style="margin-top: 0; color: #24292e;">LycheeMem Memory Architecture</h3>
    <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; margin-top: 15px;">
      <div style="border: 1px solid #d1d5da; border-radius: 6px; padding: 15px; background: white; min-width: 180px; text-align: left; box-shadow: 0 1px 3px rgba(27,31,35,0.04);">
        <h4 style="margin: 0 0 10px 0; color: #0366d6;">Working Memory</h4>
        <p style="margin: 0; font-size: 13px; color: #586069;">(Episodic)</p>
        <ul style="margin: 10px 0 0 0; padding-left: 18px; font-size: 13px; line-height: 1.6; color: #24292e;">
          <li>Session turns</li>
          <li>Summaries</li>
          <li>Token budget management</li>
        </ul>
      </div>
      <div style="border: 1px solid #d1d5da; border-radius: 6px; padding: 15px; background: white; min-width: 180px; text-align: left; box-shadow: 0 1px 3px rgba(27,31,35,0.04);">
        <h4 style="margin: 0 0 10px 0; color: #0366d6;">Semantic Memory</h4>
        <p style="margin: 0; font-size: 13px; color: #586069;">(Knowledge Graph)</p>
        <ul style="margin: 10px 0 0 0; padding-left: 18px; font-size: 13px; line-height: 1.6; color: #24292e;">
          <li>Entity nodes</li>
          <li>Bi-temporal facts</li>
          <li>Communities</li>
          <li>Episode anchors</li>
        </ul>
      </div>
      <div style="border: 1px solid #d1d5da; border-radius: 6px; padding: 15px; background: white; min-width: 180px; text-align: left; box-shadow: 0 1px 3px rgba(27,31,35,0.04);">
        <h4 style="margin: 0 0 10px 0; color: #0366d6;">Procedural Memory</h4>
        <p style="margin: 0; font-size: 13px; color: #586069;">(Skills)</p>
        <ul style="margin: 10px 0 0 0; padding-left: 18px; font-size: 13px; line-height: 1.6; color: #24292e;">
          <li>Skill entries</li>
          <li>HyDE retrieval</li>
        </ul>
      </div>
    </div>
  </div>
</div>

### Working Memory

The working memory window holds the active conversation context for a session. It operates under a **dual-threshold token budget**:

- **Warn threshold (70%)** — triggers asynchronous background pre-compression; the current request is not blocked.
- **Block threshold (90%)** — the pipeline pauses and flushes older turns to a compressed summary before proceeding.

Compression produces *summary anchors* (past context, distilled) + *raw recent turns* (last N turns, verbatim). Both are passed downstream as the conversation history.

### Semantic Memory — Graphiti Knowledge Graph

The knowledge graph is implemented as a **Graphiti-style bi-temporal graph** backed by Neo4j. It stores the world in four node types:

| Node | Purpose |
|------|---------|
| `Episode` | A single conversation turn; all facts are traceable to source episodes |
| `Entity` | A named entity (person, project, place, concept, etc.) |
| `Fact` | A typed relation between two entities with temporal validity and transaction metadata |
| `Community` | A cluster of strongly related entities, carrying a periodically refreshed summary |

#### Bi-temporal Model

Every `Fact` carries four timestamps that separate *when something was true* from *when the system learned it*:

```
t_valid_from / t_valid_to   →  Valid time   (real-world truth interval)
t_tx_created / t_tx_expired →  Transaction time (system-side record interval)
```

This allows the graph to answer queries like *"what was the user's home address last month?"* correctly even if the address has since changed, and to distinguish genuinely contradictory facts from facts that were true at different times.

#### Graph Retrieval Pipeline

Retrieval combines three complementary signals:

1. **BM25 full-text search** — keyword-level recall against `Entity.name` and `Fact.fact_text` via a Neo4j full-text index.
2. **BFS graph traversal** — starts from the most recent episode nodes for the session and expands outward, up to a configurable depth, surfacing semantically linked facts even when they do not match keyword terms.
3. **Vector ANN search** — approximate nearest-neighbour over the `Entity.embedding` vector index (configurable dimensionality and similarity function).

After retrieval, candidates are re-ranked using **Reciprocal Rank Fusion (RRF)** across all three lists. Optionally, a **cross-encoder reranker** (driven by the same LLM adapter already in the pipeline, no extra vendor SDK) refines the top-N results, followed by **Maximal Marginal Relevance (MMR)** diversification to avoid near-duplicate context in the final prompt.

#### Community Detection

A background sweep runs `refresh_all_communities()` every *N* episodes globally (default: 50). Community summaries are included in graph search results to provide broad contextual framing even when no specific fact directly matches a query.

### Procedural Memory — Skill Store

The skill store preserves reusable *how-to* knowledge as structured skill entries, each carrying:

- **Intent** — a short description of what the skill does.
- **`doc_markdown`** — a full Markdown document describing the procedure, commands, parameters, and caveats.
- **Embedding** — a dense vector of the intent text, used for similarity search.
- **Metadata** — usage counters, last-used timestamp, preconditions.

Skill retrieval uses **HyDE (Hypothetical Document Embeddings)**: the query is first expanded into a *hypothetical ideal answer* by the LLM, then that draft text is embedded to produce a query vector that matches well against stored procedure descriptions, even when the user's original phrasing is vague.

---

## Pipeline

Every request passes through a fixed sequence of five agents. Four are synchronous stages in the LangGraph pipeline; one is a background post-processing task.

<div align="center">
  <div style="display: flex; flex-direction: column; align-items: center; font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif; gap: 8px;">
    <div style="font-weight: bold; color: #586069; font-size: 14px;">START</div>
    <div style="font-size: 18px; color: #d1d5da; line-height: 1;">▼</div>
    <div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 15px; background-color: #f6f8fa; width: 100%; max-width: 600px; box-shadow: inset 0 1px 3px rgba(27,31,35,0.02);">
      <div style="display: flex; flex-direction: column; gap: 8px; text-align: left;">
        <div style="padding: 12px; border-left: 5px solid #0366d6; background: white; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); color: #24292e;">
          <strong style="color: #0366d6;">1. WMManager</strong> — Token budget check + compress/render
        </div>
        <div style="text-align: center; color: #d1d5da; font-size: 16px; margin: -4px 0;">↓</div>
        <div style="padding: 12px; border-left: 5px solid #0366d6; background: white; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); color: #24292e;">
          <strong style="color: #0366d6;">2. SearchCoordinator</strong> — Multi-query → Graph + Skill retrieval
        </div>
        <div style="text-align: center; color: #d1d5da; font-size: 16px; margin: -4px 0;">↓</div>
        <div style="padding: 12px; border-left: 5px solid #0366d6; background: white; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); color: #24292e;">
          <strong style="color: #0366d6;">3. SynthesizerAgent</strong> — LLM-as-Judge scoring + context fusion
        </div>
        <div style="text-align: center; color: #d1d5da; font-size: 16px; margin: -4px 0;">↓</div>
        <div style="padding: 12px; border-left: 5px solid #28a745; background: white; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); color: #24292e;">
          <strong style="color: #28a745;">4. ReasoningAgent</strong> — Final response generation
        </div>
      </div>
    </div>
    <div style="font-size: 18px; color: #d1d5da; line-height: 1;">▼</div>
    <div style="font-weight: bold; color: #586069; font-size: 14px;">END</div>
    <div style="display: flex; align-items: center; gap: 8px; margin-top: 10px; padding: 8px 12px; background: #eef9ff; border-radius: 6px; border: 1px dashed #0366d6; font-size: 13px; color: #24292e;">
      <span style="background: #0366d6; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold;">Background</span>
      <span>asyncio.create_task( <strong style="color: #0366d6;">ConsolidatorAgent</strong> )</span>
    </div>
  </div>
</div>

### Stage 1 — WMManager

Rule-based agent (no LLM prompt). Appends the user turn to the session log, counts tokens, and fires compression if either threshold is crossed. Produces `compressed_history` and `raw_recent_turns` for downstream stages.

### Stage 2 — SearchCoordinator

Decomposes the user's query into multiple *sub-queries* (one or more per source type: `graph` / `skill`) using an LLM planning step. This Multi-Query strategy ensures that a compound question touching several topics retrieves relevant facts for each topic independently, rather than conflating them into a single vector search.

Sub-queries for the graph are sent to `GraphitiEngine.search()`; sub-queries for skills use HyDE embedding before querying the skill store.

### Stage 3 — SynthesizerAgent

Acts as an **LLM-as-Judge**: scores every retrieved memory fragment on an absolute 0–1 relevance scale, discards fragments below the threshold (default 0.6), and fuses the survivors into a single dense `background_context` string. It also identifies `skill_reuse_plan` entries that can directly guide the final response. This stage outputs `provenance` — a per-fact citation list with full retrieval metadata including RRF scores, BFS/BM25 ranks, and back-references to the originating episodes.

### Stage 4 — ReasoningAgent

Receives `compressed_history`, `background_context`, and `skill_reuse_plan` and generates the final assistant reply. It appends the assistant turn back to the session store, completing the feedback loop.

### Background — ConsolidatorAgent

Triggered immediately after `ReasoningAgent` completes, runs in a thread pool and **does not block the response**. It:

1. Performs a **novelty check** — judges whether the conversation introduced new information worth persisting. Skips consolidation for pure retrieval exchanges.
2. Calls `GraphitiEngine.ingest_episode()` to extract entities and facts from the conversation turns and commit them into the Neo4j graph (with bi-temporal timestamps derived from wall-clock time).
3. Extracts skill entries from successful tool-usage patterns in the conversation and adds them to the skill store.
4. Optionally triggers periodic `refresh_communities_for_session()`.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Neo4j 5.x with the **GDS plugin** installed and a **vector index** created
- An LLM API key (OpenAI, Gemini, or any litellm-compatible provider)

### Installation

```bash
git clone https://github.com/LycheeMem/LycheeMem.git
cd LycheeMem
pip install -e ".[dev]"
```

### Configuration

Copy `.env.example` to `.env` and fill in your values:

```dotenv
# LLM — litellm format: provider/model
LLM_MODEL=openai/gpt-4o-mini
LLM_API_KEY=sk-...
LLM_API_BASE=                     # optional

# Embedder
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIM=1536

# Neo4j
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

> **Supported LLM providers** (via [litellm](https://github.com/BerriAI/litellm)):  
> `openai/gpt-4o-mini` · `gemini/gemini-3.0-flash` · `ollama_chat/qwen2.5` · any OpenAI-compatible endpoint

### Start the Server

```bash
python main.py
# with hot-reload:
python main.py --reload
```

The API is served at `http://localhost:8000`. Interactive docs at `/docs`.

---

## API Reference

### `POST /memory/search` — Unified Memory Retrieval

Query both the knowledge graph and the skill store in a single call.

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
  "graph_results": [ { "fact_id": "...", "summary": "...", "relevance": 0.91, ... } ],
  "skill_results": [ { "id": "...", "intent": "pg_dump backup to S3", "score": 0.87, ... } ],
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
  "graph_results": [...],   // from /memory/search
  "skill_results": [...]
}

// Response
{
  "background_context": "User regularly uses pg_dump with a cron job...",
  "skill_reuse_plan": [ { "skill_id": "...", "intent": "...", "doc_markdown": "..." } ],
  "provenance": [ { "fact_id": "...", "relevance": 0.91, "rrf_score": 0.72, ... } ],
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
  "final_response": "You typically use pg_dump scheduled via cron...",
  "session_id": "my-session",
  "wm_token_usage": 3412
}
```

---

### `POST /memory/consolidate/{session_id}` — Trigger Consolidation

Manually trigger memory consolidation for a session (normally runs automatically in the background after each chat turn).

```bash
curl -X POST http://localhost:8000/memory/consolidate/my-session
```

```json
// Response
{ "message": "Consolidation done: 5 entities, 2 skills extracted." }
```

### Usage Examples

```bash
# Basic single-turn demo (automatically registers 'demo_user')
python examples/api_pipeline_demo.py

# Multi-turn chat demo (3 consecutive turns, followed by consolidation)
python examples/api_pipeline_demo.py --multi-turn

# Custom query and user credentials
python examples/api_pipeline_demo.py --username alice --password secret123 \
  --query "How do I backup my database with pg_dump?"

# Use a fixed session_id (useful for accumulating history across multiple runs)
python examples/api_pipeline_demo.py --session-id my-test-session
```

---

## Web Demo

A frontend demo is included under `web-demo/`. It provides a chat interface alongside live views of the knowledge graph, skill library, and working memory state.

```bash
cd web-demo
npm install
npm run dev      # served at http://localhost:5173
```

Key panels:
- **Chat** — full conversation with per-step trace expansion (click any pipeline stage to inspect what was retrieved and scored).
- **Graph Memory** — interactive visualization of the Neo4j knowledge graph; search by entity name or relation type; filter by valid-time range.
- **Skills** — browse and search the procedural skill library.
- **Working Memory** — inspect the current session's token usage, compressed summaries, and raw recent turns.

> Make sure the backend is running on port 8000 (or update proxy settings in `web-demo/vite.config.ts`) before starting the frontend.