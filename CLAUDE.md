# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LycheeMemOS is a training-free agentic cognitive memory framework. It implements four cognitive memory types (Working, Graph/Semantic, Procedural, Sensory) orchestrated through six cognitive agents via a LangGraph pipeline, exposed as a FastAPI service. The project documentation and LLM prompts are primarily in Chinese.

## Commands

```bash
# Install (with dev dependencies)
pip install -e ".[dev]"

# Run server
python -m lychee_memos --llm openai          # OpenAI backend
python -m lychee_memos --llm ollama          # Ollama local
python -m lychee_memos --reload              # Dev mode with hot reload

# Tests
pytest tests/ -v                        # All tests
pytest tests/test_graph_store.py -v     # Single test file
pytest tests/test_graph_store.py::test_name -v  # Single test

# Lint
ruff check src/
ruff format src/
```

## Configuration

All config via environment variables or `.env` file (copy `.env.example`). Key settings:
- `OPENAI_API_KEY`, `OPENAI_MODEL` — LLM backend
- `SESSION_BACKEND` — memory/sqlite
- `GRAPH_BACKEND` — memory/neo4j
- `SKILL_BACKEND` — memory/file/lancedb
- Production backends (Neo4j, LanceDB) require `pip install -e ".[production]"`

## Architecture

### Pipeline Flow (LangGraph StateGraph)

```
START → WMManager → Router → [conditional]
                                ├─ need_retrieval → SearchCoordinator → Synthesizer → Reasoner → END
                                └─ direct_answer  →                                   Reasoner → END
```

After each interaction, `ConsolidatorAgent` runs in a background thread to extract entities and skills.

### Key Layers

**`src/core/`** — Pipeline orchestration and configuration
- `config.py`: `Settings` class (pydantic-settings), singleton `settings`. All env vars map here.
- `state.py`: `PipelineState` TypedDict — the LangGraph shared state schema.
- `factory.py`: `create_pipeline()` — dependency injection hub. Creates storage backends and agents based on settings, returns `LycheePipeline`.
- `graph.py`: `LycheePipeline` — wraps the LangGraph `StateGraph`, provides `run()`/`arun()` methods.

**`src/agents/`** — Six cognitive agents
- `base_agent.py`: ABC with `_call_llm()` and `_parse_json()` helpers. All agents except WMManager inherit from this.
- `wm_manager.py`: Rule-based (not LLM). Dual-threshold compression: warn at 70% token budget triggers async pre-compression, block at 90% forces sync compression.
- `router_agent.py`: Decides which memory modules to activate (`need_graph`, `need_skills`, `need_sensory`).
- `search_coordinator.py`: Orchestrates retrieval across graph, skill, and sensory stores. Uses HyDE (hypothetical document embeddings) for skill search.
- `synthesizer_agent.py`: LLM-as-Judge scoring of retrieved memories. Drops fragments below 0.6 relevance, fuses the rest.
- `reasoning_agent.py`: Generates the user-facing response from compressed history + background context.
- `consolidator_agent.py`: Background entity extraction → graph store, skill extraction → skill store.

**`src/memory/`** — Four memory substrates, each with swappable backends
- `working/`: Session stores (InMemory, SQLite) + `WorkingMemoryCompressor` (tiktoken-based token counting, LLM-driven summarization).
- `graph/`: `NetworkXGraphStore` (in-memory) / `Neo4jGraphStore` (persistent). Embedding-based semantic search, keyword fallback, N-hop expansion, semantic entity merge.
- `procedural/`: Skill stores (InMemory, FileSkillStore, LanceDB). Each skill has intent embedding, doc_markdown, success_count.
- `sensory/`: `SensoryBuffer` — FIFO deque of recent inputs (text/image/audio).

**`src/llm/`** and **`src/embedder/`** — Adapter pattern for LLM and embedding providers (OpenAI, Gemini, Ollama). All implement a common base interface (`BaseLLM`, `BaseEmbedder`).

**`src/api/server.py`** — FastAPI app factory. SSE streaming at `/chat`, non-streaming at `/chat/complete`. Full CRUD for graph nodes/edges, skills, sessions. Demo UI at `/demo`.

### Programmatic Usage

```python
from src.core.factory import create_pipeline
pipeline = create_pipeline(llm=my_llm, embedder=my_embedder)
result = pipeline.run(user_query="hello", session_id="s1")
print(result["final_response"])
```

## Testing Patterns

Tests use `FakeLLM` and `FakeEmbedder` mock classes (defined in test files) that return predetermined responses based on system prompt keywords, enabling full pipeline testing without real API calls. Async tests run automatically via `pytest-asyncio` with `asyncio_mode = "auto"`.

## Code Style

- Python 3.11+
- Ruff: line-length 100, target py311
- No CI/CD configured; run `ruff check` and `pytest` locally before committing
