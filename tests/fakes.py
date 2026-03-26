"""Shared test doubles for MCP and plugin integration tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.memory.working.session_store import InMemorySessionStore


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [float(len(query))]


class FakeGraphitiEngine:
    def search(
        self,
        *,
        query: str,
        session_id: str | None,
        top_k: int,
        query_embedding: list[float] | None,
        include_communities: bool,
        user_id: str,
    ) -> Any:
        context = (
            f"Historical graph context for {query}. "
            f"This project has prior continuity signals and remembered facts."
        )
        provenance = [
            {
                "fact_id": "fact-1",
                "fact_text": f"{query} historical fact with lots of extra detail and supporting explanation",
                "rrf": 0.91,
            }
        ][:top_k]
        return SimpleNamespace(context=context, provenance=provenance)


class FakeSkillStore:
    def search(
        self,
        query: str,
        *,
        top_k: int,
        query_embedding: list[float] | None,
        user_id: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": "skill-1",
                "intent": f"How we solved {query} before",
                "doc_markdown": "Step 1: inspect. Step 2: compress. Step 3: reuse.",
                "score": 0.88,
                "reusable": True,
            }
        ][:top_k]


class FakeSearchCoordinator:
    def __init__(self):
        self.embedder = FakeEmbedder()
        self.graphiti_engine = FakeGraphitiEngine()
        self.skill_store = FakeSkillStore()
        self.graph_top_k = 3
        self.skill_top_k = 3
        self.skill_reuse_threshold = 0.85

    def _plan_retrieval(self, query: str) -> dict[str, list[str]]:
        return {"graph": [query], "skill": [query]}

    def _search_graph(
        self,
        queries: list[str],
        *,
        session_id: str | None = None,
        user_id: str = "",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        query = queries[0]
        graph_top_k = top_k if top_k is not None else self.graph_top_k
        result = self.graphiti_engine.search(
            query=query,
            session_id=session_id,
            top_k=graph_top_k,
            query_embedding=self.embedder.embed_query(query),
            include_communities=True,
            user_id=user_id,
        )
        return [
            {
                "anchor": {
                    "node_id": "graphiti_context",
                    "name": "GraphitiContext",
                    "label": "Context",
                    "score": 1.0,
                },
                "subgraph": {"nodes": [], "edges": []},
                "constructed_context": result.context,
                "provenance": result.provenance,
            }
        ]

    def _search_skills(
        self,
        query: str,
        user_id: str = "",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        skill_top_k = top_k if top_k is not None else self.skill_top_k
        results = self.skill_store.search(
            query,
            top_k=skill_top_k,
            query_embedding=self.embedder.embed_query(query),
            user_id=user_id,
        )
        for skill in results:
            skill["reusable"] = skill.get("score", 0) >= self.skill_reuse_threshold
        return results


class FakeSynthesizer:
    def run(
        self,
        *,
        user_query: str,
        retrieved_graph_memories: list[dict[str, Any]],
        retrieved_skills: list[dict[str, Any]],
    ) -> dict[str, Any]:
        context = (
            f"Compact background for {user_query}: "
            f"{len(retrieved_graph_memories)} graph hits, {len(retrieved_skills)} skill hits."
        )
        provenance_items = []
        if retrieved_graph_memories:
            provenance_items.append({"items": retrieved_graph_memories[:1], "source": "graph"})
        if retrieved_skills:
            provenance_items.append({"items": retrieved_skills[:1], "source": "skills"})
        return {
            "background_context": context,
            "skill_reuse_plan": [{"skill_id": "skill-1", "reason": "matches prior workflow"}],
            "provenance": provenance_items,
        }


class FakeConsolidator:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        retrieved_context: str,
        user_id: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "turns": turns,
                "session_id": session_id,
                "retrieved_context": retrieved_context,
                "user_id": user_id,
            }
        )
        return {
            "entities_added": 1,
            "skills_added": 1,
            "facts_added": max(1, len(turns) // 2),
            "has_novelty": True,
            "skipped_reason": None,
            "steps": [{"name": "extract", "status": "done", "detail": "ok"}],
        }


class FakeWMManager:
    def __init__(self):
        self.session_store = InMemorySessionStore()

    def append_assistant_turn(self, session_id: str, content: str, user_id: str = "") -> None:
        self.session_store.append_turn(session_id, "assistant", content, user_id=user_id)


class FakePipeline:
    def __init__(self):
        self.search_coordinator = FakeSearchCoordinator()
        self.synthesizer = FakeSynthesizer()
        self.consolidator = FakeConsolidator()
        self.wm_manager = FakeWMManager()

    def run(self, user_query: str, session_id: str, user_id: str = "") -> dict[str, Any]:
        self.wm_manager.session_store.append_turn(session_id, "user", user_query, user_id=user_id)
        answer = f"Answer for: {user_query}"
        self.wm_manager.append_assistant_turn(session_id, answer, user_id=user_id)
        retrieved_graph_memories = self.search_coordinator._search_graph(
            [user_query],
            session_id=session_id,
            user_id=user_id,
            top_k=1,
        )
        retrieved_skills = self.search_coordinator._search_skills(
            user_query,
            user_id=user_id,
            top_k=1,
        )
        synthesis = self.synthesizer.run(
            user_query=user_query,
            retrieved_graph_memories=retrieved_graph_memories,
            retrieved_skills=retrieved_skills,
        )
        return {
            "compressed_history": [],
            "raw_recent_turns": self.wm_manager.session_store.get_or_create(session_id).turns,
            "wm_token_usage": 42,
            "retrieved_graph_memories": retrieved_graph_memories,
            "retrieved_skills": retrieved_skills,
            "background_context": synthesis["background_context"],
            "skill_reuse_plan": synthesis["skill_reuse_plan"],
            "provenance": synthesis["provenance"],
            "final_response": answer,
        }
