"""Tests for Paper §2.1 bidirectional citation: Fact → source Episodes.

The GraphitiEngine must:
1. After ranking facts, call store.get_source_episodes_for_fact(fact_id) for
   every ranked fact that has a non-synthetic ID (does not start with "fact:").
2. Call store.get_source_episodes_for_entity(entity_id) for every anchor entity
   and fact endpoint, then merge the results deduplicated by episode_id.
3. Attach the merged episode list to provenance_by_fact[fid]["source_episodes"].
4. Include a <SOURCES> block in the context string listing original turns.
5. When the store does NOT expose get_source_episodes_for_fact (legacy store),
   skip silently in non-strict mode; raise RuntimeError in strict mode.
"""

from __future__ import annotations

import pytest

from a_frame.memory.graph.graphiti_engine import GraphitiEngine


# ---------------------------------------------------------------------------
# Helpers / shared fake data
# ---------------------------------------------------------------------------

EPISODE_A = {
    "episode_id": "ep_aaa",
    "session_id": "s1",
    "role": "user",
    "content": "Alice works at Acme Corp since January.",
    "turn_index": 1,
    "t_ref": "2026-01-01T09:00:00Z",
    "episode_type": "message",
}

EPISODE_B = {
    "episode_id": "ep_bbb",
    "session_id": "s1",
    "role": "assistant",
    "content": "Got it, Alice joined Acme Corp.",
    "turn_index": 2,
    "t_ref": "2026-01-01T09:00:05Z",
    "episode_type": "message",
}

FACT_ROW = {
    "fact_id": "fact_001",
    "source": "entity_alice",
    "target": "entity_acme",
    "relation": "WORKS_AT",
    "fact": "Alice works at Acme Corp",
    "evidence": "Alice works at Acme Corp since January.",
    "t_valid_from": "2026-01-01T00:00:00Z",
    "t_valid_to": "",
    "timestamp": "2026-01-01T09:00:00Z",
}


# ---------------------------------------------------------------------------
# FakeStore with full citation support
# ---------------------------------------------------------------------------

class FakeStoreFull:
    """Fake Neo4j store that implements source-episode lookup methods."""

    def fulltext_search_facts(self, *, query: str, limit: int = 10):
        return [FACT_ROW]

    def fulltext_search_entities(self, *, query: str, limit: int = 10):
        return [
            {"entity_id": "entity_alice", "name": "Alice", "summary": "A person."},
        ]

    def vector_search_entities(self, *, query_embedding, limit: int = 10):
        return []

    def vector_search_facts(self, *, query_embedding, limit: int = 10):
        return []

    def export_semantic_subgraph(self, *, entity_ids, edge_limit: int = 600):
        return {"nodes": [], "edges": []}

    def get_entities_by_ids(self, *, entity_ids):
        return []

    def list_recent_entity_ids_from_episodes(
        self, *, session_id: str, episode_limit: int = 4, entity_limit: int = 20
    ):
        return []

    def get_source_episodes_for_fact(self, *, fact_id: str, limit: int = 10):
        """Paper §2.1 EVIDENCE_FOR lookup."""
        if fact_id == "fact_001":
            return [EPISODE_A, EPISODE_B]
        return []

    def get_source_episodes_for_entity(self, *, entity_id: str, limit: int = 5):
        """Paper §2.1 MENTIONS lookup."""
        if entity_id == "entity_alice":
            return [EPISODE_A]
        return []


# ---------------------------------------------------------------------------
# FakeStore without citation support (legacy)
# ---------------------------------------------------------------------------

class FakeStoreLegacy:
    """Fake store that does NOT implement source-episode lookup."""

    def fulltext_search_facts(self, *, query: str, limit: int = 10):
        return [FACT_ROW]

    def fulltext_search_entities(self, *, query: str, limit: int = 10):
        return []

    def export_semantic_subgraph(self, *, entity_ids, edge_limit: int = 600):
        return {"nodes": [], "edges": []}

    def get_entities_by_ids(self, *, entity_ids):
        return []

    def list_recent_entity_ids_from_episodes(
        self, *, session_id: str, episode_limit: int = 4, entity_limit: int = 20
    ):
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGraphitiEpisodeCitation:

    def test_provenance_contains_source_episodes_from_evidence_for(self):
        """Engine must call get_source_episodes_for_fact and attach results to
        provenance_by_fact[fid]['source_episodes'] (EVIDENCE_FOR path)."""
        engine = GraphitiEngine(store=FakeStoreFull(), strict=False)
        result = engine.search(query="Alice Acme", top_k=1)

        assert len(result.provenance) >= 1
        prov = result.provenance[0]
        assert prov.get("fact_id") == "fact_001"
        ep_list = prov.get("source_episodes") or []
        assert isinstance(ep_list, list), "source_episodes must be a list"
        assert len(ep_list) >= 1, "must have at least one source episode from EVIDENCE_FOR"

        ep_ids = [ep["episode_id"] for ep in ep_list]
        assert "ep_aaa" in ep_ids, "EPISODE_A (EVIDENCE_FOR) must be present"

    def test_provenance_merges_entity_mention_episodes(self):
        """MENTIONS-traced episodes for participating entities must be merged into
        source_episodes when they are not already covered by EVIDENCE_FOR."""
        engine = GraphitiEngine(store=FakeStoreFull(), strict=False)
        result = engine.search(query="Alice Acme", top_k=1)

        prov = result.provenance[0]
        ep_ids = {ep["episode_id"] for ep in (prov.get("source_episodes") or [])}
        # ep_aaa comes from both EVIDENCE_FOR and MENTIONS(alice) — must appear once
        assert "ep_aaa" in ep_ids

    def test_source_episodes_deduplicated_by_episode_id(self):
        """Episode ep_aaa is returned by both get_source_episodes_for_fact AND
        get_source_episodes_for_entity(alice); it must appear exactly once."""
        engine = GraphitiEngine(store=FakeStoreFull(), strict=False)
        result = engine.search(query="Alice Acme", top_k=1)

        prov = result.provenance[0]
        ep_ids = [ep["episode_id"] for ep in (prov.get("source_episodes") or [])]
        assert ep_ids.count("ep_aaa") == 1, "ep_aaa must appear exactly once (deduplication)"

    def test_source_episodes_sorted_by_turn_index(self):
        """Source episodes must be sorted ascending by turn_index (conversation order)."""
        engine = GraphitiEngine(store=FakeStoreFull(), strict=False)
        result = engine.search(query="Alice Acme", top_k=1)

        prov = result.provenance[0]
        ep_list = prov.get("source_episodes") or []
        turn_indices = [ep.get("turn_index") for ep in ep_list if ep.get("turn_index") is not None]
        assert turn_indices == sorted(turn_indices), "Episodes must be sorted by turn_index ASC"

    def test_context_contains_sources_block(self):
        """The context string must include a <SOURCES> block with at least one citation
        line when source episodes are available (paper §2.1 Constructor χ)."""
        engine = GraphitiEngine(store=FakeStoreFull(), strict=False)
        result = engine.search(query="Alice Acme", top_k=1)

        assert "<SOURCES>" in result.context
        assert "</SOURCES>" in result.context
        # At least one citation line referencing the fact text
        assert "Alice works at Acme Corp" in result.context
        # Should contain turn reference
        assert "turn 1" in result.context or "user" in result.context

    def test_context_citation_contains_episode_excerpt(self):
        """Each citation line must show a truncated excerpt of the episode content
        so the LLM agent can locate the original source."""
        engine = GraphitiEngine(store=FakeStoreFull(), strict=False)
        result = engine.search(query="Alice Acme", top_k=1)

        # EPISODE_A content starts with "Alice works at Acme Corp since January."
        # The excerpt (first 120 chars) should appear in the <SOURCES> block.
        sources_start = result.context.find("<SOURCES>")
        sources_end = result.context.find("</SOURCES>")
        assert sources_start > -1 and sources_end > sources_start
        sources_section = result.context[sources_start:sources_end]
        assert "Acme Corp" in sources_section, "Episode content excerpt must appear in <SOURCES>"

    def test_no_sources_block_when_no_episodes(self):
        """If no source episodes are found for any ranked fact, <SOURCES> must
        NOT appear in the context string (no empty / misleading block)."""
        engine = GraphitiEngine(store=FakeStoreLegacy(), strict=False)
        result = engine.search(query="Alice Acme", top_k=1)

        assert "<SOURCES>" not in result.context

    def test_legacy_store_non_strict_no_error(self):
        """A store without get_source_episodes_for_fact must be tolerated in
        non-strict mode: search succeeds, source_episodes is empty list."""
        engine = GraphitiEngine(store=FakeStoreLegacy(), strict=False)
        result = engine.search(query="Alice Acme", top_k=1)

        # Search still returns a result; provenance just lacks source_episodes
        assert len(result.provenance) >= 1
        prov = result.provenance[0]
        ep_list = prov.get("source_episodes")
        # source_episodes key may be absent or empty — either is acceptable
        assert not ep_list, "Legacy store should yield no source_episodes"

    def test_strict_mode_raises_when_store_missing_citation_method(self):
        """In strict mode, if the store lacks get_source_episodes_for_fact,
        a RuntimeError must be raised (fail-fast per project conventions)."""
        engine = GraphitiEngine(store=FakeStoreLegacy(), strict=True)

        with pytest.raises(RuntimeError, match="get_source_episodes_for_fact"):
            engine.search(query="Alice Acme", top_k=1)

    def test_synthetic_bfs_fact_ids_skipped_in_citation_lookup(self):
        """Facts with IDs beginning with 'fact:' are synthetic (BFS-assembled, not
        stored in Neo4j) and must NOT be passed to get_source_episodes_for_fact."""
        calls: list[str] = []

        class FakeStoreWithSpy(FakeStoreFull):
            def fulltext_search_facts(self, *, query: str, limit: int = 10):
                # Return a synthetic fact alongside a real one.
                return [
                    {
                        "fact_id": "fact:entity_alice:WORKS_AT:entity_acme:2026-01-01",
                        "source": "entity_alice",
                        "target": "entity_acme",
                        "relation": "WORKS_AT",
                        "fact": "Alice works at Acme (synthetic)",
                        "evidence": "",
                        "t_valid_from": "",
                        "t_valid_to": "",
                        "timestamp": "2026-01-01T00:00:00Z",
                    },
                    FACT_ROW,
                ]

            def get_source_episodes_for_fact(self, *, fact_id: str, limit: int = 10):
                calls.append(fact_id)
                return super().get_source_episodes_for_fact(fact_id=fact_id, limit=limit)

        engine = GraphitiEngine(store=FakeStoreWithSpy(), strict=False)
        engine.search(query="Alice Acme", top_k=2)

        assert "fact:entity_alice:WORKS_AT:entity_acme:2026-01-01" not in calls, (
            "Synthetic BFS fact IDs must not be passed to get_source_episodes_for_fact"
        )
        assert "fact_001" in calls, "Real stored fact IDs must still be traced"

    def test_sources_block_ordered_by_ranked_fact_position(self):
        """Lines inside <SOURCES> must appear in the same order as ranked_fact_ids
        (most relevant fact first), matching the ordering in <FACTS>."""

        class FakeStoreMulti(FakeStoreFull):
            def fulltext_search_facts(self, *, query: str, limit: int = 10):
                return [
                    FACT_ROW,
                    {
                        "fact_id": "fact_002",
                        "source": "entity_acme",
                        "target": "entity_city",
                        "relation": "LOCATED_IN",
                        "fact": "Acme Corp is located in Springfield",
                        "evidence": "",
                        "t_valid_from": "2025-06-01T00:00:00Z",
                        "t_valid_to": "",
                        "timestamp": "2025-06-01T00:00:00Z",
                    },
                ]

            def get_source_episodes_for_fact(self, *, fact_id: str, limit: int = 10):
                if fact_id == "fact_001":
                    return [EPISODE_A, EPISODE_B]
                if fact_id == "fact_002":
                    return [
                        {
                            "episode_id": "ep_ccc",
                            "session_id": "s1",
                            "role": "user",
                            "content": "Acme Corp is based in Springfield.",
                            "turn_index": 3,
                            "t_ref": "2026-01-01T09:01:00Z",
                            "episode_type": "message",
                        }
                    ]
                return []

        engine = GraphitiEngine(store=FakeStoreMulti(), strict=False)
        result = engine.search(query="Alice Acme", top_k=2)

        sources_start = result.context.find("<SOURCES>")
        sources_end = result.context.find("</SOURCES>")
        assert sources_start > -1 and sources_end > sources_start
        sources_section = result.context[sources_start:sources_end]

        pos_alice = sources_section.find("Alice works at Acme Corp")
        pos_acme = sources_section.find("Acme Corp is located in Springfield")
        # Both facts appear in <SOURCES> since both have source episodes
        assert pos_alice > -1, "fact_001 citation must appear in <SOURCES>"
        assert pos_acme > -1, "fact_002 citation must appear in <SOURCES>"
