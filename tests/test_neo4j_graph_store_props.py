from __future__ import annotations

import re

from src.memory.graph.neo4j_graph_store import Neo4jGraphStore


def test_neo4j_add_passes_fact_evidence_and_metadata() -> None:
    store = Neo4jGraphStore.__new__(Neo4jGraphStore)

    captured: dict[str, object] = {}

    def add_node_stub(node_id: str, label: str, properties=None) -> None:  # noqa: ANN001
        return

    def add_edge_stub(source_id: str, target_id: str, relation: str, properties=None) -> None:  # noqa: ANN001
        captured.update(
            {
                "source_id": source_id,
                "target_id": target_id,
                "relation": relation,
                "properties": dict(properties or {}),
            }
        )

    store.add_node = add_node_stub  # type: ignore[method-assign]
    store.add_edge = add_edge_stub  # type: ignore[method-assign]

    store.add(
        [
            {
                "subject": {"name": "北京会议", "label": "Event"},
                "predicate": "happens_in",
                "object": {"name": "北京电视台", "label": "Place"},
                "confidence": 0.77,
                "timestamp": "2026-03-16T00:00:00+00:00",
                "source_session": "s1",
                "fact": "北京会议在北京电视台举行。",
                "evidence": "北京会议在北京电视台",
            }
        ]
    )

    assert captured["source_id"] == "北京会议"
    assert captured["target_id"] == "北京电视台"
    assert captured["relation"] == "happens_in"

    props = captured["properties"]
    assert props["confidence"] == 0.77
    assert props["source_session"] == "s1"
    assert props["fact"] == "北京会议在北京电视台举行。"
    assert props["evidence"] == "北京会议在北京电视台"
    assert re.match(r"^2026-03-16T00:00:00\+00:00$", str(props["timestamp"]))
