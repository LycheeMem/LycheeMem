from __future__ import annotations

from src.memory.graph.graphiti_schema import schema_statements
from src.memory.graph.graphiti_neo4j_store import GraphitiNeo4jStore


def test_graphiti_schema_has_core_constraints_and_fulltext_indexes() -> None:
    stmts = schema_statements()
    names = {s.name for s in stmts}

    assert "constraint_episode_id" in names
    assert "constraint_entity_id" in names
    assert "constraint_fact_id" in names
    assert "constraint_community_id" in names

    assert "fulltext_entity" in names
    assert "fulltext_fact" in names
    assert "fulltext_community" in names

    cypher = "\n".join(s.cypher for s in stmts)
    assert "CREATE CONSTRAINT" in cypher
    assert "CREATE FULLTEXT INDEX" in cypher


def test_semantic_graph_export_query_is_stable() -> None:
    q = GraphitiNeo4jStore.semantic_graph_export_query()
    assert "MATCH (e:Entity)" in q.nodes_cypher
    assert "RETURN" in q.nodes_cypher

    assert "MATCH (f:Fact)" in q.edges_cypher
    assert "subject_entity_id AS source" in q.edges_cypher
    assert "object_entity_id AS target" in q.edges_cypher
    assert "relation" in q.edges_cypher
    assert "fact" in q.edges_cypher
    assert "evidence" in q.edges_cypher
    assert "timestamp" in q.edges_cypher
