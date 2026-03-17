"""Graphiti(论文) 风格的 Neo4j Schema 定义。

PR1 目标：仅提供 schema/索引/约束的 Cypher 语句生成，便于后续 store 初始化。

注意：这些语句会在 `GraphitiNeo4jStore` 初始化时按需执行；本模块不执行任何 I/O。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CypherStatement:
    name: str
    cypher: str


def schema_statements() -> list[CypherStatement]:
    """返回 Graphiti 所需的约束与索引创建语句。

    约定：
    - Episode/Entity/Fact/Community 四类节点。
    - 使用 *业务 id* 字段做唯一约束（而非 Neo4j 内部 id）。
    - 全文索引用于 BM25/关键词召回。

    兼容性：
    - `IF NOT EXISTS` 需要较新的 Neo4j 版本；若环境较旧，可在后续 PR
      增加降级路径。
    """

    return [
        # ── Uniqueness constraints ──
        CypherStatement(
            name="constraint_episode_id",
            cypher=(
                "CREATE CONSTRAINT episode_id_unique IF NOT EXISTS "
                "FOR (e:Episode) REQUIRE e.episode_id IS UNIQUE"
            ),
        ),
        CypherStatement(
            name="constraint_entity_id",
            cypher=(
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
            ),
        ),
        CypherStatement(
            name="constraint_fact_id",
            cypher=(
                "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS "
                "FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE"
            ),
        ),
        CypherStatement(
            name="constraint_community_id",
            cypher=(
                "CREATE CONSTRAINT community_id_unique IF NOT EXISTS "
                "FOR (c:Community) REQUIRE c.community_id IS UNIQUE"
            ),
        ),
        # ── Property indexes ──
        CypherStatement(
            name="index_episode_session_id",
            cypher=(
                "CREATE INDEX episode_session_id IF NOT EXISTS FOR (e:Episode) ON (e.session_id)"
            ),
        ),
        CypherStatement(
            name="index_fact_subject_object",
            cypher=(
                "CREATE INDEX fact_subject_object IF NOT EXISTS "
                "FOR (f:Fact) ON (f.subject_entity_id, f.object_entity_id)"
            ),
        ),
        CypherStatement(
            name="index_fact_temporal",
            cypher=(
                "CREATE INDEX fact_temporal IF NOT EXISTS "
                "FOR (f:Fact) ON (f.t_valid, f.t_invalid, f.t_created, f.t_expired)"
            ),
        ),
        # ── Fulltext indexes (BM25 / keyword) ──
        CypherStatement(
            name="fulltext_entity",
            cypher=(
                "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
                "FOR (e:Entity) ON EACH [e.name, e.summary, e.aliases]"
            ),
        ),
        CypherStatement(
            name="fulltext_fact",
            cypher=(
                "CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS "
                "FOR (f:Fact) ON EACH [f.fact_text, f.relation_type, f.evidence_text]"
            ),
        ),
        CypherStatement(
            name="fulltext_community",
            cypher=(
                "CREATE FULLTEXT INDEX community_fulltext IF NOT EXISTS "
                "FOR (c:Community) ON EACH [c.name, c.summary]"
            ),
        ),
    ]
