"""Graphiti(论文) 风格的 Neo4j Store。

PR1 目标：
- 提供 Neo4j schema 初始化（约束/索引）。
- 提供一个“语义图兼容视图”的导出查询骨架：把 Fact-node 模型映射成
  web-demo 现有的 edges 形态（source/target/relation/fact/evidence/timestamp/source_session）。

注意：
- 本 PR 不接入 Pipeline，不改变现有对话行为。
- 本 Store 采用惰性导入 neo4j driver：允许在未安装 production extras 的环境中 import。
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
from typing import Any

from a_frame.memory.graph.graphiti_schema import CypherStatement, schema_statements


class GraphitiNeo4jStore:
    """Graphiti Neo4j store（以 Neo4j 为主后端）。"""

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        init_schema: bool = True,
    ):
        try:
            from neo4j import GraphDatabase  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                'neo4j driver 未安装：请使用 `pip install -e ".[production]"` 安装生产依赖'
            ) from exc

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

        if init_schema:
            self.init_schema()

    # ──────────────────────────────────────
    # Schema
    # ──────────────────────────────────────

    @staticmethod
    def schema_statements() -> list[CypherStatement]:
        return schema_statements()

    def init_schema(self) -> None:
        """创建约束与索引（幂等）。"""
        with self._driver.session(database=self._database) as session:
            for stmt in self.schema_statements():
                session.run(stmt.cypher)

    # ──────────────────────────────────────
    # Episode ingestion (PR2)
    # ──────────────────────────────────────

    @staticmethod
    def episode_upsert_cypher() -> str:
        """返回 Episode 幂等写入的 Cypher。

        说明：
        - 以 `episode_id` 做幂等 MERGE。
        - `t_created` 只在首次写入时设置；后续更新只刷新 `t_updated`。
        - 目前用 ISO 字符串存储时间；后续 PR 可升级为 Neo4j datetime 类型。
        """

        return (
            "MERGE (e:Episode {episode_id: $episode_id}) "
            "SET e.session_id = $session_id, "
            "    e.role = $role, "
            "    e.content = $content, "
            "    e.turn_index = $turn_index, "
            "    e.t_ref = $t_ref, "
            "    e.t_created = coalesce(e.t_created, $t_created), "
            "    e.t_updated = $t_updated "
            "RETURN e.episode_id AS episode_id"
        )

    def upsert_episode(
        self,
        *,
        episode_id: str,
        session_id: str,
        role: str,
        content: str,
        turn_index: int,
        t_ref: str,
        t_created: str | None = None,
    ) -> str:
        """写入/更新一个 Episode（幂等）。

        Returns:
            episode_id
        """

        if t_created is None:
            t_created = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t_updated = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.episode_upsert_cypher(),
                episode_id=episode_id,
                session_id=session_id,
                role=role,
                content=content,
                turn_index=turn_index,
                t_ref=t_ref,
                t_created=t_created,
                t_updated=t_updated,
            )
            record = rs.single()

        return (record or {}).get("episode_id") or episode_id

    @staticmethod
    def episodes_by_session_cypher() -> str:
        return (
            "MATCH (e:Episode {session_id: $session_id}) "
            "RETURN e.episode_id AS episode_id, e.role AS role, e.content AS content, "
            "e.turn_index AS turn_index, e.t_ref AS t_ref, e.t_created AS t_created "
            "ORDER BY e.turn_index ASC "
            "LIMIT $limit"
        )

    def list_episodes(self, *, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """按 session 读取 Episode 列表（主要用于调试/验证）。"""
        with self._driver.session(database=self._database) as session:
            rs = session.run(self.episodes_by_session_cypher(), session_id=session_id, limit=limit)
            return [dict(r) for r in rs]

    # ──────────────────────────────────────
    # Compatibility export
    # ──────────────────────────────────────

    @dataclass(frozen=True, slots=True)
    class SemanticGraphExportQuery:
        nodes_cypher: str
        edges_cypher: str

    @staticmethod
    def semantic_graph_export_query() -> "GraphitiNeo4jStore.SemanticGraphExportQuery":
        """返回“语义图视图”的查询语句。

        约定：
        - 节点返回 Entity；字段尽量贴近 web-demo 解析逻辑：`id`/`name`/`label`。
        - 边返回 Fact 映射后的边：subject_entity_id → object_entity_id。

        后续 PR 会把 Fact 的 evidence 追溯到 Episode（并可返回 citation）。
        """

        nodes_cypher = (
            "MATCH (e:Entity) "
            "RETURN e.entity_id AS id, coalesce(e.name, e.entity_id) AS name, "
            "coalesce(e.type_label, e.label, '') AS label, e AS properties"
        )

        edges_cypher = (
            "MATCH (f:Fact) "
            "WHERE exists(f.subject_entity_id) AND exists(f.object_entity_id) "
            "RETURN "
            "f.subject_entity_id AS source, "
            "f.object_entity_id AS target, "
            "coalesce(f.relation_type, '') AS relation, "
            "coalesce(f.confidence, 1.0) AS confidence, "
            "coalesce(f.fact_text, '') AS fact, "
            "coalesce(f.evidence_text, '') AS evidence, "
            "coalesce(f.source_session, '') AS source_session, "
            "coalesce(f.t_created, '') AS timestamp"
        )

        return GraphitiNeo4jStore.SemanticGraphExportQuery(
            nodes_cypher=nodes_cypher,
            edges_cypher=edges_cypher,
        )

    def export_semantic_graph(self) -> dict[str, list[dict[str, Any]]]:
        """导出“语义图视图”。

        返回结构与 `/memory/graph` 的 GraphResponse 兼容：
        `{ "nodes": [...], "edges": [...] }`
        """
        q = self.semantic_graph_export_query()
        with self._driver.session(database=self._database) as session:
            nodes_rs = session.run(q.nodes_cypher)
            edges_rs = session.run(q.edges_cypher)

            nodes = [dict(r) for r in nodes_rs]
            edges = [dict(r) for r in edges_rs]

        return {"nodes": nodes, "edges": edges}

    def close(self) -> None:
        self._driver.close()
