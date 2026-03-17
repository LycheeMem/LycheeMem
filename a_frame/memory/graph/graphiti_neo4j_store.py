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
import uuid
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
        vector_dim: int | None = None,
        vector_similarity_function: str = "cosine",
    ):
        try:
            from neo4j import GraphDatabase  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                'neo4j driver 未安装：请使用 `pip install -e ".[production]"` 安装生产依赖'
            ) from exc

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

        self._vector_dim = int(vector_dim) if vector_dim is not None else None
        self._vector_similarity_function = str(vector_similarity_function or "cosine")

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
            for stmt in schema_statements(
                vector_dim=self._vector_dim,
                vector_similarity_function=self._vector_similarity_function,
            ):
                session.run(stmt.cypher)

    # ──────────────────────────────────────
    # Strict preflight (fail-fast)
    # ──────────────────────────────────────

    def preflight(
        self,
        *,
        require_gds: bool = True,
        require_vector_index: bool = True,
        vector_dim: int | None = None,
    ) -> None:
        """启动前检查：Neo4j 连通性 / GDS / 向量索引。

        严格模式下用于 fail-fast：任何缺失都直接抛异常，禁止静默降级。
        """

        # 1) Connectivity
        with self._driver.session(database=self._database) as session:
            session.run("RETURN 1 AS ok").single()

        # 2) GDS availability
        if require_gds:
            with self._driver.session(database=self._database) as session:
                # If GDS is not installed, this will raise a ClientError.
                # NOTE: Different GDS versions expose different return column names
                # (e.g. `gdsVersion` vs `version`). Use `YIELD *` for compatibility.
                session.run("RETURN gds.version() AS version").single()

        # 3) Vector index presence + dimensions
        if require_vector_index:
            expected_dim = int(vector_dim) if vector_dim is not None else self._vector_dim
            if expected_dim is None or expected_dim <= 0:
                raise RuntimeError("Graphiti preflight requires a positive vector_dim")

            required = {
                "entity_embedding": ("NODE", "Entity", "embedding"),
                "fact_embedding": ("NODE", "Fact", "embedding"),
                "community_embedding": ("NODE", "Community", "embedding"),
            }

            with self._driver.session(database=self._database) as session:
                rs = session.run(
                    "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options "
                    "WHERE type = 'VECTOR' "
                    "RETURN name, entityType, labelsOrTypes, properties, options"
                )
                indexes = [dict(r) for r in rs]

            by_name = {str(i.get("name") or ""): i for i in indexes}
            missing = [n for n in required if n not in by_name]
            if missing:
                raise RuntimeError(f"Missing required vector indexes: {missing}")

            for name, (entity_type, label, prop) in required.items():
                idx = by_name[name]

                if str(idx.get("entityType") or "").upper() != str(entity_type).upper():
                    raise RuntimeError(
                        f"Vector index {name} has unexpected entityType={idx.get('entityType')!r}"
                    )

                labels = idx.get("labelsOrTypes")
                props = idx.get("properties")
                if isinstance(labels, list) and label not in labels:
                    raise RuntimeError(
                        f"Vector index {name} labelsOrTypes={labels!r} missing {label!r}"
                    )
                if isinstance(props, list) and prop not in props:
                    raise RuntimeError(f"Vector index {name} properties={props!r} missing {prop!r}")

                options = idx.get("options") or {}
                index_config = options.get("indexConfig") if isinstance(options, dict) else None
                dims = None
                if isinstance(index_config, dict):
                    dims = index_config.get("vector.dimensions")
                if dims is not None and int(dims) != expected_dim:
                    raise RuntimeError(
                        f"Vector index {name} dimension mismatch: expected {expected_dim}, got {dims}"
                    )

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
    # Entity (PR3)
    # ──────────────────────────────────────

    @staticmethod
    def entity_upsert_cypher() -> str:
        return (
            "MERGE (e:Entity {entity_id: $entity_id}) "
            "SET e.name = $name, "
            "    e.summary = $summary, "
            "    e.aliases = $aliases, "
            "    e.type_label = $type_label, "
            "    e.embedding = $embedding, "
            "    e.source_session = $source_session, "
            "    e.t_created = coalesce(e.t_created, $t_created), "
            "    e.t_updated = $t_updated "
            "RETURN e.entity_id AS entity_id"
        )

    def upsert_entity(
        self,
        *,
        entity_id: str,
        name: str,
        summary: str = "",
        aliases: list[str] | None = None,
        type_label: str = "",
        embedding: list[float] | None = None,
        source_session: str = "",
        t_created: str | None = None,
    ) -> str:
        if t_created is None:
            t_created = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t_updated = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.entity_upsert_cypher(),
                entity_id=entity_id,
                name=name,
                summary=summary,
                aliases=list(aliases or []),
                type_label=type_label,
                embedding=embedding,
                source_session=source_session,
                t_created=t_created,
                t_updated=t_updated,
            )
            record = rs.single()
        return (record or {}).get("entity_id") or entity_id

    @staticmethod
    def fulltext_search_entities_cypher() -> str:
        return (
            "CALL db.index.fulltext.queryNodes('entity_fulltext', $q) "
            "YIELD node, score "
            "RETURN node.entity_id AS entity_id, node.name AS name, node.summary AS summary, "
            "node.aliases AS aliases, node.type_label AS type_label, score AS score "
            "ORDER BY score DESC "
            "LIMIT $limit"
        )

    @staticmethod
    def vector_search_entities_cypher() -> str:
        return (
            "CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding) "
            "YIELD node, score "
            "RETURN node.entity_id AS entity_id, node.name AS name, node.summary AS summary, "
            "node.aliases AS aliases, node.type_label AS type_label, score AS score "
            "ORDER BY score DESC "
            "LIMIT $k"
        )

    def fulltext_search_entities(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(self.fulltext_search_entities_cypher(), q=query, limit=limit)
            return [dict(r) for r in rs]

    def vector_search_entities(
        self, *, query_embedding: list[float], limit: int = 10
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            return []
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.vector_search_entities_cypher(),
                embedding=list(query_embedding),
                k=int(limit),
            )
            return [dict(r) for r in rs]

    @staticmethod
    def scan_entities_with_embeddings_cypher() -> str:
        return (
            "MATCH (e:Entity) WHERE exists(e.embedding) "
            "RETURN e.entity_id AS entity_id, e.name AS name, e.summary AS summary, "
            "e.aliases AS aliases, e.type_label AS type_label, e.embedding AS embedding "
            "LIMIT $limit"
        )

    def scan_entities_with_embeddings(self, *, limit: int = 2000) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(self.scan_entities_with_embeddings_cypher(), limit=limit)
            return [dict(r) for r in rs]

    @staticmethod
    def entity_embeddings_by_ids_cypher() -> str:
        return (
            "MATCH (e:Entity) "
            "WHERE e.entity_id IN $entity_ids "
            "RETURN e.entity_id AS entity_id, e.embedding AS embedding"
        )

    def get_entity_embeddings_by_ids(self, *, entity_ids: list[str]) -> list[dict[str, Any]]:
        entity_ids = [str(e).strip() for e in (entity_ids or []) if str(e).strip()]
        if not entity_ids:
            return []
        with self._driver.session(database=self._database) as session:
            rs = session.run(self.entity_embeddings_by_ids_cypher(), entity_ids=entity_ids)
            return [dict(r) for r in rs]

    @staticmethod
    def recent_entity_ids_for_session_cypher() -> str:
        return (
            "MATCH (e:Entity) "
            "WHERE e.source_session = $session_id "
            "RETURN e.entity_id AS entity_id "
            "ORDER BY coalesce(e.t_updated, e.t_created, '') DESC "
            "LIMIT $limit"
        )

    def list_recent_entity_ids_for_session(self, *, session_id: str, limit: int = 50) -> list[str]:
        if not str(session_id or "").strip():
            return []
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.recent_entity_ids_for_session_cypher(), session_id=session_id, limit=int(limit)
            )
            return [
                str(r.get("entity_id") or "").strip()
                for r in rs
                if str(r.get("entity_id") or "").strip()
            ]

    # ──────────────────────────────────────
    # Mentions frequency (session-wide; PR6)
    # ──────────────────────────────────────

    @staticmethod
    def fact_mentions_in_session_cypher() -> str:
        return (
            "UNWIND $fact_ids AS fid "
            "MATCH (f:Fact {fact_id: fid}) "
            "OPTIONAL MATCH (e:Episode {session_id: $session_id})-[:EVIDENCE_FOR]->(f) "
            "RETURN fid AS fact_id, count(DISTINCT e) AS mentions"
        )

    @staticmethod
    def entity_mentions_in_session_cypher() -> str:
        return (
            "UNWIND $entity_ids AS eid "
            "OPTIONAL MATCH (e:Episode {session_id: $session_id})-[:EVIDENCE_FOR]->(f:Fact) "
            "WHERE f.subject_entity_id = eid OR f.object_entity_id = eid "
            "RETURN eid AS entity_id, count(DISTINCT e) AS mentions"
        )

    def count_mentions_in_session(
        self,
        *,
        session_id: str,
        entity_ids: list[str] | None = None,
        fact_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        session_id = str(session_id or "").strip()
        if not session_id:
            return {"entities": {}, "facts": {}}

        entity_ids = [str(e).strip() for e in (entity_ids or []) if str(e).strip()]
        fact_ids = [str(f).strip() for f in (fact_ids or []) if str(f).strip()]

        entity_counts: dict[str, int] = {}
        fact_counts: dict[str, int] = {}

        with self._driver.session(database=self._database) as session:
            if entity_ids:
                rs = session.run(
                    self.entity_mentions_in_session_cypher(),
                    session_id=session_id,
                    entity_ids=entity_ids,
                )
                for r in rs:
                    eid = str(r.get("entity_id") or "").strip()
                    if eid:
                        entity_counts[eid] = int(r.get("mentions") or 0)

            if fact_ids:
                rs = session.run(
                    self.fact_mentions_in_session_cypher(),
                    session_id=session_id,
                    fact_ids=fact_ids,
                )
                for r in rs:
                    fid = str(r.get("fact_id") or "").strip()
                    if fid:
                        fact_counts[fid] = int(r.get("mentions") or 0)

        return {"entities": entity_counts, "facts": fact_counts}

    # ──────────────────────────────────────
    # Subgraph export for search (PR3 closing)
    # ──────────────────────────────────────

    @staticmethod
    def entities_by_ids_cypher() -> str:
        return (
            "MATCH (e:Entity) "
            "WHERE e.entity_id IN $entity_ids "
            "RETURN e.entity_id AS id, coalesce(e.name, e.entity_id) AS name, "
            "coalesce(e.type_label, e.label, '') AS label, properties(e) AS properties"
        )

    def get_entities_by_ids(self, *, entity_ids: list[str]) -> list[dict[str, Any]]:
        entity_ids = [str(e).strip() for e in (entity_ids or []) if str(e).strip()]
        if not entity_ids:
            return []
        with self._driver.session(database=self._database) as session:
            rs = session.run(self.entities_by_ids_cypher(), entity_ids=entity_ids)
            return [dict(r) for r in rs]

    @staticmethod
    def fact_edges_incident_to_entities_cypher() -> str:
        return (
            "MATCH (f:Fact) "
            "WHERE f.subject_entity_id IS NOT NULL AND f.object_entity_id IS NOT NULL "
            "  AND (f.subject_entity_id IN $entity_ids OR f.object_entity_id IN $entity_ids) "
            "OPTIONAL MATCH (e:Episode)-[:EVIDENCE_FOR]->(f) "
            "WITH f, collect(e.episode_id) AS episode_ids "
            "RETURN "
            "f.fact_id AS fact_id, "
            "f.subject_entity_id AS source, "
            "f.object_entity_id AS target, "
            "coalesce(f.relation_type, '') AS relation, "
            "coalesce(f.confidence, 1.0) AS confidence, "
            "coalesce(f.fact_text, '') AS fact, "
            "coalesce(f.evidence_text, '') AS evidence, "
            "coalesce(f.source_session, '') AS source_session, "
            "coalesce(f.t_valid_from, f.t_created, '') AS timestamp, "
            "coalesce(f.t_valid_from, '') AS t_valid_from, "
            "coalesce(f.t_valid_to, '') AS t_valid_to, "
            "episode_ids AS episode_ids "
            "ORDER BY f.t_created DESC "
            "LIMIT $edge_limit"
        )

    def export_semantic_subgraph(
        self,
        *,
        entity_ids: list[str],
        edge_limit: int = 200,
    ) -> dict[str, list[dict[str, Any]]]:
        """导出以给定实体为锚点的语义子图。

        返回结构与 `/memory/graph` 的 GraphResponse 兼容：
        `{ "nodes": [...], "edges": [...] }`

        Notes:
        - 当前仅返回 Entity 作为 nodes，Fact 映射为 edges。
        - edges 额外返回 `episode_ids`（如果存在 Episode→Fact 证据关系）。
        """

        entity_ids = [str(e).strip() for e in (entity_ids or []) if str(e).strip()]
        if not entity_ids:
            return {"nodes": [], "edges": []}

        with self._driver.session(database=self._database) as session:
            edges_rs = session.run(
                self.fact_edges_incident_to_entities_cypher(),
                entity_ids=entity_ids,
                edge_limit=int(edge_limit),
            )
            edges = [dict(r) for r in edges_rs]

            expanded_ids = set(entity_ids)
            for e in edges:
                s = str(e.get("source") or "").strip()
                t = str(e.get("target") or "").strip()
                if s:
                    expanded_ids.add(s)
                if t:
                    expanded_ids.add(t)

            nodes_rs = session.run(
                self.entities_by_ids_cypher(),
                entity_ids=list(expanded_ids),
            )
            nodes = [dict(r) for r in nodes_rs]

        return {"nodes": nodes, "edges": edges}

    # ──────────────────────────────────────
    # Community dynamic extension (PR6)
    # ──────────────────────────────────────

    @staticmethod
    def expand_communities_via_entities_cypher() -> str:
        return (
            "MATCH (c:Community) "
            "WHERE c.community_id IN $community_ids "
            "MATCH (c)<-[:IN_COMMUNITY]-(e:Entity)-[:IN_COMMUNITY]->(c2:Community) "
            "WHERE NOT c2.community_id IN $community_ids "
            "RETURN DISTINCT c2.community_id AS community_id, "
            "coalesce(c2.name, c2.community_id) AS name, "
            "coalesce(c2.summary, '') AS summary, 1.0 AS score "
            "LIMIT $limit"
        )

    def expand_communities_via_entities(
        self, *, community_ids: list[str], limit: int = 5
    ) -> list[dict[str, Any]]:
        community_ids = [str(c).strip() for c in (community_ids or []) if str(c).strip()]
        if not community_ids:
            return []
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.expand_communities_via_entities_cypher(),
                community_ids=community_ids,
                limit=int(limit),
            )
            return [dict(r) for r in rs]

    # ──────────────────────────────────────
    # Fact (PR3)
    # ──────────────────────────────────────

    @staticmethod
    def fact_upsert_cypher() -> str:
        return (
            "MERGE (f:Fact {fact_id: $fact_id}) "
            "SET f.subject_entity_id = $subject_entity_id, "
            "    f.object_entity_id = $object_entity_id, "
            "    f.relation_type = $relation_type, "
            "    f.fact_text = $fact_text, "
            "    f.evidence_text = $evidence_text, "
            "    f.embedding = CASE WHEN $embedding IS NULL THEN f.embedding ELSE $embedding END, "
            "    f.confidence = $confidence, "
            "    f.source_session = $source_session, "
            "    f.t_created = coalesce(f.t_created, $t_created), "
            "    f.t_valid_from = coalesce(f.t_valid_from, $t_valid_from), "
            "    f.t_valid_to = coalesce(f.t_valid_to, $t_valid_to), "
            "    f.t_tx_created = coalesce(f.t_tx_created, $t_tx_created), "
            "    f.t_updated = $t_updated "
            "WITH f "
            "MATCH (s:Entity {entity_id: $subject_entity_id}) "
            "MATCH (o:Entity {entity_id: $object_entity_id}) "
            "MERGE (s)-[:SUBJECT_OF]->(f) "
            "MERGE (f)-[:OBJECT_OF]->(o) "
            "RETURN f.fact_id AS fact_id"
        )

    def upsert_fact(
        self,
        *,
        fact_id: str,
        subject_entity_id: str,
        object_entity_id: str,
        relation_type: str,
        fact_text: str,
        evidence_text: str = "",
        embedding: list[float] | None = None,
        confidence: float = 1.0,
        source_session: str = "",
        t_created: str | None = None,
        t_valid_from: str | None = None,
        t_valid_to: str | None = None,
        t_tx_created: str | None = None,
    ) -> str:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if t_created is None:
            t_created = now_iso
        if t_valid_from is None:
            t_valid_from = t_created
        if t_tx_created is None:
            t_tx_created = t_created
        t_updated = now_iso
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.fact_upsert_cypher(),
                fact_id=fact_id,
                subject_entity_id=subject_entity_id,
                object_entity_id=object_entity_id,
                relation_type=relation_type,
                fact_text=fact_text,
                evidence_text=evidence_text,
                embedding=embedding,
                confidence=float(confidence),
                source_session=source_session,
                t_created=t_created,
                t_valid_from=t_valid_from,
                t_valid_to=t_valid_to,
                t_tx_created=t_tx_created,
                t_updated=t_updated,
            )
            record = rs.single()
        return (record or {}).get("fact_id") or fact_id

    @staticmethod
    def list_facts_between_cypher() -> str:
        return (
            "MATCH (f:Fact) "
            "WHERE f.subject_entity_id = $subject_entity_id AND f.object_entity_id = $object_entity_id "
            "RETURN f.fact_id AS fact_id, f.relation_type AS relation_type, f.fact_text AS fact_text, "
            "f.evidence_text AS evidence_text, f.confidence AS confidence, f.t_created AS t_created "
            "ORDER BY f.t_created DESC "
            "LIMIT $limit"
        )

    def list_facts_between(
        self,
        *,
        subject_entity_id: str,
        object_entity_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.list_facts_between_cypher(),
                subject_entity_id=subject_entity_id,
                object_entity_id=object_entity_id,
                limit=limit,
            )
            return [dict(r) for r in rs]

    # ──────────────────────────────────────
    # Temporal / Bi-temporal helpers (PR4)
    # ──────────────────────────────────────

    @staticmethod
    def expire_fact_cypher() -> str:
        return (
            "MATCH (f:Fact {fact_id: $fact_id}) "
            "SET f.t_valid_to = $t_valid_to, "
            "    f.t_tx_expired = $t_tx_expired, "
            "    f.t_updated = $t_updated "
            "RETURN f.fact_id AS fact_id"
        )

    def expire_fact(
        self, *, fact_id: str, t_valid_to: str, t_tx_expired: str | None = None
    ) -> None:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if t_tx_expired is None:
            t_tx_expired = now_iso
        with self._driver.session(database=self._database) as session:
            session.run(
                self.expire_fact_cypher(),
                fact_id=fact_id,
                t_valid_to=t_valid_to,
                t_tx_expired=t_tx_expired,
                t_updated=now_iso,
            )

    @staticmethod
    def list_active_facts_for_subject_relation_cypher() -> str:
        return (
            "MATCH (f:Fact) "
            "WHERE f.subject_entity_id = $subject_entity_id "
            "  AND f.relation_type = $relation_type "
            "  AND (f.t_valid_to IS NULL OR f.t_valid_to = '') "
            "RETURN f.fact_id AS fact_id, f.subject_entity_id AS subject_entity_id, "
            "f.object_entity_id AS object_entity_id, f.relation_type AS relation_type, "
            "f.fact_text AS fact_text, f.evidence_text AS evidence_text, f.confidence AS confidence, "
            "coalesce(f.t_valid_from, f.t_created, '') AS t_valid_from, "
            "coalesce(f.t_valid_to, '') AS t_valid_to, "
            "coalesce(f.t_tx_created, '') AS t_tx_created, "
            "coalesce(f.t_tx_expired, '') AS t_tx_expired "
            "ORDER BY coalesce(f.t_valid_from, f.t_created, '') DESC "
            "LIMIT $limit"
        )

    def list_active_facts_for_subject_relation(
        self,
        *,
        subject_entity_id: str,
        relation_type: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.list_active_facts_for_subject_relation_cypher(),
                subject_entity_id=subject_entity_id,
                relation_type=relation_type,
                limit=limit,
            )
            return [dict(r) for r in rs]

    @staticmethod
    def list_active_facts_for_subject_cypher() -> str:
        return (
            "MATCH (f:Fact) "
            "WHERE f.subject_entity_id = $subject_entity_id "
            "  AND (f.t_valid_to IS NULL OR f.t_valid_to = '') "
            "RETURN f.fact_id AS fact_id, f.subject_entity_id AS subject_entity_id, "
            "f.object_entity_id AS object_entity_id, f.relation_type AS relation_type, "
            "f.fact_text AS fact_text, f.evidence_text AS evidence_text, f.confidence AS confidence, "
            "coalesce(f.t_valid_from, f.t_created, '') AS t_valid_from, "
            "coalesce(f.t_valid_to, '') AS t_valid_to, "
            "coalesce(f.t_tx_created, '') AS t_tx_created, "
            "coalesce(f.t_tx_expired, '') AS t_tx_expired "
            "ORDER BY coalesce(f.t_valid_from, f.t_created, '') DESC "
            "LIMIT $limit"
        )

    def list_active_facts_for_subject(
        self,
        *,
        subject_entity_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.list_active_facts_for_subject_cypher(),
                subject_entity_id=subject_entity_id,
                limit=limit,
            )
            return [dict(r) for r in rs]

    # ──────────────────────────────────────
    # Historical facts (PR4)
    # ──────────────────────────────────────

    @staticmethod
    def list_facts_for_subject_relation_cypher() -> str:
        return (
            "MATCH (f:Fact) "
            "WHERE f.subject_entity_id = $subject_entity_id "
            "  AND f.relation_type = $relation_type "
            "RETURN f.fact_id AS fact_id, f.subject_entity_id AS subject_entity_id, "
            "f.object_entity_id AS object_entity_id, f.relation_type AS relation_type, "
            "f.fact_text AS fact_text, f.evidence_text AS evidence_text, f.confidence AS confidence, "
            "coalesce(f.t_valid_from, f.t_created, '') AS t_valid_from, "
            "coalesce(f.t_valid_to, '') AS t_valid_to, "
            "coalesce(f.t_tx_created, '') AS t_tx_created, "
            "coalesce(f.t_tx_expired, '') AS t_tx_expired "
            "ORDER BY coalesce(f.t_valid_from, f.t_created, '') DESC "
            "LIMIT $limit"
        )

    def list_facts_for_subject_relation(
        self,
        *,
        subject_entity_id: str,
        relation_type: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.list_facts_for_subject_relation_cypher(),
                subject_entity_id=subject_entity_id,
                relation_type=relation_type,
                limit=limit,
            )
            return [dict(r) for r in rs]

    @staticmethod
    def list_facts_for_subject_cypher() -> str:
        return (
            "MATCH (f:Fact) "
            "WHERE f.subject_entity_id = $subject_entity_id "
            "RETURN f.fact_id AS fact_id, f.subject_entity_id AS subject_entity_id, "
            "f.object_entity_id AS object_entity_id, f.relation_type AS relation_type, "
            "f.fact_text AS fact_text, f.evidence_text AS evidence_text, f.confidence AS confidence, "
            "coalesce(f.t_valid_from, f.t_created, '') AS t_valid_from, "
            "coalesce(f.t_valid_to, '') AS t_valid_to, "
            "coalesce(f.t_tx_created, '') AS t_tx_created, "
            "coalesce(f.t_tx_expired, '') AS t_tx_expired "
            "ORDER BY coalesce(f.t_valid_from, f.t_created, '') DESC "
            "LIMIT $limit"
        )

    def list_facts_for_subject(
        self,
        *,
        subject_entity_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.list_facts_for_subject_cypher(),
                subject_entity_id=subject_entity_id,
                limit=limit,
            )
            return [dict(r) for r in rs]

    @staticmethod
    def search_facts_by_relation_cypher() -> str:
        return (
            "MATCH (f:Fact) "
            "WHERE toUpper(coalesce(f.relation_type, '')) = toUpper($relation) "
            "RETURN "
            "f.fact_id AS fact_id, "
            "f.subject_entity_id AS source, "
            "f.object_entity_id AS target, "
            "coalesce(f.relation_type, '') AS relation, "
            "coalesce(f.confidence, 1.0) AS confidence, "
            "coalesce(f.fact_text, '') AS fact, "
            "coalesce(f.evidence_text, '') AS evidence, "
            "coalesce(f.source_session, '') AS source_session, "
            "coalesce(f.t_valid_from, f.t_created, '') AS timestamp, "
            "coalesce(f.t_valid_from, '') AS t_valid_from, "
            "coalesce(f.t_valid_to, '') AS t_valid_to "
            "ORDER BY coalesce(f.t_valid_from, f.t_created, '') DESC "
            "LIMIT $limit"
        )

    def search_facts_by_relation(self, *, relation: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(self.search_facts_by_relation_cypher(), relation=relation, limit=limit)
            return [dict(r) for r in rs]

    @staticmethod
    def search_facts_by_time_cypher() -> str:
        # Interval overlap semantics for validity time:
        # - treat missing t_valid_from as t_created
        # - treat missing t_valid_to as open interval (+infinity)
        # Query window is [since, until].
        # Overlap when: f_from <= until AND f_to >= since.
        return (
            "MATCH (f:Fact) "
            "WITH f, "
            "  coalesce(f.t_valid_from, f.t_created, '') AS f_from, "
            "  coalesce(f.t_valid_to, '9999-12-31T23:59:59+00:00') AS f_to "
            "WHERE "
            "  ($until IS NULL OR f_from <= $until) "
            "  AND ($since IS NULL OR f_to >= $since) "
            "RETURN "
            "f.fact_id AS fact_id, "
            "f.subject_entity_id AS source, "
            "f.object_entity_id AS target, "
            "coalesce(f.relation_type, '') AS relation, "
            "coalesce(f.confidence, 1.0) AS confidence, "
            "coalesce(f.fact_text, '') AS fact, "
            "coalesce(f.evidence_text, '') AS evidence, "
            "coalesce(f.source_session, '') AS source_session, "
            "f_from AS timestamp, "
            "coalesce(f.t_valid_from, '') AS t_valid_from, "
            "coalesce(f.t_valid_to, '') AS t_valid_to "
            "ORDER BY f_from DESC "
            "LIMIT $limit"
        )

    # ──────────────────────────────────────
    # Fulltext search: Facts / Communities (PR5)
    # ──────────────────────────────────────

    @staticmethod
    def fulltext_search_facts_cypher() -> str:
        return (
            "CALL db.index.fulltext.queryNodes('fact_fulltext', $q) "
            "YIELD node, score "
            "RETURN node.fact_id AS fact_id, node.subject_entity_id AS source, node.object_entity_id AS target, "
            "coalesce(node.relation_type, '') AS relation, coalesce(node.fact_text, '') AS fact, "
            "coalesce(node.evidence_text, '') AS evidence, coalesce(node.confidence, 1.0) AS confidence, "
            "coalesce(node.source_session, '') AS source_session, "
            "coalesce(node.t_valid_from, node.t_created, '') AS timestamp, "
            "coalesce(node.t_valid_from, '') AS t_valid_from, coalesce(node.t_valid_to, '') AS t_valid_to, "
            "score AS score "
            "ORDER BY score DESC "
            "LIMIT $limit"
        )

    @staticmethod
    def vector_search_facts_cypher() -> str:
        return (
            "CALL db.index.vector.queryNodes('fact_embedding', $k, $embedding) "
            "YIELD node, score "
            "RETURN node.fact_id AS fact_id, node.subject_entity_id AS source, node.object_entity_id AS target, "
            "coalesce(node.relation_type, '') AS relation, coalesce(node.fact_text, '') AS fact, "
            "coalesce(node.evidence_text, '') AS evidence, coalesce(node.confidence, 1.0) AS confidence, "
            "coalesce(node.source_session, '') AS source_session, "
            "coalesce(node.t_valid_from, node.t_created, '') AS timestamp, "
            "coalesce(node.t_valid_from, '') AS t_valid_from, coalesce(node.t_valid_to, '') AS t_valid_to, "
            "score AS score "
            "ORDER BY score DESC "
            "LIMIT $k"
        )

    def fulltext_search_facts(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(self.fulltext_search_facts_cypher(), q=query, limit=limit)
            return [dict(r) for r in rs]

    def vector_search_facts(
        self, *, query_embedding: list[float], limit: int = 10
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            return []
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.vector_search_facts_cypher(),
                embedding=list(query_embedding),
                k=int(limit),
            )
            return [dict(r) for r in rs]

    @staticmethod
    def community_upsert_cypher() -> str:
        return (
            "MERGE (c:Community {community_id: $community_id}) "
            "SET c.name = $name, "
            "    c.summary = $summary, "
            "    c.embedding = $embedding, "
            "    c.t_created = coalesce(c.t_created, $t_created), "
            "    c.t_updated = $t_updated "
            "RETURN c.community_id AS community_id"
        )

    def upsert_community(
        self,
        *,
        community_id: str,
        name: str,
        summary: str = "",
        embedding: list[float] | None = None,
        t_created: str | None = None,
    ) -> str:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if t_created is None:
            t_created = now_iso
        t_updated = now_iso
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.community_upsert_cypher(),
                community_id=community_id,
                name=name,
                summary=summary,
                embedding=embedding,
                t_created=t_created,
                t_updated=t_updated,
            )
            record = rs.single()
        return (record or {}).get("community_id") or community_id

    @staticmethod
    def link_entity_to_community_cypher() -> str:
        return (
            "MATCH (e:Entity {entity_id: $entity_id}) "
            "MATCH (c:Community {community_id: $community_id}) "
            "MERGE (e)-[:IN_COMMUNITY]->(c)"
        )

    def link_entity_to_community(self, *, entity_id: str, community_id: str) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(
                self.link_entity_to_community_cypher(),
                entity_id=entity_id,
                community_id=community_id,
            )

    @staticmethod
    def fulltext_search_communities_cypher() -> str:
        return (
            "CALL db.index.fulltext.queryNodes('community_fulltext', $q) "
            "YIELD node, score "
            "RETURN node.community_id AS community_id, coalesce(node.name, node.community_id) AS name, "
            "coalesce(node.summary, '') AS summary, score AS score "
            "ORDER BY score DESC "
            "LIMIT $limit"
        )

    @staticmethod
    def vector_search_communities_cypher() -> str:
        return (
            "CALL db.index.vector.queryNodes('community_embedding', $k, $embedding) "
            "YIELD node, score "
            "RETURN node.community_id AS community_id, coalesce(node.name, node.community_id) AS name, "
            "coalesce(node.summary, '') AS summary, score AS score "
            "ORDER BY score DESC "
            "LIMIT $k"
        )

    def fulltext_search_communities(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(self.fulltext_search_communities_cypher(), q=query, limit=limit)
            return [dict(r) for r in rs]

    def vector_search_communities(
        self, *, query_embedding: list[float], limit: int = 10
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            return []
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.vector_search_communities_cypher(),
                embedding=list(query_embedding),
                k=int(limit),
            )
            return [dict(r) for r in rs]

    # ──────────────────────────────────────
    # GDS helpers (strict parity)
    # ──────────────────────────────────────

    @staticmethod
    def _gds_project_entity_graph_cypher() -> str:
        return (
            "CALL gds.graph.project.cypher(\n"
            "  $graph_name,\n"
            "  $node_query,\n"
            "  $rel_query,\n"
            "  {parameters: {entity_ids: $entity_ids}}\n"
            ")\n"
            "YIELD graphName\n"
            "RETURN graphName"
        )

    @staticmethod
    def _gds_drop_graph_cypher() -> str:
        return "CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName"

    @staticmethod
    def _entity_neo_ids_by_entity_ids_cypher() -> str:
        return (
            "MATCH (e:Entity) "
            "WHERE e.entity_id IN $entity_ids "
            "RETURN e.entity_id AS entity_id, id(e) AS neo_id"
        )

    def gds_min_distances_from_anchors(
        self,
        *,
        anchor_entity_ids: list[str],
        entity_ids: list[str],
        max_depth: int = 4,
    ) -> dict[str, int]:
        """用 GDS BFS 计算 anchor→entity 的最短 hop 距离（取多个 anchor 的最小值）。

        - 图投影：Entity 作为节点，若存在 Fact 连接两实体则连边（无向）。
        - 仅在给定的 entity_ids 子集上投影（减少开销）。
        """

        anchor_entity_ids = [str(e).strip() for e in (anchor_entity_ids or []) if str(e).strip()]
        entity_ids = [str(e).strip() for e in (entity_ids or []) if str(e).strip()]
        if not anchor_entity_ids or not entity_ids:
            return {}

        all_ids = sorted(set(anchor_entity_ids) | set(entity_ids))
        max_depth = max(1, int(max_depth))

        node_query = "MATCH (e:Entity) WHERE e.entity_id IN $entity_ids RETURN id(e) AS id"
        rel_query = (
            "MATCH (s:Entity)-[:SUBJECT_OF]->(:Fact)-[:OBJECT_OF]->(o:Entity) "
            "WHERE s.entity_id IN $entity_ids AND o.entity_id IN $entity_ids "
            "RETURN id(s) AS source, id(o) AS target, 'RELATED' AS type "
            "UNION ALL "
            "MATCH (s:Entity)-[:SUBJECT_OF]->(:Fact)-[:OBJECT_OF]->(o:Entity) "
            "WHERE s.entity_id IN $entity_ids AND o.entity_id IN $entity_ids "
            "RETURN id(o) AS source, id(s) AS target, 'RELATED' AS type"
        )

        graph_name = f"graphiti_entity_tmp_{uuid.uuid4().hex}"
        with self._driver.session(database=self._database) as session:
            # Project
            session.run(
                self._gds_project_entity_graph_cypher(),
                graph_name=graph_name,
                node_query=node_query,
                rel_query=rel_query,
                entity_ids=all_ids,
            ).single()

            try:
                # Map anchors to neo ids
                rs = session.run(
                    self._entity_neo_ids_by_entity_ids_cypher(), entity_ids=anchor_entity_ids
                )
                anchor_map = {str(r.get("entity_id") or ""): int(r.get("neo_id")) for r in rs}
                anchor_neo_ids = [anchor_map[e] for e in anchor_entity_ids if e in anchor_map]
                if not anchor_neo_ids:
                    return {}

                # Run BFS for each anchor and keep min depth
                depths: dict[str, int] = {}
                for src in anchor_neo_ids:
                    bfs_rs = session.run(
                        "CALL gds.allShortestPaths.dijkstra.stream($graph_name, {sourceNode: $source}) "
                        "YIELD targetNode AS nodeId, totalCost AS depth "
                        "WHERE depth <= $max_depth "
                        "RETURN gds.util.asNode(nodeId).entity_id AS entity_id, toInteger(depth) AS depth",
                        graph_name=graph_name,
                        source=int(src),
                        max_depth=max_depth,
                    )
                    for r in bfs_rs:
                        entity_id = str(r.get("entity_id") or "").strip()
                        if not entity_id:
                            continue
                        depth = int(r.get("depth"))
                        if entity_id not in depths or depth < depths[entity_id]:
                            depths[entity_id] = depth

                return depths
            finally:
                # Drop projected graph
                session.run(self._gds_drop_graph_cypher(), graph_name=graph_name).single()

    def gds_label_propagation_groups(
        self,
        *,
        entity_ids: list[str],
        max_iterations: int = 10,
    ) -> dict[str, list[str]]:
        """用 GDS label propagation 在 entity 子图上做社区发现。

        返回：community_id(str) -> [entity_id,...]
        """

        entity_ids = [str(e).strip() for e in (entity_ids or []) if str(e).strip()]
        if not entity_ids:
            return {}
        max_iterations = max(1, int(max_iterations))

        node_query = "MATCH (e:Entity) WHERE e.entity_id IN $entity_ids RETURN id(e) AS id"
        rel_query = (
            "MATCH (s:Entity)-[:SUBJECT_OF]->(:Fact)-[:OBJECT_OF]->(o:Entity) "
            "WHERE s.entity_id IN $entity_ids AND o.entity_id IN $entity_ids "
            "RETURN id(s) AS source, id(o) AS target, 'RELATED' AS type "
            "UNION ALL "
            "MATCH (s:Entity)-[:SUBJECT_OF]->(:Fact)-[:OBJECT_OF]->(o:Entity) "
            "WHERE s.entity_id IN $entity_ids AND o.entity_id IN $entity_ids "
            "RETURN id(o) AS source, id(s) AS target, 'RELATED' AS type"
        )

        graph_name = f"graphiti_comm_tmp_{uuid.uuid4().hex}"
        with self._driver.session(database=self._database) as session:
            session.run(
                self._gds_project_entity_graph_cypher(),
                graph_name=graph_name,
                node_query=node_query,
                rel_query=rel_query,
                entity_ids=sorted(set(entity_ids)),
            ).single()

            try:
                lp_rs = session.run(
                    "CALL gds.labelPropagation.stream($graph_name, {maxIterations: $max_iter}) "
                    "YIELD nodeId, communityId "
                    "RETURN gds.util.asNode(nodeId).entity_id AS entity_id, communityId",
                    graph_name=graph_name,
                    max_iter=max_iterations,
                )
                groups: dict[str, list[str]] = {}
                for r in lp_rs:
                    eid = str(r.get("entity_id") or "").strip()
                    cid = str(r.get("communityId") or "").strip()
                    if eid and cid:
                        groups.setdefault(cid, []).append(eid)

                for cid in list(groups.keys()):
                    groups[cid] = sorted(set(groups[cid]))
                return groups
            finally:
                session.run(self._gds_drop_graph_cypher(), graph_name=graph_name).single()

    @staticmethod
    def scan_communities_with_embeddings_cypher() -> str:
        return (
            "MATCH (c:Community) WHERE exists(c.embedding) "
            "RETURN c.community_id AS community_id, c.name AS name, c.summary AS summary, c.embedding AS embedding "
            "LIMIT $limit"
        )

    def scan_communities_with_embeddings(self, *, limit: int = 2000) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(self.scan_communities_with_embeddings_cypher(), limit=limit)
            return [dict(r) for r in rs]

    def search_facts_by_time(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.search_facts_by_time_cypher(),
                since=since,
                until=until,
                limit=limit,
            )
            return [dict(r) for r in rs]

    # ──────────────────────────────────────
    # Episode ↔ Fact linking (PR3)
    # ──────────────────────────────────────

    @staticmethod
    def link_episode_to_fact_cypher() -> str:
        return (
            "MATCH (e:Episode {episode_id: $episode_id}) "
            "MATCH (f:Fact {fact_id: $fact_id}) "
            "MERGE (e)-[:EVIDENCE_FOR]->(f)"
        )

    def link_episode_to_fact(self, *, episode_id: str, fact_id: str) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(self.link_episode_to_fact_cypher(), episode_id=episode_id, fact_id=fact_id)

    # ──────────────────────────────────────
    # Episode ↔ Entity linking (paper: episodic edges)
    # ──────────────────────────────────────

    @staticmethod
    def link_episode_to_entity_cypher() -> str:
        return (
            "MATCH (ep:Episode {episode_id: $episode_id}) "
            "MATCH (en:Entity {entity_id: $entity_id}) "
            "MERGE (ep)-[:MENTIONS]->(en)"
        )

    def link_episode_to_entity(self, *, episode_id: str, entity_id: str) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(
                self.link_episode_to_entity_cypher(),
                episode_id=str(episode_id),
                entity_id=str(entity_id),
            )

    # ──────────────────────────────────────
    # Recent-episode entity seeds (paper: BFS seeded by recent episodes)
    # ──────────────────────────────────────

    @staticmethod
    def recent_entity_ids_from_episodes_cypher() -> str:
        return (
            "MATCH (ep:Episode {session_id: $session_id}) "
            "WITH ep ORDER BY ep.turn_index DESC "
            "LIMIT $episode_limit "
            "OPTIONAL MATCH (ep)-[:MENTIONS]->(en:Entity) "
            "WITH en, max(ep.turn_index) AS last_turn "
            "WHERE en IS NOT NULL "
            "RETURN en.entity_id AS entity_id "
            "ORDER BY last_turn DESC "
            "LIMIT $entity_limit"
        )

    def list_recent_entity_ids_from_episodes(
        self,
        *,
        session_id: str,
        episode_limit: int = 4,
        entity_limit: int = 20,
    ) -> list[str]:
        session_id = str(session_id or "").strip()
        if not session_id:
            return []
        with self._driver.session(database=self._database) as session:
            rs = session.run(
                self.recent_entity_ids_from_episodes_cypher(),
                session_id=session_id,
                episode_limit=max(1, int(episode_limit)),
                entity_limit=max(1, int(entity_limit)),
            )
            return [
                str(r.get("entity_id") or "").strip()
                for r in rs
                if str(r.get("entity_id") or "").strip()
            ]

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
            "coalesce(e.type_label, e.label, '') AS label, properties(e) AS properties"
        )

        edges_cypher = (
            "MATCH (f:Fact) "
            "WHERE f.subject_entity_id IS NOT NULL AND f.object_entity_id IS NOT NULL "
            "RETURN "
            "f.subject_entity_id AS source, "
            "f.object_entity_id AS target, "
            "coalesce(f.relation_type, '') AS relation, "
            "coalesce(f.confidence, 1.0) AS confidence, "
            "coalesce(f.fact_text, '') AS fact, "
            "coalesce(f.evidence_text, '') AS evidence, "
            "coalesce(f.source_session, '') AS source_session, "
            "coalesce(f.t_valid_from, f.t_created, '') AS timestamp, "
            "coalesce(f.t_valid_from, '') AS t_valid_from, "
            "coalesce(f.t_valid_to, '') AS t_valid_to"
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
