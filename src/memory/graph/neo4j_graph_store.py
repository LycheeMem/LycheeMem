"""
Neo4j 图谱存储。

持久化版本的知识图谱存储，兼容 NetworkXGraphStore/BaseMemoryStore 接口。
使用 Neo4j Python Driver 与 Neo4j 通信。
"""

from __future__ import annotations

import datetime
from typing import Any
from typing import Iterable

import numpy as np

from neo4j import GraphDatabase

from src.memory.base import BaseMemoryStore
from src.embedder.base import BaseEmbedder


class Neo4jGraphStore(BaseMemoryStore):
    """基于 Neo4j 的持久化图谱存储。"""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
        *,
        embedder: BaseEmbedder | None = None,
        enable_semantic_search: bool = True,
        enable_semantic_merge: bool = True,
        semantic_merge_threshold: float = 0.88,
        semantic_search_threshold: float = 0.55,
        semantic_degeneracy_epsilon: float = 1e-3,
        semantic_scan_limit: int = 5000,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="NONE")
        self._database = database
        self._embedder = embedder
        self.enable_semantic_search = enable_semantic_search
        self.enable_semantic_merge = enable_semantic_merge
        self.semantic_merge_threshold = semantic_merge_threshold
        self.semantic_search_threshold = semantic_search_threshold
        self.semantic_degeneracy_epsilon = semantic_degeneracy_epsilon
        self.semantic_scan_limit = semantic_scan_limit
        self._canonical_id_map: dict[str, str] = {}
        self._init_db()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-9
        return float(np.dot(a, b) / denom)

    def _resolve_canonical_id(self, raw_id: str) -> str:
        canonical_map = getattr(self, "_canonical_id_map", None)
        if not isinstance(canonical_map, dict):
            return raw_id
        return canonical_map.get(raw_id, raw_id)

    def _iter_node_embeddings(self) -> Iterable[tuple[str, list[float], str]]:
        """遍历 Neo4j 中已有 embedding 的节点（受 semantic_scan_limit 限制）。"""
        # 允许测试中通过 __new__ 构造（未初始化 driver）
        driver = getattr(self, "_driver", None)
        database = getattr(self, "_database", None)
        if driver is None or database is None:
            return

        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (n:Entity)
                WHERE n.embedding IS NOT NULL
                RETURN n.node_id AS node_id, n.embedding AS embedding, coalesce(n.name, '') AS name
                LIMIT $limit
                """,
                limit=getattr(self, "semantic_scan_limit", 5000),
            )
            for record in result:
                node_id = record.get("node_id") or ""
                emb = record.get("embedding")
                name = record.get("name") or ""
                if not node_id or not isinstance(emb, list) or not emb:
                    continue
                yield node_id, emb, name

    def _maybe_semantic_merge(self, raw_id: str, name: str, embedding: list[float] | None) -> str:
        if not getattr(self, "enable_semantic_merge", True):
            return raw_id
        if embedding is None:
            return raw_id
        candidates = list(self._iter_node_embeddings())
        if not candidates:
            return raw_id

        q_vec = np.array(embedding, dtype=np.float32)
        sims: list[tuple[float, str]] = []
        for node_id, emb, _ in candidates:
            try:
                sims.append(
                    (self._cosine_similarity(q_vec, np.array(emb, dtype=np.float32)), node_id)
                )
            except Exception:
                continue

        if len(sims) >= 3:
            sim_values = [s for s, _ in sims]
            if (max(sim_values) - min(sim_values)) < getattr(
                self, "semantic_degeneracy_epsilon", 1e-3
            ):
                # embedding 缺少区分度（例如固定向量测试），避免误合并
                return raw_id

        sims.sort(key=lambda x: x[0], reverse=True)
        best_sim, best_id = sims[0]
        if best_sim < getattr(self, "semantic_merge_threshold", 0.88):
            return raw_id

        canonical = self._resolve_canonical_id(best_id)
        canonical_map = getattr(self, "_canonical_id_map", None)
        if not isinstance(canonical_map, dict):
            canonical_map = {}
            setattr(self, "_canonical_id_map", canonical_map)
        canonical_map[raw_id] = canonical

        # 记录 alias（不依赖 APOC，允许少量重复）
        # 如果 driver 不存在（例如 __new__ 的桩测试），跳过 alias 回写
        if getattr(self, "_driver", None) is None:
            return canonical

        with self._driver.session(database=self._database) as session:
            if name and name != canonical:
                session.run(
                    """
                    MATCH (n:Entity {node_id: $node_id})
                    SET n.aliases = coalesce(n.aliases, []) + $alias
                    """,
                    node_id=canonical,
                    alias=name,
                )
            if raw_id != canonical:
                session.run(
                    """
                    MATCH (n:Entity {node_id: $node_id})
                    SET n.aliases = coalesce(n.aliases, []) + $alias
                    """,
                    node_id=canonical,
                    alias=raw_id,
                )
        return canonical

    def _upsert_node(
        self, node_id: str, label: str, properties: dict[str, Any] | None = None
    ) -> str:
        raw_id = str(node_id)
        raw_id = self._resolve_canonical_id(raw_id)
        props = dict(properties or {})
        name = str(props.get("name") or raw_id)

        embedding: list[float] | None = None
        embedder = getattr(self, "_embedder", None)
        if embedder is not None:
            try:
                embedding = embedder.embed_query(name)
            except Exception:
                embedding = None

        canonical_id = self._maybe_semantic_merge(raw_id=raw_id, name=name, embedding=embedding)
        if canonical_id != raw_id:
            # 合并到已有 canonical：只补充属性即可
            self.add_node(canonical_id, label=label, properties=props)
            return canonical_id

        # 新节点：写入并附带 embedding
        if embedding is not None:
            props.setdefault("embedding", embedding)
        self.add_node(canonical_id, label=label, properties=props)
        return canonical_id

    def _init_db(self) -> None:
        """创建全文索引以支持搜索。"""
        with self._driver.session(database=self._database) as session:
            # 为所有节点的 name 属性创建索引
            session.run("CREATE INDEX node_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)")

    def add_node(self, node_id: str, label: str, properties: dict[str, Any] | None = None) -> None:
        props = dict(properties or {})
        props["node_id"] = node_id
        props["label"] = label

        # 清理不可序列化的值（Neo4j 支持 list 基本类型）
        def _is_neo4j_value(v: Any) -> bool:
            if isinstance(v, (str, int, float, bool)):
                return True
            if isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v):
                return True
            return False

        clean_props = {k: v for k, v in props.items() if _is_neo4j_value(v)}
        with self._driver.session(database=self._database) as session:
            session.run(
                "MERGE (n:Entity {node_id: $node_id}) SET n += $props",
                node_id=node_id,
                props=clean_props,
            )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        clean_props = {}
        if properties:
            clean_props = {
                k: v for k, v in properties.items() if isinstance(v, (str, int, float, bool))
            }
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (a:Entity {node_id: $source_id})
                MATCH (b:Entity {node_id: $target_id})
                MERGE (a)-[r:RELATES {relation: $relation}]->(b)
                SET r += $props
                """,
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                props=clean_props,
            )

    def add(self, items: list[dict[str, Any]], *, user_id: str = "") -> None:
        """批量添加三元组 (subject, predicate, object)。"""
        for item in items:
            subj = item["subject"]
            pred = item["predicate"]
            obj = item["object"]
            subj_id = subj.get("id", subj.get("name", ""))
            obj_id = obj.get("id", obj.get("name", ""))
            subj_props = dict(subj)
            obj_props = dict(obj)
            if user_id:
                subj_props["user_id"] = user_id
                obj_props["user_id"] = user_id
            subj_id = self._upsert_node(subj_id, label=subj.get("label", "Entity"), properties=subj_props)
            obj_id = self._upsert_node(obj_id, label=obj.get("label", "Entity"), properties=obj_props)
            edge_props = dict(item.get("properties", {}) or {})
            edge_props.setdefault(
                "timestamp",
                item.get("timestamp") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            edge_props.setdefault("confidence", item.get("confidence", 1.0))
            edge_props.setdefault("source_session", item.get("source_session", ""))
            edge_props.setdefault(
                "fact",
                item.get("fact") or f"{subj.get('name', subj_id)} {pred} {obj.get('name', obj_id)}",
            )
            edge_props.setdefault("evidence", item.get("evidence", ""))
            if user_id:
                edge_props["user_id"] = user_id
            self.add_edge(subj_id, obj_id, relation=pred, properties=edge_props)

    def search(
        self,
        query: str,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
        *,
        user_id: str = "",
    ) -> list[dict[str, Any]]:
        """优先 embedding 相似度检索，fallback 到双向关键词匹配。"""

        if query_embedding is not None and self.enable_semantic_search:
            candidates = list(self._iter_node_embeddings())
            if candidates:
                q_vec = np.array(query_embedding, dtype=np.float32)
                sims: list[tuple[float, str, str]] = []
                for node_id, emb, name in candidates:
                    try:
                        sims.append(
                            (
                                self._cosine_similarity(q_vec, np.array(emb, dtype=np.float32)),
                                node_id,
                                name,
                            )
                        )
                    except Exception:
                        continue

                if len(sims) >= 3:
                    sim_values = [s for s, _, _ in sims]
                    if (max(sim_values) - min(sim_values)) < self.semantic_degeneracy_epsilon:
                        # embedding 缺少区分度时，退回关键词（避免无关命中）
                        candidates = []

                if sims:
                    sims.sort(key=lambda x: x[0], reverse=True)
                    results: list[dict[str, Any]] = []
                    for score, node_id, _ in sims:
                        if score < self.semantic_search_threshold:
                            continue
                        results.append({"node_id": node_id, "_score": score})
                        if len(results) >= top_k:
                            break
                    if results:
                        # 回填节点属性
                        ids = [r["node_id"] for r in results]
                        with self._driver.session(database=self._database) as session:
                            rs = session.run(
                                "MATCH (n:Entity) WHERE n.node_id IN $ids RETURN n",
                                ids=ids,
                            )
                            by_id = {
                                record["n"].get("node_id", ""): dict(record["n"]) for record in rs
                            }

                        merged: list[dict[str, Any]] = []
                        for r in results:
                            node_id = r["node_id"]
                            props = by_id.get(node_id, {})
                            if user_id and props.get("user_id", "") != user_id:
                                continue
                            merged.append(
                                {
                                    "node_id": node_id,
                                    **props,
                                    "_score": r.get("_score"),
                                }
                            )
                        return merged

        with self._driver.session(database=self._database) as session:
            if user_id:
                result = session.run(
                    """
                    MATCH (n:Entity)
                    WHERE (toLower(n.name) CONTAINS toLower($search_term)
                       OR toLower($search_term) CONTAINS toLower(n.name))
                      AND n.user_id = $user_id
                    RETURN n
                    LIMIT $top_k
                    """,
                    search_term=query,
                    top_k=top_k,
                    user_id=user_id,
                )
            else:
                result = session.run(
                    """
                    MATCH (n:Entity)
                    WHERE toLower(n.name) CONTAINS toLower($search_term)
                       OR toLower($search_term) CONTAINS toLower(n.name)
                    RETURN n
                    LIMIT $top_k
                    """,
                    search_term=query,
                    top_k=top_k,
                )
            return [
                {"node_id": record["n"].get("node_id", ""), **dict(record["n"])}
                for record in result
            ]

    def get_neighbors(self, node_id: str, depth: int = 1) -> dict[str, Any]:
        """获取节点的 N 跳邻居子图。"""
        with self._driver.session(database=self._database) as session:
            # 使用可变长度路径获取 N 跳邻居
            result = session.run(
                """
                MATCH (start:Entity {node_id: $node_id})
                OPTIONAL MATCH path = (start)-[*1.."""
                + str(depth)
                + """]->(neighbor)
                WITH start, collect(DISTINCT neighbor) AS out_neighbors
                OPTIONAL MATCH path2 = (ancestor)-[*1.."""
                + str(depth)
                + """]->(start)
                WITH start, out_neighbors, collect(DISTINCT ancestor) AS in_neighbors
                WITH start, out_neighbors + in_neighbors + [start] AS all_nodes
                UNWIND all_nodes AS n
                WITH collect(DISTINCT n) AS nodes
                UNWIND nodes AS a
                UNWIND nodes AS b
                OPTIONAL MATCH (a)-[r:RELATES]->(b)
                WHERE r IS NOT NULL
                RETURN
                    collect(DISTINCT {id: a.node_id, label: a.label, name: a.name}) AS nodes,
                    collect(DISTINCT {
                        source: a.node_id,
                        target: b.node_id,
                        relation: r.relation,
                        timestamp: r.timestamp,
                        confidence: r.confidence,
                        source_session: r.source_session,
                        fact: r.fact,
                        evidence: r.evidence
                    }) AS edges
                """,
                node_id=node_id,
            )
            record = result.single()
            if record is None:
                return {"nodes": [], "edges": []}
            nodes = [n for n in record["nodes"] if n.get("id")]
            edges = [e for e in record["edges"] if e.get("source") and e.get("relation")]
            return {"nodes": nodes, "edges": edges}

    def delete(self, ids: list[str], *, user_id: str = "") -> None:
        with self._driver.session(database=self._database) as session:
            if user_id:
                session.run(
                    "MATCH (n:Entity) WHERE n.node_id IN $ids"
                    " AND (n.user_id = $user_id OR n.user_id IS NULL OR n.user_id = '')"
                    " DETACH DELETE n",
                    ids=ids,
                    user_id=user_id,
                )
            else:
                session.run(
                    "MATCH (n:Entity) WHERE n.node_id IN $ids DETACH DELETE n",
                    ids=ids,
                )

    def delete_all(self, *, user_id: str = "") -> None:
        """删除所有 Entity 节点及关联边。user_id 非空时只删该用户的节点。"""
        with self._driver.session(database=self._database) as session:
            if user_id:
                session.run(
                    "MATCH (n:Entity) "
                    "WHERE n.user_id = $user_id OR n.user_id IS NULL OR n.user_id = '' "
                    "DETACH DELETE n",
                    user_id=user_id,
                )
            else:
                session.run("MATCH (n:Entity) DETACH DELETE n")

    def get_all(self, *, user_id: str = "") -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            if user_id:
                result = session.run(
                    "MATCH (n:Entity) WHERE n.user_id = $user_id RETURN n",
                    user_id=user_id,
                )
            else:
                result = session.run("MATCH (n:Entity) RETURN n")
            return [
                {"id": record["n"].get("node_id", ""), **dict(record["n"])} for record in result
            ]

    def get_all_edges(self, *, user_id: str = "") -> list[dict[str, Any]]:
        """获取所有边（用于 API 的 /memory/graph 端点）。"""
        with self._driver.session(database=self._database) as session:
            if user_id:
                result = session.run(
                    """
                    MATCH (a:Entity)-[r:RELATES]->(b:Entity)
                    WHERE r.user_id = $user_id
                    RETURN
                        a.node_id AS source,
                        b.node_id AS target,
                        r.relation AS relation,
                        r.timestamp AS timestamp,
                        r.confidence AS confidence,
                        r.source_session AS source_session,
                        r.fact AS fact,
                        r.evidence AS evidence
                    """,
                    user_id=user_id,
                )
            else:
                result = session.run(
                    """
                    MATCH (a:Entity)-[r:RELATES]->(b:Entity)
                    RETURN
                        a.node_id AS source,
                        b.node_id AS target,
                        r.relation AS relation,
                        r.timestamp AS timestamp,
                        r.confidence AS confidence,
                        r.source_session AS source_session,
                        r.fact AS fact,
                        r.evidence AS evidence
                    """
                )
            return [
                {
                    "source": record["source"],
                    "target": record["target"],
                    "relation": record["relation"],
                    "timestamp": record.get("timestamp"),
                    "confidence": record.get("confidence"),
                    "source_session": record.get("source_session"),
                    "fact": record.get("fact"),
                    "evidence": record.get("evidence"),
                }
                for record in result
            ]

    def close(self) -> None:
        self._driver.close()
