"""
Neo4j 图谱存储。

持久化版本的知识图谱存储，兼容 NetworkXGraphStore/BaseMemoryStore 接口。
使用 Neo4j Python Driver 与 Neo4j 通信。
"""

from __future__ import annotations

import json
from typing import Any

from neo4j import GraphDatabase

from a_frame.memory.base import BaseMemoryStore


class Neo4jGraphStore(BaseMemoryStore):
    """基于 Neo4j 的持久化图谱存储。"""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._init_db()

    def _init_db(self) -> None:
        """创建全文索引以支持搜索。"""
        with self._driver.session(database=self._database) as session:
            # 为所有节点的 name 属性创建索引
            session.run(
                "CREATE INDEX node_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)"
            )

    def add_node(self, node_id: str, label: str, properties: dict[str, Any] | None = None) -> None:
        props = dict(properties or {})
        props["node_id"] = node_id
        props["label"] = label
        # 清理不可序列化的值
        clean_props = {k: v for k, v in props.items() if isinstance(v, (str, int, float, bool))}
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
            clean_props = {k: v for k, v in properties.items() if isinstance(v, (str, int, float, bool))}
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

    def add(self, items: list[dict[str, Any]]) -> None:
        """批量添加三元组 (subject, predicate, object)。"""
        for item in items:
            subj = item["subject"]
            pred = item["predicate"]
            obj = item["object"]
            subj_id = subj.get("id", subj.get("name", ""))
            obj_id = obj.get("id", obj.get("name", ""))
            self.add_node(subj_id, label=subj.get("label", "Entity"), properties=subj)
            self.add_node(obj_id, label=obj.get("label", "Entity"), properties=obj)
            self.add_edge(subj_id, obj_id, relation=pred, properties=item.get("properties", {}))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """双向关键词匹配搜索。"""
        query_lower = query.lower()
        with self._driver.session(database=self._database) as session:
            # 搜索节点 name 包含在 query 中，或 query 包含在节点 name 中
            result = session.run(
                """
                MATCH (n:Entity)
                WHERE toLower(n.name) CONTAINS toLower($query)
                   OR toLower($query) CONTAINS toLower(n.name)
                RETURN n
                LIMIT $top_k
                """,
                query=query,
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
                OPTIONAL MATCH path = (start)-[*1..""" + str(depth) + """]->(neighbor)
                WITH start, collect(DISTINCT neighbor) AS out_neighbors
                OPTIONAL MATCH path2 = (ancestor)-[*1..""" + str(depth) + """]->(start)
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
                    collect(DISTINCT {source: a.node_id, target: b.node_id, relation: r.relation}) AS edges
                """,
                node_id=node_id,
            )
            record = result.single()
            if record is None:
                return {"nodes": [], "edges": []}
            nodes = [n for n in record["nodes"] if n.get("id")]
            edges = [e for e in record["edges"] if e.get("source") and e.get("relation")]
            return {"nodes": nodes, "edges": edges}

    def delete(self, ids: list[str]) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(
                "MATCH (n:Entity) WHERE n.node_id IN $ids DETACH DELETE n",
                ids=ids,
            )

    def get_all(self) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            result = session.run("MATCH (n:Entity) RETURN n")
            return [
                {"id": record["n"].get("node_id", ""), **dict(record["n"])}
                for record in result
            ]

    def get_all_edges(self) -> list[dict[str, Any]]:
        """获取所有边（用于 API 的 /memory/graph 端点）。"""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (a:Entity)-[r:RELATES]->(b:Entity)
                RETURN a.node_id AS source, b.node_id AS target, r.relation AS relation
                """
            )
            return [
                {"source": record["source"], "target": record["target"], "relation": record["relation"]}
                for record in result
            ]

    def close(self) -> None:
        self._driver.close()
