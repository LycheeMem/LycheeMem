"""
图谱存储。

开发阶段：NetworkX (纯内存，零外部依赖)
生产阶段：Neo4j
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from a_frame.memory.base import BaseMemoryStore


class NetworkXGraphStore(BaseMemoryStore):
    """基于 NetworkX 的内存图谱存储，用于开发和测试。"""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_id: str, label: str, properties: dict[str, Any] | None = None) -> None:
        props = dict(properties or {})
        props["label"] = label  # 显式覆盖，避免与 **props 冲突
        self.graph.add_node(node_id, **props)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        props = properties or {}
        self.graph.add_edge(source_id, target_id, relation=relation, **props)

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
        """双向关键词匹配搜索（开发用）。

        检查逻辑：查询词出现在节点文本中，或节点的 name/id 出现在查询中。
        生产环境应使用 Neo4j 全文索引 + embedding 相似度。
        """
        results = []
        query_lower = query.lower()
        for node_id, data in self.graph.nodes(data=True):
            node_text = " ".join(str(v) for v in data.values()).lower()
            node_name = str(data.get("name", node_id)).lower()
            # 双向匹配：查询包含节点名，或节点文本包含查询
            if node_name in query_lower or query_lower in node_text:
                results.append({"node_id": node_id, **data})
                if len(results) >= top_k:
                    break
        return results

    def get_neighbors(self, node_id: str, depth: int = 1) -> dict[str, Any]:
        """获取节点的 N 跳邻居子图。"""
        if node_id not in self.graph:
            return {"nodes": [], "edges": []}

        subgraph_nodes = {node_id}
        frontier = {node_id}
        for _ in range(depth):
            next_frontier = set()
            for n in frontier:
                next_frontier.update(self.graph.successors(n))
                next_frontier.update(self.graph.predecessors(n))
            subgraph_nodes.update(next_frontier)
            frontier = next_frontier

        nodes = [
            {"id": n, **self.graph.nodes[n]} for n in subgraph_nodes if n in self.graph.nodes
        ]
        edges = [
            {"source": u, "target": v, **d}
            for u, v, d in self.graph.edges(data=True)
            if u in subgraph_nodes and v in subgraph_nodes
        ]
        return {"nodes": nodes, "edges": edges}

    def delete(self, ids: list[str]) -> None:
        for node_id in ids:
            if node_id in self.graph:
                self.graph.remove_node(node_id)

    def get_all(self) -> list[dict[str, Any]]:
        return [{"id": n, **d} for n, d in self.graph.nodes(data=True)]
