"""
图谱存储。

开发阶段：NetworkX (纯内存，零外部依赖)
生产阶段：Neo4j
"""

from __future__ import annotations

import datetime
from typing import Any
from typing import Iterable

import networkx as nx
import numpy as np

from a_frame.memory.base import BaseMemoryStore
from a_frame.embedder.base import BaseEmbedder


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class NetworkXGraphStore(BaseMemoryStore):
    """基于 NetworkX 的内存图谱存储，用于开发和测试。"""

    def __init__(
        self,
        *,
        embedder: BaseEmbedder | None = None,
        enable_semantic_search: bool = True,
        enable_semantic_merge: bool = False,
        semantic_merge_threshold: float = 0.88,
        semantic_search_threshold: float = 0.55,
        semantic_degeneracy_epsilon: float = 1e-3,
    ):
        self.graph = nx.DiGraph()
        self._embedder = embedder
        self.enable_semantic_search = enable_semantic_search
        self.enable_semantic_merge = enable_semantic_merge
        self.semantic_merge_threshold = semantic_merge_threshold
        self.semantic_search_threshold = semantic_search_threshold
        self.semantic_degeneracy_epsilon = semantic_degeneracy_epsilon
        self._canonical_id_map: dict[str, str] = {}

    def set_embedder(self, embedder: BaseEmbedder | None) -> None:
        self._embedder = embedder

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-9
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _char_jaccard(a: str, b: str) -> float:
        a_set = {ch for ch in (a or "") if not ch.isspace()}
        b_set = {ch for ch in (b or "") if not ch.isspace()}
        if not a_set or not b_set:
            return 0.0
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        return float(inter) / float(union or 1)

    def _node_display_text(self, node_id: str, data: dict[str, Any]) -> str:
        name = str(data.get("name") or node_id)
        aliases = data.get("aliases")
        if isinstance(aliases, str):
            aliases_list = [aliases]
        elif isinstance(aliases, list):
            aliases_list = [str(x) for x in aliases if x]
        else:
            aliases_list = []
        parts = [name] + [a for a in aliases_list if a != name]
        return " ".join(parts).strip()

    def _iter_node_embeddings(self) -> Iterable[tuple[str, np.ndarray]]:
        for node_id, data in self.graph.nodes(data=True):
            emb = data.get("embedding")
            if isinstance(emb, list) and emb:
                try:
                    yield node_id, np.array(emb, dtype=np.float32)
                except Exception:
                    continue

    def _ensure_node_embedding(self, node_id: str) -> None:
        if not self._embedder:
            return
        data = self.graph.nodes.get(node_id)
        if not data:
            return
        if isinstance(data.get("embedding"), list) and data.get("embedding"):
            return
        text = self._node_display_text(node_id, data)
        if not text:
            return
        try:
            data["embedding"] = self._embedder.embed_query(text)
        except Exception:
            # embedding 失败时不阻塞写入/检索
            return

    def _resolve_canonical_id(self, raw_id: str) -> str:
        return self._canonical_id_map.get(raw_id, raw_id)

    def _add_alias(self, canonical_id: str, alias: str) -> None:
        if not alias or canonical_id not in self.graph.nodes:
            return
        data = self.graph.nodes[canonical_id]
        aliases = data.get("aliases")
        if aliases is None:
            data["aliases"] = [alias]
            return
        if isinstance(aliases, str):
            aliases_list = [aliases]
        elif isinstance(aliases, list):
            aliases_list = aliases
        else:
            aliases_list = []
        if alias not in aliases_list:
            aliases_list.append(alias)
        data["aliases"] = aliases_list

    def _maybe_semantic_merge(self, raw_id: str, name: str, embedding: list[float] | None) -> str:
        if not self.enable_semantic_merge:
            return raw_id
        if embedding is None:
            return raw_id
        existing = list(self._iter_node_embeddings())
        if not existing:
            return raw_id

        q_vec = np.array(embedding, dtype=np.float32)
        sims: list[tuple[float, str]] = []
        for node_id, vec in existing:
            sims.append((self._cosine_similarity(q_vec, vec), node_id))

        if len(sims) >= 3:
            sim_values = [s for s, _ in sims]
            if (max(sim_values) - min(sim_values)) < self.semantic_degeneracy_epsilon:
                # embedding 缺少区分度（常见于测试用 FakeEmbedder 固定向量），禁用自动合并
                return raw_id

        sims.sort(key=lambda x: x[0], reverse=True)
        best_sim, best_id = sims[0]
        if best_sim < self.semantic_merge_threshold:
            return raw_id

        canonical = self._resolve_canonical_id(best_id)
        self._canonical_id_map[raw_id] = canonical
        if name and name != canonical:
            self._add_alias(canonical, name)
        if raw_id != canonical:
            self._add_alias(canonical, raw_id)
        return canonical

    def _upsert_node(self, node_id: str, label: str, properties: dict[str, Any] | None = None) -> str:
        raw_id = str(node_id)
        raw_id = self._resolve_canonical_id(raw_id)

        props = dict(properties or {})
        name = str(props.get("name") or raw_id)

        embedding: list[float] | None = None
        if self._embedder:
            try:
                embedding = self._embedder.embed_query(name)
            except Exception:
                embedding = None

        canonical_id = self._maybe_semantic_merge(raw_id=raw_id, name=name, embedding=embedding)
        if canonical_id in self.graph.nodes:
            # 合并/已存在：补全属性（不覆盖已有的 embedding）
            existing = self.graph.nodes[canonical_id]
            for k, v in props.items():
                if k not in existing:
                    existing[k] = v
            existing["label"] = label
            if embedding is not None and not existing.get("embedding"):
                existing["embedding"] = embedding
            return canonical_id

        self.add_node(canonical_id, label=label, properties=props)
        if embedding is not None:
            self.graph.nodes[canonical_id]["embedding"] = embedding
        return canonical_id

    def add_node(self, node_id: str, label: str, properties: dict[str, Any] | None = None) -> None:
        props = dict(properties or {})
        props["label"] = label  # 显式覆盖，避免与 **props 冲突
        if "created_at" not in props:
            props["created_at"] = _now_iso()
        self.graph.add_node(node_id, **props)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        props = dict(properties or {})
        props.setdefault("timestamp", _now_iso())
        props.setdefault("confidence", 1.0)
        props.setdefault("source_session", "")
        self.graph.add_edge(source_id, target_id, relation=relation, **props)

    def add(self, items: list[dict[str, Any]]) -> None:
        """批量添加三元组 (subject, predicate, object)。"""
        for item in items:
            subj = item["subject"]
            pred = item["predicate"]
            obj = item["object"]
            subj_id = subj.get("id", subj.get("name", ""))
            obj_id = obj.get("id", obj.get("name", ""))
            subj_id = self._upsert_node(subj_id, label=subj.get("label", "Entity"), properties=subj)
            obj_id = self._upsert_node(obj_id, label=obj.get("label", "Entity"), properties=obj)
            edge_props = dict(item.get("properties", {}))
            edge_props.setdefault("confidence", item.get("confidence", 1.0))
            edge_props.setdefault("source_session", item.get("source_session", ""))
            edge_props.setdefault(
                "fact",
                item.get("fact")
                or f"{subj.get('name', subj_id)} {pred} {obj.get('name', obj_id)}",
            )
            edge_props.setdefault("evidence", item.get("evidence", ""))
            self.add_edge(subj_id, obj_id, relation=pred, properties=edge_props)

    def search(
        self,
        query: str,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """双向关键词匹配搜索（开发用）。

        检查逻辑：查询词出现在节点文本中，或节点的 name/id 出现在查询中。
        生产环境应使用 Neo4j 全文索引 + embedding 相似度。
        """
        # 1) embedding 相似度优先（如果可用）
        if query_embedding is not None and self.enable_semantic_search:
            # 确保节点有 embedding（lazy）
            if self._embedder:
                for node_id in list(self.graph.nodes):
                    self._ensure_node_embedding(node_id)

            candidates = list(self._iter_node_embeddings())
            if candidates:
                q_vec = np.array(query_embedding, dtype=np.float32)
                sims: list[tuple[float, str]] = []
                for node_id, vec in candidates:
                    sims.append((self._cosine_similarity(q_vec, vec), node_id))

                # degenerate embedding（例如测试用固定向量）时，改用字符重叠召回，避免无关命中
                if len(sims) >= 3:
                    sim_values = [s for s, _ in sims]
                    if (max(sim_values) - min(sim_values)) < self.semantic_degeneracy_epsilon:
                        scored = []
                        for node_id, data in self.graph.nodes(data=True):
                            lex = self._char_jaccard(query, self._node_display_text(node_id, data))
                            if lex > 0.0:
                                scored.append((lex, node_id, data))
                        scored.sort(key=lambda x: x[0], reverse=True)
                        return [
                            {"node_id": node_id, **data}
                            for _, node_id, data in scored[:top_k]
                        ]

                sims.sort(key=lambda x: x[0], reverse=True)
                results: list[dict[str, Any]] = []
                for score, node_id in sims:
                    if score < self.semantic_search_threshold:
                        continue
                    data = dict(self.graph.nodes[node_id])
                    results.append({"node_id": node_id, **data, "_score": score})
                    if len(results) >= top_k:
                        break
                if results:
                    return results

        # 2) 关键词 fallback
        results: list[dict[str, Any]] = []
        query_lower = query.lower()
        for node_id, data in self.graph.nodes(data=True):
            node_text = " ".join(str(v) for v in data.values()).lower()
            node_name = str(data.get("name", node_id)).lower()
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

    def get_all_edges(self) -> list[dict[str, Any]]:
        """返回所有边（含 relation, timestamp, confidence 等属性）。"""
        return [
            {"source": u, "target": v, **d}
            for u, v, d in self.graph.edges(data=True)
        ]

    def search_by_relation(self, relation: str, top_k: int = 10) -> list[dict[str, Any]]:
        """按关系类型检索边。"""
        results = []
        rel_lower = relation.lower()
        for u, v, d in self.graph.edges(data=True):
            edge_rel = str(d.get("relation", "")).lower()
            if rel_lower in edge_rel or edge_rel in rel_lower:
                results.append({"source": u, "target": v, **d})
                if len(results) >= top_k:
                    break
        return results

    def search_by_time(
        self,
        since: str | None = None,
        until: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """按时间范围检索边。

        Args:
            since: ISO 格式起始时间（含），None 表示不限。
            until: ISO 格式截止时间（含），None 表示不限。
            top_k: 最大返回数。

        Returns:
            符合时间范围的边列表，按时间降序。
        """
        edges = []
        for u, v, d in self.graph.edges(data=True):
            ts = d.get("timestamp", "")
            if not ts:
                continue
            if since and ts < since:
                continue
            if until and ts > until:
                continue
            edges.append({"source": u, "target": v, **d})

        edges.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return edges[:top_k]
