"""Graphiti(论文) 风格图谱引擎：Search/Rerank/Constructor 的对外门面。

PR1 目标：提供可注入的引擎骨架与兼容导出接口；真正的 ingest/search 能力
在后续 PR 分步实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import datetime
import hashlib

from collections import Counter

from a_frame.memory.graph.graphiti_neo4j_store import GraphitiNeo4jStore


@dataclass(slots=True)
class GraphitiSearchResult:
    context: str
    provenance: list[dict[str, Any]]


class GraphitiEngine:
    """Graphiti 引擎（面向论文的 f(α)=χ(ρ(φ(α)))）。"""

    def __init__(self, store: GraphitiNeo4jStore):
        self.store = store

    @staticmethod
    def _default_episode_id(*, session_id: str, turn_index: int, role: str, content: str) -> str:
        raw = f"{session_id}|{turn_index}|{role}|{content}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def ingest_episode(
        self,
        *,
        session_id: str,
        turn_index: int,
        role: str,
        content: str,
        t_ref: str | None = None,
        episode_id: str | None = None,
    ) -> str:
        """写入一个 Episode（幂等）。

        PR2 仅做“原始事件落盘 + 可追溯 id”，不触发实体/事实解析。
        """

        if t_ref is None:
            t_ref = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if episode_id is None:
            episode_id = self._default_episode_id(
                session_id=session_id,
                turn_index=turn_index,
                role=role,
                content=content,
            )

        return self.store.upsert_episode(
            episode_id=episode_id,
            session_id=session_id,
            role=role,
            content=content,
            turn_index=turn_index,
            t_ref=t_ref,
        )

    def search(
        self,
        *,
        query: str,
        session_id: str | None = None,
        top_k: int = 10,
        query_embedding: list[float] | None = None,
        include_communities: bool = True,
    ) -> GraphitiSearchResult:
        """论文式检索：Search→Rerank→Constructor。

        PR5 实现要点（轻量版）：
        - 召回：Fact BM25(全文索引) + Entity cosine(embedding) + BFS(从高相关实体扩展事实)
        - 复排：RRF + mentions boost + graph distance
        - 构造：将 top facts + 相关实体/社区摘要为可直接注入 LLM 的 context 文本
        """

        if not query.strip():
            return GraphitiSearchResult(context="", provenance=[])

        top_k = max(1, int(top_k))

        # ──────────────────────────────────────
        # Channel 1: BM25 fulltext over facts
        # ──────────────────────────────────────
        bm25_facts: list[dict[str, Any]] = []
        if hasattr(self.store, "fulltext_search_facts"):
            try:
                bm25_facts = self.store.fulltext_search_facts(query=query, limit=max(30, top_k * 6))
            except Exception:
                bm25_facts = []

        # ──────────────────────────────────────
        # Channel 2: cosine similarity over entities (best-effort scan)
        # ──────────────────────────────────────
        cosine_entities: list[dict[str, Any]] = []
        if query_embedding is not None:
            try:
                scanned = self.store.scan_entities_with_embeddings(limit=2000)
            except Exception:
                scanned = []

            scored: list[dict[str, Any]] = []
            for row in scanned:
                emb = row.get("embedding")
                if not emb:
                    continue
                sim = self._cosine_similarity(query_embedding, emb)
                scored.append(
                    {
                        "entity_id": row.get("entity_id"),
                        "name": row.get("name"),
                        "summary": row.get("summary", ""),
                        "type_label": row.get("type_label", ""),
                        "score": sim,
                    }
                )

            scored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
            cosine_entities = scored[: max(20, top_k * 4)]

        # ──────────────────────────────────────
        # Anchor entities for BFS expansion (mentions + high-scoring)
        # ──────────────────────────────────────
        ft_entities: list[dict[str, Any]] = []
        try:
            ft_entities = self.store.fulltext_search_entities(query=query, limit=10)
        except Exception:
            ft_entities = []

        anchor_entity_ids: list[str] = []
        for row in (ft_entities[:5] + cosine_entities[:5]):
            eid = (row.get("entity_id") or row.get("id") or "").strip()
            if eid and eid not in anchor_entity_ids:
                anchor_entity_ids.append(eid)
        anchor_entity_ids = anchor_entity_ids[:5]

        # ──────────────────────────────────────
        # Channel 3: BFS / local subgraph expansion
        # ──────────────────────────────────────
        bfs_fact_candidates: dict[str, dict[str, Any]] = {}
        bfs_ranked_fact_ids: list[str] = []
        entity_cache: dict[str, dict[str, Any]] = {}

        frontier = list(anchor_entity_ids)
        visited_entities = set(frontier)
        max_depth = 2
        for depth in range(max_depth):
            if not frontier:
                break
            try:
                subgraph = self.store.export_semantic_subgraph(entity_ids=frontier, edge_limit=600)
            except Exception:
                break

            nodes = subgraph.get("nodes") or []
            edges = subgraph.get("edges") or []

            for n in nodes:
                nid = (n.get("id") or "").strip()
                if nid:
                    entity_cache[nid] = n

            next_frontier: list[str] = []
            for e in edges:
                fact_id = (e.get("fact_id") or "").strip() or f"fact:{e.get('source','')}:{e.get('relation','')}:{e.get('target','')}:{e.get('timestamp','')}"
                if fact_id not in bfs_fact_candidates:
                    bfs_fact_candidates[fact_id] = {
                        "fact_id": fact_id,
                        "edge": e,
                        "distance": depth + 1,
                    }

                for endpoint in [e.get("source"), e.get("target")]:
                    endpoint = (endpoint or "").strip()
                    if endpoint and endpoint not in visited_entities:
                        visited_entities.add(endpoint)
                        next_frontier.append(endpoint)

            frontier = next_frontier[:20]

        # Deterministic order for BFS facts: shorter distance first, then timestamp desc
        bfs_ordered = sorted(
            bfs_fact_candidates.values(),
            key=lambda x: (
                int(x.get("distance") or 999),
                str((x.get("edge") or {}).get("timestamp") or ""),
            ),
        )
        bfs_ranked_fact_ids = [x["fact_id"] for x in bfs_ordered]

        # ──────────────────────────────────────
        # Community retrieval (optional, best-effort)
        # ──────────────────────────────────────
        communities: list[dict[str, Any]] = []
        built_communities: list[dict[str, Any]] = []
        if include_communities:
            # 1) query-time retrieval
            if hasattr(self.store, "fulltext_search_communities"):
                try:
                    communities = self.store.fulltext_search_communities(query=query, limit=5)
                except Exception:
                    communities = []

            if query_embedding is not None and hasattr(self.store, "scan_communities_with_embeddings"):
                try:
                    scanned = self.store.scan_communities_with_embeddings(limit=2000)
                except Exception:
                    scanned = []

                scored = []
                for row in scanned:
                    emb = row.get("embedding")
                    if not emb:
                        continue
                    sim = self._cosine_similarity(query_embedding, emb)
                    scored.append(
                        {
                            "community_id": row.get("community_id"),
                            "name": row.get("name") or row.get("community_id"),
                            "summary": row.get("summary", ""),
                            "score": sim,
                        }
                    )
                scored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
                for c in scored[:5]:
                    communities.append(c)

            # 2) dynamic expansion + refresh (label propagation)
            if (not communities) and anchor_entity_ids:
                try:
                    built_communities = self.build_or_refresh_communities_for_entities(
                        anchor_entity_ids,
                        entity_cache=entity_cache,
                        edges=[(info.get("edge") or {}) for info in bfs_fact_candidates.values()],
                    )
                except Exception:
                    built_communities = []

            if not communities and built_communities:
                communities = built_communities[:5]

        # ──────────────────────────────────────
        # Merge facts + rerank
        # ──────────────────────────────────────

        def _extract_fact_id(row: dict[str, Any]) -> str:
            fid = (row.get("fact_id") or "").strip()
            if fid:
                return fid
            # fallback stable-ish id
            return f"fact:{row.get('source','')}:{row.get('relation','')}:{row.get('target','')}:{row.get('timestamp','')}"

        fact_by_id: dict[str, dict[str, Any]] = {}

        bm25_ranked_fact_ids: list[str] = []
        for row in bm25_facts:
            fid = _extract_fact_id(row)
            bm25_ranked_fact_ids.append(fid)
            fact_by_id.setdefault(fid, {}).update(row)

        # Add BFS edges as fact candidates too
        for fid, info in bfs_fact_candidates.items():
            edge = info.get("edge") or {}
            fact_by_id.setdefault(fid, {}).update(edge)
            fact_by_id[fid].setdefault("fact_id", fid)

        # Mention boost: entity name appears in query
        query_l = query.lower()
        mention_entity_ids: set[str] = set()
        for ent in (ft_entities[:10] + cosine_entities[:10]):
            eid = (ent.get("entity_id") or "").strip()
            name = str(ent.get("name") or "").strip()
            if eid and name and name.lower() in query_l:
                mention_entity_ids.add(eid)

        # RRF scores
        rrf_scores: dict[str, float] = {}
        rrf_scores = self._rrf_scores(
            {
                "bm25": bm25_ranked_fact_ids,
                "bfs": bfs_ranked_fact_ids,
            },
            k=60,
        )

        # Add distance/mention boosts
        final_scores: dict[str, float] = {}
        provenance_by_fact: dict[str, dict[str, Any]] = {}

        def _rank_index(lst: list[str]) -> dict[str, int]:
            return {fid: i + 1 for i, fid in enumerate(lst)}

        bm25_rank_index = _rank_index(bm25_ranked_fact_ids)
        bfs_rank_index = _rank_index(bfs_ranked_fact_ids)

        for fid in fact_by_id.keys() | rrf_scores.keys():
            base = float(rrf_scores.get(fid, 0.0))
            info = fact_by_id.get(fid) or {}
            edge_distance = None
            if fid in bfs_fact_candidates:
                edge_distance = int(bfs_fact_candidates[fid].get("distance") or 1)
            distance_boost = 0.0
            if edge_distance is not None:
                distance_boost = 1.0 / (1.0 + float(edge_distance))

            mention_boost = 0.0
            if mention_entity_ids and (
                (info.get("source") in mention_entity_ids) or (info.get("target") in mention_entity_ids)
            ):
                mention_boost = 0.5

            final = base + distance_boost + mention_boost
            final_scores[fid] = final
            provenance_by_fact[fid] = {
                "fact_id": fid,
                "rrf": base,
                "bm25_rank": bm25_rank_index.get(fid),
                "bfs_rank": bfs_rank_index.get(fid),
                "mentions": bool(mention_boost),
                "distance": edge_distance,
            }

        ranked_fact_ids = sorted(final_scores.keys(), key=lambda fid: final_scores[fid], reverse=True)
        ranked_fact_ids = ranked_fact_ids[:top_k]

        # Ensure we have entity names for formatting
        needed_entity_ids: set[str] = set()
        for fid in ranked_fact_ids:
            row = fact_by_id.get(fid) or {}
            for endpoint in [row.get("source"), row.get("target")]:
                endpoint = (endpoint or "").strip()
                if endpoint and endpoint not in entity_cache:
                    needed_entity_ids.add(endpoint)

        if needed_entity_ids:
            try:
                extra_nodes = self.store.get_entities_by_ids(entity_ids=sorted(needed_entity_ids))
            except Exception:
                extra_nodes = []
            for n in extra_nodes:
                nid = (n.get("id") or "").strip()
                if nid:
                    entity_cache[nid] = n

        def _entity_name(eid: str) -> str:
            n = entity_cache.get(eid) or {}
            return str(n.get("name") or eid)

        # Constructor context
        lines: list[str] = []
        lines.append("[GraphitiRetrievedFacts]")
        for fid in ranked_fact_ids:
            row = fact_by_id.get(fid) or {}
            src = str(row.get("source") or "").strip()
            tgt = str(row.get("target") or "").strip()
            rel = str(row.get("relation") or "").strip()
            fact = str(row.get("fact") or "").strip()
            evidence = str(row.get("evidence") or "").strip()
            t_from = str(row.get("t_valid_from") or "").strip()
            t_to = str(row.get("t_valid_to") or "").strip()
            time_span = ""
            if t_from or t_to:
                time_span = f" [{t_from or '?'} → {t_to or '?'}]"
            lines.append(
                f"- { _entity_name(src) } --{rel}--> { _entity_name(tgt) }{time_span}: {fact}".strip()
            )
            if evidence:
                lines.append(f"  evidence: {evidence}")

        if communities:
            lines.append("\n[GraphitiCommunities]")
            for c in communities[:5]:
                name = str(c.get("name") or c.get("community_id") or "").strip()
                summary = str(c.get("summary") or "").strip()
                if summary:
                    lines.append(f"- {name}: {summary}")
                else:
                    lines.append(f"- {name}")

        # Light entity summary section (anchors only)
        if anchor_entity_ids:
            lines.append("\n[GraphitiAnchorEntities]")
            for eid in anchor_entity_ids[:5]:
                n = entity_cache.get(eid) or {}
                name = str(n.get("name") or eid)
                summary = str((n.get("properties") or {}).get("summary") or n.get("summary") or "").strip()
                if summary:
                    lines.append(f"- {name}: {summary}")
                else:
                    lines.append(f"- {name}")

        context = "\n".join(lines).strip() + "\n"

        provenance = [provenance_by_fact[fid] for fid in ranked_fact_ids if fid in provenance_by_fact]
        return GraphitiSearchResult(context=context, provenance=provenance)

    # ──────────────────────────────────────
    # Community build/refresh (PR5)
    # ──────────────────────────────────────

    def refresh_communities_for_session(self, *, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        if not str(session_id or "").strip():
            return []
        if not hasattr(self.store, "list_recent_entity_ids_for_session"):
            return []
        try:
            entity_ids = self.store.list_recent_entity_ids_for_session(session_id=session_id, limit=limit)
        except Exception:
            return []
        if not entity_ids:
            return []
        # refresh based on a local subgraph around recent entities
        try:
            sub = self.store.export_semantic_subgraph(entity_ids=entity_ids[:10], edge_limit=800)
        except Exception:
            sub = {"nodes": [], "edges": []}
        entity_cache = {str(n.get("id") or "").strip(): n for n in (sub.get("nodes") or []) if str(n.get("id") or "").strip()}
        edges = list(sub.get("edges") or [])
        return self.build_or_refresh_communities_for_entities(
            entity_ids[:10],
            entity_cache=entity_cache,
            edges=edges,
        )

    def build_or_refresh_communities_for_entities(
        self,
        entity_ids: list[str],
        *,
        entity_cache: dict[str, dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
        min_size: int = 2,
        max_iter: int = 10,
    ) -> list[dict[str, Any]]:
        """在局部子图上做社区发现并写入 Neo4j。

        - 使用确定性的 label propagation（按节点 id 排序 + ties 取最小 label）
        - 社区 embedding：成员实体 embedding 的均值（若可取到）
        """

        entity_ids = [str(e).strip() for e in (entity_ids or []) if str(e).strip()]
        if not entity_ids:
            return []

        if entity_cache is None or edges is None:
            sub = self.store.export_semantic_subgraph(entity_ids=entity_ids[:10], edge_limit=800)
            entity_cache = {str(n.get("id") or "").strip(): n for n in (sub.get("nodes") or []) if str(n.get("id") or "").strip()}
            edges = list(sub.get("edges") or [])

        # Build adjacency
        adjacency: dict[str, set[str]] = {}
        nodes: set[str] = set(entity_cache.keys()) | set(entity_ids)
        for n in nodes:
            adjacency.setdefault(n, set())

        for e in edges or []:
            s = str(e.get("source") or "").strip()
            t = str(e.get("target") or "").strip()
            if not s or not t:
                continue
            nodes.add(s)
            nodes.add(t)
            adjacency.setdefault(s, set()).add(t)
            adjacency.setdefault(t, set()).add(s)

        if len(nodes) < min_size:
            return []

        labels: dict[str, str] = {n: n for n in nodes}
        ordered_nodes = sorted(nodes)
        for _ in range(max_iter):
            changed = 0
            for n in ordered_nodes:
                neigh = adjacency.get(n) or set()
                if not neigh:
                    continue
                counts = Counter(labels[m] for m in neigh if m in labels)
                if not counts:
                    continue
                max_freq = max(counts.values())
                candidates = sorted([lab for lab, c in counts.items() if c == max_freq])
                new_label = candidates[0]
                if labels[n] != new_label:
                    labels[n] = new_label
                    changed += 1
            if changed == 0:
                break

        groups: dict[str, list[str]] = {}
        for n, lab in labels.items():
            groups.setdefault(lab, []).append(n)

        # Prepare embeddings
        member_ids = sorted({m for members in groups.values() for m in members})
        embeddings_map: dict[str, list[float]] = {}
        if hasattr(self.store, "get_entity_embeddings_by_ids"):
            try:
                rows = self.store.get_entity_embeddings_by_ids(entity_ids=member_ids)
            except Exception:
                rows = []
            for r in rows:
                eid = str(r.get("entity_id") or "").strip()
                emb = r.get("embedding")
                if eid and emb:
                    embeddings_map[eid] = emb

        def _name_for_entity(eid: str) -> str:
            n = (entity_cache or {}).get(eid) or {}
            return str(n.get("name") or eid)

        def _avg_embedding(eids: list[str]) -> list[float] | None:
            vecs = [embeddings_map[e] for e in eids if e in embeddings_map]
            if not vecs:
                return None
            dim = min(len(v) for v in vecs)
            if dim <= 0:
                return None
            out = [0.0] * dim
            for v in vecs:
                for i in range(dim):
                    out[i] += float(v[i])
            return [x / float(len(vecs)) for x in out]

        # Persist communities
        created: list[dict[str, Any]] = []
        for _, members in groups.items():
            members = sorted([m for m in members if m in nodes])
            if len(members) < min_size:
                continue

            raw = "|".join(members).encode("utf-8")
            community_id = "comm_" + hashlib.sha1(raw).hexdigest()

            # Name: top degree entities
            deg_sorted = sorted(members, key=lambda x: len(adjacency.get(x) or set()), reverse=True)
            top_names = [
                _name_for_entity(eid)
                for eid in deg_sorted[:3]
                if _name_for_entity(eid)
            ]
            name = " / ".join(top_names) if top_names else community_id

            # Summary: key facts inside the community
            facts_in = []
            member_set = set(members)
            for e in edges or []:
                s = str(e.get("source") or "").strip()
                t = str(e.get("target") or "").strip()
                if s in member_set and t in member_set:
                    ft = str(e.get("fact") or "").strip()
                    if ft:
                        facts_in.append(ft)
            facts_in = facts_in[:5]
            summary = "；".join(facts_in)

            embedding = _avg_embedding(members)
            try:
                if hasattr(self.store, "upsert_community"):
                    self.store.upsert_community(
                        community_id=community_id,
                        name=name,
                        summary=summary,
                        embedding=embedding,
                    )
                if hasattr(self.store, "link_entity_to_community"):
                    for eid in members:
                        self.store.link_entity_to_community(entity_id=eid, community_id=community_id)
            except Exception:
                # best-effort persistence
                pass

            created.append(
                {
                    "community_id": community_id,
                    "name": name,
                    "summary": summary,
                    "score": 1.0,
                    "size": len(members),
                }
            )

        created.sort(key=lambda x: int(x.get("size") or 0), reverse=True)
        return created

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(n):
            av = float(a[i])
            bv = float(b[i])
            dot += av * bv
            na += av * av
            nb += bv * bv
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / ((na**0.5) * (nb**0.5))

    @staticmethod
    def _rrf_scores(rankings: dict[str, list[str]], *, k: int = 60) -> dict[str, float]:
        scores: dict[str, float] = {}
        for _, lst in rankings.items():
            for idx, key in enumerate(lst):
                r = idx + 1
                scores[key] = scores.get(key, 0.0) + 1.0 / float(k + r)
        return scores

    def export_semantic_graph(self) -> dict[str, list[dict[str, Any]]]:
        """为 API/web-demo 提供兼容的语义图视图导出。"""
        return self.store.export_semantic_graph()
