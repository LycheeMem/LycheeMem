"""Graphiti(论文) 风格图谱引擎：Search/Rerank/Constructor 的对外门面。

PR1 目标：提供可注入的引擎骨架与兼容导出接口；真正的 ingest/search 能力
在后续 PR 分步实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import datetime
import hashlib
import json

from collections import Counter

from src.memory.graph.graphiti_neo4j_store import GraphitiNeo4jStore
from src.memory.graph.graphiti_prompts import (
    COMMUNITY_SUMMARY_MAP_SYSTEM_PROMPT,
    COMMUNITY_SUMMARY_REDUCE_SYSTEM_PROMPT,
)


@dataclass(slots=True)
class GraphitiSearchResult:
    context: str
    provenance: list[dict[str, Any]]


class GraphitiEngine:
    """Graphiti 引擎（面向论文的 f(α)=χ(ρ(φ(α)))）。"""

    def __init__(
        self,
        store: GraphitiNeo4jStore,
        *,
        strict: bool = False,
        community_llm: Any | None = None,
        embedder: Any | None = None,
        gds_distance_max_depth: int = 4,
        cross_encoder: Any | None = None,
        cross_encoder_top_n: int = 20,
        cross_encoder_weight: float = 1.0,
        mmr_lambda: float = 0.5,
        bfs_recent_episode_limit: int = 4,
    ):
        self.store = store
        self.strict = bool(strict)
        self.community_llm = community_llm
        self.embedder = embedder
        self.gds_distance_max_depth = max(1, int(gds_distance_max_depth))
        self.cross_encoder = cross_encoder
        self.cross_encoder_top_n = max(1, int(cross_encoder_top_n))
        self.cross_encoder_weight = float(cross_encoder_weight)
        self.mmr_lambda = max(0.0, min(1.0, float(mmr_lambda)))
        # Paper §3.1: configurable number of recent episodes used as BFS seeds.
        self.bfs_recent_episode_limit = max(1, int(bfs_recent_episode_limit))

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
        episode_type: str = "message",
    ) -> str:
        """写入一个 Episode（幂等）。

        Paper §2.1: Episodes can be one of three core types: message, text, or JSON.
        Each type requires specific handling during graph construction.
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
            episode_type=str(episode_type or "message"),
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
            if self.strict:
                bm25_facts = self.store.fulltext_search_facts(query=query, limit=max(30, top_k * 6))
            else:
                try:
                    bm25_facts = self.store.fulltext_search_facts(
                        query=query, limit=max(30, top_k * 6)
                    )
                except Exception:
                    bm25_facts = []

        # ──────────────────────────────────────
        # Channel 2: vector similarity over entities (Neo4j native vector index)
        # ──────────────────────────────────────
        vector_entities: list[dict[str, Any]] = []
        if query_embedding is not None:
            if hasattr(self.store, "vector_search_entities"):
                if self.strict:
                    vector_entities = self.store.vector_search_entities(
                        query_embedding=query_embedding,
                        limit=max(20, top_k * 4),
                    )
                else:
                    try:
                        vector_entities = self.store.vector_search_entities(
                            query_embedding=query_embedding,
                            limit=max(20, top_k * 4),
                        )
                    except Exception:
                        vector_entities = []
            else:
                if self.strict:
                    raise RuntimeError(
                        "Graphiti strict search requires store.vector_search_entities()"
                    )
                # Non-strict fallback: keep legacy scan+cosine behavior if available.
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
                vector_entities = scored[: max(20, top_k * 4)]

        # ──────────────────────────────────────
        # Anchor entities for BFS expansion (mentions + high-scoring)
        # ──────────────────────────────────────
        recent_episode_entity_ids: list[str] = []
        if session_id is not None and str(session_id).strip():
            if not hasattr(self.store, "list_recent_entity_ids_from_episodes"):
                if self.strict:
                    raise RuntimeError(
                        "Graphiti strict search requires store.list_recent_entity_ids_from_episodes()"
                    )
            else:
                try:
                    recent_episode_entity_ids = self.store.list_recent_entity_ids_from_episodes(
                        session_id=str(session_id),
                        episode_limit=self.bfs_recent_episode_limit,
                        entity_limit=20,
                    )
                except Exception as e:
                    if self.strict:
                        raise RuntimeError(
                            f"Graphiti strict recent-episode seeds failed: {e}"
                        ) from e
                    recent_episode_entity_ids = []

        ft_entities: list[dict[str, Any]] = []
        if self.strict:
            ft_entities = self.store.fulltext_search_entities(query=query, limit=10)
        else:
            try:
                ft_entities = self.store.fulltext_search_entities(query=query, limit=10)
            except Exception:
                ft_entities = []

        anchor_entity_ids: list[str] = []
        for row in ft_entities[:5] + vector_entities[:5]:
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

        bfs_seed_entity_ids: list[str] = []
        for eid in (recent_episode_entity_ids or []) + (anchor_entity_ids or []):
            eid = str(eid or "").strip()
            if eid and eid not in bfs_seed_entity_ids:
                bfs_seed_entity_ids.append(eid)

        frontier = list(bfs_seed_entity_ids)
        visited_entities = set(frontier)
        max_depth = 2
        for depth in range(max_depth):
            if not frontier:
                break
            if self.strict:
                subgraph = self.store.export_semantic_subgraph(entity_ids=frontier, edge_limit=600)
            else:
                try:
                    subgraph = self.store.export_semantic_subgraph(
                        entity_ids=frontier, edge_limit=600
                    )
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
                fact_id = (
                    (e.get("fact_id") or "").strip()
                    or f"fact:{e.get('source', '')}:{e.get('relation', '')}:{e.get('target', '')}:{e.get('timestamp', '')}"
                )
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
        # Community retrieval (optional)
        # ──────────────────────────────────────
        communities: list[dict[str, Any]] = []
        if include_communities:
            # 1) query-time retrieval
            if hasattr(self.store, "fulltext_search_communities"):
                if self.strict:
                    communities = self.store.fulltext_search_communities(query=query, limit=5)
                else:
                    try:
                        communities = self.store.fulltext_search_communities(query=query, limit=5)
                    except Exception:
                        communities = []

            if query_embedding is not None:
                if hasattr(self.store, "vector_search_communities"):
                    if self.strict:
                        communities.extend(
                            self.store.vector_search_communities(
                                query_embedding=query_embedding,
                                limit=5,
                            )
                        )
                    else:
                        try:
                            communities.extend(
                                self.store.vector_search_communities(
                                    query_embedding=query_embedding,
                                    limit=5,
                                )
                            )
                        except Exception:
                            pass
                elif hasattr(self.store, "scan_communities_with_embeddings"):
                    if self.strict:
                        raise RuntimeError(
                            "Graphiti strict search requires store.vector_search_communities()"
                        )
                    # Non-strict fallback: keep legacy scan+cosine behavior if available.
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

            # 1.5) dynamic extension: expand via shared entities to fill the pool
            if communities and hasattr(self.store, "expand_communities_via_entities"):
                existing_ids = [str(c.get("community_id") or "").strip() for c in communities]
                existing_ids = [cid for cid in existing_ids if cid]
                unique_existing = list(dict.fromkeys(existing_ids))
                need = max(0, 5 - len(unique_existing))
                if need > 0 and unique_existing:
                    if self.strict:
                        extra = self.store.expand_communities_via_entities(
                            community_ids=unique_existing,
                            limit=need,
                        )
                    else:
                        try:
                            extra = self.store.expand_communities_via_entities(
                                community_ids=unique_existing,
                                limit=need,
                            )
                        except Exception:
                            extra = []
                    if extra:
                        communities.extend(list(extra))

        # ──────────────────────────────────────
        # Merge facts + rerank
        # ──────────────────────────────────────

        def _extract_fact_id(row: dict[str, Any]) -> str:
            fid = (row.get("fact_id") or "").strip()
            if fid:
                return fid
            # fallback stable-ish id
            return f"fact:{row.get('source', '')}:{row.get('relation', '')}:{row.get('target', '')}:{row.get('timestamp', '')}"

        fact_by_id: dict[str, dict[str, Any]] = {}

        bm25_ranked_fact_ids: list[str] = []
        for row in bm25_facts:
            fid = _extract_fact_id(row)
            bm25_ranked_fact_ids.append(fid)
            fact_by_id.setdefault(fid, {}).update(row)

        # Optional vector recall over facts (Neo4j native vector index)
        vector_facts: list[dict[str, Any]] = []
        vector_ranked_fact_ids: list[str] = []
        if query_embedding is not None:
            if hasattr(self.store, "vector_search_facts"):
                if self.strict:
                    vector_facts = self.store.vector_search_facts(
                        query_embedding=query_embedding,
                        limit=max(30, top_k * 6),
                    )
                else:
                    try:
                        vector_facts = self.store.vector_search_facts(
                            query_embedding=query_embedding,
                            limit=max(30, top_k * 6),
                        )
                    except Exception:
                        vector_facts = []
            elif self.strict:
                raise RuntimeError("Graphiti strict search requires store.vector_search_facts()")

        for row in vector_facts:
            fid = _extract_fact_id(row)
            vector_ranked_fact_ids.append(fid)
            fact_by_id.setdefault(fid, {}).update(row)

        # Add BFS edges as fact candidates too
        for fid, info in bfs_fact_candidates.items():
            edge = info.get("edge") or {}
            fact_by_id.setdefault(fid, {}).update(edge)
            fact_by_id[fid].setdefault("fact_id", fid)

        # Mention boost: mention frequency across the entire session
        session_id = str(session_id).strip() if session_id is not None else None
        entity_mentions: dict[str, int] = {}
        fact_mentions: dict[str, int] = {}
        if session_id:
            if not hasattr(self.store, "count_mentions_in_session"):
                if self.strict:
                    raise RuntimeError(
                        "Graphiti strict mentions rerank requires store.count_mentions_in_session()"
                    )
            else:
                candidate_entity_ids: set[str] = set(anchor_entity_ids)
                for row in fact_by_id.values():
                    for endpoint in [row.get("source"), row.get("target")]:
                        endpoint = (endpoint or "").strip()
                        if endpoint:
                            candidate_entity_ids.add(endpoint)

                try:
                    counts = self.store.count_mentions_in_session(
                        session_id=session_id,
                        entity_ids=sorted(candidate_entity_ids),
                        fact_ids=sorted(fact_by_id.keys()),
                    )
                except Exception as e:
                    if self.strict:
                        raise RuntimeError(f"Graphiti strict mentions count failed: {e}") from e
                    counts = {}

                if isinstance(counts, dict):
                    ent = counts.get("entities")
                    fac = counts.get("facts")
                    if isinstance(ent, dict):
                        entity_mentions = {str(k): int(v or 0) for k, v in ent.items()}
                    if isinstance(fac, dict):
                        fact_mentions = {str(k): int(v or 0) for k, v in fac.items()}

        # RRF scores
        rrf_scores: dict[str, float] = {}
        rrf_scores = self._rrf_scores(
            {
                "bm25": bm25_ranked_fact_ids,
                "vector": vector_ranked_fact_ids,
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

        gds_entity_distances: dict[str, int] = {}

        # If strict, now that we have fact candidates, expand the entity set and recompute distances.
        if self.strict and anchor_entity_ids:
            if not hasattr(self.store, "gds_min_distances_from_anchors"):
                raise RuntimeError(
                    "Graphiti strict search requires store.gds_min_distances_from_anchors()"
                )
            candidate_entity_ids: set[str] = set(anchor_entity_ids)
            for row in fact_by_id.values():
                for endpoint in [row.get("source"), row.get("target")]:
                    endpoint = (endpoint or "").strip()
                    if endpoint:
                        candidate_entity_ids.add(endpoint)

            try:
                gds_entity_distances = self.store.gds_min_distances_from_anchors(
                    anchor_entity_ids=anchor_entity_ids,
                    entity_ids=sorted(candidate_entity_ids),
                    max_depth=self.gds_distance_max_depth,
                )
            except Exception as e:
                raise RuntimeError(f"Graphiti strict GDS distance failed: {e}") from e

        # Precompute per-fact mention counts for normalization.
        mention_count_by_fact: dict[str, int] = {}
        for fid in fact_by_id.keys() | rrf_scores.keys():
            info = fact_by_id.get(fid) or {}
            src = str(info.get("source") or "").strip()
            tgt = str(info.get("target") or "").strip()
            mention_count_by_fact[fid] = max(
                int(fact_mentions.get(fid, 0) or 0),
                int(entity_mentions.get(src, 0) or 0),
                int(entity_mentions.get(tgt, 0) or 0),
            )

        max_mention = max(mention_count_by_fact.values(), default=0)

        for fid in fact_by_id.keys() | rrf_scores.keys():
            base = float(rrf_scores.get(fid, 0.0))
            info = fact_by_id.get(fid) or {}
            edge_distance = None
            if fid in bfs_fact_candidates:
                edge_distance = int(bfs_fact_candidates[fid].get("distance") or 1)

            # Graph distance boost
            graph_distance = None
            if self.strict and anchor_entity_ids:
                src = str(info.get("source") or "").strip()
                tgt = str(info.get("target") or "").strip()
                ds = gds_entity_distances.get(src)
                dt = gds_entity_distances.get(tgt)
                candidates = [d for d in [ds, dt] if d is not None]
                if candidates:
                    graph_distance = int(min(candidates))
            else:
                graph_distance = edge_distance

            distance_boost = 0.0
            if graph_distance is not None:
                distance_boost = 1.0 / (1.0 + float(graph_distance))

            mention_count = int(mention_count_by_fact.get(fid, 0) or 0)
            mention_boost = 0.0
            if max_mention > 0 and mention_count > 0:
                # Normalize to [0, 0.5] so it remains a boost term.
                mention_boost = 0.5 * (float(mention_count) / float(max_mention))

            final = base + distance_boost + mention_boost
            final_scores[fid] = final
            _fact_text = str(info.get("fact") or info.get("fact_text") or "").strip()
            provenance_by_fact[fid] = {
                "fact_id": fid,
                "rrf": base,
                "bm25_rank": bm25_rank_index.get(fid),
                "bfs_rank": bfs_rank_index.get(fid),
                "mentions": mention_count,
                "distance": graph_distance,
                "bfs_distance": edge_distance,
                "gds_distance": graph_distance if self.strict else None,
                "fact_text": _fact_text,
                "subject_entity_id": str(info.get("source") or "").strip(),
                "object_entity_id": str(info.get("target") or "").strip(),
                "relation_type": str(info.get("relation") or "").strip(),
            }

        # ──────────────────────────────────────
        # Cross-encoder rerank (optional; paper-parity)
        # ──────────────────────────────────────
        if self.cross_encoder is not None and self.cross_encoder_weight != 0.0:
            # Use current score order as the candidate pool.
            pool = sorted(final_scores.keys(), key=lambda fid: final_scores[fid], reverse=True)
            pool = pool[: min(len(pool), self.cross_encoder_top_n)]

            passages: list[dict[str, str]] = []
            for fid in pool:
                row = fact_by_id.get(fid) or {}
                src = str(row.get("source") or "").strip()
                tgt = str(row.get("target") or "").strip()
                rel = str(row.get("relation") or "").strip()
                fact = str(row.get("fact") or "").strip()
                evidence = str(row.get("evidence") or "").strip()
                text = " ".join([p for p in [src, rel, tgt, fact, evidence] if p]).strip()
                if text:
                    passages.append({"id": fid, "text": text})

            if passages:
                if self.strict:
                    cross_scores = self.cross_encoder.score(query=query, passages=passages)
                else:
                    try:
                        cross_scores = self.cross_encoder.score(query=query, passages=passages)
                    except Exception:
                        cross_scores = {}

                for fid in pool:
                    s = float(cross_scores.get(fid, 0.0) or 0.0)
                    final_scores[fid] = float(final_scores.get(fid, 0.0)) + (
                        self.cross_encoder_weight * s
                    )
                    if fid in provenance_by_fact:
                        provenance_by_fact[fid]["cross_encoder_score"] = s

        ranked_fact_ids = sorted(
            final_scores.keys(), key=lambda fid: final_scores[fid], reverse=True
        )

        # ──────────────────────────────────────
        # MMR diversity rerank (paper §3.2)
        # ──────────────────────────────────────
        if query_embedding is not None and self.mmr_lambda < 1.0:
            # Collect fact embeddings for MMR
            fact_embedding_map: dict[str, list[float]] = {}
            for fid in ranked_fact_ids:
                row = fact_by_id.get(fid) or {}
                emb = row.get("embedding")
                if isinstance(emb, list) and emb:
                    fact_embedding_map[fid] = emb

            if fact_embedding_map:
                # Apply MMR on the top candidate pool (2x top_k to allow diversity selection)
                mmr_pool = ranked_fact_ids[: max(top_k * 2, 40)]
                ranked_fact_ids = self._mmr_rerank(
                    mmr_pool,
                    query_embedding=query_embedding,
                    embedding_map=fact_embedding_map,
                    lam=self.mmr_lambda,
                    top_k=top_k,
                )
            else:
                ranked_fact_ids = ranked_fact_ids[:top_k]
        else:
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

        # ──────────────────────────────────────
        # Paper §2.1: Reverse Episode index — bidirectional citation lookup.
        # "Semantic artifacts can be traced to their sources for citation or quotation,
        #  while episodes can quickly retrieve their relevant entities and facts."
        #
        # For every ranked Fact: look up which Episodes have an EVIDENCE_FOR edge to it.
        # For every anchor Entity: look up which Episodes have a MENTIONS edge to it.
        # Results are stored in provenance_by_fact[fid]["source_episodes"] (list of episode
        # dicts) and in a separate entity_source_episodes map, so the Constructor χ can
        # emit a <SOURCES> block and callers can provide full citation chains.
        # ──────────────────────────────────────

        # Fact → source Episodes
        fact_source_episodes: dict[str, list[dict[str, Any]]] = {}
        if hasattr(self.store, "get_source_episodes_for_fact"):
            for fid in ranked_fact_ids:
                # Only trace canonical (non-synthetic) fact IDs stored in Neo4j.
                # Synthetic BFS-only IDs (prefixed "fact:") are skip-safe.
                if fid.startswith("fact:"):
                    continue
                if self.strict:
                    episodes = self.store.get_source_episodes_for_fact(fact_id=fid, limit=10)
                else:
                    try:
                        episodes = self.store.get_source_episodes_for_fact(
                            fact_id=fid, limit=10
                        )
                    except Exception:
                        episodes = []
                if episodes:
                    fact_source_episodes[fid] = [dict(ep) for ep in episodes]
        elif self.strict:
            raise RuntimeError(
                "Graphiti strict search requires store.get_source_episodes_for_fact() "
                "for bidirectional citation (paper §2.1)"
            )

        # Entity → source Episodes (for anchor entities and fact endpoints)
        entity_source_episodes: dict[str, list[dict[str, Any]]] = {}
        if hasattr(self.store, "get_source_episodes_for_entity"):
            # Collect all entity IDs visible in the top-ranked results.
            entity_ids_to_trace: list[str] = []
            for eid in anchor_entity_ids:
                if eid and eid not in entity_ids_to_trace:
                    entity_ids_to_trace.append(eid)
            for fid in ranked_fact_ids:
                row = fact_by_id.get(fid) or {}
                for endpoint in [row.get("source"), row.get("target")]:
                    endpoint = (endpoint or "").strip()
                    if endpoint and endpoint not in entity_ids_to_trace:
                        entity_ids_to_trace.append(endpoint)

            for eid in entity_ids_to_trace:
                if self.strict:
                    ep_list = self.store.get_source_episodes_for_entity(
                        entity_id=eid, limit=5
                    )
                else:
                    try:
                        ep_list = self.store.get_source_episodes_for_entity(
                            entity_id=eid, limit=5
                        )
                    except Exception:
                        ep_list = []
                if ep_list:
                    entity_source_episodes[eid] = [dict(ep) for ep in ep_list]

        # Enrich provenance_by_fact with the fetched Episode citations.
        for fid in ranked_fact_ids:
            if fid not in provenance_by_fact:
                continue
            row = fact_by_id.get(fid) or {}
            src_eid = str(row.get("source") or "").strip()
            tgt_eid = str(row.get("target") or "").strip()

            # Source episodes from EVIDENCE_FOR (direct Fact citation)
            fact_eps = fact_source_episodes.get(fid, [])

            # Merge entity-originated episodes as fallback (MENTIONS)
            # so even facts not directly linked via EVIDENCE_FOR can surface
            # the Episodes that introduced the participating entities.
            entity_ep_src = entity_source_episodes.get(src_eid, [])
            entity_ep_tgt = entity_source_episodes.get(tgt_eid, [])

            # Deduplicate by episode_id across all three lists.
            seen_ep_ids: set[str] = set()
            merged_eps: list[dict[str, Any]] = []
            for ep in fact_eps + entity_ep_src + entity_ep_tgt:
                ep_id = str(ep.get("episode_id") or "").strip()
                if ep_id and ep_id not in seen_ep_ids:
                    seen_ep_ids.add(ep_id)
                    merged_eps.append(ep)

            # Sort by turn_index ascending so citations appear in conversation order.
            merged_eps.sort(key=lambda e: int(e.get("turn_index") or 0))

            provenance_by_fact[fid]["source_episodes"] = merged_eps
            provenance_by_fact[fid]["fact_id"] = fid

        # Constructor context (paper template)
        lines: list[str] = []
        lines.append("FACTS and ENTITIES represent relevant context to the current conversation.")
        lines.append(
            "These are the most relevant facts and their valid date ranges. If the fact is about an event, the event takes place during this time."
        )
        lines.append("format: FACT (Date range: from - to)")
        lines.append("<FACTS>")
        for fid in ranked_fact_ids:
            row = fact_by_id.get(fid) or {}
            fact_text = str(row.get("fact") or row.get("fact_text") or "").strip()
            if not fact_text:
                src = str(row.get("source") or "").strip()
                tgt = str(row.get("target") or "").strip()
                rel = str(row.get("relation") or "").strip()
                fact_text = f"{_entity_name(src)} --{rel}--> {_entity_name(tgt)}".strip()

            evidence = str(row.get("evidence") or row.get("evidence_text") or "").strip()
            t_from = str(row.get("t_valid_from") or "").strip() or "null"
            t_to = str(row.get("t_valid_to") or "").strip() or "null"
            lines.append(f"- {fact_text} (Date range: {t_from} - {t_to})")
            if evidence:
                lines.append(f"  evidence: {evidence}")
        lines.append("</FACTS>")

        # Entities: include anchors + endpoints + top retrieved entities
        relevant_entity_ids: list[str] = []
        for eid in anchor_entity_ids:
            if eid and eid not in relevant_entity_ids:
                relevant_entity_ids.append(eid)

        for fid in ranked_fact_ids:
            row = fact_by_id.get(fid) or {}
            for endpoint in [row.get("source"), row.get("target")]:
                endpoint = (endpoint or "").strip()
                if endpoint and endpoint not in relevant_entity_ids:
                    relevant_entity_ids.append(endpoint)

        for ent in ft_entities[:10] + vector_entities[:10]:
            eid = (ent.get("entity_id") or ent.get("id") or "").strip()
            if eid and eid not in relevant_entity_ids:
                relevant_entity_ids.append(eid)

        lines.append("These are the most relevant entities")
        lines.append("ENTITY_NAME: entity summary")
        lines.append("<ENTITIES>")
        for eid in relevant_entity_ids[:15]:
            n = entity_cache.get(eid) or {}
            name = str(n.get("name") or eid).strip() or eid
            summary = str(
                (n.get("properties") or {}).get("summary") or n.get("summary") or ""
            ).strip()
            if summary:
                lines.append(f"- {name}: {summary}")
            else:
                lines.append(f"- {name}")
        lines.append("</ENTITIES>")

        # Communities (constructor definition in paper includes community summaries)
        if communities:
            lines.append("<COMMUNITIES>")
            for c in communities[:5]:
                name = str(c.get("name") or c.get("community_id") or "").strip()
                summary = str(c.get("summary") or "").strip()
                if summary:
                    lines.append(f"- {name}: {summary}")
                else:
                    lines.append(f"- {name}")
            lines.append("</COMMUNITIES>")

        # Paper §2.1: Sources block — bidirectional citation from semantic artifacts
        # back to raw Episode data.  Each ranked Fact is annotated with the turn(s)
        # that originally introduced it (via EVIDENCE_FOR) and the turns that first
        # mentioned its participating entities (via MENTIONS).  This lets the LLM
        # agent cite or quote source material rather than rely on its own paraphrase.
        has_citations = any(
            provenance_by_fact.get(fid, {}).get("source_episodes")
            for fid in ranked_fact_ids
        )
        if has_citations:
            lines.append(
                "The following shows the original conversation turns each fact was extracted from."
            )
            lines.append("format: FACT_TEXT [turn N (role): excerpt]")
            lines.append("<SOURCES>")
            for fid in ranked_fact_ids:
                prov = provenance_by_fact.get(fid) or {}
                source_eps: list[dict[str, Any]] = prov.get("source_episodes") or []
                if not source_eps:
                    continue
                row = fact_by_id.get(fid) or {}
                fact_text = str(row.get("fact") or row.get("fact_text") or "").strip()
                if not fact_text:
                    src_eid = str(row.get("source") or "").strip()
                    tgt_eid = str(row.get("target") or "").strip()
                    rel = str(row.get("relation") or "").strip()
                    fact_text = (
                        f"{_entity_name(src_eid)} --{rel}--> {_entity_name(tgt_eid)}"
                    ).strip()

                # Render source citations: show at most 3 Episodes per Fact to
                # avoid bloating the context window.
                citation_parts: list[str] = []
                for ep in source_eps[:3]:
                    turn_idx = ep.get("turn_index")
                    role = str(ep.get("role") or "").strip()
                    content = str(ep.get("content") or "").strip()
                    t_ref = str(ep.get("t_ref") or "").strip()

                    # Trim content to a representative excerpt (first 120 chars).
                    excerpt = content[:120].replace("\n", " ").strip()
                    if len(content) > 120:
                        excerpt += "…"

                    if turn_idx is not None:
                        label = f"turn {turn_idx}"
                        if role:
                            label += f" ({role})"
                        if t_ref:
                            label += f" @ {t_ref}"
                    else:
                        label = role or "unknown"

                    citation_parts.append(f"[{label}: {excerpt}]")

                if citation_parts:
                    lines.append(f"- {fact_text} " + " ".join(citation_parts))
            lines.append("</SOURCES>")

        context = "\n".join(lines).strip() + "\n"

        provenance = [
            provenance_by_fact[fid] for fid in ranked_fact_ids if fid in provenance_by_fact
        ]
        return GraphitiSearchResult(context=context, provenance=provenance)

    @staticmethod
    def _safe_json_loads(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    def _community_map_reduce(self, *, facts: list[str]) -> dict[str, str]:
        if self.community_llm is None:
            if self.strict:
                raise RuntimeError("Graphiti strict community build requires community_llm")
            return {"name": "", "summary": ""}

        facts = [str(f).strip() for f in (facts or []) if str(f).strip()]
        if not facts:
            return {"name": "", "summary": ""}

        chunk_size = 10
        partials: list[str] = []
        for i in range(0, len(facts), chunk_size):
            chunk = facts[i : i + chunk_size]
            prompt = COMMUNITY_SUMMARY_MAP_SYSTEM_PROMPT.format(
                facts="\n".join(f"- {x}" for x in chunk)
            )
            raw = self.community_llm.generate(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Return JSON only."},
                ],
            )
            data = self._safe_json_loads(raw)
            if isinstance(data, dict) and str(data.get("summary") or "").strip():
                partials.append(str(data.get("summary") or "").strip())

        reduce_prompt = COMMUNITY_SUMMARY_REDUCE_SYSTEM_PROMPT.format(
            partial_summaries="\n".join(f"- {s}" for s in partials) if partials else ""
        )
        raw = self.community_llm.generate(
            [
                {"role": "system", "content": reduce_prompt},
                {"role": "user", "content": "Return JSON only."},
            ],
        )
        data = self._safe_json_loads(raw)
        if not isinstance(data, dict):
            return {"name": "", "summary": ""}
        name = str(data.get("name") or "").strip()
        summary = str(data.get("summary") or "").strip()
        return {"name": name, "summary": summary}

    # ──────────────────────────────────────
    # Community build/refresh (PR5)
    # ──────────────────────────────────────

    def refresh_communities_for_session(
        self, *, session_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        if not str(session_id or "").strip():
            return []
        if not hasattr(self.store, "list_recent_entity_ids_for_session"):
            return []
        try:
            entity_ids = self.store.list_recent_entity_ids_for_session(
                session_id=session_id, limit=limit
            )
        except Exception:
            return []
        if not entity_ids:
            return []
        # refresh based on a local subgraph around recent entities
        try:
            sub = self.store.export_semantic_subgraph(entity_ids=entity_ids[:10], edge_limit=800)
        except Exception:
            sub = {"nodes": [], "edges": []}
        entity_cache = {
            str(n.get("id") or "").strip(): n
            for n in (sub.get("nodes") or [])
            if str(n.get("id") or "").strip()
        }
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

        if self.strict and self.community_llm is None:
            raise RuntimeError("Graphiti strict community build requires community_llm")

        if entity_cache is None or edges is None:
            sub = self.store.export_semantic_subgraph(entity_ids=entity_ids[:10], edge_limit=800)
            entity_cache = {
                str(n.get("id") or "").strip(): n
                for n in (sub.get("nodes") or [])
                if str(n.get("id") or "").strip()
            }
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

        groups: dict[str, list[str]] = {}
        if self.strict:
            if not hasattr(self.store, "gds_label_propagation_groups"):
                raise RuntimeError(
                    "Graphiti strict community build requires store.gds_label_propagation_groups()"
                )
            try:
                groups = self.store.gds_label_propagation_groups(
                    entity_ids=sorted(nodes),
                    max_iterations=max_iter,
                )
            except Exception as e:
                raise RuntimeError(f"Graphiti strict GDS community detection failed: {e}") from e
        else:
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
            top_names = [_name_for_entity(eid) for eid in deg_sorted[:3] if _name_for_entity(eid)]
            name = " / ".join(top_names) if top_names else community_id

            # Summary: facts inside the community (map-reduce)
            facts_in: list[str] = []
            member_set = set(members)
            for e in edges or []:
                s = str(e.get("source") or "").strip()
                t = str(e.get("target") or "").strip()
                if s in member_set and t in member_set:
                    ft = str(e.get("fact") or "").strip()
                    if ft:
                        facts_in.append(ft)

            summary = "；".join(facts_in[:5])
            if facts_in:
                try:
                    mr = self._community_map_reduce(facts=facts_in[:50])
                    mr_name = str(mr.get("name") or "").strip()
                    mr_summary = str(mr.get("summary") or "").strip()
                    if mr_summary:
                        summary = mr_summary
                    if mr_name:
                        name = mr_name
                except Exception:
                    if self.strict:
                        raise

            embedding = _avg_embedding(members)
            # Paper §2.3: embed the community *name* text, not member-entity average.
            if self.embedder is not None and name:
                try:
                    embedding = self.embedder.embed_query(name)
                except Exception:
                    if self.strict:
                        raise
                    # Fall back to avg member embedding if embed() fails
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
                        self.store.link_entity_to_community(
                            entity_id=eid, community_id=community_id
                        )
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

    def refresh_all_communities(
        self,
        *,
        min_size: int = 2,
        max_iter: int = 10,
    ) -> list[dict[str, Any]]:
        """Paper §2.3: periodic full-graph community refresh.

        Runs label propagation over all entities, rebuilds community
        summaries via map-reduce, and re-embeds community names.
        Should be called periodically (e.g. via a background cron job)
        to ensure full graph coverage beyond the incremental
        ingestion-time assignments.
        """
        # Gather the full entity set
        if not hasattr(self.store, "list_all_entity_ids"):
            if self.strict:
                raise RuntimeError(
                    "Graphiti strict periodic community refresh requires "
                    "store.list_all_entity_ids()"
                )
            return []

        try:
            all_entity_ids = self.store.list_all_entity_ids()
        except Exception as e:
            if self.strict:
                raise RuntimeError(
                    f"Graphiti strict periodic community refresh failed: {e}"
                ) from e
            return []

        if not all_entity_ids or len(all_entity_ids) < min_size:
            return []

        # Export the full semantic subgraph
        try:
            sub = self.store.export_semantic_subgraph(
                entity_ids=all_entity_ids,
                edge_limit=10000,
            )
        except Exception as e:
            if self.strict:
                raise RuntimeError(
                    f"Graphiti strict periodic community refresh subgraph export failed: {e}"
                ) from e
            return []

        entity_cache = {
            str(n.get("id") or "").strip(): n
            for n in (sub.get("nodes") or [])
            if str(n.get("id") or "").strip()
        }
        edges = list(sub.get("edges") or [])

        return self.build_or_refresh_communities_for_entities(
            all_entity_ids,
            entity_cache=entity_cache,
            edges=edges,
            min_size=min_size,
            max_iter=max_iter,
        )

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

    @staticmethod
    def _mmr_rerank(
        candidates: list[str],
        *,
        query_embedding: list[float],
        embedding_map: dict[str, list[float]],
        lam: float = 0.5,
        top_k: int = 10,
        cosine_fn: Any = None,
    ) -> list[str]:
        """Maximal Marginal Relevance (paper \u00a73.2).

        MMR(d) = \u03bb \u00b7 sim(d, q) - (1-\u03bb) \u00b7 max_{d_i \u2208 S} sim(d, d_i)

        Selects results that are both relevant to the query and diverse.
        """
        if not candidates or not query_embedding:
            return candidates[:top_k]

        if cosine_fn is None:
            cosine_fn = GraphitiEngine._cosine_similarity

        # Precompute query similarities
        q_sim: dict[str, float] = {}
        for fid in candidates:
            emb = embedding_map.get(fid)
            q_sim[fid] = cosine_fn(query_embedding, emb) if emb else 0.0

        selected: list[str] = []
        remaining = set(candidates)

        for _ in range(min(top_k, len(candidates))):
            best_fid = None
            best_score = float("-inf")
            for fid in remaining:
                relevance = q_sim.get(fid, 0.0)
                redundancy = 0.0
                emb = embedding_map.get(fid)
                if emb and selected:
                    redundancy = max(
                        cosine_fn(emb, embedding_map[s])
                        for s in selected
                        if s in embedding_map
                    ) if any(s in embedding_map for s in selected) else 0.0
                score = lam * relevance - (1.0 - lam) * redundancy
                if score > best_score:
                    best_score = score
                    best_fid = fid
            if best_fid is None:
                break
            selected.append(best_fid)
            remaining.discard(best_fid)

        return selected

    def export_semantic_graph(self) -> dict[str, list[dict[str, Any]]]:
        """为 API/web-demo 提供兼容的语义图视图导出。"""
        return self.store.export_semantic_graph()
