"""Compact Semantic Engine（总装引擎）。

实现 BaseSemanticMemoryEngine，串联所有子模块：
- search(): Planner → 5 通道召回 → Scorer → 格式化输出
- ingest_conversation(): Novelty Check → Encoder → 去重写入 → Fusion

这是整个 Compact Semantic Memory 的核心入口。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from src.embedder.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.memory.semantic.base import (
    BaseSemanticMemoryEngine,
    ConsolidationResult,
    SemanticSearchResult,
)
from src.memory.semantic.encoder import TypedMemoryEncoder
from src.memory.semantic.models import MemoryRecord, SearchPlan, UsageLog
from src.memory.semantic.planner import TaskAwareRetrievalPlanner
from src.memory.semantic.prompts import (
    NOVELTY_CHECK_SYSTEM,
    RETRIEVAL_ADEQUACY_CHECK_SYSTEM,
    RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM,
)
from src.memory.semantic.scorer import MemoryScorer, ScoredCandidate, ScoringWeights
from src.memory.semantic.sqlite_store import SQLiteSemanticStore
from src.memory.semantic.synthesizer import RecordFusionEngine
from src.memory.semantic.vector_index import LanceVectorIndex


class CompactSemanticEngine(BaseSemanticMemoryEngine):
    """Compact Semantic Memory 总装引擎。

    实现 BaseSemanticMemoryEngine 接口，是 SearchCoordinator
    和 ConsolidatorAgent 的直接依赖。
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        *,
        sqlite_db_path: str = "data/compact_memory.db",
        vector_db_path: str = "data/compact_vector",
        dedup_threshold: float = 0.85,
        synthesis_min_records: int = 2,
        synthesis_similarity: float = 0.75,
        scorer_weights: ScoringWeights | None = None,
        embedding_dim: int = 0,
        max_reflection_rounds: int = 2,
    ):
        self._llm = llm
        self._embedder = embedder

        # 存储层
        self._sqlite = SQLiteSemanticStore(db_path=sqlite_db_path)
        self._vector = LanceVectorIndex(
            db_path=vector_db_path, embedder=embedder, embedding_dim=embedding_dim
        )

        # 子模块
        self._encoder = TypedMemoryEncoder(llm=llm)
        self._planner = TaskAwareRetrievalPlanner(llm=llm)
        self._scorer = MemoryScorer(weights=scorer_weights)
        self._synthesizer = RecordFusionEngine(
            llm=llm,
            sqlite_store=self._sqlite,
            vector_index=self._vector,
            similarity_threshold=synthesis_similarity,
            min_records_for_synthesis=synthesis_min_records,
        )

        self._dedup_threshold = dedup_threshold
        self._max_reflection_rounds = max_reflection_rounds

    # ════════════════════════════════════════════════════════════════
    # search() — 检索管线
    # ════════════════════════════════════════════════════════════════

    def search(
        self,
        *,
        query: str,
        session_id: str | None = None,
        top_k: int = 10,
        query_embedding: list[float] | None = None,
        user_id: str = "",
        retrieval_plan: dict[str, Any] | None = None,
    ) -> SemanticSearchResult:
        """多通道检索 + 反思循环 + 打分 + 格式化。

        通道：
        1. FTS（BM25 全文）— semantic_queries 驱动
        2. 向量（semantic vector）— semantic_queries 驱动
        3. 向量（normalized/pragmatic vector）— pragmatic_queries 驱动
        4. Tag 过滤 — tool_hints / required_constraints 驱动
        5. 时间范围 — temporal_filter 驱动

        召回后去重 → Scorer 打分 → 反思循环（充分性检查 + 补充召回）→ 取 top_k → 格式化。
        """
        # Step 1: 确定检索计划
        if retrieval_plan:
            plan = self._dict_to_plan(retrieval_plan)
        else:
            plan = self._planner.plan(query)

        # Step 2: 初始多通道召回
        seen_ids: set[str] = set()
        raw_candidates = self._multi_channel_recall(
            plan=plan,
            query=query,
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        for c in raw_candidates:
            seen_ids.add(c.get("id", ""))

        if not raw_candidates:
            return SemanticSearchResult(context="", provenance=[])

        # Step 3: 初始打分
        scored = self._scorer.score_candidates(
            raw_candidates,
            plan_mode=plan.mode,
            plan_tool_hints=plan.tool_hints,
            plan_required_constraints=plan.required_constraints,
        )

        # Step 4: 反思循环（最多 max_reflection_rounds 轮）
        for _ in range(self._max_reflection_rounds):
            current_top = scored[:top_k]
            context_preview = self._format_context(current_top) or "（无检索结果）"

            adequacy = self._check_adequacy(query, context_preview)
            if adequacy["is_sufficient"]:
                break

            missing_info = adequacy.get("missing_info", "")
            additional_queries = self._generate_additional_queries(
                query, context_preview, missing_info
            )
            if not additional_queries:
                break

            # 用补充查询做额外召回（复用 _multi_channel_recall，注入 semantic_queries）
            supplement_plan = SearchPlan(
                mode=plan.mode,
                semantic_queries=additional_queries,
                pragmatic_queries=[],
                tool_hints=plan.tool_hints,
                required_constraints=plan.required_constraints,
                depth=top_k,
            )
            extra_candidates = self._multi_channel_recall(
                plan=supplement_plan,
                query=query,
                user_id=user_id,
                query_embedding=None,
                top_k=top_k,
            )

            # 去重合并
            new_candidates = [
                c for c in extra_candidates
                if c.get("id", "") not in seen_ids
            ]
            for c in new_candidates:
                seen_ids.add(c.get("id", ""))

            if not new_candidates:
                break

            raw_candidates = raw_candidates + new_candidates
            scored = self._scorer.score_candidates(
                raw_candidates,
                plan_mode=plan.mode,
                plan_tool_hints=plan.tool_hints,
                plan_required_constraints=plan.required_constraints,
            )

        # Step 5: 取 top_k
        top = scored[:top_k]

        # Step 6: 记录使用日志
        self._log_usage(
            query=query,
            session_id=session_id or "",
            user_id=user_id,
            plan=plan,
            retrieved_ids=[s.id for s in scored],
            kept_ids=[s.id for s in top],
        )

        # Step 7: 更新 retrieval_count
        all_retrieved_ids = [s.id for s in scored]
        kept_ids = [s.id for s in top]
        self._sqlite.increment_retrieval_count(all_retrieved_ids)
        self._sqlite.increment_hit_count(kept_ids)

        # Step 8: 格式化
        context = self._format_context(top)
        provenance = self._build_provenance(top)

        return SemanticSearchResult(context=context, provenance=provenance)

    def _multi_channel_recall(
        self,
        *,
        plan: SearchPlan,
        query: str,
        user_id: str,
        query_embedding: list[float] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """5 通道并行召回 + 去重合并。"""
        seen_ids: set[str] = set()
        candidates: list[dict[str, Any]] = []
        recall_limit = max(top_k * 3, 30)  # 召回量 > 最终需求量

        semantic_queries = plan.semantic_queries or [query]
        pragmatic_queries = plan.pragmatic_queries or []

        # ── 通道 1: FTS 全文检索 ──
        for sq in semantic_queries:
            fts_results = self._sqlite.fulltext_search(
                sq, user_id=user_id, limit=recall_limit,
            )
            for r in fts_results:
                uid = r.get("record_id", "")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    r["id"] = uid
                    r["source"] = "record"
                    r["semantic_distance"] = 1.0 - min(1.0, abs(r.get("fts_score", 0)) / 20.0)
                    candidates.append(r)

            # FTS on synthesized
            synth_fts = self._sqlite.fulltext_search_synthesized(
                sq, user_id=user_id, limit=10,
            )
            for r in synth_fts:
                sid = r.get("composite_id", "")
                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    r["id"] = sid
                    r["source"] = "composite"
                    r["semantic_distance"] = 1.0 - min(1.0, abs(r.get("fts_score", 0)) / 20.0)
                    candidates.append(r)

        # ── 通道 2: 语义向量检索 ──
        for sq in semantic_queries:
            vec_results = self._vector.search(
                sq,
                user_id=user_id,
                column="vector",
                limit=recall_limit,
            )
            for r in vec_results:
                uid = r.get("record_id", "")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    # 从 sqlite 获取完整数据
                    full = self._sqlite.get_record(uid)
                    if full:
                        c = self._record_to_candidate(full)
                        c["semantic_distance"] = r.get("_distance", 1.0)
                        candidates.append(c)

            # 向量 on synthesized
            synth_vec = self._vector.search_synthesized(
                sq, user_id=user_id, column="vector", limit=10,
            )
            for r in synth_vec:
                sid = r.get("composite_id", "")
                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    # 直接按 composite_id 从 sqlite 获取完整数据
                    su = self._sqlite.get_synthesized(sid)
                    if su:
                        s = {
                            "id": sid,
                            "source": "composite",
                            "semantic_distance": r.get("_distance", 1.0),
                            "memory_type": su.memory_type,
                            "semantic_text": su.semantic_text,
                            "normalized_text": su.normalized_text,
                            "tool_tags": su.tool_tags,
                            "constraint_tags": su.constraint_tags,
                            "task_tags": su.task_tags,
                            "failure_tags": su.failure_tags,
                            "affordance_tags": su.affordance_tags,
                            "created_at": su.created_at,
                            "retrieval_count": su.retrieval_count,
                            "retrieval_hit_count": su.retrieval_hit_count,
                            "action_success_count": su.action_success_count,
                            "action_fail_count": su.action_fail_count,
                            "evidence_turn_range": [],
                            "temporal": su.temporal,
                            "entities": su.entities,
                        }
                        candidates.append(s)

        # ── 通道 3: 实用向量检索（normalized_vector） ──
        for pq in pragmatic_queries:
            prag_results = self._vector.search(
                pq,
                user_id=user_id,
                column="normalized_vector",
                limit=recall_limit,
            )
            for r in prag_results:
                uid = r.get("record_id", "")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    full = self._sqlite.get_record(uid)
                    if full:
                        c = self._record_to_candidate(full)
                        c["semantic_distance"] = r.get("_distance", 1.0)
                        candidates.append(c)

        # ── 通道 4: Tag 过滤 ──
        if plan.tool_hints or plan.required_constraints:
            tag_results = self._sqlite.search_by_tags(
                tool_tags=plan.tool_hints or None,
                constraint_tags=plan.required_constraints or None,
                user_id=user_id,
                limit=recall_limit,
            )
            for r in tag_results:
                uid = r.get("record_id", "")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    r["id"] = uid
                    r["source"] = "record"
                    r["semantic_distance"] = 0.5  # tag 匹配给中等距离
                    candidates.append(r)

        # ── 通道 5: 时间范围 ──
        if plan.temporal_filter:
            time_results = self._sqlite.search_by_time(
                since=plan.temporal_filter.get("since"),
                until=plan.temporal_filter.get("until"),
                user_id=user_id,
                limit=recall_limit,
            )
            for r in time_results:
                uid = r.get("record_id", "")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    r["id"] = uid
                    r["source"] = "record"
                    r["semantic_distance"] = 0.6  # 时间匹配给中等偏高距离
                    candidates.append(r)

        return candidates

    # ════════════════════════════════════════════════════════════════
    # ingest_conversation() — 固化管线
    # ════════════════════════════════════════════════════════════════

    def ingest_conversation(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        user_id: str = "",
        retrieved_context: str = "",
        reference_timestamp: str | None = None,
    ) -> ConsolidationResult:
        """完整固化流程：新颖性检查 → 编码 → 去重写入 → 合成。"""
        if not turns:
            return ConsolidationResult(
                records_added=0, records_merged=0, records_expired=0, steps=[]
            )

        steps: list[dict[str, Any]] = []

        # Step 1: 新颖性检查
        has_novelty = self._check_novelty(turns, retrieved_context)
        steps.append({
            "name": "novelty_check",
            "status": "done",
            "detail": "检测到新信息" if has_novelty else "无新信息，跳过固化",
        })
        if not has_novelty:
            return ConsolidationResult(
                records_added=0, records_merged=0, records_expired=0, steps=steps
            )

        # Step 2: Compact Encoding
        # 取最近 4 轮作为上文，其余全部作为当前轮
        previous = turns[:-4] if len(turns) > 4 else []
        current = turns[-4:] if len(turns) > 4 else turns

        new_records = self._encoder.encode_conversation(
            current,
            previous_turns=previous,
            session_id=session_id,
            user_id=user_id,
        )

        steps.append({
            "name": "compact_encoding",
            "status": "done",
            "detail": f"抽取 {len(new_records)} 条 MemoryRecord",
        })

        if not new_records:
            return ConsolidationResult(
                records_added=0, records_merged=0, records_expired=0, steps=steps
            )

        # Step 3: 去重 + 写入
        actually_added = 0
        for record in new_records:
            # 检查是否已存在相似条目
            similar = self._sqlite.find_similar_by_normalized_text(
                record.normalized_text, user_id=user_id, limit=3,
            )
            is_duplicate = False
            for s in similar:
                if s.record_id == record.record_id:
                    is_duplicate = True
                    break
                # 用向量相似度判断重复
                try:
                    vecs = self._embedder.embed([record.normalized_text, s.normalized_text])
                    sim = self._cosine_similarity(vecs[0], vecs[1])
                    if sim >= self._dedup_threshold:
                        is_duplicate = True
                        # 更新已有条目而非插入新的
                        s.updated_at = datetime.now(timezone.utc).isoformat()
                        s.confidence = min(1.0, s.confidence + 0.1)
                        self._sqlite.upsert_record(s)
                        break
                except Exception:
                    pass

            if not is_duplicate:
                self._sqlite.upsert_record(record)
                try:
                    self._vector.upsert(
                        record_id=record.record_id,
                        user_id=record.user_id,
                        memory_type=record.memory_type,
                        semantic_text=record.semantic_text,
                        normalized_text=record.normalized_text,
                    )
                except Exception:
                    # 向量写入失败不阻断固化；已写 SQLite，FTS 仍可召回
                    import logging
                    logging.getLogger("src.memory.semantic.engine").warning(
                        "vector upsert failed for record %s", record.record_id, exc_info=True
                    )
                actually_added += 1

        steps.append({
            "name": "dedup_and_store",
            "status": "done",
            "detail": f"去重后写入 {actually_added}/{len(new_records)} 条",
        })

        # Step 4: Record Fusion（在线聚合）
        composite_records = []
        if actually_added > 0:
            composite_records = self._synthesizer.synthesize_on_ingest(
                [u for u in new_records if not any(
                    s.record_id == u.record_id
                    for s in self._sqlite.find_similar_by_normalized_text(
                        u.normalized_text, user_id=user_id, limit=1,
                    )
                    if s.record_id != u.record_id
                )],
                user_id=user_id,
            )

        steps.append({
            "name": "record_fusion",
            "status": "done",
            "detail": f"聚合 {len(composite_records)} 条 CompositeRecord",
        })

        return ConsolidationResult(
            records_added=actually_added,
            records_merged=len(composite_records),
            records_expired=0,
            steps=steps,
        )

    # ════════════════════════════════════════════════════════════════
    # delete / export
    # ════════════════════════════════════════════════════════════════

    def delete_all_for_user(self, user_id: str) -> dict[str, int]:
        result = self._sqlite.delete_all_for_user(user_id)
        self._vector.delete_all_for_user(user_id)
        return result

    def export_debug(self, *, user_id: str = "") -> dict[str, Any]:
        return self._sqlite.export_all(user_id=user_id)

    # ════════════════════════════════════════════════════════════════
    # 内部工具方法
    # ════════════════════════════════════════════════════════════════

    def _check_adequacy(self, query: str, context_text: str) -> dict[str, Any]:
        """LLM 判断当前检索结果是否足以回应查询。

        Returns:
            {"is_sufficient": bool, "missing_info": str}
        """
        user_content = (
            f"<USER_QUERY>\n{query}\n</USER_QUERY>\n\n"
            f"<RETRIEVED_MEMORY>\n{context_text}\n</RETRIEVED_MEMORY>"
        )
        response = self._llm.generate([
            {"role": "system", "content": RETRIEVAL_ADEQUACY_CHECK_SYSTEM},
            {"role": "user", "content": user_content},
        ])
        try:
            parsed = json.loads(
                response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            )
            return {
                "is_sufficient": bool(parsed.get("is_sufficient", True)),
                "missing_info": str(parsed.get("missing_info", "")),
            }
        except (json.JSONDecodeError, ValueError):
            # 解析失败保守地视为充分，避免无限循环
            return {"is_sufficient": True, "missing_info": ""}

    def _generate_additional_queries(
        self, query: str, context_text: str, missing_info: str,
    ) -> list[str]:
        """LLM 根据缺失信息生成补充检索查询。

        Returns:
            补充查询字符串列表（2~4 条），失败时返回空列表。
        """
        user_content = (
            f"<USER_QUERY>\n{query}\n</USER_QUERY>\n\n"
            f"<CURRENT_MEMORY>\n{context_text}\n</CURRENT_MEMORY>\n\n"
            f"<MISSING_INFO>\n{missing_info}\n</MISSING_INFO>"
        )
        response = self._llm.generate([
            {"role": "system", "content": RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM},
            {"role": "user", "content": user_content},
        ])
        try:
            parsed = json.loads(
                response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            )
            queries = parsed.get("additional_queries", [])
            if isinstance(queries, list):
                return [q for q in queries if isinstance(q, str) and q.strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    def _check_novelty(
        self, turns: list[dict[str, Any]], retrieved_context: str,
    ) -> bool:
        """LLM 判断对话是否有新信息。"""
        conversation_text = "\n".join(
            f"{t.get('role', '')}: {t.get('content', '')}" for t in turns
        )
        user_content = (
            f"<EXISTING_MEMORY>\n{retrieved_context or '（无已有记忆）'}\n</EXISTING_MEMORY>\n\n"
            f"<CONVERSATION>\n{conversation_text}\n</CONVERSATION>"
        )
        response = self._llm.generate([
            {"role": "system", "content": NOVELTY_CHECK_SYSTEM},
            {"role": "user", "content": user_content},
        ])
        try:
            parsed = json.loads(response.strip().lstrip("```json").rstrip("```").strip())
            return bool(parsed.get("has_novelty", True))
        except (json.JSONDecodeError, ValueError):
            return True  # 保守策略：解析失败则认为有新信息

    def _dict_to_plan(self, d: dict[str, Any]) -> SearchPlan:
        """dict → SearchPlan。"""
        mode = d.get("mode", "answer")
        if mode not in ("answer", "action", "mixed"):
            mode = "answer"
        temporal_filter = d.get("temporal_filter")
        if temporal_filter and not isinstance(temporal_filter, dict):
            temporal_filter = None
        return SearchPlan(
            mode=mode,
            semantic_queries=d.get("semantic_queries", []),
            pragmatic_queries=d.get("pragmatic_queries", []),
            temporal_filter=temporal_filter,
            tool_hints=d.get("tool_hints", []),
            required_constraints=d.get("required_constraints", []),
            missing_slots=d.get("missing_slots", []),
            depth=int(d.get("depth", 5)),
            reasoning=d.get("reasoning", ""),
        )

    @staticmethod
    def _record_to_candidate(record: MemoryRecord) -> dict[str, Any]:
        """MemoryRecord → scorer 需要的 candidate dict。"""
        return {
            "id": record.record_id,
            "source": "record",
            "semantic_distance": 0.5,  # 默认，由调用方覆盖
            "memory_type": record.memory_type,
            "semantic_text": record.semantic_text,
            "normalized_text": record.normalized_text,
            "tool_tags": record.tool_tags,
            "constraint_tags": record.constraint_tags,
            "task_tags": record.task_tags,
            "failure_tags": record.failure_tags,
            "affordance_tags": record.affordance_tags,
            "created_at": record.created_at,
            "retrieval_count": record.retrieval_count,
            "retrieval_hit_count": record.retrieval_hit_count,
            "action_success_count": record.action_success_count,
            "action_fail_count": record.action_fail_count,
            "evidence_turn_range": record.evidence_turn_range,
            "temporal": record.temporal,
            "entities": record.entities,
        }

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _format_context(self, scored: list[ScoredCandidate]) -> str:
        """将 top-k 候选格式化为 LLM 可注入的文本。"""
        if not scored:
            return ""

        parts: list[str] = []
        for i, sc in enumerate(scored, 1):
            d = sc.data
            mt = d.get("memory_type", "unknown")
            text = d.get("semantic_text", d.get("normalized_text", ""))
            entities = d.get("entities", [])
            score = f"{sc.final_score:.3f}"

            header = f"[{i}] ({mt}, score={score})"
            if entities:
                header += f" entities=[{', '.join(entities)}]"

            parts.append(f"{header}\n{text}")

        return "\n\n".join(parts)

    @staticmethod
    def _build_provenance(scored: list[ScoredCandidate]) -> list[dict[str, Any]]:
        """构建溯源信息。"""
        provenance = []
        for sc in scored:
            d = sc.data
            provenance.append({
                "record_id": sc.id,
                "source": sc.source,
                "memory_type": d.get("memory_type", ""),
                "score": sc.final_score,
                "score_breakdown": sc.score_breakdown,
                "semantic_text": d.get("semantic_text", ""),
                "entities": d.get("entities", []),
            })
        return provenance

    def _log_usage(
        self,
        *,
        query: str,
        session_id: str,
        user_id: str,
        plan: SearchPlan,
        retrieved_ids: list[str],
        kept_ids: list[str],
    ) -> None:
        """记录一次检索使用日志。"""
        log = UsageLog(
            log_id=uuid.uuid4().hex,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            retrieval_plan={
                "mode": plan.mode,
                "semantic_queries": plan.semantic_queries,
                "pragmatic_queries": plan.pragmatic_queries,
                "depth": plan.depth,
            },
            retrieved_record_ids=retrieved_ids,
            kept_record_ids=kept_ids,
        )
        self._sqlite.insert_usage_log(log)
