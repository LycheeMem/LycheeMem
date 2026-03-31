"""Compact Semantic Engine（总装引擎）。

实现 BaseSemanticMemoryEngine，串联所有子模块：
- search(): Planner → 5 通道召回 → Scorer → 格式化输出
- ingest_conversation(): Novelty Check → Encoder → 去重写入 → Fusion

这是整个 Compact Semantic Memory 的核心入口。
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from src.embedder.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.memory.semantic.base import (
    BaseSemanticMemoryEngine,
    ConsolidationResult,
    SemanticSearchResult,
)
from src.memory.semantic.encoder import CompactSemanticEncoder
from src.memory.semantic.models import (
    ActionState,
    CompositeRecord,
    MemoryRecord,
    SearchPlan,
    UsageLog,
    MEMORY_TYPE_CONSTRAINT,
    MEMORY_TYPE_FAILURE_PATTERN,
    MEMORY_TYPE_PROCEDURE,
    MEMORY_TYPE_TOOL_AFFORDANCE,
    SYNTH_TYPE_CONSTRAINT,
    SYNTH_TYPE_PATTERN,
    SYNTH_TYPE_USAGE,
)
from src.memory.semantic.planner import ActionAwareSearchPlanner
from src.memory.semantic.prompts import (
    NOVELTY_CHECK_SYSTEM,
    RETRIEVAL_ADEQUACY_CHECK_SYSTEM,
    RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM,
)
from src.memory.semantic.scorer import MemoryScorer, ScoredCandidate, ScoringWeights
from src.memory.semantic.sqlite_store import SQLiteSemanticStore
from src.memory.semantic.synthesizer import RecordFusionEngine
from src.memory.semantic.vector_index import LanceVectorIndex

# ──────────────────────────────────────────────────────────────────
# 敏感信息模式：匹配到任意一项即跳过写入长期记忆
# ──────────────────────────────────────────────────────────────────
_SENSITIVE_PATTERNS: list[re.Pattern] = [
    # 明文密码赋值：password = xxx / passwd: xxx
    re.compile(r"(?i)\b(password|passwd|pwd)\s*[=:]\s*\S{4,}"),
    # API key / secret / token 赋值
    re.compile(r"(?i)\b(api[_\-]?key|secret[_\-]?key|access[_\-]?token|auth[_\-]?token"
               r"|private[_\-]?key|client[_\-]?secret)\s*[=:]\s*\S{6,}"),
    # Bearer Token
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/=]{20,}"),
    # OpenAI sk- / GitHub ghp_ / AWS AKIA
    re.compile(r"\b(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}"
               r"|gho_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,}"
               r"|AKIA[A-Z0-9]{16,})\b"),
    # PEM 私钥块
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    # 信用卡号（Luhn 格式检测前缀 + 长度）
    re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?"           # Visa
        r"|5[1-5][0-9]{14}"                        # MasterCard
        r"|3[47][0-9]{13}"                         # Amex
        r"|6(?:011|5[0-9]{2})[0-9]{12})\b"        # Discover
    ),
]


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
        self._encoder = CompactSemanticEncoder(llm=llm)
        self._planner = ActionAwareSearchPlanner(llm=llm)
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
    # 敏感信息过滤
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_sensitive_text(text: str) -> bool:
        """检查文本是否包含密码/API key/私钥/信用卡号等敏感信息。

        匹配任意一项即返回 True，对应 record 将不写入长期记忆。
        """
        for pattern in _SENSITIVE_PATTERNS:
            if pattern.search(text):
                return True
        return False

    # ════════════════════════════════════════════════════════════════
    # search() — 检索管线
    # ════════════════════════════════════════════════════════════════

    def search(
        self,
        *,
        query: str,
        session_id: str | None = None,
        top_k: int = 0,
        query_embedding: list[float] | None = None,
        user_id: str = "",
        recent_context: str = "",
        action_state: dict[str, Any] | None = None,
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
        action_state_obj = self._dict_to_action_state(action_state)

        if retrieval_plan:
            plan = self._dict_to_plan(retrieval_plan)
        else:
            plan = self._planner.plan(
                query,
                recent_context=recent_context,
                action_state=action_state_obj,
            )

        top_k = max(1, int(top_k or plan.depth or 5))
        resolved_action_state = self._merge_action_state_with_plan(action_state_obj, plan)
        focus_terms = self._derive_query_focus_terms(
            query=query,
            plan=plan,
            action_state=resolved_action_state,
        )
        scoring_slots = self._merge_unique(plan.missing_slots, focus_terms)

        # Step 2: 初始多通道召回
        seen_ids: set[str] = set()
        raw_candidates = self._multi_channel_recall(
            plan=plan,
            query=query,
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k,
            record_type_bias=self._derive_record_type_bias(
                plan=plan,
                action_state=resolved_action_state,
            ),
            synth_type_bias=self._derive_synth_type_bias(
                plan=plan,
                action_state=resolved_action_state,
            ),
            focus_terms=focus_terms,
        )
        for c in raw_candidates:
            seen_ids.add(c.get("id", ""))

        # Step 3: 初始打分
        scored = self._scorer.score_candidates(
            raw_candidates,
            plan_mode=plan.mode,
            plan_tool_hints=plan.tool_hints,
            plan_required_constraints=plan.required_constraints,
            plan_required_affordances=plan.required_affordances,
            plan_missing_slots=scoring_slots,
        ) if raw_candidates else []

        # Step 4: 反思循环（最多 max_reflection_rounds 轮）
        for _ in range(self._max_reflection_rounds):
            current_top = scored[:top_k]
            context_preview = self._format_context(current_top) or "（无检索结果）"

            adequacy = self._check_adequacy(
                query,
                context_preview,
                plan=plan,
                action_state=resolved_action_state,
            )
            if adequacy["is_sufficient"]:
                break

            supplement = self._generate_additional_queries(
                query,
                context_preview,
                adequacy=adequacy,
                plan=plan,
                action_state=resolved_action_state,
            )
            supplement_semantic_queries = supplement.get("semantic_queries", [])
            supplement_pragmatic_queries = supplement.get("pragmatic_queries", [])
            supplement_tool_hints = supplement.get("tool_hints", [])
            supplement_constraints = supplement.get("required_constraints", [])
            supplement_affordances = supplement.get("required_affordances", [])
            supplement_slots = supplement.get("missing_slots", [])

            if not any(
                [
                    supplement_semantic_queries,
                    supplement_pragmatic_queries,
                    supplement_tool_hints,
                    supplement_constraints,
                    supplement_affordances,
                    supplement_slots,
                ]
            ):
                break

            # 用补充查询做额外召回（补充 plan 也必须继承新的 action 缺口）
            supplement_plan = SearchPlan(
                mode=plan.mode,
                semantic_queries=supplement_semantic_queries,
                pragmatic_queries=supplement_pragmatic_queries,
                tool_hints=self._merge_unique(plan.tool_hints, supplement_tool_hints),
                required_constraints=self._merge_unique(plan.required_constraints, supplement_constraints),
                required_affordances=self._merge_unique(plan.required_affordances, supplement_affordances),
                missing_slots=self._merge_unique(plan.missing_slots, supplement_slots),
                depth=top_k,
            )
            extra_candidates = self._multi_channel_recall(
                plan=supplement_plan,
                query=query,
                user_id=user_id,
                query_embedding=None,
                top_k=top_k,
                record_type_bias=self._derive_record_type_bias(
                    plan=supplement_plan,
                    action_state=resolved_action_state,
                    adequacy=adequacy,
                ),
                synth_type_bias=self._derive_synth_type_bias(
                    plan=supplement_plan,
                    action_state=resolved_action_state,
                    adequacy=adequacy,
                ),
                focus_terms=focus_terms,
            )

            plan = SearchPlan(
                mode=plan.mode,
                semantic_queries=self._merge_unique(plan.semantic_queries, supplement_semantic_queries),
                pragmatic_queries=self._merge_unique(plan.pragmatic_queries, supplement_pragmatic_queries),
                temporal_filter=plan.temporal_filter,
                tool_hints=list(supplement_plan.tool_hints),
                required_constraints=list(supplement_plan.required_constraints),
                required_affordances=list(supplement_plan.required_affordances),
                missing_slots=list(supplement_plan.missing_slots),
                depth=max(plan.depth, top_k),
                reasoning=plan.reasoning,
            )
            resolved_action_state = self._merge_action_state_with_plan(resolved_action_state, plan)
            scoring_slots = self._merge_unique(plan.missing_slots, focus_terms)
            reflection_scoring_context = self._build_reflection_scoring_context(
                adequacy=adequacy,
                action_state=resolved_action_state,
            )

            # 去重合并
            new_candidates = [
                c for c in extra_candidates
                if c.get("id", "") not in seen_ids
            ]
            for c in new_candidates:
                seen_ids.add(c.get("id", ""))

            raw_candidates = raw_candidates + new_candidates
            scored = self._scorer.score_candidates(
                raw_candidates,
                plan_mode=plan.mode,
                plan_tool_hints=plan.tool_hints,
                plan_required_constraints=plan.required_constraints,
                plan_required_affordances=plan.required_affordances,
                plan_missing_slots=scoring_slots,
                reflection_context=reflection_scoring_context,
            )

            if not new_candidates:
                continue

        # Step 5: 取 top_k
        top = self._select_final_candidates(
            scored,
            top_k=top_k,
            focus_terms=focus_terms,
        )

        # Step 6: 记录使用日志
        log_id = self._log_usage(
            query=query,
            session_id=session_id or "",
            user_id=user_id,
            plan=plan,
            action_state=resolved_action_state,
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

        return SemanticSearchResult(
            context=context,
            provenance=provenance,
            retrieval_plan=self._plan_to_dict(plan),
            action_state=self._action_state_to_dict(resolved_action_state),
            usage_log_id=log_id,
            mode=plan.mode,
        )

    def _multi_channel_recall(
        self,
        *,
        plan: SearchPlan,
        query: str,
        user_id: str,
        query_embedding: list[float] | None,
        top_k: int,
        record_type_bias: list[str] | None = None,
        synth_type_bias: list[str] | None = None,
        focus_terms: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """5 通道并行召回 + 去重合并。"""
        seen_ids: set[str] = set()
        candidates: list[dict[str, Any]] = []
        recall_limit = max(top_k * 3, 30)  # 召回量 > 最终需求量

        semantic_queries = plan.semantic_queries or [query]
        pragmatic_queries = plan.pragmatic_queries or []
        record_type_bias = [t for t in (record_type_bias or []) if str(t or "").strip()]
        synth_type_bias = [t for t in (synth_type_bias or []) if str(t or "").strip()]
        focus_terms = [t for t in (focus_terms or []) if str(t or "").strip()]

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
                            "source_record_ids": su.source_record_ids,
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

        # ── 通道 4: Tag 过滤（tool / constraint / affordance）──
        if plan.tool_hints or plan.required_constraints or plan.required_affordances:
            tag_results = self._sqlite.search_by_tags(
                tool_tags=plan.tool_hints or None,
                constraint_tags=plan.required_constraints or None,
                affordance_tags=plan.required_affordances or None,
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

        # ── 通道 4.5: Slot 补全召回 ──
        if plan.missing_slots:
            slot_results = self._sqlite.search_by_slot_hints(
                slot_terms=plan.missing_slots,
                user_id=user_id,
                limit=recall_limit,
            )
            for r in slot_results:
                uid = r.get("record_id", "")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    r["id"] = uid
                    r["source"] = "record"
                    r["semantic_distance"] = 0.45
                    candidates.append(r)

            slot_query = " ".join(plan.missing_slots)
            if slot_query.strip():
                synth_slot_results = self._sqlite.fulltext_search_synthesized(
                    slot_query,
                    user_id=user_id,
                    limit=10,
                )
                for r in synth_slot_results:
                    sid = r.get("composite_id", "")
                    if sid and sid not in seen_ids:
                        seen_ids.add(sid)
                        r["id"] = sid
                        r["source"] = "composite"
                        r["semantic_distance"] = 0.45
                        candidates.append(r)

        # ── 通道 4.6: query focus hints（分工/负责/截止等答案导向槽位）──
        if focus_terms:
            focus_results = self._sqlite.search_by_slot_hints(
                slot_terms=focus_terms,
                user_id=user_id,
                limit=max(top_k * 2, 12),
            )
            for r in focus_results:
                uid = r.get("record_id", "")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    r["id"] = uid
                    r["source"] = "record"
                    r["semantic_distance"] = 0.4
                    candidates.append(r)

            focus_query = " ".join(focus_terms)
            if focus_query.strip():
                synth_focus_results = self._sqlite.fulltext_search_synthesized(
                    focus_query,
                    user_id=user_id,
                    limit=max(6, top_k),
                )
                for r in synth_focus_results:
                    sid = r.get("composite_id", "")
                    if sid and sid not in seen_ids:
                        seen_ids.add(sid)
                        r["id"] = sid
                        r["source"] = "composite"
                        r["semantic_distance"] = 0.4
                        candidates.append(r)

        # ── 通道 4.8: 按类型偏置召回（failure / affordance / procedure / constraint）──
        if record_type_bias or synth_type_bias:
            focused_limit = max(top_k * 2, 12)
            focused_queries = self._merge_unique(
                self._merge_unique(semantic_queries, pragmatic_queries),
                plan.missing_slots,
            )

            for fq in focused_queries:
                if record_type_bias:
                    focused_records = self._sqlite.fulltext_search(
                        fq,
                        user_id=user_id,
                        limit=focused_limit,
                        memory_types=record_type_bias,
                    )
                    for r in focused_records:
                        uid = r.get("record_id", "")
                        if uid and uid not in seen_ids:
                            seen_ids.add(uid)
                            r["id"] = uid
                            r["source"] = "record"
                            r["semantic_distance"] = 0.35
                            candidates.append(r)

                if synth_type_bias:
                    focused_synth = self._sqlite.fulltext_search_synthesized(
                        fq,
                        user_id=user_id,
                        limit=max(6, top_k),
                        memory_types=synth_type_bias,
                    )
                    for r in focused_synth:
                        sid = r.get("composite_id", "")
                        if sid and sid not in seen_ids:
                            seen_ids.add(sid)
                            r["id"] = sid
                            r["source"] = "composite"
                            r["semantic_distance"] = 0.35
                            candidates.append(r)

            if record_type_bias and (plan.tool_hints or plan.required_constraints or plan.required_affordances):
                focused_tags = self._sqlite.search_by_tags(
                    tool_tags=plan.tool_hints or None,
                    constraint_tags=plan.required_constraints or None,
                    affordance_tags=plan.required_affordances or None,
                    memory_types=record_type_bias,
                    user_id=user_id,
                    limit=focused_limit,
                )
                for r in focused_tags:
                    uid = r.get("record_id", "")
                    if uid and uid not in seen_ids:
                        seen_ids.add(uid)
                        r["id"] = uid
                        r["source"] = "record"
                        r["semantic_distance"] = 0.35
                        candidates.append(r)

            if record_type_bias and plan.missing_slots:
                focused_slots = self._sqlite.search_by_slot_hints(
                    slot_terms=plan.missing_slots,
                    memory_types=record_type_bias,
                    user_id=user_id,
                    limit=focused_limit,
                )
                for r in focused_slots:
                    uid = r.get("record_id", "")
                    if uid and uid not in seen_ids:
                        seen_ids.add(uid)
                        r["id"] = uid
                        r["source"] = "record"
                        r["semantic_distance"] = 0.35
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

        # ── 补召回：如果已经召回到 child record，则把其 covering composite 一并拉回 ──
        child_best_distance: dict[str, float] = {}
        for candidate in candidates:
            if candidate.get("source") != "record":
                continue
            cid = str(candidate.get("id") or "")
            if not cid:
                continue
            distance = float(candidate.get("semantic_distance", 1.0) or 1.0)
            child_best_distance[cid] = min(child_best_distance.get(cid, distance), distance)

        if child_best_distance:
            parent_composites = self._sqlite.get_synthesized_by_source(list(child_best_distance))
            for composite in parent_composites:
                sid = composite.composite_id
                if sid in seen_ids:
                    continue
                candidate = self._synth_to_candidate(composite)
                child_distances = [
                    child_best_distance[src_id]
                    for src_id in composite.source_record_ids
                    if src_id in child_best_distance
                ]
                candidate["semantic_distance"] = (
                    max(0.0, min(child_distances) - 0.05)
                    if child_distances
                    else 0.4
                )
                seen_ids.add(sid)
                candidates.append(candidate)

        return candidates

    @staticmethod
    def _derive_query_focus_terms(
        *,
        query: str,
        plan: SearchPlan,
        action_state: ActionState,
    ) -> list[str]:
        text = str(query or "").strip().lower()
        focus_terms: list[str] = []

        def _contains_any(*keywords: str) -> bool:
            return any(keyword in text for keyword in keywords)

        if _contains_any("分工", "负责", "职责", "安排", "谁做", "assignment", "owner", "responsible"):
            focus_terms.extend(["分工", "负责", "职责", "安排"])
        if _contains_any("成员", "组员", "小组", "团队", "team member"):
            focus_terms.extend(["成员", "组员"])
        if _contains_any("主题", "课题", "topic"):
            focus_terms.extend(["主题", "课题"])
        if _contains_any("截止", "日期", "时间", "什么时候", "deadline", "due"):
            focus_terms.extend(["截止", "日期", "时间"])

        focus_terms = CompactSemanticEngine._merge_unique(focus_terms, plan.missing_slots)
        focus_terms = CompactSemanticEngine._merge_unique(focus_terms, action_state.missing_slots)
        return focus_terms

    def _select_final_candidates(
        self,
        scored: list[ScoredCandidate],
        *,
        top_k: int,
        focus_terms: list[str] | None = None,
    ) -> list[ScoredCandidate]:
        """最终候选选择：优先 composite，并折叠被覆盖或近重复的 record/composite。"""
        if not scored:
            return []

        focus_terms = [str(term or "").strip() for term in (focus_terms or []) if str(term or "").strip()]
        composite_index = {
            candidate.id: candidate
            for candidate in scored
            if candidate.source == "composite"
        }
        composite_source_sets = {
            candidate_id: self._candidate_source_record_set(candidate.data)
            for candidate_id, candidate in composite_index.items()
        }
        covering_composites: dict[str, list[ScoredCandidate]] = {}
        for candidate in composite_index.values():
            for source_id in candidate.data.get("source_record_ids", []) or []:
                source_key = str(source_id or "").strip()
                if not source_key:
                    continue
                covering_composites.setdefault(source_key, []).append(candidate)
        for items in covering_composites.values():
            items.sort(key=lambda item: item.final_score, reverse=True)

        covering_parent_composites: dict[str, list[ScoredCandidate]] = {}
        for child_id, child_sources in composite_source_sets.items():
            if not child_sources:
                continue
            parents = [
                candidate
                for candidate_id, candidate in composite_index.items()
                if candidate_id != child_id
                and child_sources.issubset(composite_source_sets.get(candidate_id, set()))
                and len(composite_source_sets.get(candidate_id, set())) > len(child_sources)
            ]
            if parents:
                parents.sort(key=lambda item: item.final_score, reverse=True)
                covering_parent_composites[child_id] = parents

        ordered = sorted(
            scored,
            key=lambda item: item.final_score + (0.02 if item.source == "composite" else 0.0),
            reverse=True,
        )

        selected: list[ScoredCandidate] = []
        selected_ids: set[str] = set()
        covered_record_ids: set[str] = set()

        for candidate in ordered:
            if candidate.id in selected_ids:
                continue

            if candidate.source == "composite":
                candidate_sources = composite_source_sets.get(candidate.id, set())
                selected_covering_parents = [
                    item for item in selected
                    if item.source == "composite"
                    and candidate_sources
                    and candidate_sources.issubset(composite_source_sets.get(item.id, set()))
                    and len(composite_source_sets.get(item.id, set())) > len(candidate_sources)
                ]
                if selected_covering_parents and not self._record_adds_unique_value(
                    candidate,
                    selected_covering_parents,
                    focus_terms=focus_terms,
                ):
                    continue

                best_parent = next(iter(covering_parent_composites.get(candidate.id, [])), None)
                if (
                    best_parent is not None
                    and best_parent.id not in selected_ids
                    and best_parent.final_score >= candidate.final_score - 0.05
                    and not self._record_adds_unique_value(
                        candidate,
                        [best_parent],
                        focus_terms=focus_terms,
                    )
                ):
                    if not self._is_duplicate_candidate(best_parent, selected, focus_terms=focus_terms):
                        selected.append(best_parent)
                        selected_ids.add(best_parent.id)
                        covered_record_ids.update(
                            str(source_id or "").strip()
                            for source_id in (best_parent.data.get("source_record_ids", []) or [])
                            if str(source_id or "").strip()
                        )
                    continue

            if candidate.source == "record":
                selected_covering = [
                    item for item in selected
                    if item.source == "composite"
                    and candidate.id in set(item.data.get("source_record_ids", []) or [])
                ]
                if selected_covering and not self._record_adds_unique_value(
                    candidate,
                    selected_covering,
                    focus_terms=focus_terms,
                ):
                    continue

                best_cover = next(iter(covering_composites.get(candidate.id, [])), None)
                if (
                    best_cover is not None
                    and best_cover.id not in selected_ids
                    and best_cover.final_score >= candidate.final_score - 0.05
                    and not self._record_adds_unique_value(
                        candidate,
                        [best_cover],
                        focus_terms=focus_terms,
                    )
                ):
                    if not self._is_duplicate_candidate(best_cover, selected, focus_terms=focus_terms):
                        selected.append(best_cover)
                        selected_ids.add(best_cover.id)
                        covered_record_ids.update(
                            str(source_id or "").strip()
                            for source_id in (best_cover.data.get("source_record_ids", []) or [])
                            if str(source_id or "").strip()
                        )
                    continue

                if candidate.id in covered_record_ids:
                    continue

            if self._is_duplicate_candidate(candidate, selected, focus_terms=focus_terms):
                continue

            selected.append(candidate)
            selected_ids.add(candidate.id)
            if candidate.source == "composite":
                covered_record_ids.update(
                    str(source_id or "").strip()
                    for source_id in (candidate.data.get("source_record_ids", []) or [])
                    if str(source_id or "").strip()
                )

            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            for candidate in ordered:
                if candidate.id in selected_ids:
                    continue
                if self._is_duplicate_candidate(candidate, selected, focus_terms=focus_terms):
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.id)
                if len(selected) >= top_k:
                    break

        return selected[:top_k]

    def _record_adds_unique_value(
        self,
        record: ScoredCandidate,
        covering_composites: list[ScoredCandidate],
        *,
        focus_terms: list[str] | None = None,
    ) -> bool:
        if not covering_composites:
            return True

        focus_terms = [str(term or "").strip() for term in (focus_terms or []) if str(term or "").strip()]
        record_data = record.data
        record_entities = {str(v or "").strip() for v in (record_data.get("entities") or []) if str(v or "").strip()}

        composite_entities: set[str] = set()
        composite_tags: dict[str, set[str]] = {
            "task_tags": set(),
            "tool_tags": set(),
            "constraint_tags": set(),
            "failure_tags": set(),
            "affordance_tags": set(),
        }
        composite_temporal_values: set[str] = set()
        composite_texts: list[str] = []
        composite_structured_tokens: set[str] = set()

        for composite in covering_composites:
            data = composite.data
            composite_entities.update(
                str(v or "").strip() for v in (data.get("entities") or []) if str(v or "").strip()
            )
            for tag_field in composite_tags:
                composite_tags[tag_field].update(
                    str(v or "").strip() for v in (data.get(tag_field) or []) if str(v or "").strip()
                )
            temporal = data.get("temporal") or {}
            if isinstance(temporal, dict):
                composite_temporal_values.update(
                    str(v or "").strip() for v in temporal.values() if str(v or "").strip()
                )
            text_blob = self._candidate_text_blob(data)
            if text_blob:
                composite_texts.append(text_blob)
                composite_structured_tokens.update(self._extract_structured_tokens(text_blob))

        if record_entities - composite_entities:
            return True

        for tag_field, covered_values in composite_tags.items():
            record_values = {
                str(v or "").strip()
                for v in (record_data.get(tag_field) or [])
                if str(v or "").strip()
            }
            if record_values - covered_values:
                return True

        temporal = record_data.get("temporal") or {}
        if isinstance(temporal, dict):
            record_temporal_values = {
                str(v or "").strip()
                for v in temporal.values()
                if str(v or "").strip()
            }
            if record_temporal_values - composite_temporal_values:
                return True

        record_text = self._candidate_text_blob(record_data)
        combined_composite_text = "\n".join(composite_texts)
        if focus_terms:
            record_focus_terms = [term for term in focus_terms if term in record_text]
            if record_focus_terms and any(term not in combined_composite_text for term in record_focus_terms):
                return True

        record_structured_tokens = self._extract_structured_tokens(record_text)
        if record_structured_tokens - composite_structured_tokens:
            return True

        return False

    def _is_duplicate_candidate(
        self,
        candidate: ScoredCandidate,
        selected: list[ScoredCandidate],
        *,
        focus_terms: list[str] | None = None,
    ) -> bool:
        for existing in selected:
            if candidate.id == existing.id:
                return True

            if candidate.source == "record" and existing.source == "composite":
                if candidate.id in self._candidate_source_record_set(existing.data):
                    if not self._record_adds_unique_value(candidate, [existing], focus_terms=focus_terms):
                        return True

            if candidate.source == "composite" and existing.source == "record":
                if existing.id in self._candidate_source_record_set(candidate.data):
                    if not self._record_adds_unique_value(existing, [candidate], focus_terms=focus_terms):
                        return True

            if candidate.source == "composite" and existing.source == "composite":
                candidate_sources = self._candidate_source_record_set(candidate.data)
                existing_sources = self._candidate_source_record_set(existing.data)
                if candidate_sources and existing_sources:
                    if (
                        candidate_sources.issubset(existing_sources)
                        and len(existing_sources) > len(candidate_sources)
                        and not self._record_adds_unique_value(candidate, [existing], focus_terms=focus_terms)
                    ):
                        return True
                    if (
                        existing_sources.issubset(candidate_sources)
                        and len(candidate_sources) > len(existing_sources)
                        and not self._record_adds_unique_value(existing, [candidate], focus_terms=focus_terms)
                    ):
                        return True

            if self._looks_like_near_duplicate(candidate.data, existing.data):
                return True

        return False

    def _looks_like_near_duplicate(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        left_text = self._candidate_text_blob(left)
        right_text = self._candidate_text_blob(right)
        if not left_text or not right_text:
            return False

        left_norm = str(left.get("normalized_text") or left.get("semantic_text") or "").strip()
        right_norm = str(right.get("normalized_text") or right.get("semantic_text") or "").strip()
        semantic_ratio = SequenceMatcher(None, left_text, right_text).ratio()
        norm_ratio = SequenceMatcher(None, left_norm, right_norm).ratio() if left_norm and right_norm else 0.0
        max_ratio = max(semantic_ratio, norm_ratio)

        left_entities = {str(v or "").strip() for v in (left.get("entities") or []) if str(v or "").strip()}
        right_entities = {str(v or "").strip() for v in (right.get("entities") or []) if str(v or "").strip()}
        entity_overlap = 0.0
        if left_entities and right_entities:
            entity_overlap = len(left_entities & right_entities) / max(1, len(left_entities | right_entities))

        left_type = str(left.get("memory_type") or "")
        right_type = str(right.get("memory_type") or "")
        if max_ratio >= 0.94:
            return True
        if max_ratio >= 0.88 and entity_overlap >= 0.5:
            return True
        if left_type == right_type and max_ratio >= 0.84 and entity_overlap >= 0.7:
            return True

        return False

    @staticmethod
    def _extract_structured_tokens(text: str) -> set[str]:
        if not text:
            return set()
        return {
            token
            for token in re.findall(r"\b(?:\d{2,}|[A-Za-z_][A-Za-z0-9_.:-]{1,})\b", text)
            if token
        }

    @staticmethod
    def _candidate_text_blob(data: dict[str, Any]) -> str:
        return " ".join(
            part
            for part in [
                str(data.get("normalized_text") or "").strip(),
                str(data.get("semantic_text") or "").strip(),
                " ".join(str(v) for v in (data.get("entities") or []) if str(v or "").strip()),
                " ".join(str(v) for v in (data.get("task_tags") or []) if str(v or "").strip()),
                " ".join(str(v) for v in (data.get("tool_tags") or []) if str(v or "").strip()),
                " ".join(str(v) for v in (data.get("constraint_tags") or []) if str(v or "").strip()),
                " ".join(str(v) for v in (data.get("failure_tags") or []) if str(v or "").strip()),
                " ".join(str(v) for v in (data.get("affordance_tags") or []) if str(v or "").strip()),
            ]
            if part
        )

    @staticmethod
    def _candidate_source_record_set(data: dict[str, Any]) -> set[str]:
        return {
            str(value or "").strip()
            for value in (data.get("source_record_ids") or [])
            if str(value or "").strip()
        }

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

        # Step 2 分桶提前确定，供 Step 1 和 Step 2 共用
        # 取最近 4 轮作为当前轮（供编码器做指代消解），其余作为上文
        previous = turns[:-4] if len(turns) > 4 else []
        current = turns[-4:] if len(turns) > 4 else turns

        # Step 1: 新颖性检查
        # 只检查最近一次 user-assistant 交换（最后 2 条），避免历史已固化轮次
        # 的重复内容干扰判断，导致"本轮新信息"被误判为"已有记忆覆盖"。
        last_exchange = turns[-2:] if len(turns) >= 2 else turns
        has_novelty = self._check_novelty(last_exchange, retrieved_context)
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
        sensitive_skipped = 0
        for record in new_records:
            # 3a. 敏感信息过滤：密码/API key/私钥/信用卡号不写入长期记忆
            check_text = record.semantic_text + " " + record.normalized_text
            if self._is_sensitive_text(check_text):
                import logging as _logging
                _logging.getLogger("src.memory.semantic.engine").warning(
                    "Skipping sensitive content record %s (memory_type=%s)",
                    record.record_id, record.memory_type,
                )
                sensitive_skipped += 1
                continue

            # 3b. FTS 候选 + 向量候选合并去重
            similar = self._sqlite.find_similar_by_normalized_text(
                record.normalized_text, user_id=user_id, limit=3,
            )
            # 向量补充去重：FTS 对中文近重复召回弱，用 normalized_vector 兜底
            try:
                vec_hits = self._vector.search(
                    record.normalized_text,
                    user_id=user_id,
                    column="normalized_vector",
                    limit=5,
                )
                seen_dedup_ids = {s.record_id for s in similar}
                for vh in vec_hits:
                    rid = vh.get("record_id", "")
                    if rid and rid not in seen_dedup_ids:
                        full_rec = self._sqlite.get_record(rid)
                        if full_rec and not full_rec.expired:
                            similar.append(full_rec)
                            seen_dedup_ids.add(rid)
            except Exception:
                pass

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

        sensitive_note = f"，{sensitive_skipped} 条含敏感信息跳过" if sensitive_skipped else ""
        steps.append({
            "name": "dedup_and_store",
            "status": "done",
            "detail": f"去重后写入 {actually_added}/{len(new_records)} 条{sensitive_note}",
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

    def finalize_usage_log(
        self,
        *,
        log_id: str,
        final_response_excerpt: str = "",
    ) -> None:
        if not log_id:
            return
        excerpt = str(final_response_excerpt or "").strip()
        if len(excerpt) > 1000:
            excerpt = excerpt[:1000] + "…"
        self._sqlite.update_usage_log(
            log_id,
            final_response_excerpt=excerpt,
        )

    def apply_feedback_from_user_turn(
        self,
        *,
        session_id: str,
        user_turn: str,
        user_id: str = "",
    ) -> dict[str, Any]:
        if not session_id or not str(user_turn or "").strip():
            return {}

        logs = self._sqlite.get_recent_usage_logs(session_id=session_id, limit=10)
        pending_log = None
        for log in logs:
            if user_id and log.user_id and log.user_id != user_id:
                continue
            mode = str((log.retrieval_plan or {}).get("mode") or "").strip().lower()
            if mode not in {"action", "mixed"}:
                continue
            outcome = str(log.action_outcome or "").strip().lower()
            if outcome and outcome != "unknown":
                continue
            pending_log = log
            break

        if pending_log is None:
            return {}

        feedback = self._classify_feedback_from_user_turn(user_turn)
        if feedback["outcome"] == "unknown":
            return {}

        self._sqlite.update_usage_log(
            pending_log.log_id,
            user_feedback=feedback["feedback"],
            action_outcome=feedback["outcome"],
        )
        if feedback["outcome"] == "success":
            self._sqlite.increment_action_success(pending_log.kept_record_ids)
        elif feedback["outcome"] == "fail":
            self._sqlite.increment_action_fail(pending_log.kept_record_ids)

        return {
            "log_id": pending_log.log_id,
            "outcome": feedback["outcome"],
            "feedback": feedback["feedback"],
        }

    # ════════════════════════════════════════════════════════════════
    # 内部工具方法
    # ════════════════════════════════════════════════════════════════

    def _check_adequacy(
        self,
        query: str,
        context_text: str,
        *,
        plan: SearchPlan,
        action_state: ActionState,
    ) -> dict[str, Any]:
        """LLM 判断当前检索结果是否足以回应查询/支撑当前动作。

        Returns:
            {
                "is_sufficient": bool,
                "missing_info": str,
                "missing_constraints": list[str],
                "missing_slots": list[str],
                "missing_affordances": list[str],
                "needs_failure_avoidance": bool,
                "needs_tool_selection_basis": bool,
            }
        """
        user_content = (
            f"<USER_QUERY>\n{query}\n</USER_QUERY>\n\n"
            f"<SEARCH_PLAN>\n{json.dumps(self._plan_to_dict(plan), ensure_ascii=False)}\n</SEARCH_PLAN>\n\n"
            f"<ACTION_STATE>\n{json.dumps(self._action_state_to_dict(action_state), ensure_ascii=False)}\n</ACTION_STATE>\n\n"
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
                "missing_constraints": [
                    str(x) for x in (parsed.get("missing_constraints") or []) if str(x or "").strip()
                ],
                "missing_slots": [
                    str(x) for x in (parsed.get("missing_slots") or []) if str(x or "").strip()
                ],
                "missing_affordances": [
                    str(x) for x in (parsed.get("missing_affordances") or []) if str(x or "").strip()
                ],
                "needs_failure_avoidance": bool(parsed.get("needs_failure_avoidance", False)),
                "needs_tool_selection_basis": bool(parsed.get("needs_tool_selection_basis", False)),
            }
        except (json.JSONDecodeError, ValueError):
            # 解析失败保守地视为充分，避免无限循环
            return {
                "is_sufficient": True,
                "missing_info": "",
                "missing_constraints": [],
                "missing_slots": [],
                "missing_affordances": [],
                "needs_failure_avoidance": False,
                "needs_tool_selection_basis": False,
            }

    def _generate_additional_queries(
        self,
        query: str,
        context_text: str,
        *,
        adequacy: dict[str, Any],
        plan: SearchPlan,
        action_state: ActionState,
    ) -> dict[str, Any]:
        """LLM 根据 action-grounded 缺失信息生成补充检索查询和补充计划。

        Returns:
            dict，包含 semantic/pragmatic queries 以及需补充的约束/slot/affordance。
        """
        missing_info = str(adequacy.get("missing_info", ""))
        user_content = (
            f"<USER_QUERY>\n{query}\n</USER_QUERY>\n\n"
            f"<SEARCH_PLAN>\n{json.dumps(self._plan_to_dict(plan), ensure_ascii=False)}\n</SEARCH_PLAN>\n\n"
            f"<ACTION_STATE>\n{json.dumps(self._action_state_to_dict(action_state), ensure_ascii=False)}\n</ACTION_STATE>\n\n"
            f"<CURRENT_MEMORY>\n{context_text}\n</CURRENT_MEMORY>\n\n"
            f"<MISSING_INFO>\n{missing_info}\n</MISSING_INFO>\n\n"
            f"<MISSING_CONSTRAINTS>\n{json.dumps(adequacy.get('missing_constraints', []), ensure_ascii=False)}\n</MISSING_CONSTRAINTS>\n\n"
            f"<MISSING_SLOTS>\n{json.dumps(adequacy.get('missing_slots', []), ensure_ascii=False)}\n</MISSING_SLOTS>\n\n"
            f"<MISSING_AFFORDANCES>\n{json.dumps(adequacy.get('missing_affordances', []), ensure_ascii=False)}\n</MISSING_AFFORDANCES>\n\n"
            f"<NEEDS_FAILURE_AVOIDANCE>\n{json.dumps(bool(adequacy.get('needs_failure_avoidance', False)), ensure_ascii=False)}\n</NEEDS_FAILURE_AVOIDANCE>\n\n"
            f"<NEEDS_TOOL_SELECTION_BASIS>\n{json.dumps(bool(adequacy.get('needs_tool_selection_basis', False)), ensure_ascii=False)}\n</NEEDS_TOOL_SELECTION_BASIS>"
        )
        response = self._llm.generate([
            {"role": "system", "content": RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM},
            {"role": "user", "content": user_content},
        ])
        try:
            parsed = json.loads(
                response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            )
            return {
                "semantic_queries": [
                    q for q in (parsed.get("semantic_queries") or [])
                    if isinstance(q, str) and q.strip()
                ],
                "pragmatic_queries": [
                    q for q in (parsed.get("pragmatic_queries") or [])
                    if isinstance(q, str) and q.strip()
                ],
                "tool_hints": [
                    q for q in (parsed.get("tool_hints") or [])
                    if isinstance(q, str) and q.strip()
                ],
                "required_constraints": [
                    q for q in (parsed.get("required_constraints") or [])
                    if isinstance(q, str) and q.strip()
                ],
                "required_affordances": [
                    q for q in (parsed.get("required_affordances") or [])
                    if isinstance(q, str) and q.strip()
                ],
                "missing_slots": [
                    q for q in (parsed.get("missing_slots") or [])
                    if isinstance(q, str) and q.strip()
                ],
            }
        except (json.JSONDecodeError, ValueError):
            pass
        return {
            "semantic_queries": [],
            "pragmatic_queries": [],
            "tool_hints": [],
            "required_constraints": [],
            "required_affordances": [],
            "missing_slots": [],
        }

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
            required_affordances=d.get("required_affordances", []),
            missing_slots=d.get("missing_slots", []),
            depth=int(d.get("depth", 5)),
            reasoning=d.get("reasoning", ""),
        )

    @staticmethod
    def _dict_to_action_state(data: dict[str, Any] | None) -> ActionState:
        if not isinstance(data, dict):
            return ActionState()
        return ActionState(
            current_subgoal=str(data.get("current_subgoal") or ""),
            tentative_action=str(data.get("tentative_action") or ""),
            last_tool_name=str(data.get("last_tool_name") or ""),
            last_tool_result=str(data.get("last_tool_result") or ""),
            missing_slots=[str(x) for x in (data.get("missing_slots") or []) if str(x or "").strip()],
            known_constraints=[str(x) for x in (data.get("known_constraints") or []) if str(x or "").strip()],
            available_tools=[str(x) for x in (data.get("available_tools") or []) if str(x or "").strip()],
            failure_signal=str(data.get("failure_signal") or ""),
            token_budget=int(data.get("token_budget") or 0),
            recent_context_excerpt=str(data.get("recent_context_excerpt") or ""),
        )

    @staticmethod
    def _action_state_to_dict(action_state: ActionState) -> dict[str, Any]:
        return {
            "current_subgoal": action_state.current_subgoal,
            "tentative_action": action_state.tentative_action,
            "last_tool_name": action_state.last_tool_name,
            "last_tool_result": action_state.last_tool_result,
            "missing_slots": list(action_state.missing_slots),
            "known_constraints": list(action_state.known_constraints),
            "available_tools": list(action_state.available_tools),
            "failure_signal": action_state.failure_signal,
            "token_budget": action_state.token_budget,
            "recent_context_excerpt": action_state.recent_context_excerpt,
        }

    @staticmethod
    def _plan_to_dict(plan: SearchPlan) -> dict[str, Any]:
        return {
            "mode": plan.mode,
            "semantic_queries": list(plan.semantic_queries),
            "pragmatic_queries": list(plan.pragmatic_queries),
            "temporal_filter": plan.temporal_filter,
            "tool_hints": list(plan.tool_hints),
            "required_constraints": list(plan.required_constraints),
            "required_affordances": list(plan.required_affordances),
            "missing_slots": list(plan.missing_slots),
            "depth": plan.depth,
            "reasoning": plan.reasoning,
        }

    @staticmethod
    def _merge_action_state_with_plan(
        action_state: ActionState,
        plan: SearchPlan,
    ) -> ActionState:
        return ActionState(
            current_subgoal=action_state.current_subgoal,
            tentative_action=action_state.tentative_action,
            last_tool_name=action_state.last_tool_name,
            last_tool_result=action_state.last_tool_result,
            missing_slots=CompactSemanticEngine._merge_unique(
                action_state.missing_slots,
                plan.missing_slots,
            ),
            known_constraints=CompactSemanticEngine._merge_unique(
                action_state.known_constraints,
                plan.required_constraints,
            ),
            available_tools=CompactSemanticEngine._merge_unique(
                action_state.available_tools,
                plan.tool_hints,
            ),
            failure_signal=action_state.failure_signal,
            token_budget=action_state.token_budget,
            recent_context_excerpt=action_state.recent_context_excerpt,
        )

    @staticmethod
    def _merge_unique(primary: list[str], secondary: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for value in list(primary or []) + list(secondary or []):
            item = str(value or "").strip()
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    @staticmethod
    def _derive_record_type_bias(
        *,
        plan: SearchPlan,
        action_state: ActionState,
        adequacy: dict[str, Any] | None = None,
    ) -> list[str]:
        adequacy = adequacy or {}
        bias: list[str] = []

        if plan.mode in {"action", "mixed"}:
            bias.extend([MEMORY_TYPE_PROCEDURE, MEMORY_TYPE_CONSTRAINT])

        missing_constraints = adequacy.get("missing_constraints") or plan.required_constraints
        if missing_constraints:
            bias.extend([MEMORY_TYPE_CONSTRAINT, MEMORY_TYPE_PROCEDURE])

        missing_slots = adequacy.get("missing_slots") or plan.missing_slots or action_state.missing_slots
        if missing_slots:
            bias.extend([MEMORY_TYPE_PROCEDURE, MEMORY_TYPE_TOOL_AFFORDANCE])

        missing_affordances = adequacy.get("missing_affordances") or plan.required_affordances
        if missing_affordances or adequacy.get("needs_tool_selection_basis"):
            bias.extend([MEMORY_TYPE_TOOL_AFFORDANCE, MEMORY_TYPE_PROCEDURE, MEMORY_TYPE_CONSTRAINT])

        if adequacy.get("needs_failure_avoidance") or action_state.failure_signal:
            bias.extend([MEMORY_TYPE_FAILURE_PATTERN, MEMORY_TYPE_CONSTRAINT, MEMORY_TYPE_PROCEDURE])

        return CompactSemanticEngine._merge_unique(bias, [])

    @staticmethod
    def _derive_synth_type_bias(
        *,
        plan: SearchPlan,
        action_state: ActionState,
        adequacy: dict[str, Any] | None = None,
    ) -> list[str]:
        adequacy = adequacy or {}
        bias: list[str] = []

        if plan.mode in {"action", "mixed"}:
            bias.extend([SYNTH_TYPE_USAGE, SYNTH_TYPE_CONSTRAINT])

        missing_constraints = adequacy.get("missing_constraints") or plan.required_constraints
        if missing_constraints:
            bias.extend([SYNTH_TYPE_CONSTRAINT, SYNTH_TYPE_USAGE])

        missing_slots = adequacy.get("missing_slots") or plan.missing_slots or action_state.missing_slots
        if missing_slots:
            bias.extend([SYNTH_TYPE_USAGE, SYNTH_TYPE_PATTERN])

        missing_affordances = adequacy.get("missing_affordances") or plan.required_affordances
        if missing_affordances or adequacy.get("needs_tool_selection_basis"):
            bias.extend([SYNTH_TYPE_USAGE, SYNTH_TYPE_PATTERN])

        if adequacy.get("needs_failure_avoidance") or action_state.failure_signal:
            bias.extend([SYNTH_TYPE_PATTERN, SYNTH_TYPE_CONSTRAINT, SYNTH_TYPE_USAGE])

        return CompactSemanticEngine._merge_unique(bias, [])

    @staticmethod
    def _build_reflection_scoring_context(
        *,
        adequacy: dict[str, Any],
        action_state: ActionState,
    ) -> dict[str, Any]:
        return {
            "missing_constraints": list(adequacy.get("missing_constraints") or []),
            "missing_slots": list(adequacy.get("missing_slots") or []),
            "missing_affordances": list(adequacy.get("missing_affordances") or []),
            "needs_failure_avoidance": bool(adequacy.get("needs_failure_avoidance", False)),
            "needs_tool_selection_basis": bool(adequacy.get("needs_tool_selection_basis", False)),
            "available_tools": list(action_state.available_tools),
        }

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
    def _synth_to_candidate(record: CompositeRecord) -> dict[str, Any]:
        """CompositeRecord → scorer 需要的 candidate dict。"""
        return {
            "id": record.composite_id,
            "source": "composite",
            "semantic_distance": 0.5,
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
            "evidence_turn_range": [],
            "temporal": record.temporal,
            "entities": record.entities,
            "source_record_ids": list(record.source_record_ids),
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
                "semantic_source_type": sc.source,
                "score": sc.final_score,
                "score_breakdown": sc.score_breakdown,
                "semantic_text": d.get("semantic_text", ""),
                "entities": d.get("entities", []),
                "source_record_ids": d.get("source_record_ids", []),
            })
        return provenance

    def _log_usage(
        self,
        *,
        query: str,
        session_id: str,
        user_id: str,
        plan: SearchPlan,
        action_state: ActionState,
        retrieved_ids: list[str],
        kept_ids: list[str],
    ) -> str:
        """记录一次检索使用日志。"""
        log_id = uuid.uuid4().hex
        log = UsageLog(
            log_id=log_id,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            retrieval_plan={
                "mode": plan.mode,
                "semantic_queries": plan.semantic_queries,
                "pragmatic_queries": plan.pragmatic_queries,
                "required_constraints": plan.required_constraints,
                "required_affordances": plan.required_affordances,
                "missing_slots": plan.missing_slots,
                "depth": plan.depth,
            },
            action_state=self._action_state_to_dict(action_state),
            retrieved_record_ids=retrieved_ids,
            kept_record_ids=kept_ids,
        )
        self._sqlite.insert_usage_log(log)
        return log_id

    @staticmethod
    def _classify_feedback_from_user_turn(user_turn: str) -> dict[str, str]:
        text = str(user_turn or "").strip().lower()
        if not text:
            return {"feedback": "", "outcome": "unknown"}

        positive_markers = (
            "好了", "可以了", "搞定", "成功", "解决了", "生效了", "没问题了",
            "可以用了", "worked", "that works", "resolved", "fixed",
        )
        negative_markers = (
            "还是不行", "不行", "失败", "报错", "错误", "异常", "没生效",
            "不对", "有问题", "超时", "冲突", "无法", "崩溃", "failed",
            "error", "wrong", "not work", "doesn't work",
        )
        correction_markers = ("不是", "不对", "更正", "应该是", "改成", "其实")

        if any(marker in text for marker in correction_markers):
            return {"feedback": "correction", "outcome": "fail"}
        if any(marker in text for marker in negative_markers):
            return {"feedback": "negative", "outcome": "fail"}
        if any(marker in text for marker in positive_markers):
            return {"feedback": "positive", "outcome": "success"}
        return {"feedback": "", "outcome": "unknown"}
