"""Compact Semantic Engine（总装引擎）。

实现 BaseSemanticMemoryEngine，串联所有子模块：
- search(): 三阶段 LLM 驱动检索（Composite 过滤 → 记忆树下钻 → 反思 + FTS/向量补召回）
- ingest_conversation(): Novelty Check → Encoder → 去重写入 → Fusion

这是整个 Compact Semantic Memory 的核心入口。
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid

from src.llm.base import set_llm_call_source
from collections import deque
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
from src.memory.semantic.prompts import (
    COMPOSITE_FILTER_SYSTEM,
    NOVELTY_CHECK_SYSTEM,
    RETRIEVAL_ADEQUACY_CHECK_SYSTEM,
    RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM,
    FEEDBACK_CLASSIFICATION_SYSTEM,
)
from src.evolve.prompt_registry import get_prompt
from src.memory.semantic.scorer import ScoredCandidate, ScoringWeights
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
        session_store: Any | None = None,
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
        self._session_store = session_store

        # 存储层
        self._sqlite = SQLiteSemanticStore(db_path=sqlite_db_path)
        self._vector = LanceVectorIndex(
            db_path=vector_db_path, embedder=embedder, embedding_dim=embedding_dim
        )

        # 子模块
        self._encoder = CompactSemanticEncoder(llm=llm)
        self._synthesizer = RecordFusionEngine(
            sqlite_store=self._sqlite,
            vector_index=self._vector,
            synthesis_similarity=synthesis_similarity,
            dedup_threshold=dedup_threshold,
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
    # search() — 三阶段 LLM 驱动检索管线
    # ════════════════════════════════════════════════════════════════

    def search(
        self,
        *,
        query: str,
        session_id: str | None = None,
        top_k: int = 0,
        query_embedding: list[float] | None = None,
        recent_context: str = "",
        action_state: dict[str, Any] | None = None,
        retrieval_plan: dict[str, Any] | None = None,
        prompt_versions_used: dict[str, int] | None = None,
    ) -> SemanticSearchResult:
        """三阶段 LLM 驱动检索（已替换旧的多通道打分管线）。

        Phase 1: LLM 过滤全量 CompositeRecord（分块，每块 20 条）
        Phase 2: 对标记为 needs_detail 的 composite 下钻到叶子 MemoryRecord
        Phase 3: 反思循环 — LLM 判断当前记忆是否充分；若不充分则 FTS + 向量
                 检索补充 MemoryRecord + 原始对话 turns
        """
        action_state_obj = self._dict_to_action_state(action_state)
        top_k = max(1, int(top_k or 5))

        if retrieval_plan:
            plan = self._dict_to_plan(retrieval_plan)
        else:
            plan = SearchPlan(
                mode="answer",
                semantic_queries=[query],
                pragmatic_queries=[],
                depth=top_k,
                include_episodic_context=True,
            )

        selected_candidates: list[ScoredCandidate] = []
        seen_ids: set[str] = set()
        adequacy_history: list[dict[str, Any]] = []

        # ─── Phase 1: ANN 预过滤 + LLM 过滤 CompositeRecord ─────
        # 先用向量 ANN 检索 top-20 相关 composites，再做一次 LLM 过滤
        ann_composite_results = self._vector.search_synthesized(
            query, column="vector", limit=20,
        )
        ann_composite_ids = [
            str(r.get("composite_id", "")).strip()
            for r in ann_composite_results
            if str(r.get("composite_id", "")).strip()
        ]
        ann_composites: list[CompositeRecord] = []
        for cid in ann_composite_ids:
            comp = self._sqlite.get_synthesized(cid)
            if comp:
                ann_composites.append(comp)

        needs_detail_ids: set[str] = set()

        if ann_composites:
            sel_ids, det_ids = self._llm_filter_composites(
                query=query,
                recent_context=recent_context,
                composites=ann_composites,
            )
            for composite in ann_composites:
                cid = composite.composite_id
                if cid in sel_ids and cid not in seen_ids:
                    seen_ids.add(cid)
                    data = self._synth_to_candidate(composite)
                    selected_candidates.append(ScoredCandidate(
                        id=cid,
                        source="composite",
                        final_score=1.0,
                        score_breakdown={"llm_selected": 1.0},
                        data=data,
                    ))
                if cid in det_ids:
                    needs_detail_ids.add(cid)

        # ─── Phase 2: 下钻叶子 MemoryRecord ──────────────────────
        for composite_id in list(needs_detail_ids):
            leaf_ids = self._collect_leaf_record_ids(composite_id)
            records = self._sqlite.get_records_by_ids(leaf_ids)
            for record in records:
                if record.expired:
                    continue
                rid = record.record_id
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                data = self._record_to_candidate(record)
                selected_candidates.append(ScoredCandidate(
                    id=rid,
                    source="record",
                    final_score=0.9,
                    score_breakdown={"tree_descent": 1.0},
                    data=data,
                ))

        # ─── Phase 3: 反思循环 ────────────────────────────────────
        for _round in range(self._max_reflection_rounds):
            context_preview = (
                self._format_context(selected_candidates) or "(no retrieved results)"
            )
            adequacy = self._check_adequacy(
                query,
                context_preview,
                plan=plan,
                action_state=action_state_obj,
            )
            adequacy_history.append({
                "round": int(_round),
                "is_sufficient": bool(adequacy.get("is_sufficient", True)),
                "missing_info": str(adequacy.get("missing_info", ""))[:200],
            })
            if adequacy["is_sufficient"]:
                break

            # Fallback: FTS + 语义向量 + 实用向量检索 MemoryRecord + 原始 turns
            fallback_candidates = self._fallback_record_search(
                query=query,
                top_k=max(top_k, 10),
            )
            raw_turn_candidates = self._search_raw_turns_direct(
                query=query,
                session_id=session_id,
                top_k=max(top_k, 10),
            )
            new_added = 0
            for cdata in fallback_candidates:
                cid = str(cdata.get("id", ""))
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)
                selected_candidates.append(ScoredCandidate(
                    id=cid,
                    source=cdata.get("source", "record"),
                    final_score=0.7,
                    score_breakdown={"fallback_search": 1.0},
                    data=cdata,
                ))
                new_added += 1

            for cdata in raw_turn_candidates:
                cid = str(cdata.get("id", ""))
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)
                selected_candidates.append(ScoredCandidate(
                    id=cid,
                    source="episode",
                    final_score=0.65,
                    score_breakdown={"fallback_raw_turn": 1.0},
                    data=cdata,
                ))
                new_added += 1

            if not new_added:
                break

        # 情节上下文增强：强制开启，且保证至少 window=1
        # 这样即使 answer 模式下 _normalize_tree_retrieval_plan 将 window 设为 0，
        # 此处也会回溯每条已召回记录的原始对话 turns（含相邻 1 条 turn），
        # 从而补偿 encoder 在抽取时丢失的细节。
        plan.include_episodic_context = True
        plan.episodic_turn_window = max(plan.episodic_turn_window, 1)
        self._enrich_candidates_with_episodic_context(
            selected_candidates,
            plan=plan,
            focus_terms=[],
        )

        # 按 final_score 降序排列，取 top_k
        selected_candidates.sort(key=lambda c: c.final_score, reverse=True)
        top = selected_candidates[:top_k]

        # 记录使用日志
        log_id = self._log_usage(
            query=query,
            session_id=session_id or "",
            plan=plan,
            action_state=action_state_obj,
            retrieved_ids=[c.id for c in selected_candidates],
            kept_ids=[c.id for c in top],
            prompt_versions_used=prompt_versions_used or {},
        )

        # 更新统计（仅对 SQLite 中存储的 record / composite）
        sqlite_ids = [
            c.id for c in selected_candidates
            if c.source in {"record", "composite"}
        ]
        self._sqlite.increment_retrieval_count(sqlite_ids)
        self._sqlite.increment_hit_count([
            c.id for c in top if c.source in {"record", "composite"}
        ])

        # 格式化
        context = self._format_context(top)
        provenance = self._build_provenance(top)

        return SemanticSearchResult(
            context=context,
            provenance=provenance,
            retrieval_plan=self._plan_to_dict(plan),
            action_state=self._action_state_to_dict(action_state_obj),
            usage_log_id=log_id,
            mode=plan.mode,
            diagnostics={
                "adequacy_history": adequacy_history,
                "reflection_rounds": len(adequacy_history),
                "final_is_sufficient": bool(adequacy_history[-1]["is_sufficient"]) if adequacy_history else True,
                "max_reflection_rounds": int(self._max_reflection_rounds),
            },
        )

    def _llm_filter_composites(
        self,
        *,
        query: str,
        recent_context: str,
        composites: list[CompositeRecord],
    ) -> tuple[set[str], set[str]]:
        """LLM 过滤一批 CompositeRecord，返回 (selected_ids, needs_detail_ids)。

        LLM 根据 query 判断哪些 composite 相关（selected_ids），
        以及哪些需要展开到叶子 MemoryRecord（needs_detail_ids ⊆ selected_ids）。
        解析失败时保守返回空集，避免影响后续流程。
        """
        if not composites:
            return set(), set()

        lines: list[str] = []
        for composite in composites:
            entities_str = ", ".join(composite.entities[:6]) if composite.entities else "-"
            summary = str(composite.normalized_text or composite.semantic_text or "").strip()
            if len(summary) > 300:
                summary = summary[:300] + "…"
            lines.append(
                f"id={composite.composite_id}\n"
                f"  type: {composite.memory_type}\n"
                f"  summary: {summary}\n"
                f"  entities: [{entities_str}]"
            )

        composites_text = "\n---\n".join(lines)
        user_content = f"<USER_QUERY>\n{query}\n</USER_QUERY>\n\n"
        if recent_context:
            user_content += f"<RECENT_CONTEXT>\n{recent_context}\n</RECENT_CONTEXT>\n\n"
        user_content += f"<MEMORY_SUMMARIES>\n{composites_text}\n</MEMORY_SUMMARIES>"

        set_llm_call_source("composite_filter")
        response = self._llm.generate([
            {"role": "system", "content": get_prompt("composite_filter", COMPOSITE_FILTER_SYSTEM)},
            {"role": "user", "content": user_content},
        ])

        try:
            raw = response.strip()
            # 剥除可能存在的 markdown 代码块标记
            if raw.startswith("```"):
                parts = raw.split("```")
                # parts[1] is the content between first and second ```
                raw = parts[1].lstrip("json").strip() if len(parts) >= 3 else parts[-1].strip()
            parsed = json.loads(raw)
            selected_ids = {
                str(v or "").strip()
                for v in (parsed.get("selected_ids") or [])
                if str(v or "").strip()
            }
            needs_detail_ids = {
                str(v or "").strip()
                for v in (parsed.get("needs_detail") or [])
                if str(v or "").strip()
            } & selected_ids  # needs_detail 必须是 selected 的子集
            return selected_ids, needs_detail_ids
        except (json.JSONDecodeError, ValueError):
            return set(), set()

    def _collect_leaf_record_ids(self, composite_id: str) -> list[str]:
        """递归收集从给定 CompositeRecord 向下到叶子的全部 MemoryRecord ID。

        遍历 source_record_ids（直接叶子）和 child_composite_ids（子 composite 树）。
        使用 BFS 防止循环引用。
        """
        leaf_ids: set[str] = set()
        visited: set[str] = set()
        queue = [composite_id]
        while queue:
            cid = queue.pop()
            if cid in visited:
                continue
            visited.add(cid)
            composite = self._sqlite.get_synthesized(cid)
            if composite is None:
                continue
            for record_id in (composite.source_record_ids or []):
                rid = str(record_id or "").strip()
                if rid:
                    leaf_ids.add(rid)
            for child_id in (composite.child_composite_ids or []):
                child_id = str(child_id or "").strip()
                if child_id and child_id not in visited:
                    queue.append(child_id)
        return list(leaf_ids)

    def _fallback_record_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """反思 fallback：FTS + 语义向量 + 实用向量召回 MemoryRecord。

        用于反思循环中当前记忆不充分时的补充检索。
        """
        seen_ids: set[str] = set()
        candidates: list[dict[str, Any]] = []
        limit = max(top_k * 2, 20)

        # FTS（BM25 全文检索）
        for r in self._sqlite.fulltext_search(query, limit=limit):
            uid = str(r.get("record_id", "") or "").strip()
            if uid and uid not in seen_ids:
                seen_ids.add(uid)
                full = self._sqlite.get_record(uid)
                if full:
                    c = self._record_to_candidate(full)
                    c["semantic_distance"] = 1.0 - min(
                        1.0, abs(float(r.get("fts_score", 0))) / 20.0
                    )
                    candidates.append(c)

        # 语义向量检索
        try:
            for r in self._vector.search(query, column="vector", limit=limit):
                uid = str(r.get("record_id", "") or "").strip()
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    full = self._sqlite.get_record(uid)
                    if full:
                        c = self._record_to_candidate(full)
                        c["semantic_distance"] = float(r.get("_distance", 1.0))
                        candidates.append(c)
        except Exception:
            pass

        # 实用向量检索（normalized_vector）
        try:
            for r in self._vector.search(query, column="normalized_vector", limit=limit):
                uid = str(r.get("record_id", "") or "").strip()
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    full = self._sqlite.get_record(uid)
                    if full:
                        c = self._record_to_candidate(full)
                        c["semantic_distance"] = float(r.get("_distance", 1.0))
                        candidates.append(c)
        except Exception:
            pass

        return candidates

    # ════════════════════════════════════════════════════════════════
    # ingest_conversation() — 固化管线
    # ════════════════════════════════════════════════════════════════

    def ingest_conversation(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        retrieved_context: str = "",
        turn_index_offset: int = 0,
        reference_timestamp: str | None = None,
    ) -> ConsolidationResult:
        """完整固化流程：新颖性检查 → 编码 → 去重写入 → 合成。"""
        if not turns:
            return ConsolidationResult(
                records_added=0,
                records_merged=0,
                records_expired=0,
                has_novelty=False,
                steps=[],
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
                records_added=0,
                records_merged=0,
                records_expired=0,
                has_novelty=False,
                steps=steps,
            )

        # Step 2: Compact Encoding
        new_records = self._encoder.encode_conversation(
            current,
            previous_turns=previous,
            session_id=session_id,
            turn_index_offset=turn_index_offset + max(0, len(turns) - len(current)),
            session_date=reference_timestamp,
        )

        steps.append({
            "name": "compact_encoding",
            "status": "done",
            "detail": f"抽取 {len(new_records)} 条 MemoryRecord",
        })

        if not new_records:
            return ConsolidationResult(
                records_added=0,
                records_merged=0,
                records_expired=0,
                has_novelty=True,
                steps=steps,
            )

        # Step 3: 去重 + 写入
        actually_added = 0
        sensitive_skipped = 0
        persisted_records: list[MemoryRecord] = []
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
                record.semantic_text, limit=3,
            )
            # 向量补充去重：FTS 对中文近重复召回弱，用 vector（semantic_text）兜底
            try:
                vec_hits = self._vector.search(
                    record.semantic_text,
                    column="vector",
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
                # 用向量相似度判断重复（使用 semantic_text，保留完整语义）
                try:
                    vecs = self._embedder.embed([record.semantic_text, s.semantic_text])
                    sim = self._cosine_similarity(vecs[0], vecs[1])
                    if sim >= self._dedup_threshold:
                        is_duplicate = True
                        # 更新已有条目而非插入新的；合并 temporal，以新记录为准补充缺失字段
                        s.updated_at = datetime.now(timezone.utc).isoformat()
                        s.confidence = min(1.0, s.confidence + 0.1)
                        s.temporal = self._merge_temporal(s.temporal, record.temporal)
                        self._sqlite.upsert_record(s)
                        break
                except Exception:
                    pass

            if not is_duplicate:
                self._sqlite.upsert_record(record)
                try:
                    self._vector.upsert(
                        record_id=record.record_id,
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
                persisted_records.append(record)

        sensitive_note = f"，{sensitive_skipped} 条含敏感信息跳过" if sensitive_skipped else ""
        steps.append({
            "name": "dedup_and_store",
            "status": "done",
            "detail": f"去重后写入 {actually_added}/{len(new_records)} 条{sensitive_note}",
        })

        # Step 4: Record Fusion（在线聚合）
        composite_records = []
        fusion_stats: dict[str, Any] = {}
        if actually_added > 0:
            composite_records = self._synthesizer.synthesize_on_ingest(
                persisted_records,
            )
            fusion_stats = self._synthesizer.get_last_run_stats()

        expired_count = len(fusion_stats.get("expired_record_ids", []))
        updated_count = len(fusion_stats.get("updated_record_ids", []))
        invalidated_composite_count = len(fusion_stats.get("invalidated_composite_ids", []))
        fusion_detail = f"聚合 {len(composite_records)} 条 CompositeRecord"
        if updated_count or expired_count or invalidated_composite_count:
            extras: list[str] = []
            if updated_count:
                extras.append(f"更新 {updated_count} 条冲突旧记忆")
            if expired_count:
                extras.append(f"过期 {expired_count} 条冲突新记忆")
            if invalidated_composite_count:
                extras.append(f"失效 {invalidated_composite_count} 条受影响 composite")
            fusion_detail += "，" + "，".join(extras)

        steps.append({
            "name": "record_fusion",
            "status": "done",
            "detail": fusion_detail,
        })

        return ConsolidationResult(
            records_added=actually_added,
            records_merged=len(composite_records),
            records_expired=expired_count,
            has_novelty=True,
            steps=steps,
        )

    # ════════════════════════════════════════════════════════════════
    # delete / export
    # ════════════════════════════════════════════════════════════════

    def delete_all(self) -> dict[str, int]:
        result = self._sqlite.delete_all()
        self._vector.delete_all()
        return result

    def export_debug(self) -> dict[str, Any]:
        return self._sqlite.export_all()

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
    ) -> dict[str, Any]:
        if not session_id or not str(user_turn or "").strip():
            return {}

        logs = self._sqlite.get_recent_usage_logs(session_id=session_id, limit=10)
        pending_log = None
        for log in logs:
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
            "prompt_versions_used": dict(getattr(pending_log, "prompt_versions_used", {}) or {}),
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
        set_llm_call_source("adequacy_check")
        response = self._llm.generate([
            {"role": "system", "content": get_prompt("retrieval_adequacy_check", RETRIEVAL_ADEQUACY_CHECK_SYSTEM)},
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
            A dict containing semantic/pragmatic queries and any constraints / slots / affordances that still need to be supplemented.
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
        set_llm_call_source("additional_queries")
        response = self._llm.generate([
            {"role": "system", "content": get_prompt("retrieval_additional_queries", RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM)},
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
            f"<EXISTING_MEMORY>\n{retrieved_context or '(no existing memory)'}\n</EXISTING_MEMORY>\n\n"
            f"<CONVERSATION>\n{conversation_text}\n</CONVERSATION>"
        )
        set_llm_call_source("novelty_check")
        response = self._llm.generate([
            {"role": "system", "content": get_prompt("novelty_check", NOVELTY_CHECK_SYSTEM)},
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
            tree_retrieval_mode=str(d.get("tree_retrieval_mode", "balanced") or "balanced"),
            tree_expansion_depth=int(d.get("tree_expansion_depth", 1) or 0),
            include_leaf_records=bool(d.get("include_leaf_records", False)),
            include_episodic_context=bool(d.get("include_episodic_context", False)),
            episodic_turn_window=max(0, int(d.get("episodic_turn_window", 0) or 0)),
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
            "tree_retrieval_mode": plan.tree_retrieval_mode,
            "tree_expansion_depth": plan.tree_expansion_depth,
            "include_leaf_records": plan.include_leaf_records,
            "include_episodic_context": plan.include_episodic_context,
            "episodic_turn_window": plan.episodic_turn_window,
            "depth": plan.depth,
            "reasoning": plan.reasoning,
        }

    @staticmethod
    def _normalize_tree_retrieval_plan(
        plan: SearchPlan,
        action_state: ActionState,
        *,
        focus_terms: list[str] | None = None,
        adequacy: dict[str, Any] | None = None,
    ) -> SearchPlan:
        valid_modes = {"root_only", "balanced", "descend"}
        if plan.tree_retrieval_mode not in valid_modes:
            plan.tree_retrieval_mode = "balanced"

        adequacy = adequacy or {}
        detail_signals = any([
            focus_terms,
            plan.pragmatic_queries,
            plan.required_constraints,
            plan.required_affordances,
            plan.missing_slots,
            action_state.missing_slots,
            action_state.known_constraints,
            action_state.failure_signal,
            adequacy.get("missing_constraints"),
            adequacy.get("missing_affordances"),
            adequacy.get("missing_slots"),
            adequacy.get("needs_failure_avoidance"),
            adequacy.get("needs_tool_selection_basis"),
        ])

        if plan.mode == "answer" and not detail_signals:
            plan.tree_retrieval_mode = "root_only"
            plan.tree_expansion_depth = 0
            plan.include_leaf_records = False
            plan.include_episodic_context = False
            plan.episodic_turn_window = 0
            return plan

        if plan.mode == "mixed":
            if plan.tree_retrieval_mode == "root_only":
                plan.tree_retrieval_mode = "balanced"
            if detail_signals and adequacy.get("needs_failure_avoidance"):
                plan.tree_retrieval_mode = "descend"
            plan.tree_expansion_depth = max(
                1,
                min(
                    int(plan.tree_expansion_depth or 1),
                    3 if plan.tree_retrieval_mode == "descend" else 2,
                ),
            )
            plan.include_leaf_records = bool(plan.include_leaf_records or detail_signals)
            plan.include_episodic_context = bool(
                plan.include_episodic_context
                or plan.include_leaf_records
                or detail_signals
                or adequacy.get("missing_info")
            )
            plan.episodic_turn_window = max(0, min(int(plan.episodic_turn_window or 1), 2))
            return plan

        plan.tree_retrieval_mode = "descend"
        plan.tree_expansion_depth = max(2, min(int(plan.tree_expansion_depth or 2), 3))
        plan.include_leaf_records = True
        plan.include_episodic_context = True
        plan.episodic_turn_window = max(1, min(int(plan.episodic_turn_window or 1), 2))
        return plan

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
            "tags": record.tags,
            "created_at": record.created_at,
            "retrieval_count": record.retrieval_count,
            "retrieval_hit_count": record.retrieval_hit_count,
            "action_success_count": record.action_success_count,
            "action_fail_count": record.action_fail_count,
            "evidence_turn_range": record.evidence_turn_range,
            "source_session": record.source_session,
            "source_role": record.source_role,
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
            "tags": record.tags,
            "created_at": record.created_at,
            "retrieval_count": record.retrieval_count,
            "retrieval_hit_count": record.retrieval_hit_count,
            "action_success_count": record.action_success_count,
            "action_fail_count": record.action_fail_count,
            "evidence_turn_range": [],
            "temporal": record.temporal,
            "entities": record.entities,
            "source_record_ids": list(record.source_record_ids),
            "child_composite_ids": list(record.child_composite_ids),
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

    @staticmethod
    def _merge_temporal(
        existing: dict[str, str] | None,
        incoming: dict[str, str] | None,
    ) -> dict[str, str]:
        """合并两条记录的 temporal 字段：以 incoming 的非空值覆盖 existing 的空值。

        规则：
        - incoming 的任意非空字段填补 existing 中的空字段；
        - 若两者均非空且不同，incoming（较新）优先；
        - 始终返回含 t_ref / t_valid_from / t_valid_to 三个键的 dict。
        """
        base = {"t_ref": "", "t_valid_from": "", "t_valid_to": ""}
        for k in base:
            ev = str((existing or {}).get(k) or "").strip()
            iv = str((incoming or {}).get(k) or "").strip()
            # incoming 有值则覆盖（不管 existing 是否有值）；否则保留 existing
            base[k] = iv if iv else ev
        return base

    def index_unvectorized_turns(self) -> int:
        """将 session_store 中尚未向量化的 turns 写入 episode_turns 向量索引。

        在引擎启动后调用一次，对历史数据补全索引；后续每次对话写入新 turn 后
        也可增量调用。返回本次新索引的 turn 数量。
        """
        if self._session_store is None:
            return 0
        if not hasattr(self._session_store, "list_sessions"):
            return 0

        # 1. 获取已索引的 episode_ids（增量判断）
        indexed_ids: set[str] = self._vector.get_all_episode_ids()

        # 2. 遍历所有 session，收集未索引 turns
        batch: list[dict[str, Any]] = []
        try:
            sessions = self._session_store.list_sessions(offset=0, limit=100_000)
        except Exception:
            return 0

        for sess_info in sessions:
            sid = str(sess_info.get("session_id", "")).strip()
            if not sid:
                continue
            try:
                all_turns = self._session_store.get_turns(sid)
            except Exception:
                continue
            for idx, turn in enumerate(all_turns):
                ep_id = self._make_episode_id(sid, idx)
                if ep_id in indexed_ids:
                    continue
                content = str(turn.get("content", "")).strip()
                if not content or turn.get("deleted", False):
                    continue
                batch.append({
                    "episode_id": ep_id,
                    "session_id": sid,
                    "turn_index": idx,
                    "role": str(turn.get("role", "unknown")),
                    "content": content,
                    "created_at": str(turn.get("created_at", "")),
                })

        if not batch:
            return 0

        # 3. 分批写入（每批 8 条，符合部分 embedding API 的批量限制）
        _CHUNK = 8
        total = 0
        for i in range(0, len(batch), _CHUNK):
            chunk = batch[i : i + _CHUNK]
            try:
                self._vector.upsert_turns_batch(chunk)
                total += len(chunk)
            except Exception:
                pass  # 局部失败继续下一批

        return total

    def _search_raw_turns_direct(
        self,
        query: str,
        session_id: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """在 episode_turns 向量索引中直接检索原始对话 turns。

        不依赖 MemoryRecord 作为中介锚点，可以召回尚未被提炼成记忆记录、
        或提炼质量不足的对话内容。反思循环 round 1+ 作为 fallback 通道使用。
        """
        if not str(query or "").strip():
            return []

        try:
            hits = self._vector.search_turns(
                query,
                limit=top_k,
                session_id=session_id,
            )
        except Exception:
            return []

        results: list[dict[str, Any]] = []
        for hit in hits:
            ep_id = str(hit.get("episode_id", ""))
            sid = str(hit.get("session_id", ""))
            idx = int(hit.get("turn_index", 0))
            content = str(hit.get("content", "")).strip()
            role = str(hit.get("role", "unknown"))
            distance = float(hit.get("_distance", 1.0))
            if not content:
                continue
            display = f"[{role}]: {content}"
            results.append({
                "id": ep_id,
                "source": "episode",
                "memory_type": "raw_turn",
                "semantic_text": display,
                "normalized_text": content,
                "display_text": display,
                "semantic_distance": min(1.0, max(0.0, distance)),
                "source_session": sid,
                "evidence_turn_range": [idx],
                "entities": [],
                "tags": [],
                "temporal": {},
                "created_at": str(hit.get("created_at", "")),
                "retrieval_count": 0,
                "retrieval_hit_count": 0,
                "action_success_count": 0,
                "action_fail_count": 0,
                "source_record_ids": [],
                "tree_parent_id": "",
                "tree_depth": 0,
            })
        return results

    def _enrich_candidates_with_episodic_context(
        self,
        scored: list[ScoredCandidate],
        *,
        plan: SearchPlan,
        focus_terms: list[str] | None = None,
    ) -> None:
        if not scored or not plan.include_episodic_context or self._session_store is None:
            return

        seen_episode_ids: set[str] = set()
        for candidate in scored:
            for ref in candidate.data.get("episode_refs") or []:
                if not isinstance(ref, dict):
                    continue
                episode_id = str(ref.get("episode_id") or "").strip()
                if episode_id:
                    seen_episode_ids.add(episode_id)

        for candidate in scored:
            self._attach_episodic_context(
                candidate.data,
                plan=plan,
                focus_terms=focus_terms,
                seen_episode_ids=seen_episode_ids,
            )

    def _attach_episodic_context(
        self,
        candidate_data: dict[str, Any],
        *,
        plan: SearchPlan,
        focus_terms: list[str] | None = None,
        seen_episode_ids: set[str] | None = None,
    ) -> None:
        if candidate_data.get("_episodic_attached"):
            return

        base_text = str(
            candidate_data.get("semantic_text")
            or candidate_data.get("normalized_text")
            or ""
        ).strip()
        episode_refs = self._collect_episode_refs_for_candidate(
            candidate_data,
            plan=plan,
            focus_terms=focus_terms,
            seen_episode_ids=seen_episode_ids,
        )
        episodic_context = self._render_episode_refs(episode_refs)

        candidate_data["episode_refs"] = episode_refs
        candidate_data["episodic_context"] = episodic_context
        candidate_data["display_text"] = base_text
        if episodic_context:
            candidate_data["display_text"] = (
                f"{base_text}\n\n[Original Dialogue Context]\n{episodic_context}"
                if base_text else episodic_context
            )
        candidate_data["_episodic_attached"] = True

    def _collect_episode_refs_for_candidate(
        self,
        candidate_data: dict[str, Any],
        *,
        plan: SearchPlan,
        focus_terms: list[str] | None = None,
        seen_episode_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self._session_store is None:
            return []

        source = str(candidate_data.get("source") or "").strip().lower()
        if source == "record":
            return self._episode_refs_for_record_data(
                candidate_data,
                window=plan.episodic_turn_window,
                seen_episode_ids=seen_episode_ids,
            )

        if source != "composite":
            return []

        record_ids = [
            str(record_id or "").strip()
            for record_id in (candidate_data.get("source_record_ids") or [])
            if str(record_id or "").strip()
        ]
        if not record_ids:
            return []

        ranked_records: list[tuple[int, dict[str, Any]]] = []
        for record_id in record_ids[:12]:
            record = self._sqlite.get_record(record_id)
            if record is None or record.expired:
                continue
            record_data = self._record_to_candidate(record)
            if not record_data.get("source_session") or not record_data.get("evidence_turn_range"):
                continue
            focus_score = self._episode_focus_score(record_data, focus_terms)
            ranked_records.append((focus_score, record_data))

        ranked_records = sorted(
            ranked_records,
            key=lambda item: (
                str(item[1].get("created_at") or ""),
                str(item[1].get("id") or ""),
            ),
            reverse=True,
        )
        ranked_records = sorted(
            ranked_records,
            key=lambda item: item[0],
            reverse=True,
        )

        # 对所有模式统一取 3 条叶子 record 的原始对话：
        # answer 模式之所以之前限制为 1，是担心 token 过多，但这导致 composite 只展示
        # 代表记录的 turn，而具体细节（书名、物品描述、精确数字等）往往在其他叶子
        # record 的原始对话里，造成细节特异性丢失。改为统一 3 条。
        max_records = 3
        episode_refs: list[dict[str, Any]] = []
        seen_turn_keys: set[tuple[str, int]] = set()
        for _, record_data in ranked_records[:max_records]:
            refs = self._episode_refs_for_record_data(
                record_data,
                window=plan.episodic_turn_window,
                seen_turn_keys=seen_turn_keys,
                seen_episode_ids=seen_episode_ids,
            )
            episode_refs.extend(refs)
            if len(episode_refs) >= 6:
                break

        return episode_refs[:6]

    def _episode_refs_for_record_data(
        self,
        record_data: dict[str, Any],
        *,
        window: int = 0,
        seen_turn_keys: set[tuple[str, int]] | None = None,
        seen_episode_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        session_id = str(record_data.get("source_session") or "").strip()
        evidence_turns = self._normalize_turn_indices(record_data.get("evidence_turn_range"))
        if not session_id or not evidence_turns or self._session_store is None:
            return []

        seen_turn_keys = seen_turn_keys if seen_turn_keys is not None else set()
        try:
            turns = self._session_store.get_turn_window(
                session_id,
                min(evidence_turns),
                max(evidence_turns),
                window=max(0, int(window or 0)),
            )
        except Exception:
            return []

        refs: list[dict[str, Any]] = []
        for turn in turns:
            try:
                turn_index = int(turn.get("turn_index"))
            except (TypeError, ValueError):
                continue
            episode_id = self._make_episode_id(session_id, turn_index)
            if seen_episode_ids is not None and episode_id in seen_episode_ids:
                continue
            turn_key = (session_id, turn_index)
            if turn_key in seen_turn_keys:
                continue
            seen_turn_keys.add(turn_key)
            if seen_episode_ids is not None:
                seen_episode_ids.add(episode_id)
            refs.append({
                "episode_id": episode_id,
                "session_id": session_id,
                "turn_index": turn_index,
                "role": str(turn.get("role") or "unknown"),
                "content": str(turn.get("content") or ""),
                "created_at": str(turn.get("created_at") or ""),
                "deleted": bool(turn.get("deleted", False)),
                "source_record_id": str(record_data.get("id") or ""),
            })
        return refs

    def _episode_focus_score(
        self,
        record_data: dict[str, Any],
        focus_terms: list[str] | None,
    ) -> int:
        if not focus_terms:
            return 0
        blob = " ".join(
            part
            for part in [
                str(record_data.get("normalized_text") or "").strip(),
                str(record_data.get("semantic_text") or "").strip(),
            ]
            if part
        ).lower()
        score = 0
        for term in focus_terms:
            normalized = str(term or "").strip().lower()
            if not normalized:
                continue
            if normalized in blob:
                score += 2
                continue
            tokens = [tok for tok in re.split(r"[\s/,_:：\-]+", normalized) if tok]
            if tokens and any(tok in blob for tok in tokens):
                score += 1
        return score

    @staticmethod
    def _normalize_turn_indices(raw_value: Any) -> list[int]:
        if not isinstance(raw_value, list):
            return []
        result: list[int] = []
        seen: set[int] = set()
        for raw in raw_value:
            try:
                index = int(raw)
            except (TypeError, ValueError):
                continue
            if index < 0 or index in seen:
                continue
            seen.add(index)
            result.append(index)
        result.sort()
        return result

    @staticmethod
    def _make_episode_id(session_id: str, turn_index: int) -> str:
        return f"episode:{session_id}:{int(turn_index)}"

    def _render_episode_refs(self, refs: list[dict[str, Any]]) -> str:
        if not refs:
            return ""
        lines: list[str] = []
        multi_session = len({str(ref.get("session_id") or "") for ref in refs}) > 1
        for ref in refs:
            session_id = str(ref.get("session_id") or "")
            session_prefix = f"[{session_id[:8]}]" if multi_session and session_id else ""
            turn_index = ref.get("turn_index")
            role = str(ref.get("role") or "unknown")
            created_at = self._format_time_label(str(ref.get("created_at") or ""))
            content = self._truncate_text(str(ref.get("content") or ""), max_chars=220)
            turn_prefix = f"t{turn_index}" if turn_index is not None else "t?"
            time_prefix = f"[{created_at}]" if created_at else ""
            lines.append(f"{session_prefix}{time_prefix}[{turn_prefix}][{role}] {content}")
        return "\n".join(lines)

    @staticmethod
    def _truncate_text(text: str, *, max_chars: int = 220) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1].rstrip() + "…"

    @staticmethod
    def _format_time_label(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        return re.sub(r"\.\d+(?=(?:Z|[+-]\d{2}:?\d{2})?$)", "", raw)

    def _format_context(self, scored: list[ScoredCandidate]) -> str:
        """将 top-k 候选格式化为 LLM 可注入的文本。"""
        if not scored:
            return ""

        parts: list[str] = []
        for i, sc in enumerate(scored, 1):
            d = sc.data
            mt = d.get("memory_type", "unknown")
            text = d.get("display_text") or d.get("semantic_text", d.get("normalized_text", ""))
            entities = d.get("entities", [])
            score = f"{sc.final_score:.3f}"

            header = f"[{i}] ({mt}, score={score})"
            if entities:
                header += f" entities=[{', '.join(entities)}]"

            # 拼接 temporal 字段：只输出非空的时间值，让 LLM 能感知有效时间窗口
            temporal = d.get("temporal") or {}
            if isinstance(temporal, dict):
                t_parts = []
                if temporal.get("t_ref"):
                    t_parts.append(f"t_ref={self._format_time_label(temporal['t_ref'])}")
                if temporal.get("t_valid_from"):
                    t_parts.append(f"valid_from={self._format_time_label(temporal['t_valid_from'])}")
                if temporal.get("t_valid_to"):
                    t_parts.append(f"valid_to={self._format_time_label(temporal['t_valid_to'])}")
                if t_parts:
                    header += f" temporal=[{', '.join(t_parts)}]"

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
                "display_text": d.get("display_text") or d.get("semantic_text", ""),
                "created_at": d.get("created_at", ""),
                "temporal": d.get("temporal", {}),
                "episodic_context": d.get("episodic_context", ""),
                "episode_refs": d.get("episode_refs", []),
                "entities": d.get("entities", []),
                "source_record_ids": d.get("source_record_ids", []),
                "source_session": d.get("source_session", ""),
                "source_role": d.get("source_role", ""),
                "evidence_turn_range": d.get("evidence_turn_range", []),
                "tree_parent_id": d.get("tree_parent_id", ""),
                "tree_depth": d.get("tree_depth", 0),
                "tree_expanded": bool(d.get("tree_expanded", False)),
            })
        return provenance

    def _log_usage(
        self,
        *,
        query: str,
        session_id: str,
        plan: SearchPlan,
        action_state: ActionState,
        retrieved_ids: list[str],
        kept_ids: list[str],
        prompt_versions_used: dict[str, int] | None = None,
    ) -> str:
        """记录一次检索使用日志。"""
        log_id = uuid.uuid4().hex
        log = UsageLog(
            log_id=log_id,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            retrieval_plan={
                "mode": plan.mode,
                "semantic_queries": plan.semantic_queries,
                "pragmatic_queries": plan.pragmatic_queries,
                "required_constraints": plan.required_constraints,
                "required_affordances": plan.required_affordances,
                "missing_slots": plan.missing_slots,
                "tree_retrieval_mode": plan.tree_retrieval_mode,
                "tree_expansion_depth": plan.tree_expansion_depth,
                "include_leaf_records": plan.include_leaf_records,
                "include_episodic_context": plan.include_episodic_context,
                "episodic_turn_window": plan.episodic_turn_window,
                "depth": plan.depth,
            },
            action_state=self._action_state_to_dict(action_state),
            retrieved_record_ids=retrieved_ids,
            kept_record_ids=kept_ids,
            prompt_versions_used=dict(prompt_versions_used or {}),
        )
        self._sqlite.insert_usage_log(log)
        return log_id

    def _classify_feedback_from_user_turn(self, user_turn: str) -> dict[str, str]:
        text = str(user_turn or "").strip()
        if not text:
            return {"feedback": "", "outcome": "unknown"}

        user_content = f"<USER_TURN>\n{text}\n</USER_TURN>"
        
        set_llm_call_source("feedback_classification")
        response = self._llm.generate([
            {"role": "system", "content": get_prompt("feedback_classification", FEEDBACK_CLASSIFICATION_SYSTEM)},
            {"role": "user", "content": user_content},
        ])
        
        try:
            import json
            raw = response.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) >= 3 else parts[-1].strip()
            parsed = json.loads(raw)
            feedback = str(parsed.get("feedback") or "").strip()
            outcome = str(parsed.get("outcome") or "unknown").strip()
            if feedback not in {"positive", "negative", "correction"}:
                feedback = ""
            if outcome not in {"success", "fail", "unknown"}:
                outcome = "unknown"
            return {"feedback": feedback, "outcome": outcome}
        except Exception:
            return {"feedback": "", "outcome": "unknown"}
