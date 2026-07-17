"""Ingestion helpers for CompactSemanticEngine."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.embedder.base import set_embedding_call_source
from src.memory.semantic.base import ConsolidationResult
from src.memory.semantic.chunker import SemanticChunk
from src.memory.semantic.models import MemoryRecord


class SemanticIngestionMixin:
    def ingest_conversation(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        turn_index_offset: int = 0,
        reference_timestamp: str | None = None,
        flush_session: bool = False,
    ) -> ConsolidationResult:
        """完整固化流程：原始 turn 索引 → 在线分块 → 编码 → 证据组织。"""
        if not turns and not flush_session:
            return ConsolidationResult(
                records_added=0,
                records_merged=0,
                records_expired=0,
                steps=[],
            )

        steps: list[dict[str, Any]] = []

        if turns:
            indexed_turns = self._index_ingested_episode_turns(
                turns=turns,
                session_id=session_id,
                turn_index_offset=turn_index_offset,
                reference_timestamp=reference_timestamp,
            )
            steps.append({
                "name": "index_episode_turns",
                "status": "done",
                "detail": f"写入 {indexed_turns} 条原始对话 turn 向量",
            })
        elif flush_session:
            steps.append({
                "name": "index_episode_turns",
                "status": "skipped",
                "detail": "session flush，无新增原始 turn 需要索引",
            })

        chunking = self._chunker.add_turns(
            session_id=session_id,
            turns=turns,
            turn_index_offset=turn_index_offset,
            flush=flush_session,
        )
        steps.append({
            "name": "online_semantic_chunking",
            "status": "done",
            "detail": (
                f"产出 {len(chunking.finalized_chunks)} 个待编码 chunk；"
                f"pending={chunking.pending_exchange_count} exchanges/"
                f"{chunking.pending_token_count} tokens"
                + ("；session flush 已触发" if flush_session else "")
            ),
            **self._chunking_debug_payload(chunking.finalized_chunks, chunking.decisions),
            "pending": {
                "exchange_count": chunking.pending_exchange_count,
                "token_count": chunking.pending_token_count,
            },
        })

        if not chunking.finalized_chunks:
            return ConsolidationResult(
                records_added=0,
                records_merged=0,
                records_expired=0,
                steps=steps,
            )

        self._persist_semantic_chunks(session_id, chunking.finalized_chunks)

        records_added = 0
        consumed_until = max(
            (
                chunk.turn_end_index + 1
                for chunk in chunking.finalized_chunks
            ),
            default=turn_index_offset,
        )
        turns_consumed = max(0, consumed_until - max(0, int(turn_index_offset or 0)))
        for index, chunk in enumerate(chunking.finalized_chunks, 1):
            result = self._ingest_semantic_chunk(
                chunk,
                session_id=session_id,
                reference_timestamp=reference_timestamp,
                chunk_index=index,
                chunk_count=len(chunking.finalized_chunks),
            )
            steps.extend(result.steps)
            records_added += result.records_added

        return ConsolidationResult(
            records_added=records_added,
            records_merged=0,
            records_expired=0,
            turns_consumed=turns_consumed,
            steps=steps,
        )

    def _ingest_semantic_chunk(
        self,
        chunk: SemanticChunk,
        *,
        session_id: str,
        reference_timestamp: str | None,
        chunk_index: int,
        chunk_count: int,
    ) -> ConsolidationResult:
        """Encode and store one finalized semantic chunk."""
        steps: list[dict[str, Any]] = []
        current = chunk.turns
        previous: list[dict[str, Any]] = []
        chunk_label = f"chunk {chunk_index}/{chunk_count}"

        encoder_reference_context = self._get_encoder_reference_context(session_id)
        encode_with_context = getattr(
            self._encoder,
            "encode_conversation_with_disambiguation",
            None,
        )
        if callable(encode_with_context):
            new_records, encoder_disambiguation_context = encode_with_context(
                current,
                previous_turns=previous,
                reference_context=encoder_reference_context,
                session_id=session_id,
                turn_index_offset=chunk.turn_index_offset,
                session_date=reference_timestamp,
            )
        else:
            new_records = self._encoder.encode_conversation(
                current,
                previous_turns=previous,
                reference_context=encoder_reference_context,
                session_id=session_id,
                turn_index_offset=chunk.turn_index_offset,
                session_date=reference_timestamp,
            )
            encoder_disambiguation_context = self._encoder.last_disambiguation_context

        steps.append({
            "name": "compact_encoding",
            "status": "done",
            "detail": (
                f"{chunk_label}: 抽取 {len(new_records)} 条 MemoryRecord "
                f"(reason={chunk.reason}, exchanges={chunk.exchange_count}, tokens={chunk.token_count})"
                + ("；使用同 session 消歧上下文" if encoder_reference_context else "")
            ),
        })

        if not new_records:
            self._update_encoder_reference_context(
                session_id=session_id,
                disambiguation_context=encoder_disambiguation_context,
                records=[],
            )
            return ConsolidationResult(
                records_added=0,
                records_merged=0,
                records_expired=0,
                steps=steps,
            )

        # Step 3: 精确写入原子记录。这里不再做 embedding 近似去重，
        # 因为句级 embedding 相近不能可靠说明两个事实重复。
        actually_added = 0
        persisted_records: list[MemoryRecord] = []

        self._update_encoder_reference_context(
            session_id=session_id,
            disambiguation_context=encoder_disambiguation_context,
            records=new_records,
        )

        # 3a. 仅按 record_id 做精确去重；record_id = SHA256(semantic_text)。
        exact_unique: dict[str, MemoryRecord] = {}
        for record in new_records:
            if record.record_id:
                if reference_timestamp:
                    record.created_at = reference_timestamp
                exact_unique[record.record_id] = record
        persisted_records = list(exact_unique.values())

        # 3b. 批量 embed record semantic_text + normalized_text，用于直接记录检索。
        _record_vec_map: dict[str, list[float]] = {}   # record_id → semantic_vector
        _record_norm_vec_map: dict[str, list[float]] = {}  # record_id → normalized_vector
        if persisted_records:
            set_embedding_call_source("record_ingest")
            _N = len(persisted_records)
            _all_texts = (
                [r.semantic_text for r in persisted_records]
                + [r.normalized_text for r in persisted_records]
            )
            _all_vecs = self._embedder.embed(_all_texts)
            _record_vec_map = {
                r.record_id: _all_vecs[i] for i, r in enumerate(persisted_records)
            }
            _record_norm_vec_map = {
                r.record_id: _all_vecs[_N + i] for i, r in enumerate(persisted_records)
            }

        # 3d. 写入 SQLite + 向量索引。
        for record in persisted_records:
            self._sqlite.upsert_record(record)
        if persisted_records:
            self._vector.upsert_batch([
                {
                    "record_id": r.record_id,
                    "memory_type": r.memory_type,
                    "semantic_text": r.semantic_text,
                    "normalized_text": r.normalized_text,
                    "semantic_vector": _record_vec_map.get(r.record_id),
                    "normalized_vector": _record_norm_vec_map.get(r.record_id),
                }
                for r in persisted_records
            ])

        actually_added = len(persisted_records)

        steps.append({
            "name": "store_atomic_records",
            "status": "done",
            "detail": f"精确写入 {actually_added}/{len(new_records)} 条原子记录",
        })

        # Step 4: Fielded Evidence Graph（无 LLM、无 record embedding 聚类）
        evidence_stats: dict[str, Any] = {}
        if actually_added > 0:
            evidence_stats = self._evidence_organizer.organize_on_ingest(persisted_records)

        steps.append({
            "name": "fielded_evidence_graph",
            "status": "done",
            "detail": (
                f"构建/更新 {evidence_stats.get('evidence_nodes_upserted', 0)} 个 evidence nodes "
                f"{evidence_stats.get('by_type', {})}"
            ),
        })

        return ConsolidationResult(
            records_added=actually_added,
            records_merged=0,
            records_expired=0,
            steps=steps,
        )

    def _persist_semantic_chunks(
        self,
        session_id: str,
        chunks: list[SemanticChunk],
    ) -> None:
        if self._session_store is None or not chunks:
            return
        add_chunks = getattr(self._session_store, "add_semantic_chunks", None)
        if not callable(add_chunks):
            return
        payload = [
            {
                "turn_start": chunk.turn_index_offset,
                "turn_end": chunk.turn_end_index,
                "token_count": chunk.token_count,
                "exchange_count": chunk.exchange_count,
                "reason": chunk.reason,
            }
            for chunk in chunks
        ]
        try:
            add_chunks(session_id, payload)
        except Exception:
            logging.getLogger("src.memory.semantic.engine").exception(
                "Failed to persist semantic chunk boundaries for session=%s",
                session_id,
            )

    # ════════════════════════════════════════════════════════════════
    # delete / export
    # ════════════════════════════════════════════════════════════════

    @classmethod
    def _chunking_debug_payload(
        cls,
        chunks: list[SemanticChunk],
        decisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "chunks": [
                {
                    "index": idx,
                    "reason": chunk.reason,
                    "turn_start": chunk.turn_index_offset,
                    "turn_end": chunk.turn_end_index,
                    "turn_count": len(chunk.turns),
                    "exchange_count": chunk.exchange_count,
                    "token_count": chunk.token_count,
                    "boundary_score": round(float(chunk.boundary_score or 0.0), 6),
                    "boundary_probability": round(
                        float(chunk.boundary_probability or 0.0), 6
                    ),
                    "preview": cls._chunk_preview(chunk.turns),
                }
                for idx, chunk in enumerate(chunks, 1)
            ],
            "decisions": [
                cls._compact_chunk_decision(decision)
                for decision in decisions[-40:]
                if isinstance(decision, dict)
            ],
        }

    @staticmethod
    def _chunk_preview(turns: list[dict[str, Any]], *, max_chars: int = 260) -> str:
        text = "\n".join(
            f"{turn.get('role', 'unknown')}: {turn.get('content', '')}".strip()
            for turn in turns[:4]
            if str(turn.get("content", "")).strip()
        ).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)].rstrip() + "..."

    @staticmethod
    def _compact_chunk_decision(decision: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "action",
            "cut",
            "reason",
            "turn_index",
            "chunk_tokens_before",
            "new_exchange_tokens",
            "chunk_exchanges_before",
            "probability",
            "score",
            "semantic_surprise",
            "cohesion_drop",
            "length_hazard",
            "centroid_similarity",
            "tail_similarity",
        )
        return {key: decision.get(key) for key in keys if key in decision}

    def _get_encoder_reference_context(self, session_id: str) -> str | None:
        """Return compact reference context for the same session only."""
        sid = str(session_id or "").strip()
        if not sid:
            return None
        return self._encoder_reference_by_session.get(sid) or None

    def _update_encoder_reference_context(
        self,
        *,
        session_id: str,
        disambiguation_context: str,
        records: list[MemoryRecord],
        max_records: int = 12,
        max_chars: int = 2400,
    ) -> None:
        """Keep only sufficient same-session context for later coreference resolution."""
        sid = str(session_id or "").strip()
        if not sid:
            return

        parts: list[str] = []
        note = str(disambiguation_context or "").strip()
        if note:
            parts.append(f"Resolved references: {note}")

        stored_texts = list(self._encoder_record_texts_by_session.get(sid, []))
        seen: set[str] = set(stored_texts)
        for record in records:
            text = str(record.semantic_text or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            stored_texts.append(text)
        stored_texts = stored_texts[-max_records:]
        if stored_texts:
            self._encoder_record_texts_by_session[sid] = stored_texts
            parts.append(
                "Recent same-session records:\n"
                + "\n".join(f"- {text}" for text in stored_texts)
            )

        context = "\n".join(parts).strip()
        if len(context) > max_chars:
            context = context[:max_chars].rstrip()
        if context:
            self._encoder_reference_by_session[sid] = context

    def _index_ingested_episode_turns(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        turn_index_offset: int,
        reference_timestamp: str | None = None,
    ) -> int:
        """Index raw dialogue turns as first-class retrievable evidence."""
        if not turns:
            return 0
        sid = str(session_id or "").strip()
        if not sid:
            return 0

        batch: list[dict[str, Any]] = []
        for local_index, turn in enumerate(turns):
            content = str(turn.get("content", "")).strip()
            if not content or turn.get("deleted", False):
                continue
            try:
                turn_index = int(turn.get("turn_index", turn_index_offset + local_index))
            except (TypeError, ValueError):
                turn_index = turn_index_offset + local_index
            source_dialogue_time = str(reference_timestamp or turn.get("created_at") or "")
            indexed_content = self._content_with_speaker(content, turn.get("speaker"))
            batch.append({
                "episode_id": self._make_episode_id(sid, turn_index),
                "session_id": sid,
                "turn_index": turn_index,
                "role": str(turn.get("role", "unknown")),
                "content": indexed_content,
                "created_at": source_dialogue_time,
            })

        if not batch:
            return 0
        try:
            self._vector.upsert_turns_batch(batch)
        except Exception:
            return 0
        return len(batch)

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
                indexed_content = self._content_with_speaker(content, turn.get("speaker"))
                batch.append({
                    "episode_id": ep_id,
                    "session_id": sid,
                    "turn_index": idx,
                    "role": str(turn.get("role", "unknown")),
                    "content": indexed_content,
                    "created_at": str(turn.get("created_at", "")),
                })

        if not batch:
            return 0

        self._vector.upsert_turns_batch(batch)
        return len(batch)

    def _search_raw_turns_direct(
        self,
        query: str,
        top_k: int,
        *,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """在 episode_turns 向量索引中直接检索原始对话 turns。

        不依赖 MemoryRecord 作为中介锚点，可以召回尚未被提炼成记忆记录、
        或提炼质量不足的对话内容。
        """
        if not str(query or "").strip():
            return []
        if query_vector is None:
            return []

        try:
            hits = self._vector.search_turns(
                query,
                limit=top_k,
                session_id=None,
                query_vector=query_vector,
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
                "source_role": role,
                "role": role,
                "evidence_turn_range": [idx],
                "entities": [],
                "tags": [],
                "temporal": {},
                "created_at": str(hit.get("created_at", "")),
                "source_dialogue_time": str(hit.get("created_at", "")),
            })
        return results

    @staticmethod
    def _content_with_speaker(content: str, speaker: Any) -> str:
        """Make participant identity searchable without changing vector schema."""
        normalized_content = str(content or "").strip()
        normalized_speaker = str(speaker or "").strip()
        if not normalized_speaker or not normalized_content:
            return normalized_content
        prefix = f"{normalized_speaker}:"
        if normalized_content.casefold().startswith(prefix.casefold()):
            return normalized_content
        return f"{prefix} {normalized_content}"

    @staticmethod
    def _make_episode_id(session_id: str, turn_index: int) -> str:
        return f"episode:{session_id}:{int(turn_index)}"


