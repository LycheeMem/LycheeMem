"""Compact Semantic Engine（总装引擎）。

实现 BaseSemanticMemoryEngine，串联：
- search(): evidence-node / record / raw-turn 多通道召回 + 可选 rerank + coverage selection。
- ingest_conversation(): raw-turn index → online semantic chunking → Encoder → fielded evidence organization。

这是整个 Compact Semantic Memory 的核心入口。
"""

from __future__ import annotations

from collections import OrderedDict
import threading

from typing import Any

from src.embedder.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.memory.semantic.base import (
    BaseSemanticMemoryEngine,
    SemanticSearchResult,
)
from src.memory.semantic.chunker import OnlineSemanticChunker
from src.memory.semantic.debug_trace import (
    make_trace_id,
    semantic_trace_enabled,
)
from src.memory.semantic.encoder import CompactSemanticEncoder
from src.memory.semantic.models import SearchPlan
from src.memory.semantic.reranker import (
    LocalCrossEncoderReranker,
    RemoteHTTPReranker,
)
from src.memory.semantic.sqlite_store import SQLiteSemanticStore
from src.memory.semantic.evidence_graph import FieldedEvidenceOrganizer
from src.memory.semantic.vector_index import LanceVectorIndex
from src.memory.semantic.retrieval import SemanticRetrievalMixin
from src.memory.semantic.ingestion import SemanticIngestionMixin
from src.memory.semantic.planning import RetrievalPlanningMixin
from src.memory.semantic.formatting import RetrievalFormattingMixin


class CompactSemanticEngine(
    SemanticRetrievalMixin,
    SemanticIngestionMixin,
    RetrievalPlanningMixin,
    RetrievalFormattingMixin,
    BaseSemanticMemoryEngine,
):
    """Compact Semantic Memory 总装引擎。

    实现 BaseSemanticMemoryEngine 接口，负责语义记忆固化、索引与检索。
    """

    MAX_SEMANTIC_QUERIES = 16
    MAX_EVIDENCE_ROUTES = 8
    MAX_ROUTE_QUERIES = 8
    MAX_CONSTRAINTS = 12

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        *,
        session_store: Any | None = None,
        sqlite_db_path: str = "data/compact_memory.db",
        vector_db_path: str = "data/compact_vector",
        embedding_dim: int = 0,
        reranker_enabled: bool = False,
        reranker_backend: str = "local",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        reranker_api_base: str = "",
        reranker_api_key: str = "",
        reranker_device: str = "auto",
        reranker_batch_size: int = 16,
        reranker_max_length: int = 512,
        reranker_candidate_limit: int = 100,
        **_compat_kwargs: Any,
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
        self._chunker = OnlineSemanticChunker(embedder=embedder)
        self._evidence_organizer = FieldedEvidenceOrganizer(
            sqlite_store=self._sqlite,
            vector_index=self._vector,
        )

        self._reranker_enabled = bool(reranker_enabled)
        self._reranker_candidate_limit = max(20, int(reranker_candidate_limit or 100))
        self._encoder_reference_by_session: dict[str, str] = {}
        self._encoder_record_texts_by_session: dict[str, list[str]] = {}
        self._query_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_embedding_cache_lock = threading.Lock()
        self._query_embedding_inflight: dict[str, threading.Event] = {}
        self._query_embedding_cache_max = 4096
        self._reranker = None
        if self._reranker_enabled:
            backend = str(reranker_backend or "local").lower()
            if backend == "http":
                self._reranker = RemoteHTTPReranker(
                    api_base=reranker_api_base,
                    api_key=reranker_api_key or None,
                    model_name=reranker_model,
                )
            else:
                self._reranker = LocalCrossEncoderReranker(
                    model_name=reranker_model,
                    device=reranker_device,
                    batch_size=reranker_batch_size,
                    max_length=reranker_max_length,
                )

    # ════════════════════════════════════════════════════════════════
    # search() — semantic memory retrieval
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
        reference_time: str | None = None,
    ) -> SemanticSearchResult:
        """Semantic memory retrieval over plan-generated query variants."""
        requested_top_k = max(0, int(top_k or 0))
        trace_id = ""
        if isinstance(retrieval_plan, dict):
            trace_id = str(retrieval_plan.get("_trace_id") or "").strip()
        if not trace_id and semantic_trace_enabled():
            trace_id = make_trace_id("search")

        if retrieval_plan:
            plan = self._dict_to_plan(retrieval_plan)
        else:
            plan = SearchPlan(
                semantic_queries=[query],
            )

        plan = self._normalize_plan_for_query(query, plan)
        top_k = max(requested_top_k, int(plan.depth or 0), 1)

        routes = self._build_evidence_routes(query, plan)
        strategy = self._resolve_retrieval_strategy(plan)

        result = self._search_fielded_evidence(
            query=query,
            session_id=session_id,
            top_k=top_k,
            routes=routes,
            plan=plan,
            strategy=strategy,
            query_embedding=query_embedding,
            reference_time=reference_time,
            trace_id=trace_id,
        )
        result.action_state = dict(action_state or {})
        result.diagnostics = dict(result.diagnostics or {})

        return result

    def delete_all(self) -> dict[str, int]:
        result = self._sqlite.delete_all()
        self._vector.delete_all()
        self._chunker.reset_all()
        return result

    def export_debug(self) -> dict[str, Any]:
        return self._sqlite.export_all()

    # ════════════════════════════════════════════════════════════════
    # 内部工具方法
    # ════════════════════════════════════════════════════════════════
