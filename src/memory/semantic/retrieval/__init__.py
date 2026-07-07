"""Retrieval helper composition for CompactSemanticEngine."""

from __future__ import annotations

from src.memory.semantic.retrieval.candidate_utils import RetrievalCandidateUtilsMixin
from src.memory.semantic.retrieval.embedding_cache import RetrievalEmbeddingCacheMixin
from src.memory.semantic.retrieval.expansion import RetrievalExpansionMixin
from src.memory.semantic.retrieval.fusion import RetrievalFusionMixin
from src.memory.semantic.retrieval.recall import RecallRetrievalMixin
from src.memory.semantic.retrieval.route_helpers import RetrievalRouteHelperMixin
from src.memory.semantic.retrieval.routes import RouteRetrievalMixin
from src.memory.semantic.retrieval.selection import RetrievalSelectionMixin
from src.memory.semantic.retrieval.strategy_policy import RetrievalStrategyPolicyMixin


class SemanticRetrievalMixin(
    RouteRetrievalMixin,
    RetrievalFusionMixin,
    RetrievalRouteHelperMixin,
    RetrievalStrategyPolicyMixin,
    RecallRetrievalMixin,
    RetrievalEmbeddingCacheMixin,
    RetrievalExpansionMixin,
    RetrievalCandidateUtilsMixin,
    RetrievalSelectionMixin,
):
    """Composes retrieval helpers used by CompactSemanticEngine."""
