"""Compact Semantic Memory — 替代 Graphiti 的长期语义记忆模块。"""

from src.memory.semantic.base import (
    BaseSemanticMemoryEngine,
    ConsolidationResult,
    SemanticSearchResult,
)
from src.memory.semantic.engine import CompactSemanticEngine

__all__ = [
    "BaseSemanticMemoryEngine",
    "CompactSemanticEngine",
    "ConsolidationResult",
    "SemanticSearchResult",
]
