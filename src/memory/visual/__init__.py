"""视觉记忆模块。"""

from __future__ import annotations

from src.memory.visual.models import VisualMemoryRecord
from src.memory.visual.visual_extractor import VisualExtractor
from src.memory.visual.visual_store import VisualStore
from src.memory.visual.visual_retriever import VisualRetriever
from src.memory.visual.visual_forgetter import VisualForgetter

from src.memory.visual.visual_extractor_fast import VisualExtractorFast
from src.memory.visual.multimodal_embedder_fast import MultimodalEmbedderFast

__all__ = [
    "VisualMemoryRecord",
    "VisualExtractor",
    "VisualStore",
    "VisualRetriever",
    "VisualForgetter",
    "VisualExtractorFast",
    "MultimodalEmbedderFast",
]
