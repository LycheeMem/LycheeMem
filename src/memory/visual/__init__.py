"""视觉记忆模块。

提供 LycheeMem 的多模态视觉记忆能力：
- VisualMemoryRecord: 数据模型
- VisualExtractor: VLM 驱动的图片理解
- VisualStore: SQLite + LanceDB + 文件系统存储
- VisualRetriever: 视觉记忆检索
- VisualForgetter: 视觉记忆遗忘管理

极速版组件（性能优化）:
- VisualExtractorFast: 超快速图片理解（512px 压缩，15s 超时）
- MultimodalEmbedderFast: 快速多模态嵌入（FP16 加速，批量处理）
"""

from __future__ import annotations

from src.memory.visual.models import VisualMemoryRecord
from src.memory.visual.visual_extractor import VisualExtractor
from src.memory.visual.visual_store import VisualStore
from src.memory.visual.visual_retriever import VisualRetriever
from src.memory.visual.visual_forgetter import VisualForgetter

# 极速版组件
from src.memory.visual.visual_extractor_fast import VisualExtractorFast
from src.memory.visual.multimodal_embedder_fast import MultimodalEmbedderFast

__all__ = [
    # 标准版
    "VisualMemoryRecord",
    "VisualExtractor",
    "VisualStore",
    "VisualRetriever",
    "VisualForgetter",
    # 极速版
    "VisualExtractorFast",
    "MultimodalEmbedderFast",
]
