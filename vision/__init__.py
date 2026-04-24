"""Vision Module - 视觉记忆模块入口。

使用示例:
    from vision import VisualExtractor, VisualStore
    from vision.models import VisualMemoryRecord
"""

# 模块版本
__version__ = '1.0.0'

# 导出主要类（从原路径导入，保持代码复用）
from src.memory.visual.models import VisualMemoryRecord
from src.memory.visual.visual_store import VisualStore
from src.memory.visual.visual_extractor import VisualExtractor
from src.memory.visual.visual_extractor_fast import VisualExtractorFast
from src.memory.visual.visual_retriever import VisualRetriever
from src.memory.visual.visual_forgetter import VisualForgetter

# 导出常量
from src.memory.visual.visual_store import (
    VISUAL_MEMORY_TYPE_IMAGE,
    VISUAL_MEMORY_TYPE_SCREENSHOT,
    VISUAL_MEMORY_TYPE_CHART,
    VISUAL_MEMORY_TYPE_DOCUMENT,
)

__all__ = [
    # 版本
    '__version__',
    # 数据模型
    'VisualMemoryRecord',
    # 存储
    'VisualStore',
    # 提取器
    'VisualExtractor',
    'VisualExtractorFast',
    # 检索
    'VisualRetriever',
    # 遗忘
    'VisualForgetter',
    # 常量
    'VISUAL_MEMORY_TYPE_IMAGE',
    'VISUAL_MEMORY_TYPE_SCREENSHOT',
    'VISUAL_MEMORY_TYPE_CHART',
    'VISUAL_MEMORY_TYPE_DOCUMENT',
]
