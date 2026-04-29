"""视觉记忆数据模型。

定义 VisualMemoryRecord：经过 VLM 理解后的最小自洽视觉记忆记录。
存储于 SQLite + LanceDB + 本地文件系统。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VisualMemoryRecord:
    """视觉记忆记录（Visual Memory Record）。

    每条记录对应一张被 VLM 理解过的图片。
    包含：原始图片引用、描述文本、结构化实体、embedding 向量。
    """

    record_id: str                          # SHA256(image_hash + session_id + timestamp)
    session_id: str                         # 来源会话 ID
    timestamp: str                          # ISO 8601 时间戳

    # ── 图片存储 ──
    image_path: str                         # 本地文件系统路径
    image_url: str = ""                     # 可选的外部 URL 引用
    image_hash: str = ""                    # 图片内容 MD5/SHA256（用于去重）
    image_size: int = 0                     # 文件大小（字节）
    image_mime_type: str = ""               # image/jpeg, image/png 等

    # ── VLM 理解结果 ──
    caption: str = ""                       # 图片描述文本（主要检索依据）
    entities: list[dict[str, Any]] = field(default_factory=list)
    # 每个实体: {"type": str, "name": str, "confidence": float, "bbox": dict}
    scene_type: str = ""                    # screenshot / chart / photo / document / ui / other
    structured_data: dict[str, Any] = field(default_factory=dict)
    # 场景相关的结构化提取，如 chart 的 {chart_type, axes, data_points}

    # ── Embedding（双嵌入：文本 + 视觉）──
    caption_embedding: list[float] = field(default_factory=list)
    # 基于 caption 的文本 embedding（向后兼容）
    
    visual_embedding: list[float] = field(default_factory=list)
    # 基于 CLIP 的视觉 embedding（与文本同一空间，支持跨模态检索）

    # ── 元数据 ──
    confidence: float = 1.0
    importance_score: float = 1.0           # 重要性评分（用于遗忘策略）
    source_role: str = ""                   # "user" | "assistant"

    # ── 关联文本记忆 ──
    related_memory_record_ids: list[str] = field(default_factory=list)
    # 关联的 MemoryRecord record_id 列表

    # ── 使用统计 ──
    retrieval_count: int = 0
    last_retrieved_at: str = ""

    # ── 遗忘 ──
    expired: bool = False
    expired_at: str = ""
    ttl: Optional[str] = None               # 过期时间（ISO 8601）
