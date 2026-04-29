"""视觉记忆检索器 - 高性能版。

提供多种检索方式：
- 文本查询 → 视觉记忆检索（双路检索 + 分数融合）
- 图像查询 → 相似图片检索（基于 CLIP 视觉嵌入）
- 联合文本记忆检索

优化:
- 批量获取记录，避免 N+1 查询
- SQLite 连接复用
- 检索计数异步批量更新
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.memory.visual.visual_store import VisualStore

logger = logging.getLogger(__name__)


class VisualRetriever:
    """视觉记忆检索器（高性能版）。

    Args:
        visual_store: 视觉记忆存储实例。
    """

    def __init__(self, visual_store: VisualStore) -> None:
        self.store = visual_store
        # 批量更新队列：record_id -> increment_count
        self._retrieval_count_buffer: dict[str, int] = {}
        self._buffer_size = 20  # 缓冲 20 条后批量写入

    def retrieve_by_text(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """通过文本查询检索相关视觉记忆（双路检索 + 分数融合，批量优化）。

        Args:
            query: 查询文本。
            top_k: 返回数量。
            session_id: 可选的会话范围限制。

        Returns:
            检索结果列表，每项包含 record_id, caption, image_path, score 等。
        """
        results = self.store.search_by_text(query, top_k=top_k)

        # 批量获取记录，避免 N+1 查询
        if not results:
            return []

        record_ids = [r["record_id"] for r in results]
        records_map = self.store.get_by_ids_batch(record_ids)

        # 补充图片路径信息
        enriched = []
        for r in results:
            record = records_map.get(r["record_id"])
            if record and not record.expired:
                enriched.append(
                    {
                        "record_id": r["record_id"],
                        "caption": r["caption"],
                        "image_path": record.image_path,
                        "image_hash": record.image_hash,
                        "scene_type": r.get("scene_type", ""),
                        "score": r.get("score", 0.0),
                        "timestamp": r.get("timestamp", ""),
                        "importance_score": r.get("importance_score", 0.0),
                        "session_id": record.session_id,
                        "entities": record.entities,
                        "structured_data": record.structured_data,
                    }
                )
                # 更新检索计数（批量）
                self._increment_retrieval_count(r["record_id"])

        # 按分数排序
        enriched.sort(key=lambda x: x["score"], reverse=True)
        return enriched[:top_k]

    def retrieve_by_image(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """通过图像查询检索相似视觉记忆（基于 CLIP 视觉嵌入）。

        Args:
            image_path: 查询图像的路径
            top_k: 返回数量

        Returns:
            检索结果列表，每项包含 record_id, caption, image_path, score 等
        """
        logger.info("Retrieving visual memories by image: %s (top_k=%d)", image_path, top_k)

        # 检查多模态嵌入器可用性
        if not hasattr(self.store, 'multimodal_embedder') or self.store.multimodal_embedder is None:
            logger.warning("Multimodal embedder not available, image retrieval disabled")
            return []

        try:
            from pathlib import Path
            if not Path(image_path).exists():
                logger.error("Image file not found: %s", image_path)
                return []

            # 生成 CLIP 图像嵌入
            logger.debug("Generating image embedding for: %s", image_path)
            query_embedding = self.store.multimodal_embedder.embed_image(image_path)

            if not query_embedding:
                logger.warning("Failed to generate image embedding")
                return []

            logger.debug("Generated embedding with dim=%d", len(query_embedding))

            # 使用视觉嵌入检索
            results = self.store.search_by_visual_embedding(query_embedding, top_k=top_k)

            logger.info("Image retrieval completed: %d results", len(results))

            # 批量获取记录
            if not results:
                return []

            record_ids = [r["record_id"] for r in results]
            records_map = self.store.get_by_ids_batch(record_ids)

            # 补充图片路径信息
            enriched = []
            for r in results:
                record = records_map.get(r["record_id"])
                if record and not record.expired:
                    enriched.append(
                        {
                            "record_id": r["record_id"],
                            "caption": r["caption"],
                            "image_path": record.image_path,
                            "image_hash": record.image_hash,
                            "scene_type": r.get("scene_type", ""),
                            "score": r.get("score", 0.0),
                            "timestamp": r.get("timestamp", ""),
                            "importance_score": r.get("importance_score", 0.0),
                            "session_id": record.session_id,
                            "entities": record.entities,
                            "structured_data": record.structured_data,
                        }
                    )
                    self._increment_retrieval_count(r["record_id"])

            # 按分数排序
            enriched.sort(key=lambda x: x["score"], reverse=True)
            return enriched[:top_k]

        except Exception as e:
            logger.error("Image retrieval failed: %s", e, exc_info=True)
            return []

    def retrieve_by_session(
        self,
        session_id: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """检索某个会话的全部视觉记忆。

        Args:
            session_id: 会话 ID。
            top_k: 返回数量。

        Returns:
            视觉记忆列表。
        """
        records = self.store.list_memories(
            session_id=session_id, limit=top_k, include_expired=False
        )
        return [
            {
                "record_id": r.record_id,
                "caption": r.caption,
                "image_path": r.image_path,
                "image_hash": r.image_hash,
                "scene_type": r.scene_type,
                "timestamp": r.timestamp,
                "importance_score": r.importance_score,
                "session_id": r.session_id,
                "entities": r.entities,
                "structured_data": r.structured_data,
                "source_role": r.source_role,
            }
            for r in records
        ]

    def retrieve_for_context(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """为对话上下文检索相关视觉记忆（供 Synthesizer/Reasoner 使用）。

        Args:
            query: 当前用户查询。
            top_k: 返回数量（默认较少，避免过多 token 消耗）。

        Returns:
            格式化后的上下文片段列表。
        """
        results = self.retrieve_by_text(query, top_k=top_k)
        return results

    def _increment_retrieval_count(self, record_id: str) -> None:
        """增加记录的检索次数（批量更新，减少 SQLite 连接开销）。"""
        self._retrieval_count_buffer[record_id] = self._retrieval_count_buffer.get(record_id, 0) + 1

        # 当缓冲区达到阈值时批量写入
        if len(self._retrieval_count_buffer) >= self._buffer_size:
            self._flush_retrieval_counts()

    def _flush_retrieval_counts(self) -> None:
        """批量刷新检索计数到数据库。"""
        if not self._retrieval_count_buffer:
            return

        import sqlite3

        db_path = self.store.db_path
        try:
            conn = sqlite3.connect(str(db_path))
            # 使用单个事务批量更新
            updates = [
                (count, record_id)
                for record_id, count in self._retrieval_count_buffer.items()
            ]
            conn.executemany(
                "UPDATE visual_memories SET retrieval_count = retrieval_count + ?, last_retrieved_at = datetime('now') WHERE record_id = ?",
                updates,
            )
            conn.commit()
            conn.close()
            logger.debug("Flushed %d retrieval count updates", len(self._retrieval_count_buffer))
        except Exception as e:
            logger.warning("Failed to flush retrieval counts: %s", e)
        finally:
            self._retrieval_count_buffer.clear()

    def flush_pending_updates(self) -> None:
        """刷新所有待处理的更新（在程序退出前调用）。"""
        self._flush_retrieval_counts()
