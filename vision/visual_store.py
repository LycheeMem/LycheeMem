"""视觉记忆存储引擎。

使用 SQLite（元数据）+ LanceDB（向量索引）+ 本地文件系统（原始图片）
三层存储架构管理视觉记忆记录。
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.memory.visual.models import VisualMemoryRecord

logger = logging.getLogger(__name__)


# ── SQLite 建表 SQL ──
_CREATE_VISUAL_MEMORY_TABLE = """\
CREATE TABLE IF NOT EXISTS visual_memories (
    record_id       TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    image_path      TEXT NOT NULL,
    image_url       TEXT DEFAULT '',
    image_hash      TEXT NOT NULL,
    image_size      INTEGER DEFAULT 0,
    image_mime_type TEXT DEFAULT '',
    caption         TEXT NOT NULL DEFAULT '',
    entities        TEXT DEFAULT '[]',
    scene_type      TEXT DEFAULT '',
    structured_data TEXT DEFAULT '{}',
    caption_embedding TEXT,
    visual_embedding TEXT,
    confidence      REAL DEFAULT 1.0,
    importance_score REAL DEFAULT 1.0,
    source_role     TEXT DEFAULT '',
    related_memory_record_ids TEXT DEFAULT '[]',
    retrieval_count INTEGER DEFAULT 0,
    last_retrieved_at TEXT DEFAULT '',
    expired         INTEGER DEFAULT 0,
    expired_at      TEXT DEFAULT '',
    ttl             TEXT DEFAULT NULL
);
"""

_CREATE_IMAGE_HASH_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_visual_hash ON visual_memories(image_hash, expired);
"""

_CREATE_SESSION_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_visual_session ON visual_memories(session_id, expired);
"""

_CREATE_TIMESTAMP_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_visual_timestamp ON visual_memories(timestamp DESC);
"""


class VisualStore:
    """视觉记忆存储管理器。

    Args:
        db_path: SQLite 数据库文件路径。
        vector_db_path: LanceDB 向量存储目录。
        image_storage_path: 图片存储目录。
        embedding_dim: Embedding 向量维度。
        embedder: Embedding 适配器（可选，传 None 则不做向量索引）。
    """

    def __init__(
        self,
        db_path: str = "data/visual_memory.db",
        vector_db_path: str = "data/visual_vector",
        image_storage_path: str = "data/visual_memory",
        embedding_dim: int = 1024,
        embedder=None,
        multimodal_embedder=None,
    ) -> None:
        """初始化视觉记忆存储。

        Args:
            db_path: SQLite 数据库文件路径。
            vector_db_path: LanceDB 向量存储目录。
            image_storage_path: 图片存储目录。
            embedding_dim: Embedding 向量维度。
            embedder: 文本 Embedding 适配器（可选，向后兼容）。
            multimodal_embedder: 多模态 Embedder（新增，支持跨模态检索）。
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)

        self.image_storage_path = Path(image_storage_path)
        self.image_storage_path.mkdir(parents=True, exist_ok=True)

        self.embedding_dim = embedding_dim
        self.embedder = embedder  # 文本 embedder（向后兼容）
        self.multimodal_embedder = multimodal_embedder  # 多模态 embedder（新增）

        # 初始化 SQLite
        self._init_sqlite()

        # 初始化 LanceDB
        self._init_lancedb()

        # 向量索引缓冲区（批量异步写入）
        self._vector_buffer: list[VisualMemoryRecord] = []
        self._buffer_size = 10  # 缓冲 10 条后批量写入

    def _init_sqlite(self) -> None:
        """初始化 SQLite 数据库和表结构。"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(_CREATE_VISUAL_MEMORY_TABLE)
            conn.execute(_CREATE_IMAGE_HASH_INDEX)
            conn.execute(_CREATE_SESSION_INDEX)
            conn.execute(_CREATE_TIMESTAMP_INDEX)
            conn.commit()
            logger.info("Visual memory SQLite initialized: %s", self.db_path)
        finally:
            conn.close()

    def _init_lancedb(self) -> None:
        """初始化 LanceDB 向量存储。"""
        try:
            import lancedb

            self._lancedb = lancedb.connect(str(self.vector_db_path))

            # 创建或打开 visual_records 表
            try:
                self._visual_table = self._lancedb.open_table("visual_records")
                logger.info("LanceDB visual_records table opened")
            except Exception:
                # 表不存在，需要创建
                self._visual_table = None
                logger.info("LanceDB visual_records table will be created on first insert")
        except ImportError:
            logger.warning("LanceDB not available, vector indexing disabled")
            self._lancedb = None
            self._visual_table = None

    # ──────────────────────────────────────────────
    #  图片文件管理
    # ──────────────────────────────────────────────

    def save_image_file(
        self,
        image_b64: str,
        image_hash: str,
        mime_type: str = "image/jpeg",
        session_id: str = "",
    ) -> str:
        """将 Base64 图片保存到本地文件系统。

        Args:
            image_b64: Base64 编码的图片。
            image_hash: 图片内容哈希。
            mime_type: MIME 类型。
            session_id: 来源会话 ID。

        Returns:
            保存后的文件相对路径。
        """
        import base64

        # 按会话创建子目录
        session_dir = self.image_storage_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # 确定文件扩展名
        ext_map = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }
        ext = ext_map.get(mime_type, ".jpg")
        filename = f"{image_hash}{ext}"
        filepath = session_dir / filename

        if filepath.exists():
            logger.debug("Image already exists: %s", filepath)
            return str(filepath.relative_to(self.image_storage_path.parent.parent))

        image_bytes = base64.b64decode(image_b64)
        filepath.write_bytes(image_bytes)
        logger.info("Image saved: %s (%d bytes)", filepath, len(image_bytes))

        return str(filepath.relative_to(self.image_storage_path.parent.parent))

    # ──────────────────────────────────────────────
    #  CRUD 操作
    # ──────────────────────────────────────────────

    def store(self, record: VisualMemoryRecord) -> str:
        """存储一条视觉记忆记录。

        Args:
            record: 视觉记忆记录对象。

        Returns:
            record_id。
        """
        import json

        # 检查去重（相同 image_hash 且同一 session 则跳过）
        if self._check_duplicate(record.image_hash, record.session_id):
            logger.info(
                "Duplicate visual memory skipped: hash=%s, session=%s",
                record.image_hash, record.session_id,
            )
            return ""

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """INSERT INTO visual_memories (
                    record_id, session_id, timestamp,
                    image_path, image_url, image_hash, image_size, image_mime_type,
                    caption, entities, scene_type, structured_data,
                    caption_embedding, visual_embedding,
                    confidence, importance_score, source_role,
                    related_memory_record_ids,
                    retrieval_count, last_retrieved_at,
                    expired, expired_at, ttl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.record_id,
                    record.session_id,
                    record.timestamp,
                    record.image_path,
                    record.image_url,
                    record.image_hash,
                    record.image_size,
                    record.image_mime_type,
                    record.caption,
                    json.dumps(record.entities, ensure_ascii=False),
                    record.scene_type,
                    json.dumps(record.structured_data, ensure_ascii=False),
                    json.dumps(record.caption_embedding) if record.caption_embedding else None,
                    json.dumps(record.visual_embedding) if record.visual_embedding else None,
                    record.confidence,
                    record.importance_score,
                    record.source_role,
                    json.dumps(record.related_memory_record_ids),
                    record.retrieval_count,
                    record.last_retrieved_at,
                    int(record.expired),
                    record.expired_at,
                    record.ttl,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # 写入 LanceDB 向量索引（批量异步写入）
        self._index_vector_batch(record)

        logger.info("Visual memory stored: id=%s, caption=%s", record.record_id, record.caption[:60])
        return record.record_id

    def _check_duplicate(self, image_hash: str, session_id: str) -> bool:
        """检查是否存在重复图片。"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM visual_memories WHERE image_hash = ? AND session_id = ? AND expired = 0",
                (image_hash, session_id),
            )
            count = cursor.fetchone()[0]
            return count > 0
        finally:
            conn.close()

    def _index_vector_batch(self, record: VisualMemoryRecord) -> None:
        """批量将视觉记忆写入 LanceDB 向量索引（异步非阻塞）。

        使用缓冲区累积记录，达到阈值后批量写入。
        """
        if self._visual_table is None:
            logger.debug("LanceDB table not initialized, skipping vector indexing")
            return

        # 添加到缓冲区
        self._vector_buffer.append(record)

        # 当缓冲区达到阈值时批量写入
        if len(self._vector_buffer) >= self._buffer_size:
            self._flush_vector_buffer()

    def _flush_vector_buffer(self) -> None:
        """刷新向量缓冲区到 LanceDB。"""
        if not self._vector_buffer:
            return

        try:
            import numpy as np

            # 准备批量数据
            batch_data = []
            for record in self._vector_buffer:
                data = {
                    "record_id": record.record_id,
                    "caption": record.caption,
                    "scene_type": record.scene_type,
                    "timestamp": record.timestamp,
                    "importance_score": record.importance_score,
                }

                # 添加 caption embedding（向后兼容）
                if record.caption_embedding:
                    data["vector"] = np.array(record.caption_embedding, dtype=np.float32)

                # 添加 visual embedding（用于跨模态检索）
                if record.visual_embedding:
                    data["visual_vector"] = np.array(record.visual_embedding, dtype=np.float32)

                # 只有当至少有一个向量时才添加
                if "vector" in data or "visual_vector" in data:
                    batch_data.append(data)

            if batch_data:
                self._visual_table.add(batch_data)
                logger.info(
                    "Vector indexing completed: %d records (buffer flushed)",
                    len(batch_data),
                )
            else:
                logger.warning("No embeddings in buffer to index")

        except Exception as e:
            logger.error("Failed to flush vector buffer: %s", e)
        finally:
            self._vector_buffer.clear()

    def flush_pending_vectors(self) -> None:
        """刷新所有待处理的向量索引（程序退出前调用）。"""
        self._flush_vector_buffer()

    def get_by_id(self, record_id: str) -> Optional[VisualMemoryRecord]:
        """按 ID 获取视觉记忆记录。"""
        import json

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM visual_memories WHERE record_id = ?", (record_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)
        finally:
            conn.close()

    def get_by_ids_batch(self, record_ids: list[str]) -> dict[str, VisualMemoryRecord]:
        """批量获取多个记录，避免 N+1 查询问题。

        Args:
            record_ids: record_id 列表。

        Returns:
            record_id -> VisualMemoryRecord 字典，不存在的 ID 不会包含在结果中。
        """
        if not record_ids:
            return {}

        import json

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" * len(record_ids))
            cursor = conn.execute(
                f"SELECT * FROM visual_memories WHERE record_id IN ({placeholders})",
                record_ids,
            )
            rows = cursor.fetchall()
            return {row["record_id"]: self._row_to_record(row) for row in rows}
        finally:
            conn.close()

    def list_memories(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        include_expired: bool = False,
    ) -> list[VisualMemoryRecord]:
        """列出视觉记忆记录。

        Args:
            session_id: 可选的会话过滤。
            limit: 返回数量限制。
            offset: 分页偏移。
            include_expired: 是否包含已过期的记录。

        Returns:
            视觉记忆记录列表。
        """
        import json

        conditions = []
        params: list[Any] = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if not include_expired:
            conditions.append("expired = 0")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                f"SELECT * FROM visual_memories WHERE {where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            )
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            conn.close()

    def search_by_text(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """通过文本查询检索视觉记忆（双路检索 + 分数融合）。

        使用多模态嵌入器时：
        - 第一路：CLIP 文本嵌入 → visual_vector 检索
        - 第二路：文本嵌入 → vector 检索
        - 融合：加权分数融合

        不使用多模态嵌入器时（向后兼容）：
        - 优先使用 LanceDB 向量检索
        - 回退到 SQLite LIKE 搜索

        Args:
            query: 查询文本。
            top_k: 返回数量。

        Returns:
            匹配结果列表，每项为 dict，包含 record_id, caption, score, metadata。
        """
        logger.info("Searching visual memories with query: '%s' (top_k=%d)", query, top_k)

        # 方案 A：使用多模态嵌入器进行双路检索
        if self.multimodal_embedder is not None and self._visual_table is not None:
            try:
                import asyncio
                import numpy as np

                loop = asyncio.new_event_loop()
                try:
                    # 生成 CLIP 文本嵌入（用于 visual_vector 检索）
                    clip_query_vec = loop.run_until_complete(
                        asyncio.to_thread(self.multimodal_embedder.embed_text, query)
                    )
                    # 生成文本嵌入（用于 caption vector 检索，向后兼容）
                    if self.embedder:
                        text_query_vec = loop.run_until_complete(
                            asyncio.to_thread(self.embedder.embed_query, query)
                        )
                    else:
                        text_query_vec = clip_query_vec.copy()
                finally:
                    loop.close()

                logger.debug(
                    "Generated embeddings: clip_dim=%d, text_dim=%d",
                    len(clip_query_vec), len(text_query_vec),
                )

                results = []

                # 第一路：CLIP 文本嵌入 → visual_vector 检索
                try:
                    visual_results = (
                        self._visual_table.search(np.array(clip_query_vec, dtype=np.float32), vector_column_name="visual_vector")
                        .limit(top_k * 2)
                        .to_list()
                    )
                    logger.debug("Visual vector search returned %d results", len(visual_results))
                    for r in visual_results:
                        results.append({
                            "record_id": r["record_id"],
                            "caption": r["caption"],
                            "scene_type": r.get("scene_type", ""),
                            "visual_score": float(1.0 - r.get("_distance", 0.0)),
                            "text_score": 0.0,
                            "timestamp": r.get("timestamp", ""),
                            "importance_score": r.get("importance_score", 0.0),
                        })
                except Exception as e:
                    logger.warning("Visual vector search failed: %s", e)

                # 第二路：文本嵌入 → caption vector 检索
                try:
                    text_results = (
                        self._visual_table.search(np.array(text_query_vec, dtype=np.float32), vector_column_name="vector")
                        .limit(top_k * 2)
                        .to_list()
                    )
                    logger.debug("Text vector search returned %d results", len(text_results))
                    for r in text_results:
                        # 检查是否已存在
                        existing = next((x for x in results if x["record_id"] == r["record_id"]), None)
                        if existing:
                            # 更新 text_score
                            existing["text_score"] = float(1.0 - r.get("_distance", 0.0))
                        else:
                            results.append({
                                "record_id": r["record_id"],
                                "caption": r["caption"],
                                "scene_type": r.get("scene_type", ""),
                                "visual_score": 0.0,
                                "text_score": float(1.0 - r.get("_distance", 0.0)),
                                "timestamp": r.get("timestamp", ""),
                                "importance_score": r.get("importance_score", 0.0),
                            })
                except Exception as e:
                    logger.warning("Text vector search failed: %s", e)

                # 分数融合：加权求和
                text_weight = 0.4
                visual_weight = 0.6

                for r in results:
                    r["final_score"] = (
                        text_weight * r["text_score"] +
                        visual_weight * r["visual_score"]
                    )
                    # 重要性加权
                    r["final_score"] *= (0.5 + 0.5 * r["importance_score"])

                # 按最终分数排序
                results.sort(key=lambda x: x["final_score"], reverse=True)

                logger.info("Multimodal search completed: %d results (before dedup)", len(results))

                # 返回 top_k
                return [
                    {
                        "record_id": r["record_id"],
                        "caption": r["caption"],
                        "scene_type": r["scene_type"],
                        "score": r["final_score"],
                        "timestamp": r["timestamp"],
                        "importance_score": r["importance_score"],
                    }
                    for r in results[:top_k]
                ]

            except Exception as e:
                logger.warning("Multimodal search failed, falling back to text-only: %s", e)

        # 向后兼容：仅使用文本嵌入
        if self._visual_table is not None and self.embedder is not None:
            try:
                import asyncio
                import numpy as np

                loop = asyncio.new_event_loop()
                try:
                    query_vec = loop.run_until_complete(
                        asyncio.to_thread(self.embedder.embed_query, query)
                    )
                finally:
                    loop.close()

                if query_vec:
                    results = (
                        self._visual_table.search(np.array(query_vec, dtype=np.float32))
                        .limit(top_k)
                        .to_list()
                    )
                    logger.info("Text-only vector search completed: %d results", len(results))
                    return [
                        {
                            "record_id": r["record_id"],
                            "caption": r["caption"],
                            "scene_type": r.get("scene_type", ""),
                            "score": float(1.0 - r.get("_distance", 0.0)),
                            "timestamp": r.get("timestamp", ""),
                            "importance_score": r.get("importance_score", 0.0),
                        }
                        for r in results
                    ]
            except Exception as e:
                logger.warning("Vector search failed, falling back to SQLite: %s", e)

        # 回退到 SQLite LIKE 搜索
        logger.info("Falling back to SQLite LIKE search")
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """SELECT record_id, caption, scene_type, timestamp, importance_score
                   FROM visual_memories
                   WHERE expired = 0
                     AND (caption LIKE ? OR scene_type LIKE ?)
                   ORDER BY importance_score DESC, timestamp DESC
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", top_k),
            )
            rows = cursor.fetchall()
            return [
                {
                    "record_id": row["record_id"],
                    "caption": row["caption"],
                    "scene_type": row["scene_type"],
                    "score": 0.5,  # LIKE 搜索给固定分
                    "timestamp": row["timestamp"],
                    "importance_score": row["importance_score"],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def search_by_visual_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """通过视觉嵌入检索视觉记忆（新增，支持图像查询）。
        
        Args:
            query_embedding: 查询图像的 CLIP 嵌入向量
            top_k: 返回数量
            
        Returns:
            匹配结果列表
        """
        if self._visual_table is None or not query_embedding:
            return []
        
        try:
            import numpy as np
            
            results = (
                self._visual_table.search(np.array(query_embedding, dtype=np.float32), vector_column_name="visual_vector")
                .limit(top_k)
                .to_list()
            )
            return [
                {
                    "record_id": r["record_id"],
                    "caption": r["caption"],
                    "scene_type": r.get("scene_type", ""),
                    "score": float(1.0 - r.get("_distance", 0.0)),
                    "timestamp": r.get("timestamp", ""),
                    "importance_score": r.get("importance_score", 0.0),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("Visual embedding search failed: %s", e)
            return []

    def mark_expired(self, record_id: str, expired_at: str = "") -> bool:
        """标记视觉记忆为已过期（软删除）。"""
        import json
        from datetime import datetime, timezone

        if not expired_at:
            expired_at = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                "UPDATE visual_memories SET expired = 1, expired_at = ? WHERE record_id = ?",
                (expired_at, record_id),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def get_image_path(self, record_id: str) -> Optional[str]:
        """获取视觉记忆对应的图片文件路径。"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "SELECT image_path FROM visual_memories WHERE record_id = ?", (record_id,)
            )
            row = cursor.fetchone()
            if row:
                return row[0]
            return None
        finally:
            conn.close()

    def get_stats(self) -> dict[str, Any]:
        """获取视觉记忆统计信息。"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM visual_memories"
            ).fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM visual_memories WHERE expired = 0"
            ).fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM visual_memories WHERE expired = 1"
            ).fetchone()[0]

            # 按场景类型统计
            cursor = conn.execute(
                "SELECT scene_type, COUNT(*) as cnt FROM visual_memories WHERE expired = 0 GROUP BY scene_type"
            )
            by_scene = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total": total,
                "active": active,
                "expired": expired,
                "by_scene_type": by_scene,
            }
        finally:
            conn.close()

    def delete_record(self, record_id: str) -> bool:
        """彻底删除视觉记忆记录（包括图片文件）。"""
        import os

        # 先获取图片路径
        image_path = self.get_image_path(record_id)

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("DELETE FROM visual_memories WHERE record_id = ?", (record_id,))
            conn.commit()
        finally:
            conn.close()

        # 删除图片文件
        if image_path:
            full_path = self.image_storage_path / image_path
            if full_path.exists():
                full_path.unlink()
                logger.info("Image file deleted: %s", full_path)

        return True

    def _row_to_record(self, row: sqlite3.Row) -> VisualMemoryRecord:
        """将 SQLite Row 转为 VisualMemoryRecord 对象。"""
        import json

        return VisualMemoryRecord(
            record_id=row["record_id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            image_path=row["image_path"],
            image_url=row["image_url"] or "",
            image_hash=row["image_hash"],
            image_size=row["image_size"] or 0,
            image_mime_type=row["image_mime_type"] or "",
            caption=row["caption"] or "",
            entities=json.loads(row["entities"] or "[]"),
            scene_type=row["scene_type"] or "",
            structured_data=json.loads(row["structured_data"] or "{}"),
            caption_embedding=json.loads(row["caption_embedding"]) if row["caption_embedding"] else [],
            visual_embedding=json.loads(row["visual_embedding"]) if row["visual_embedding"] else [],
            confidence=row["confidence"] or 1.0,
            importance_score=row["importance_score"] or 1.0,
            source_role=row["source_role"] or "",
            related_memory_record_ids=json.loads(row["related_memory_record_ids"] or "[]"),
            retrieval_count=row["retrieval_count"] or 0,
            last_retrieved_at=row["last_retrieved_at"] or "",
            expired=bool(row["expired"] or 0),
            expired_at=row["expired_at"] or "",
            ttl=row["ttl"],
        )
