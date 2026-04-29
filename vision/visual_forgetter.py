"""视觉记忆遗忘管理器。

基于时间衰减和重要性评分实现视觉记忆的自动遗忘。
参考艾宾浩斯遗忘曲线，结合信息量、可替代性等因素。
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from src.memory.visual.visual_store import VisualStore

logger = logging.getLogger(__name__)

# ── 遗忘曲线参数 ──
# 基础衰减半衰期（小时）
DEFAULT_HALF_LIFE_HOURS = 168  # 7 天
# 重要性调节因子：importance_score=1.0 时半衰期延长至 30 天
IMPORTANCE_HALF_LIFE_MULTIPLIER = 30 * 24 / DEFAULT_HALF_LIFE_HOURS  # ~4.28
# 最大 TTL（天）
MAX_TTL_DAYS = 90


class VisualForgetter:
    """视觉记忆遗忘管理器。

    Args:
        visual_store: 视觉记忆存储实例。
        half_life_hours: 基础半衰期（小时），默认 168（7 天）。
    """

    def __init__(
        self,
        visual_store: VisualStore,
        half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
    ) -> None:
        self.store = visual_store
        self.half_life_hours = half_life_hours

    def compute_decay_score(self, record: Any, now: datetime | None = None) -> float:
        """计算视觉记忆的当前衰减分数。

        分数范围 0.0-1.0，1.0 表示完全保持，0.0 表示应被遗忘。

        公式:
            effective_half_life = base_half_life * (1 + importance * (multiplier - 1))
            decay = 0.5 ** (elapsed_hours / effective_half_life)

        Args:
            record: VisualMemoryRecord 对象或 dict。
            now: 当前时间（默认 now）。

        Returns:
            衰减后的分数。
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # 解析时间戳
        ts_str = record.timestamp if hasattr(record, "timestamp") else record.get("timestamp", "")
        try:
            created_at = datetime.fromisoformat(ts_str)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            return 0.0  # 无效时间戳，标记为应遗忘

        elapsed = (now - created_at).total_seconds() / 3600.0  # 小时

        # 获取重要性评分
        importance = (
            record.importance_score
            if hasattr(record, "importance_score")
            else record.get("importance_score", 0.5)
        )

        # 有效半衰期
        effective_half_life = self.half_life_hours * (
            1 + importance * (IMPORTANCE_HALF_LIFE_MULTIPLIER - 1)
        )

        # 衰减计算
        decay = 0.5 ** (elapsed / effective_half_life)

        # 检索次数加成（被检索过的记忆衰减更慢）
        retrieval_count = (
            record.retrieval_count
            if hasattr(record, "retrieval_count")
            else record.get("retrieval_count", 0)
        )
        if retrieval_count > 0:
            boost = min(0.3, retrieval_count * 0.05)  # 最多 +30%
            decay = min(1.0, decay + boost)

        return round(decay, 4)

    def should_forget(self, record: Any, threshold: float = 0.1) -> bool:
        """判断是否应该遗忘某条视觉记忆。

        Args:
            record: VisualMemoryRecord 或 dict。
            threshold: 遗忘阈值（低于此值则遗忘）。

        Returns:
            True 表示应遗忘。
        """
        score = self.compute_decay_score(record)
        return score < threshold

    def cleanup_expired(self) -> int:
        """清理已过期的视觉记忆。

        Returns:
            清理的记录数量。
        """
        now = datetime.now(timezone.utc)
        count = 0

        conn = sqlite3.connect(str(self.store.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM visual_memories WHERE expired = 0 ORDER BY timestamp ASC"
            )
            rows = cursor.fetchall()

            for row in rows:
                record = self.store._row_to_record(row)
                if self.should_forget(record, now=now):
                    self.store.mark_expired(
                        record.record_id,
                        expired_at=now.isoformat(),
                    )
                    count += 1
                    logger.info(
                        "Visual memory expired: id=%s, score=%.4f",
                        record.record_id,
                        self.compute_decay_score(record, now=now),
                    )

        finally:
            conn.close()

        logger.info("Visual forgetter cleanup: %d records expired", count)
        return count

    def schedule_ttl(self, record: Any) -> str | None:
        """为视觉记忆计算并设置 TTL。

        Args:
            record: VisualMemoryRecord 对象。

        Returns:
            计算的 TTL 时间（ISO 8601），如果不需要 TTL 则返回 None。
        """
        importance = getattr(record, "importance_score", 0.5)

        # 根据重要性决定 TTL
        if importance >= 0.8:
            ttl_days = MAX_TTL_DAYS
        elif importance >= 0.5:
            ttl_days = 30
        elif importance >= 0.3:
            ttl_days = 14
        else:
            ttl_days = 7

        ttl = datetime.now(timezone.utc) + timedelta(days=ttl_days)
        return ttl.isoformat()
