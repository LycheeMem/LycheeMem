"""
LanceDB 技能库（向量存储）。

持久化版本的程序记忆 / 技能库，兼容 InMemorySkillStore/BaseMemoryStore 接口。
使用 LanceDB 做向量相似度检索，数据自动持久化到本地目录。
"""

from __future__ import annotations

import uuid
from typing import Any

import lancedb
import pyarrow as pa

from src.memory.base import BaseMemoryStore


class LanceDBSkillStore(BaseMemoryStore):
    """基于 LanceDB 的持久化技能库。

    LanceDB 是一个嵌入式向量数据库，数据存储在本地文件中，
    无需额外服务端，适合从开发到中等规模生产使用。
    """

    TABLE_NAME = "skills"

    def __init__(self, db_path: str = "lychee_memos_lancedb", embedding_dim: int = 768):
        self._db = lancedb.connect(db_path)
        self._embedding_dim = embedding_dim
        self._ensure_table()

    def _ensure_table(self) -> None:
        """确保表存在，不存在则创建。"""
        if self.TABLE_NAME not in self._db.table_names():
            schema = pa.schema(
                [
                    pa.field("id", pa.utf8()),
                    pa.field("intent", pa.utf8()),
                    pa.field("doc_markdown", pa.utf8()),  # Markdown 文档
                    pa.field("metadata", pa.utf8()),  # JSON 序列化
                    pa.field("user_id", pa.utf8()),
                    pa.field("vector", pa.list_(pa.float32(), self._embedding_dim)),
                ]
            )
            self._db.create_table(self.TABLE_NAME, schema=schema)

    def _get_table(self):
        return self._db.open_table(self.TABLE_NAME)

    def add(self, items: list[dict[str, Any]], *, user_id: str = "") -> None:
        """添加技能条目。每个 item 需包含 intent, embedding, doc_markdown。"""
        import json

        rows = []
        for item in items:
            rows.append(
                {
                    "id": item.get("id", str(uuid.uuid4())),
                    "intent": item["intent"],
                    "doc_markdown": item["doc_markdown"],
                    "metadata": json.dumps(item.get("metadata", {}), ensure_ascii=False),
                    "user_id": user_id,
                    "vector": item["embedding"],
                }
            )
        if rows:
            table = self._get_table()
            table.add(rows)

    def search(
        self,
        query: str,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
        *,
        user_id: str = "",
    ) -> list[dict[str, Any]]:
        """向量相似度检索。需传入 query_embedding。"""
        import json

        if query_embedding is None:
            return []

        table = self._get_table()
        try:
            search_q = table.search(query_embedding)
            if user_id:
                search_q = search_q.where(f"user_id = '{user_id}'")
            results = search_q.limit(top_k).to_list()
        except Exception:
            return []

        return [
            {
                "id": r["id"],
                "intent": r["intent"],
                "doc_markdown": r.get("doc_markdown", ""),
                "score": 1.0 - r.get("_distance", 0.0),
                "metadata": json.loads(r.get("metadata", "{}")),
            }
            for r in results
        ]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        table = self._get_table()
        # LanceDB 使用 SQL-like where 子句删除
        id_list = ", ".join(f"'{id_}'" for id_ in ids)
        table.delete(f"id IN ({id_list})")

    def get_all(self, *, user_id: str = "") -> list[dict[str, Any]]:
        import json

        table = self._get_table()
        rows = table.to_pandas()
        return [
            {
                "id": row["id"],
                "intent": row["intent"],
                "doc_markdown": row.get("doc_markdown", ""),
                "metadata": json.loads(row.get("metadata", "{}")),
            }
            for _, row in rows.iterrows()
            if not user_id or row.get("user_id", "") == user_id
        ]
