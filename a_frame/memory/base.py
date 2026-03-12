"""记忆存储抽象基类。"""

from abc import ABC, abstractmethod
from typing import Any


class BaseMemoryStore(ABC):
    """所有记忆存储的统一接口。"""

    @abstractmethod
    def add(self, items: list[dict[str, Any]]) -> None:
        """写入记忆条目。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """检索记忆。"""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """删除指定记忆。"""

    @abstractmethod
    def get_all(self) -> list[dict[str, Any]]:
        """获取所有记忆（调试用）。"""
