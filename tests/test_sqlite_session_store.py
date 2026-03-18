"""SQLite 会话存储测试。

使用临时数据库文件测试持久化存储的完整生命周期。
"""

import os
import tempfile

from src.memory.working.sqlite_session_store import SQLiteSessionStore


class TestSQLiteSessionStore:
    def _make_store(self, tmp_path: str) -> SQLiteSessionStore:
        db_path = os.path.join(tmp_path, "test_sessions.db")
        return SQLiteSessionStore(db_path=db_path)

    def test_append_and_get_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.append_turn("s1", "user", "你好")
            store.append_turn("s1", "assistant", "你好！")

            turns = store.get_turns("s1")
            assert len(turns) == 2
            assert turns[0]["role"] == "user"
            assert turns[0]["content"] == "你好"
            assert turns[0].get("created_at")

            assert turns[1]["role"] == "assistant"
            assert turns[1]["content"] == "你好！"
            assert turns[1].get("created_at")
            store.close()

    def test_get_or_create_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            log = store.get_or_create("new-session")
            assert log.session_id == "new-session"
            assert log.turns == []
            assert log.summaries == []
            store.close()

    def test_add_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.append_turn("s1", "user", "hello")
            store.add_summary("s1", 0, "Summary text")

            log = store.get_or_create("s1")
            assert len(log.summaries) == 1
            assert log.summaries[0]["boundary_index"] == 0
            assert log.summaries[0]["content"] == "Summary text"
            store.close()

    def test_delete_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.append_turn("s1", "user", "hello")
            store.add_summary("s1", 0, "summary")

            store.delete_session("s1")

            log = store.get_or_create("s1")
            assert log.turns == []
            assert log.summaries == []
            store.close()

    def test_persistence_across_instances(self):
        """验证数据在创建新实例后仍然存在（持久化）。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "persist.db")

            # 第一个实例写入数据
            store1 = SQLiteSessionStore(db_path=db_path)
            store1.append_turn("s1", "user", "持久化测试")
            store1.append_turn("s1", "assistant", "数据已保存")
            store1.close()

            # 第二个实例读取数据
            store2 = SQLiteSessionStore(db_path=db_path)
            turns = store2.get_turns("s1")
            assert len(turns) == 2
            assert turns[0]["content"] == "持久化测试"
            store2.close()

    def test_multiple_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.append_turn("s1", "user", "session 1")
            store.append_turn("s2", "user", "session 2")

            assert len(store.get_turns("s1")) == 1
            assert len(store.get_turns("s2")) == 1
            assert store.get_turns("s1")[0]["content"] == "session 1"
            assert store.get_turns("s2")[0]["content"] == "session 2"
            store.close()
