"""测试会话存储。"""

from src.memory.working.session_store import InMemorySessionStore


class TestSessionStore:
    def test_get_or_create(self):
        store = InMemorySessionStore()
        log = store.get_or_create("s1")
        assert log.session_id == "s1"
        assert log.turns == []

    def test_append_and_get_turns(self):
        store = InMemorySessionStore()
        store.append_turn("s1", "user", "hello")
        store.append_turn("s1", "assistant", "hi")
        turns = store.get_turns("s1")
        assert len(turns) == 2
        assert turns[0]["role"] == "user"

    def test_add_summary(self):
        store = InMemorySessionStore()
        store.get_or_create("s1")
        store.add_summary("s1", 4, "summary text")
        log = store.get_or_create("s1")
        assert len(log.summaries) == 1
        assert log.summaries[0]["content"] == "summary text"

    def test_delete_session(self):
        store = InMemorySessionStore()
        store.append_turn("s1", "user", "hello")
        store.delete_session("s1")
        turns = store.get_turns("s1")
        assert turns == []
