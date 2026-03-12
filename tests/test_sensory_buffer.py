"""测试感觉记忆缓冲区。"""

from a_frame.memory.sensory.buffer import SensoryBuffer


class TestSensoryBuffer:
    def test_push_and_get(self):
        buf = SensoryBuffer(max_size=5)
        buf.push("hello")
        buf.push("world")
        items = buf.get_recent()
        assert len(items) == 2
        assert items[0].content == "hello"

    def test_max_size(self):
        buf = SensoryBuffer(max_size=3)
        for i in range(5):
            buf.push(f"item_{i}")
        assert len(buf) == 3
        items = buf.get_recent()
        assert items[0].content == "item_2"

    def test_get_recent_n(self):
        buf = SensoryBuffer(max_size=10)
        for i in range(5):
            buf.push(f"item_{i}")
        items = buf.get_recent(2)
        assert len(items) == 2
        assert items[0].content == "item_3"

    def test_clear(self):
        buf = SensoryBuffer()
        buf.push("test")
        buf.clear()
        assert len(buf) == 0
