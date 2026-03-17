"""测试工作记忆管理器。"""

from a_frame.agents.wm_manager import WMManager
from a_frame.memory.working.compressor import WorkingMemoryCompressor
from a_frame.memory.working.session_store import InMemorySessionStore


class FakeLLM:
    def generate(self, messages, **kwargs):
        return "## Intent Mapping\n测试\n## Progress Assessment\n进行中\n## Recent Commands Analysis\n无"


class TestWMManager:
    def setup_method(self):
        self.store = InMemorySessionStore()
        self.compressor = WorkingMemoryCompressor(
            llm=FakeLLM(),
            max_tokens=500,
            warn_threshold=0.7,
            block_threshold=0.9,
            min_recent_turns=2,
        )
        self.wm = WMManager(session_store=self.store, compressor=self.compressor)

    def test_first_query(self):
        result = self.wm.run(session_id="s1", user_query="你好")
        assert "compressed_history" in result
        assert "raw_recent_turns" in result
        assert "wm_token_usage" in result
        assert result["wm_token_usage"] > 0

    def test_multiple_turns(self):
        self.wm.run(session_id="s1", user_query="第一轮")
        self.wm.append_assistant_turn("s1", "回复第一轮")
        self.wm.run(session_id="s1", user_query="第二轮")
        self.wm.append_assistant_turn("s1", "回复第二轮")

        turns = self.store.get_turns("s1")
        assert len(turns) == 4

    def test_compression_triggered(self):
        """构造长对话触发压缩。"""
        # 填充大量对话（每条消息较长以触发 token 阈值）
        long_text = "这是一段很长的文字内容。" * 50
        for i in range(10):
            self.wm.run(session_id="s1", user_query=f"{long_text} 第{i}轮")
            self.wm.append_assistant_turn("s1", f"回复 {long_text}")

        self.wm.run(session_id="s1", user_query="最新查询")

        # 应该有摘要产生
        log = self.store.get_or_create("s1")
        assert len(log.summaries) > 0

    def test_append_assistant_turn(self):
        self.wm.run(session_id="s1", user_query="你好")
        self.wm.append_assistant_turn("s1", "你好呀")
        turns = self.store.get_turns("s1")
        assert turns[-1]["role"] == "assistant"
        assert turns[-1]["content"] == "你好呀"

    def test_separate_sessions(self):
        self.wm.run(session_id="s1", user_query="会话1的消息")
        self.wm.run(session_id="s2", user_query="会话2的消息")
        assert len(self.store.get_turns("s1")) == 1
        assert len(self.store.get_turns("s2")) == 1
