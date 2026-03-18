"""测试上下文压缩器。"""

from src.memory.working.compressor import WorkingMemoryCompressor


class FakeLLM:
    """假的 LLM，用于测试。"""

    def generate(self, messages, **kwargs):
        return (
            "## Intent Mapping\n测试\n## Progress Assessment\n完成\n## Recent Commands Analysis\n无"
        )


class TestCompressor:
    def setup_method(self):
        self.llm = FakeLLM()
        self.compressor = WorkingMemoryCompressor(
            llm=self.llm,
            max_tokens=1000,
            warn_threshold=0.7,
            block_threshold=0.9,
            min_recent_turns=2,
        )

    def test_count_tokens(self):
        msgs = [{"role": "user", "content": "hello world"}]
        count = self.compressor.count_tokens(msgs)
        assert count > 0

    def test_should_compress_none(self):
        assert self.compressor.should_compress(100) == "none"

    def test_should_compress_async(self):
        assert self.compressor.should_compress(750) == "async"

    def test_should_compress_sync(self):
        assert self.compressor.should_compress(950) == "sync"

    def test_find_compression_boundary(self, sample_turns):
        boundary = self.compressor.find_compression_boundary(sample_turns)
        # 8 turns total, keep 2*2=4, boundary = 4
        assert boundary == 4

    def test_compress(self, sample_turns):
        summary = self.compressor.compress(sample_turns, boundary_index=4)
        assert "Intent Mapping" in summary

    def test_render_context_no_summaries(self, sample_turns):
        result = self.compressor.render_context(sample_turns, [])
        assert result == sample_turns

    def test_render_context_with_summary(self, sample_turns):
        summaries = [{"boundary_index": 4, "content": "测试摘要"}]
        result = self.compressor.render_context(sample_turns, summaries)
        assert result[0]["role"] == "system"
        assert "测试摘要" in result[0]["content"]
        assert len(result) == 5  # 1 summary + 4 recent turns
