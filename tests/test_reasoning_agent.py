"""测试核心推理器。"""

from a_frame.agents.reasoning_agent import ReasoningAgent


class FakeLLMReasoner:
    def generate(self, messages, **kwargs):
        # 检查是否注入了背景知识
        has_background = any("背景知识" in m.get("content", "") or "记忆" in m.get("content", "") for m in messages)
        if has_background:
            return "根据记忆，张三在 Google 工作。"
        return "你好！有什么可以帮助你的？"


class TestReasoningAgent:
    def test_simple_response(self):
        agent = ReasoningAgent(llm=FakeLLMReasoner())
        result = agent.run(user_query="你好")
        assert "final_response" in result
        assert len(result["final_response"]) > 0

    def test_with_background_context(self):
        agent = ReasoningAgent(llm=FakeLLMReasoner())
        result = agent.run(
            user_query="张三在哪工作？",
            background_context="张三在 Google 工作，是一名工程师。",
        )
        assert "final_response" in result

    def test_with_compressed_history(self):
        agent = ReasoningAgent(llm=FakeLLMReasoner())
        history = [
            {"role": "system", "content": "[历史摘要] 用户之前问了Python问题"},
            {"role": "user", "content": "继续讲讲"},
            {"role": "assistant", "content": "好的，接下来..."},
        ]
        result = agent.run(
            user_query="还有什么？",
            compressed_history=history,
        )
        assert "final_response" in result

    def test_no_duplicate_user_query(self):
        """确保当 compressed_history 末尾已包含用户查询时不重复。"""

        class SpyLLM:
            def __init__(self):
                self.last_messages = None

            def generate(self, messages, **kwargs):
                self.last_messages = messages
                return "回答"

        spy = SpyLLM()
        agent = ReasoningAgent(llm=spy)
        history = [
            {"role": "user", "content": "我的问题"},
        ]
        agent.run(user_query="我的问题", compressed_history=history)
        # 最后一条 user 消息应该只出现一次
        user_msgs = [m for m in spy.last_messages if m["role"] == "user"]
        assert len(user_msgs) == 1
