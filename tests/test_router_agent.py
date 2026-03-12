"""测试认知路由器。"""

from a_frame.agents.router_agent import RouterAgent


class FakeLLMRouter:
    """返回路由决策 JSON 的假 LLM。"""

    def generate(self, messages, **kwargs):
        # 检查用户消息内容来决定返回什么
        user_msg = messages[-1]["content"]
        if "工作" in user_msg or "实体" in user_msg:
            return '{"need_graph": true, "need_skills": false, "need_sensory": false, "reasoning": "实体关系查询"}'
        if "爬虫" in user_msg or "工具" in user_msg:
            return '{"need_graph": false, "need_skills": true, "need_sensory": false, "reasoning": "工具调用"}'
        if "刚才" in user_msg:
            return '{"need_graph": false, "need_skills": false, "need_sensory": true, "reasoning": "引用最近输入"}'
        return '{"need_graph": false, "need_skills": false, "need_sensory": false, "reasoning": "简单对话"}'


class FakeLLMBroken:
    def generate(self, messages, **kwargs):
        return "这不是JSON"


class TestRouterAgent:
    def test_route_graph(self):
        router = RouterAgent(llm=FakeLLMRouter())
        result = router.run(user_query="张三在哪里工作？")
        assert result["need_graph"] is True
        assert result["need_skills"] is False

    def test_route_skills(self):
        router = RouterAgent(llm=FakeLLMRouter())
        result = router.run(user_query="帮我写一个爬虫工具")
        assert result["need_skills"] is True

    def test_route_sensory(self):
        router = RouterAgent(llm=FakeLLMRouter())
        result = router.run(user_query="刚才说的方案")
        assert result["need_sensory"] is True

    def test_route_simple(self):
        router = RouterAgent(llm=FakeLLMRouter())
        result = router.run(user_query="你好")
        assert result["need_graph"] is False
        assert result["need_skills"] is False
        assert result["need_sensory"] is False

    def test_route_with_context(self):
        router = RouterAgent(llm=FakeLLMRouter())
        recent = [
            {"role": "user", "content": "介绍下Python"},
            {"role": "assistant", "content": "Python 是一门编程语言"},
        ]
        result = router.run(user_query="你好", recent_turns=recent)
        assert "reasoning" in result

    def test_route_broken_json_fallback(self):
        router = RouterAgent(llm=FakeLLMBroken())
        result = router.run(user_query="测试")
        # 应该返回安全默认值
        assert result["need_graph"] is False
        assert result["need_skills"] is False
        assert result["need_sensory"] is False

    def test_route_returns_all_fields(self):
        router = RouterAgent(llm=FakeLLMRouter())
        result = router.run(user_query="你好")
        assert "need_graph" in result
        assert "need_skills" in result
        assert "need_sensory" in result
        assert "reasoning" in result
