"""测试实体抽取器。"""

from a_frame.memory.graph.entity_extractor import EntityExtractor


class FakeLLMForExtraction:
    """返回固定三元组 JSON 的假 LLM。"""

    def generate(self, messages, **kwargs):
        return """[
  {
    "subject": {"name": "张三", "label": "Person"},
    "predicate": "works_at",
    "object": {"name": "Google", "label": "Organization"},
    "confidence": 0.9
  },
  {
    "subject": {"name": "张三", "label": "Person"},
    "predicate": "lives_in",
    "object": {"name": "北京", "label": "Place"},
    "confidence": 0.3
  }
]"""


class FakeLLMEmpty:
    def generate(self, messages, **kwargs):
        return "[]"


class FakeLLMBroken:
    def generate(self, messages, **kwargs):
        return "这不是有效的 JSON"


class TestEntityExtractor:
    def test_extract_filters_by_confidence(self):
        extractor = EntityExtractor(llm=FakeLLMForExtraction(), confidence_threshold=0.6)
        triples = extractor.extract("张三在 Google 工作，住在北京")
        # confidence 0.3 的应被过滤
        assert len(triples) == 1
        assert triples[0]["subject"]["name"] == "张三"
        assert triples[0]["predicate"] == "works_at"

    def test_extract_empty_text(self):
        extractor = EntityExtractor(llm=FakeLLMEmpty())
        assert extractor.extract("") == []

    def test_extract_empty_result(self):
        extractor = EntityExtractor(llm=FakeLLMEmpty())
        assert extractor.extract("你好世界") == []

    def test_extract_broken_json(self):
        extractor = EntityExtractor(llm=FakeLLMBroken())
        result = extractor.extract("一些文本")
        assert result == []

    def test_extract_from_turns(self):
        extractor = EntityExtractor(llm=FakeLLMForExtraction(), confidence_threshold=0.5)
        turns = [
            {"role": "user", "content": "张三在哪工作？"},
            {"role": "assistant", "content": "张三在 Google 工作"},
        ]
        triples = extractor.extract_from_turns(turns)
        assert len(triples) >= 1

    def test_parse_markdown_wrapped_json(self):
        """测试处理 markdown code block 包裹的 JSON。"""

        class MarkdownLLM:
            def generate(self, messages, **kwargs):
                return '```json\n[{"subject": {"name": "A", "label": "Concept"}, "predicate": "related_to", "object": {"name": "B", "label": "Concept"}, "confidence": 0.8}]\n```'

        extractor = EntityExtractor(llm=MarkdownLLM())
        result = extractor.extract("A 和 B 相关")
        assert len(result) == 1
