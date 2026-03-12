"""测试夹具。"""

import pytest


@pytest.fixture
def sample_turns():
    """示例对话数据。"""
    return [
        {"role": "user", "content": "你好，我想学习 Python"},
        {"role": "assistant", "content": "好的！你有编程基础吗？"},
        {"role": "user", "content": "有一些 Java 基础"},
        {"role": "assistant", "content": "很好，那我们从 Python 语法开始..."},
        {"role": "user", "content": "Python 的列表和 Java 数组有什么区别？"},
        {"role": "assistant", "content": "Python 列表更灵活，可以存储不同类型的元素..."},
        {"role": "user", "content": "明白了，那字典呢？"},
        {"role": "assistant", "content": "字典是 Python 中的哈希映射..."},
    ]


@pytest.fixture
def sample_triples():
    """示例三元组数据。"""
    return [
        {
            "subject": {"name": "张三", "label": "Person"},
            "predicate": "works_at",
            "object": {"name": "Google", "label": "Company"},
        },
        {
            "subject": {"name": "张三", "label": "Person"},
            "predicate": "lives_in",
            "object": {"name": "北京", "label": "City"},
        },
    ]
