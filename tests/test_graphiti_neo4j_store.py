from src.memory.graph.graphiti_neo4j_store import escape_lucene_query


def test_escape_lucene_query_escapes_slash_and_colon():
    escaped = escape_lucene_query("上次项目的CI/CD配置: 关联信息")
    assert escaped == "上次项目的CI\\/CD配置\\: 关联信息"


def test_escape_lucene_query_escapes_boolean_and_parentheses():
    escaped = escape_lucene_query("foo && bar (baz)")
    assert escaped == r"foo \&\& bar \(baz\)"


def test_escape_lucene_query_returns_empty_for_blank_input():
    assert escape_lucene_query("   ") == ""
