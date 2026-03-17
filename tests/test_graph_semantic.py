"""测试图谱的 embedding 检索与同义节点合并。

覆盖场景：
- 查询“会议”能够召回节点“开会”（同义/近义）
- 构图时将“开会/会议”归一到同一个 canonical 节点（避免分裂子图）

说明：这里用一个可控的 FakeEmbedder，让同义词映射到同一向量。
"""

from a_frame.memory.graph.graph_store import NetworkXGraphStore


class FakeEmbedder:
    def embed(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str):
        text = (text or "").strip()
        if text in {"开会", "会议"}:
            return [1.0, 0.0, 0.0, 0.0]
        if text == "吃饭":
            return [0.0, 1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]


def test_semantic_search_recalls_synonym_node():
    store = NetworkXGraphStore(embedder=FakeEmbedder(), enable_semantic_search=True)
    store.add(
        [
            {
                "subject": {"name": "开会", "label": "Event"},
                "predicate": "at",
                "object": {"name": "周一", "label": "Time"},
            }
        ]
    )

    q_emb = FakeEmbedder().embed_query("会议")
    hits = store.search("会议", top_k=5, query_embedding=q_emb)
    assert hits
    assert hits[0]["node_id"] == "开会"


def test_semantic_merge_unifies_synonym_nodes_on_add():
    store = NetworkXGraphStore(
        embedder=FakeEmbedder(),
        enable_semantic_search=True,
        enable_semantic_merge=True,
        semantic_merge_threshold=0.95,
    )

    store.add(
        [
            {
                "subject": {"name": "开会", "label": "Event"},
                "predicate": "with",
                "object": {"name": "张三", "label": "Person"},
            }
        ]
    )
    # 第二次写入使用“会议”，应被归一到 canonical 节点“开会”
    store.add(
        [
            {
                "subject": {"name": "会议", "label": "Event"},
                "predicate": "with",
                "object": {"name": "李四", "label": "Person"},
            }
        ]
    )

    # 节点应只有一个 canonical（开会），并记录 alias
    node_ids = {n for n, _ in store.graph.nodes(data=True)}
    assert "开会" in node_ids
    assert "会议" not in node_ids
    aliases = store.graph.nodes["开会"].get("aliases", [])
    assert "会议" in aliases

    # 边的 source 应指向 canonical 节点
    sources = {u for u, _, _ in store.graph.edges(data=True)}
    assert "开会" in sources
    assert "会议" not in sources
