"""测试图谱存储。"""

from src.memory.graph.graph_store import NetworkXGraphStore


class TestGraphStore:
    def test_add_and_search(self, sample_triples):
        store = NetworkXGraphStore()
        store.add(sample_triples)
        results = store.search("张三")
        assert len(results) >= 1

    def test_get_neighbors(self, sample_triples):
        store = NetworkXGraphStore()
        store.add(sample_triples)
        subgraph = store.get_neighbors("张三", depth=1)
        assert len(subgraph["nodes"]) >= 2

    def test_delete(self, sample_triples):
        store = NetworkXGraphStore()
        store.add(sample_triples)
        store.delete(["张三"])
        results = store.search("张三")
        assert len(results) == 0

    def test_get_all(self, sample_triples):
        store = NetworkXGraphStore()
        store.add(sample_triples)
        all_nodes = store.get_all()
        assert len(all_nodes) >= 3  # 张三, Google, 北京

    def test_edges_store_fact(self, sample_triples):
        store = NetworkXGraphStore()
        store.add(sample_triples)
        edges = store.get_all_edges()
        assert edges
        # 每条边都应有可读事实
        assert all(e.get("fact") for e in edges)
