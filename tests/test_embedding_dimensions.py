from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

from main import _create_embedder
from src.core.factory import _resolve_embedding_dim
from src.embedder.litellm_embedder import LiteLLMEmbedder
from src.memory.semantic.vector_index import LanceVectorIndex


class _ProbeEmbedder:
    def __init__(self, dimension: int) -> None:
        self.output_dimension = dimension
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [0.0] * self.output_dimension


def _settings(dimension: int) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_backend="litellm",
        embedding_model="openai/text-embedding-3-small",
        embedding_api_key="test-key",
        embedding_api_base="http://embedding.invalid/v1",
        embedding_local=False,
        embedding_device="cpu",
        embedding_dim=dimension,
    )


def _table_dim(index: LanceVectorIndex, table_name: str) -> int:
    vector_type = index._db.open_table(table_name).schema.field("vector").type
    assert pa.types.is_fixed_size_list(vector_type)
    return int(vector_type.list_size)


def test_litellm_embedder_receives_configured_dimensions() -> None:
    embedder = _create_embedder(_settings(1024))

    assert isinstance(embedder, LiteLLMEmbedder)
    assert embedder._dimensions == 1024


def test_factory_uses_and_caches_actual_probe_dimension() -> None:
    embedder = _ProbeEmbedder(1536)

    resolved = _resolve_embedding_dim(embedder, _settings(1024))
    resolved_again = _resolve_embedding_dim(embedder, _settings(1024))

    assert resolved == 1536
    assert resolved_again == 1536
    assert embedder.calls == 1


def test_empty_lance_tables_are_rebuilt_for_actual_dimension(tmp_path) -> None:
    db_path = str(tmp_path / "vectors")
    first = LanceVectorIndex(db_path=db_path, embedding_dim=1024)
    assert _table_dim(first, first.MEMORY_TABLE) == 1024

    second = LanceVectorIndex(db_path=db_path, embedding_dim=1536)
    assert _table_dim(second, second.MEMORY_TABLE) == 1536
    assert _table_dim(second, second.EVIDENCE_TABLE) == 1536
    assert _table_dim(second, second.EPISODE_TABLE) == 1536


def test_wrong_vector_dimension_fails_before_replacing_existing_row(tmp_path) -> None:
    index = LanceVectorIndex(db_path=str(tmp_path / "vectors"), embedding_dim=4)
    index.upsert(
        "record-1",
        "fact",
        "semantic",
        "normalized",
        semantic_vector=[1.0, 0.0, 0.0, 0.0],
        normalized_vector=[1.0, 0.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="expected=4, actual=3"):
        index.upsert(
            "record-1",
            "fact",
            "replacement",
            "replacement",
            semantic_vector=[1.0, 0.0, 0.0],
            normalized_vector=[1.0, 0.0, 0.0],
        )

    table = index._db.open_table(index.MEMORY_TABLE)
    assert table.count_rows() == 1
    assert table.to_arrow().column("record_id").to_pylist() == ["record-1"]


def test_nonempty_dimension_mismatch_requires_explicit_rebuild(tmp_path) -> None:
    db_path = str(tmp_path / "vectors")
    index = LanceVectorIndex(db_path=db_path, embedding_dim=4)
    index.upsert(
        "record-1",
        "fact",
        "semantic",
        "normalized",
        semantic_vector=[1.0, 0.0, 0.0, 0.0],
        normalized_vector=[1.0, 0.0, 0.0, 0.0],
    )

    with pytest.raises(RuntimeError, match="stored=4, active=3"):
        LanceVectorIndex(db_path=db_path, embedding_dim=3)
