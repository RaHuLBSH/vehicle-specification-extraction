"""Tests for the FAISS vector store."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from src.embeddings import EMBEDDING_DIMENSION
from src.vector_store import VectorStore, VectorStoreError, debug_vector_store


def _unit_vectors(n: int, dim: int = EMBEDDING_DIMENSION) -> np.ndarray:
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        vectors[i, i % dim] = 1.0
    return vectors


def _chunks(n: int) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk_{i + 1:04d}",
            "text": f"Spec chunk {i + 1}: torque {100 + i} Nm",
            "page": 40 + i,
            "section": "Front Brake" if i % 2 == 0 else "Engine Oil",
            "start_page": 40 + i,
            "end_page": 40 + i,
        }
        for i in range(n)
    ]


def test_add_and_search_returns_metadata(tmp_path: Path) -> None:
    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    vectors = _unit_vectors(3)
    meta = _chunks(3)
    store.add(vectors, meta)

    results = store.search(vectors[0], top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk_0001"
    assert "108 Nm" in results[0]["text"] or "100 Nm" in results[0]["text"]
    assert results[0]["page"] == 40
    assert results[0]["section"] == "Front Brake"
    assert "score" in results[0]
    assert results[0]["faiss_id"] == 0


def test_add_rejects_mismatched_counts() -> None:
    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    with pytest.raises(ValueError, match="does not match"):
        store.add(_unit_vectors(2), _chunks(1))


def test_add_rejects_wrong_dimension() -> None:
    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    bad = np.zeros((1, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add(bad, _chunks(1))


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    index_path = tmp_path / "manual.faiss"
    metadata_path = tmp_path / "metadata.json"

    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    store.add(_unit_vectors(3), _chunks(3))
    store.save(index_path=index_path, metadata_path=metadata_path)

    assert index_path.is_file()
    assert metadata_path.is_file()

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["count"] == 3
    assert payload["dimension"] == EMBEDDING_DIMENSION
    assert payload["chunks"][0]["text"]

    loaded = VectorStore.load(index_path=index_path, metadata_path=metadata_path)
    assert loaded.size == 3
    assert loaded.dimension == EMBEDDING_DIMENSION
    assert loaded.metadata == store.metadata
    assert loaded.search(_unit_vectors(1)[0], top_k=1)[0]["chunk_id"] == "chunk_0001"


def test_load_fails_when_index_missing(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"dimension": EMBEDDING_DIMENSION, "count": 0, "chunks": []}),
        encoding="utf-8",
    )
    with pytest.raises(VectorStoreError, match="FAISS index not found"):
        VectorStore.load(index_path=tmp_path / "missing.faiss", metadata_path=metadata_path)


def test_load_fails_on_count_mismatch(tmp_path: Path) -> None:
    index_path = tmp_path / "manual.faiss"
    metadata_path = tmp_path / "metadata.json"

    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    store.add(_unit_vectors(2), _chunks(2))
    store.save(index_path=index_path, metadata_path=metadata_path)

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["chunks"] = payload["chunks"][:1]  # drop one metadata row
    payload["count"] = 1
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VectorStoreError, match="Inconsistent store on load"):
        VectorStore.load(index_path=index_path, metadata_path=metadata_path)


def test_load_fails_on_corrupt_metadata_json(tmp_path: Path) -> None:
    index_path = tmp_path / "manual.faiss"
    metadata_path = tmp_path / "metadata.json"

    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    store.add(_unit_vectors(1), _chunks(1))
    store.save(index_path=index_path, metadata_path=metadata_path)
    metadata_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(VectorStoreError, match="Corrupt metadata JSON"):
        VectorStore.load(index_path=index_path, metadata_path=metadata_path)


def test_load_fails_on_corrupt_faiss_index(tmp_path: Path) -> None:
    index_path = tmp_path / "manual.faiss"
    metadata_path = tmp_path / "metadata.json"
    index_path.write_bytes(b"not-a-faiss-index")
    metadata_path.write_text(
        json.dumps(
            {
                "dimension": EMBEDDING_DIMENSION,
                "count": 1,
                "chunks": _chunks(1),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(VectorStoreError, match="Corrupt or unreadable FAISS index"):
        VectorStore.load(index_path=index_path, metadata_path=metadata_path)


def test_debug_vector_store_reload(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index_path = tmp_path / "manual.faiss"
    metadata_path = tmp_path / "metadata.json"
    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    store.add(_unit_vectors(2), _chunks(2))

    info = debug_vector_store(
        store,
        index_path=index_path,
        metadata_path=metadata_path,
        reload=True,
    )
    assert info["embedding_dimension"] == EMBEDDING_DIMENSION
    assert info["faiss_ntotal"] == 2
    assert info["metadata_count"] == 2
    assert info["reload_verified"] is True
    out = capsys.readouterr().out
    assert "FAISS ntotal: 2" in out
    assert "Reload verification: OK" in out


def test_uses_index_flat_ip() -> None:
    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    assert isinstance(store.index, faiss.IndexFlatIP)
    assert store.index.d == EMBEDDING_DIMENSION
