"""Tests for natural-language retrieval over FAISS."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.embeddings import EMBEDDING_DIMENSION
from src.retriever import (
    Retriever,
    format_retrieval_results,
    retrieve,
)
from src.vector_store import VectorStore


def _unit_vectors(n: int, dim: int = EMBEDDING_DIMENSION) -> np.ndarray:
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        vectors[i, i % dim] = 1.0
    return vectors


def _chunks() -> list[dict]:
    return [
        {
            "chunk_id": "chunk_0001",
            "text": "Front Brake caliper bolt torque: 28 Nm",
            "page": 42,
            "section": "Front Brake",
            "start_page": 42,
            "end_page": 42,
        },
        {
            "chunk_id": "chunk_0002",
            "text": "Engine oil capacity: 4.5 L grade 5W-30",
            "page": 10,
            "section": "Engine Oil",
            "start_page": 10,
            "end_page": 10,
        },
        {
            "chunk_id": "chunk_0003",
            "text": "Oil filter 90915-YZZD1",
            "page": 11,
            "section": "Lubrication",
            "start_page": 11,
            "end_page": 11,
        },
    ]


@pytest.fixture
def populated_store() -> VectorStore:
    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    store.add(_unit_vectors(3), _chunks())
    return store


def test_retrieve_returns_expected_schema(populated_store: VectorStore) -> None:
    query_vec = _unit_vectors(1)

    with patch("src.retriever.embed_query", return_value=query_vec):
        results = retrieve("Torque for brake caliper bolts", store=populated_store, top_k=2)

    assert len(results) == 2
    assert list(results[0].keys()) == ["chunk_id", "text", "page", "section", "score"]
    assert results[0]["chunk_id"] == "chunk_0001"
    assert results[0]["page"] == 42
    assert results[0]["section"] == "Front Brake"
    assert "28 Nm" in results[0]["text"]
    assert isinstance(results[0]["score"], float)


def test_results_sorted_by_score_descending(populated_store: VectorStore) -> None:
    # Query closer to second vector than first.
    query = np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)
    query[0, 1] = 1.0

    with patch("src.retriever.embed_query", return_value=query):
        results = Retriever(populated_store).retrieve("oil capacity", top_k=3)

    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["chunk_id"] == "chunk_0002"


def test_default_top_k_is_five(populated_store: VectorStore) -> None:
    # Only 3 docs exist, so top_k=5 still returns at most 3.
    with patch("src.retriever.embed_query", return_value=_unit_vectors(1)):
        results = retrieve("anything", store=populated_store)
    assert len(results) == 3


def test_rejects_blank_query(populated_store: VectorStore) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Retriever(populated_store).retrieve("   ")


def test_skips_invalid_faiss_indices(populated_store: VectorStore) -> None:
    bogus_hits = [
        {
            "chunk_id": "chunk_0001",
            "text": "ok",
            "page": 1,
            "section": "A",
            "score": 0.9,
            "faiss_id": -1,
        },
        {
            "chunk_id": "chunk_bad",
            "text": "bad",
            "page": 2,
            "section": "B",
            "score": 0.8,
            "faiss_id": 999,
        },
        {
            "chunk_id": "chunk_0002",
            "text": "kept",
            "page": 3,
            "section": "C",
            "score": 0.7,
            "faiss_id": 1,
        },
    ]
    with patch.object(populated_store, "search", return_value=bogus_hits):
        with patch("src.retriever.embed_query", return_value=_unit_vectors(1)):
            results = Retriever(populated_store).retrieve("q", top_k=5)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_0002"
    assert results[0]["text"] == "kept"


def test_format_retrieval_results_includes_debug_fields() -> None:
    text = format_retrieval_results(
        "Torque for brake caliper bolts",
        [
            {
                "chunk_id": "chunk_0001",
                "text": "caliper bolt torque: 28 Nm",
                "page": 42,
                "section": "Front Brake",
                "score": 0.87,
            }
        ],
    )
    assert "Rank 1" in text
    assert "0.8700" in text
    assert "page:    42" in text
    assert "Front Brake" in text
    assert "28 Nm" in text


def test_load_from_disk_paths(tmp_path: Path, populated_store: VectorStore) -> None:
    index_path = tmp_path / "manual.faiss"
    metadata_path = tmp_path / "metadata.json"
    populated_store.save(index_path=index_path, metadata_path=metadata_path)

    with patch("src.retriever.embed_query", return_value=_unit_vectors(1)):
        results = retrieve(
            "brake torque",
            top_k=1,
            index_path=index_path,
            metadata_path=metadata_path,
        )
    assert results[0]["section"] == "Front Brake"
