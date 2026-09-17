"""Tests for the SentenceTransformer embedding layer.

Unit tests mock the model so they run offline. An optional integration
test loads the real ``all-MiniLM-L6-v2`` weights when available.
"""

from __future__ import annotations

from typing import Sequence
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    debug_embeddings,
    embed_documents,
    embed_query,
    embed_texts,
    get_embedding_config,
    get_model,
    reset_model_cache,
)


def _fake_encode(
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False,
    convert_to_numpy: bool = True,
) -> np.ndarray:
    """Deterministic stand-in for SentenceTransformer.encode."""
    del batch_size, show_progress_bar, convert_to_numpy
    vectors = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
    for index, text in enumerate(texts):
        # Unique non-zero pattern per text; optional L2 normalize.
        vectors[index, index % EMBEDDING_DIMENSION] = float(len(text) + 1)
        vectors[index, (index + 1) % EMBEDDING_DIMENSION] = 1.0
    if normalize_embeddings:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
    return vectors.astype(np.float32)


@pytest.fixture
def mock_model():
    reset_model_cache()
    fake = MagicMock()
    fake.encode.side_effect = _fake_encode
    with patch("src.embeddings.SentenceTransformer", return_value=fake) as ctor:
        yield fake, ctor
    reset_model_cache()


@pytest.fixture
def sample_texts() -> list[str]:
    return [
        "Front Brake wheel nut torque: 108 Nm",
        "Engine oil capacity: 4.5 L using 5W-30",
        "Oil filter 90915-YZZD1 bolt M10x1.25",
    ]


def test_config_is_centralized() -> None:
    config = get_embedding_config()
    assert config["model_name"] == EMBEDDING_MODEL_NAME
    assert config["dimension"] == EMBEDDING_DIMENSION == 384
    assert config["normalize_embeddings"] is True


def test_model_loads_once(mock_model) -> None:
    fake, ctor = mock_model
    first = get_model()
    second = get_model()
    assert first is second is fake
    ctor.assert_called_once_with(EMBEDDING_MODEL_NAME)


def test_embed_texts_shape_dtype_and_dimension(mock_model, sample_texts: list[str]) -> None:
    vectors = embed_texts(sample_texts, batch_size=2)
    assert isinstance(vectors, np.ndarray)
    assert vectors.dtype == np.float32
    assert vectors.shape == (len(sample_texts), EMBEDDING_DIMENSION)
    assert vectors.shape[1] == 384
    fake, _ = mock_model
    assert fake.encode.call_args.kwargs["normalize_embeddings"] is True
    assert fake.encode.call_args.kwargs["batch_size"] == 2


def test_embeddings_are_normalized(mock_model, sample_texts: list[str]) -> None:
    vectors = embed_texts(sample_texts)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_documents_accepts_chunk_dicts(mock_model, sample_texts: list[str]) -> None:
    chunks = [
        {
            "chunk_id": "chunk_0001",
            "text": sample_texts[0],
            "page": 1,
            "section": "Front Brake",
            "start_page": 1,
            "end_page": 1,
        }
    ]
    vectors = embed_documents(chunks)
    assert vectors.shape == (1, 384)
    assert vectors.dtype == np.float32


def test_embed_query_shape(mock_model) -> None:
    vector = embed_query("What is the wheel nut torque?")
    assert vector.shape == (1, EMBEDDING_DIMENSION)
    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_embed_texts_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        embed_texts([])


def test_embed_query_rejects_blank() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        embed_query("   ")


def test_debug_embeddings(mock_model, sample_texts: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    summary = debug_embeddings(sample_texts, sample_norms=3)
    assert summary["dimension"] == 384
    assert summary["expected_dimension"] == 384
    assert summary["num_chunks"] == len(sample_texts)
    assert summary["dtype"] == "float32"
    out = capsys.readouterr().out
    assert "Embedding dimension: 384" in out
    assert "dtype: float32" in out


@pytest.mark.integration
def test_real_minilm_dimension() -> None:
    """Live check against downloaded all-MiniLM-L6-v2 weights (384-d)."""
    reset_model_cache()
    try:
        vectors = embed_texts(["wheel nut torque 108 Nm"])
    except Exception as exc:  # pragma: no cover - depends on hub access
        pytest.skip(f"Real model unavailable: {exc}")
    finally:
        reset_model_cache()

    assert vectors.shape == (1, 384)
    assert vectors.dtype == np.float32
    assert np.isclose(np.linalg.norm(vectors), 1.0, atol=1e-5)
