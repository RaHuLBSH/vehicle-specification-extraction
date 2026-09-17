"""Embedding generation with sentence-transformers.

Uses ``all-MiniLM-L6-v2`` (384-dimensional) and returns L2-normalized
``float32`` NumPy arrays suitable for FAISS inner-product search.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.chunker import Chunk

# --- Central model configuration -------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output size
NORMALIZE_EMBEDDINGS = True
DEFAULT_BATCH_SIZE = 32

EmbeddingArray = NDArray[np.float32]

_model: SentenceTransformer | None = None


def get_embedding_config() -> dict[str, object]:
    """Return the centralized embedding configuration."""
    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "dimension": EMBEDDING_DIMENSION,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "batch_size": DEFAULT_BATCH_SIZE,
    }


def get_model() -> SentenceTransformer:
    """Load and return the shared SentenceTransformer instance.

    The model is loaded once per process and reused for all subsequent
    embedding calls.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def reset_model_cache() -> None:
    """Drop the cached model (intended for tests)."""
    global _model
    _model = None


def _as_float32_matrix(vectors: np.ndarray) -> EmbeddingArray:
    """Cast embeddings to a contiguous float32 2-D array for FAISS."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return np.ascontiguousarray(array, dtype=np.float32)


def _validate_dimension(embeddings: EmbeddingArray) -> None:
    """Ensure embeddings match the expected MiniLM dimension."""
    if embeddings.shape[-1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension {EMBEDDING_DIMENSION}, "
            f"got {embeddings.shape[-1]}"
        )


def embed_texts(
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress_bar: bool = False,
) -> EmbeddingArray:
    """Embed one or more texts in batches.

    Args:
        texts: Strings to embed (queries or document passages).
        batch_size: Encode batch size passed to SentenceTransformer.
        show_progress_bar: Forwarded to the underlying encode call.

    Returns:
        Array of shape ``(n_texts, 384)`` with ``dtype=float32``,
        L2-normalized when ``NORMALIZE_EMBEDDINGS`` is True.

    Raises:
        ValueError: If ``texts`` is empty.
    """
    if not texts:
        raise ValueError("texts must contain at least one string")

    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    embeddings = _as_float32_matrix(vectors)
    _validate_dimension(embeddings)
    return embeddings


def embed_documents(
    chunks: Sequence[Chunk] | Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress_bar: bool = False,
) -> EmbeddingArray:
    """Embed document chunks for indexing.

    Args:
        chunks: Chunk dicts (uses ``text``) or raw strings.
        batch_size: Encode batch size.
        show_progress_bar: Forwarded to encode.

    Returns:
        Float32 array of shape ``(n_chunks, 384)``.
    """
    texts: list[str] = []
    for item in chunks:
        if isinstance(item, str):
            texts.append(item)
        else:
            texts.append(item["text"])
    return embed_texts(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )


def embed_query(query: str) -> EmbeddingArray:
    """Embed a single retrieval query.

    Args:
        query: Natural-language query string.

    Returns:
        Float32 array of shape ``(1, 384)``, L2-normalized.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    return embed_texts([query])


def debug_embeddings(
    texts: Sequence[str],
    *,
    sample_norms: int = 5,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, object]:
    """Embed texts and print shape / dtype / norm diagnostics.

    Args:
        texts: Chunk (or sample) strings to embed.
        sample_norms: How many vector L2 norms to print.
        batch_size: Encode batch size.

    Returns:
        Summary dict with counts, shape, dimension, dtype, and norms.
    """
    embeddings = embed_texts(texts, batch_size=batch_size)
    norms = np.linalg.norm(embeddings, axis=1)
    n_show = min(sample_norms, len(norms))

    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print(f"Number of chunks embedded: {embeddings.shape[0]}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Expected dimension: {EMBEDDING_DIMENSION}")
    print(f"dtype: {embeddings.dtype}")
    print(f"normalize_embeddings: {NORMALIZE_EMBEDDINGS}")
    print("Vector norms (first samples):")
    for index, value in enumerate(norms[:n_show]):
        print(f"  [{index}] {float(value):.6f}")

    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "num_chunks": int(embeddings.shape[0]),
        "shape": tuple(int(x) for x in embeddings.shape),
        "dimension": int(embeddings.shape[1]),
        "expected_dimension": EMBEDDING_DIMENSION,
        "dtype": str(embeddings.dtype),
        "norms": [float(v) for v in norms[:n_show]],
        "embeddings": embeddings,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug chunk embeddings with all-MiniLM-L6-v2."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=None,
        help="Optional PDF to parse/clean/chunk before embedding",
    )
    parser.add_argument(
        "--sample-texts",
        nargs="+",
        default=None,
        help="Embed these strings instead of a PDF",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sample-norms", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for embedding diagnostics."""
    args = _build_arg_parser().parse_args(argv)

    if args.sample_texts:
        texts = list(args.sample_texts)
    elif args.pdf_path:
        from src.chunker import chunk_pages
        from src.pdf_parser import extract_pages
        from src.text_cleaner import clean_pages

        try:
            pages = clean_pages(extract_pages(args.pdf_path))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        chunks = chunk_pages(pages)
        texts = [chunk["text"] for chunk in chunks]
        if not texts:
            print("No chunks produced from PDF.", file=sys.stderr)
            return 1
    else:
        texts = [
            "Wheel nut torque: 108 Nm",
            "Engine oil capacity: 4.5 L grade 5W-30",
            "Oil filter part number 90915-YZZD1",
        ]

    debug_embeddings(
        texts,
        sample_norms=args.sample_norms,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
