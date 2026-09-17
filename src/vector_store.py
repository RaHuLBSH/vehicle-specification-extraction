"""FAISS-backed vector store for chunk embeddings.

Uses ``IndexFlatIP`` so L2-normalized MiniLM vectors rank by cosine
similarity via inner product. Vectors live in FAISS; original chunk text
and page/section metadata are stored separately in JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
from numpy.typing import NDArray

from src.chunker import Chunk
from src.embeddings import EMBEDDING_DIMENSION

DEFAULT_INDEX_PATH = Path("storage/manual.faiss")
DEFAULT_METADATA_PATH = Path("storage/metadata.json")

EmbeddingArray = NDArray[np.float32]


class VectorStoreError(RuntimeError):
    """Raised when the on-disk index and metadata are missing or inconsistent."""


class VectorStore:
    """Persist and query document chunk embeddings with FAISS IndexFlatIP."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        """Create an empty in-memory store.

        Args:
            dimension: Embedding width (384 for all-MiniLM-L6-v2).
        """
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        """Number of indexed vectors / metadata rows."""
        return int(self.index.ntotal)

    def _validate_embeddings(self, embeddings: EmbeddingArray) -> EmbeddingArray:
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {array.shape}")
        if array.shape[1] != self.dimension:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.dimension}, "
                f"got {array.shape[1]}"
            )
        if not np.isfinite(array).all():
            raise ValueError("embeddings contain NaN or Inf values")
        return np.ascontiguousarray(array, dtype=np.float32)

    @staticmethod
    def _normalize_metadata_item(item: Chunk | dict[str, Any]) -> dict[str, Any]:
        """Ensure required chunk fields are present for persistence/retrieval."""
        required = ("chunk_id", "text", "page", "section", "start_page", "end_page")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"metadata missing required fields: {missing}")
        return {
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "page": item["page"],
            "section": item["section"],
            "start_page": item["start_page"],
            "end_page": item["end_page"],
        }

    def add(
        self,
        embeddings: EmbeddingArray | Sequence[Sequence[float]],
        metadata: Sequence[Chunk] | Sequence[dict[str, Any]],
    ) -> None:
        """Add embeddings and associated chunk metadata to the index.

        FAISS row ``i`` maps 1:1 to ``metadata[i]``.

        Args:
            embeddings: Array of shape ``(n, dimension)``.
            metadata: Parallel list of chunk dicts (must include text/page/section).
        """
        vectors = self._validate_embeddings(np.asarray(embeddings, dtype=np.float32))
        if len(metadata) != vectors.shape[0]:
            raise ValueError(
                f"metadata count ({len(metadata)}) does not match "
                f"embedding count ({vectors.shape[0]})"
            )
        records = [self._normalize_metadata_item(item) for item in metadata]
        self.index.add(vectors)
        self.metadata.extend(records)
        self._assert_consistent()

    def search(
        self,
        query_embedding: EmbeddingArray | Sequence[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for nearest-neighbor chunks by inner product.

        Args:
            query_embedding: Query vector of shape ``(dimension,)`` or ``(1, dimension)``.
            top_k: Number of results to return.

        Returns:
            Ranked list of dicts with chunk metadata plus ``score`` and ``rank``.
        """
        if self.size == 0:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query = self._validate_embeddings(np.asarray(query_embedding, dtype=np.float32))
        if query.shape[0] != 1:
            raise ValueError("query_embedding must contain exactly one vector")

        k = min(top_k, self.size)
        scores, indices = self.index.search(query, k)

        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or int(idx) >= len(self.metadata):
                continue
            item = dict(self.metadata[int(idx)])
            item["score"] = float(score)
            item["rank"] = rank
            item["faiss_id"] = int(idx)
            results.append(item)
        return results

    def _assert_consistent(self) -> None:
        """Fail clearly when FAISS size and metadata diverge."""
        if int(self.index.ntotal) != len(self.metadata):
            raise VectorStoreError(
                "Inconsistent vector store: "
                f"FAISS ntotal={self.index.ntotal} but metadata count={len(self.metadata)}"
            )
        if self.index.d != self.dimension:
            raise VectorStoreError(
                f"Inconsistent vector store: index dimension={self.index.d}, "
                f"expected={self.dimension}"
            )

    def save(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> None:
        """Persist the FAISS index and chunk metadata to disk.

        Args:
            index_path: Destination for ``manual.faiss``.
            metadata_path: Destination for ``metadata.json``.
        """
        self._assert_consistent()
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        payload = {
            "dimension": self.dimension,
            "count": len(self.metadata),
            "chunks": self.metadata,
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> VectorStore:
        """Load an existing index and metadata, validating consistency.

        Args:
            index_path: Path to ``manual.faiss``.
            metadata_path: Path to ``metadata.json``.

        Returns:
            Populated :class:`VectorStore`.

        Raises:
            VectorStoreError: If files are missing, unreadable, or inconsistent.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.is_file():
            raise VectorStoreError(f"FAISS index not found: {index_path}")
        if not metadata_path.is_file():
            raise VectorStoreError(f"Metadata file not found: {metadata_path}")

        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:  # faiss raises generic errors on corrupt files
            raise VectorStoreError(f"Corrupt or unreadable FAISS index: {index_path}") from exc

        if not isinstance(index, faiss.IndexFlatIP) and index.__class__.__name__ != "IndexFlatIP":
            # Some FAISS builds wrap indexes; still require IP flat metric.
            if getattr(index, "metric_type", None) not in (None, faiss.METRIC_INNER_PRODUCT):
                raise VectorStoreError(
                    f"Unsupported FAISS index type at {index_path}: {type(index)!r}"
                )

        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VectorStoreError(
                f"Corrupt metadata JSON (invalid JSON): {metadata_path}"
            ) from exc
        except OSError as exc:
            raise VectorStoreError(f"Unable to read metadata: {metadata_path}") from exc

        if not isinstance(raw, dict):
            raise VectorStoreError(
                f"Corrupt metadata JSON: expected object at {metadata_path}"
            )

        chunks = raw.get("chunks")
        if not isinstance(chunks, list):
            raise VectorStoreError(
                f"Corrupt metadata JSON: missing 'chunks' list in {metadata_path}"
            )

        dimension = int(raw.get("dimension", index.d))
        store = cls(dimension=dimension)
        store.index = index  # type: ignore[assignment]
        store.metadata = []
        for item in chunks:
            if not isinstance(item, dict):
                raise VectorStoreError("Corrupt metadata: chunk entry is not an object")
            store.metadata.append(cls._normalize_metadata_item(item))

        declared = raw.get("count")
        if declared is not None and int(declared) != len(store.metadata):
            raise VectorStoreError(
                f"Corrupt metadata: declared count={declared} "
                f"but chunks list length={len(store.metadata)}"
            )

        if int(store.index.ntotal) != len(store.metadata):
            raise VectorStoreError(
                "Inconsistent store on load: "
                f"FAISS ntotal={store.index.ntotal} != metadata count={len(store.metadata)}"
            )
        if store.index.d != store.dimension:
            raise VectorStoreError(
                f"Inconsistent store on load: index.d={store.index.d} "
                f"!= metadata dimension={store.dimension}"
            )
        return store

    def debug_info(
        self,
        *,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> dict[str, Any]:
        """Return current in-memory store diagnostics."""
        self._assert_consistent()
        info = {
            "embedding_dimension": self.dimension,
            "faiss_ntotal": int(self.index.ntotal),
            "metadata_count": len(self.metadata),
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
        }
        return info


def debug_vector_store(
    store: VectorStore,
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    reload: bool = True,
) -> dict[str, Any]:
    """Print store diagnostics and optionally verify save/load round-trip.

    Args:
        store: Populated vector store.
        index_path: Path used for save/reload checks.
        metadata_path: Metadata JSON path.
        reload: When True, save then reload and compare sizes.

    Returns:
        Diagnostic dictionary including reload verification flags.
    """
    info = store.debug_info(index_path=index_path, metadata_path=metadata_path)
    print(f"Embedding dimension: {info['embedding_dimension']}")
    print(f"FAISS ntotal: {info['faiss_ntotal']}")
    print(f"Metadata count: {info['metadata_count']}")
    print(f"Saved index path: {info['index_path']}")
    print(f"Saved metadata path: {info['metadata_path']}")

    if reload:
        store.save(index_path=index_path, metadata_path=metadata_path)
        reloaded = VectorStore.load(index_path=index_path, metadata_path=metadata_path)
        ok = (
            reloaded.size == store.size
            and reloaded.dimension == store.dimension
            and reloaded.metadata == store.metadata
        )
        info["reload_verified"] = ok
        info["reloaded_ntotal"] = reloaded.size
        info["reloaded_metadata_count"] = len(reloaded.metadata)
        print(f"Reload verification: {'OK' if ok else 'FAILED'}")
        print(f"Reloaded FAISS ntotal: {reloaded.size}")
        print(f"Reloaded metadata count: {len(reloaded.metadata)}")
    else:
        info["reload_verified"] = None

    return info


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/debug a FAISS IndexFlatIP store from sample embeddings."
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path for manual.faiss",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path for metadata.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI: create a tiny demo index, save it, and verify reload."""
    args = _build_arg_parser().parse_args(argv)

    # Deterministic unit vectors for a smoke demo (no model download required).
    dim = EMBEDDING_DIMENSION
    vectors = np.zeros((3, dim), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    vectors[2, 2] = 1.0
    chunks: list[Chunk] = [
        {
            "chunk_id": "chunk_0001",
            "text": "Front Brake wheel nut torque: 108 Nm",
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

    store = VectorStore(dimension=dim)
    store.add(vectors, chunks)
    try:
        debug_vector_store(
            store,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            reload=True,
        )
    except VectorStoreError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
