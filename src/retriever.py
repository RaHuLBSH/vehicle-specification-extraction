"""Retrieve relevant document chunks for a natural-language query.

Embedding + FAISS search only — no Gemini / LLM extraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, TypedDict

from src.embeddings import embed_query
from src.vector_store import (
    DEFAULT_INDEX_PATH,
    DEFAULT_METADATA_PATH,
    VectorStore,
    VectorStoreError,
)


class RetrievalResult(TypedDict):
    """One ranked retrieval hit."""

    chunk_id: str
    text: str
    page: int
    section: str
    score: float


class Retriever:
    """Query a persisted FAISS index with natural-language text."""

    def __init__(
        self,
        store: VectorStore | None = None,
        *,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> None:
        """Create a retriever from an in-memory store or paths on disk.

        Args:
            store: Optional already-loaded :class:`VectorStore`.
            index_path: Path to ``manual.faiss`` when ``store`` is omitted.
            metadata_path: Path to ``metadata.json`` when ``store`` is omitted.
        """
        if store is not None:
            self.store = store
        else:
            self.store = VectorStore.load(
                index_path=index_path,
                metadata_path=metadata_path,
            )
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Embed ``query`` and return the top-k most similar chunks.

        Args:
            query: Natural-language question or lookup string.
            top_k: Maximum number of chunks to return (default 5).

        Returns:
            Results sorted by descending similarity score, each containing
            ``chunk_id``, ``text``, ``page``, ``section``, and ``score``.

        Raises:
            ValueError: If ``query`` is blank or ``top_k`` is invalid.
        """
        if not query or not str(query).strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_vector = embed_query(str(query).strip())
        raw_hits = self.store.search(query_vector, top_k=top_k)
        return _map_hits(raw_hits, metadata_size=self.store.size)


def _map_hits(
    raw_hits: list[dict[str, Any]],
    *,
    metadata_size: int,
) -> list[RetrievalResult]:
    """Map FAISS hits to the public retrieval schema; skip invalid indices."""
    results: list[RetrievalResult] = []
    for hit in raw_hits:
        faiss_id = hit.get("faiss_id")
        if faiss_id is not None:
            idx = int(faiss_id)
            if idx < 0 or idx >= metadata_size:
                continue

        try:
            score = float(hit["score"])
            page = int(hit["page"])
            chunk_id = str(hit["chunk_id"])
            text = str(hit["text"])
            section = str(hit["section"])
        except (KeyError, TypeError, ValueError):
            continue

        results.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "page": page,
                "section": section,
                "score": score,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def retrieve(
    query: str,
    top_k: int = 5,
    *,
    store: VectorStore | None = None,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> list[RetrievalResult]:
    """Embed a query and return the most relevant chunks from the vector store.

    Convenience wrapper around :class:`Retriever`.

    Args:
        query: Natural-language question or lookup string.
        top_k: Number of chunks to retrieve.
        store: Optional in-memory store (skips disk load).
        index_path: FAISS index path when loading from disk.
        metadata_path: Metadata JSON path when loading from disk.

    Returns:
        Ranked list of ``chunk_id`` / ``text`` / ``page`` / ``section`` / ``score``.
    """
    retriever = Retriever(
        store,
        index_path=index_path,
        metadata_path=metadata_path,
    )
    return retriever.retrieve(query, top_k=top_k)


def format_retrieval_results(
    query: str,
    results: list[RetrievalResult],
    *,
    max_text_chars: int = 500,
) -> str:
    """Pretty-print retrieval hits for CLI debugging."""
    lines = [
        f'Query: "{query}"',
        f"Hits: {len(results)}",
        "",
    ]
    if not results:
        lines.append("No results.")
        return "\n".join(lines)

    for rank, item in enumerate(results, start=1):
        text = item["text"]
        preview = text if len(text) <= max_text_chars else text[:max_text_chars] + "..."
        lines.extend(
            [
                f"--- Rank {rank} ---",
                f"score:   {item['score']:.4f}",
                f"page:    {item['page']}",
                f"section: {item['section']}",
                f"chunk:   {item['chunk_id']}",
                "text:",
                preview,
                "",
            ]
        )
    return "\n".join(lines)


def _run_interactive(retriever: Retriever, top_k: int) -> int:
    print("Retrieval debug mode (no Gemini).")
    print("Enter a query, or 'quit' to exit.")
    print()
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            return 0
        try:
            results = retriever.retrieve(query, top_k=top_k)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue
        print(format_retrieval_results(query, results))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug FAISS retrieval independently from Gemini extraction."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help='Optional one-shot query, e.g. "Torque for brake caliper bolts"',
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of hits (default: 5)")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to manual.faiss",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to metadata.json",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for multiple queries",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for retrieval-quality evaluation."""
    args = _build_arg_parser().parse_args(argv)
    try:
        retriever = Retriever(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
        )
    except VectorStoreError as exc:
        print(f"Error loading vector store: {exc}", file=sys.stderr)
        return 1

    if args.interactive or args.query is None:
        if args.query:
            results = retriever.retrieve(args.query, top_k=args.top_k)
            print(format_retrieval_results(args.query, results))
            print()
        return _run_interactive(retriever, top_k=args.top_k)

    try:
        results = retriever.retrieve(args.query, top_k=args.top_k)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_retrieval_results(args.query, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
