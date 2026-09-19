"""Retrieve relevant document chunks for a natural-language query.

Embedding + FAISS search only — no Gemini / LLM extraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, TypedDict
from typing_extensions import NotRequired

from src.embeddings import embed_query
from src.vector_store import (
    DEFAULT_INDEX_PATH,
    DEFAULT_METADATA_PATH,
    VectorStore,
    VectorStoreError,
)
from src.bm25_store import BM25Store

DEFAULT_CANDIDATE_K = 20
RRF_K = 60


class RetrievalResult(TypedDict):
    chunk_id: str
    text: str
    page: int
    section: str
    score: float

    faiss_score: NotRequired[float]
    faiss_rank: NotRequired[int]

    bm25_score: NotRequired[float]
    bm25_rank: NotRequired[int]

    rrf_score: NotRequired[float]

class Retriever:
    """Hybrid FAISS + BM25 retriever using Reciprocal Rank Fusion."""

    def __init__(
        self,
        store: VectorStore | None = None,
        *,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> None:

        if store is not None:
            self.store = store
        else:
            self.store = VectorStore.load(
                index_path=index_path,
                metadata_path=metadata_path,
            )

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.bm25_store = BM25Store(self.store.metadata)


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> list[RetrievalResult]:
        """Retrieve and fuse semantic FAISS and lexical BM25 candidates."""

        if not query or not str(query).strip():
            raise ValueError(
                "query must be a non-empty string"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be a positive integer"
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be a positive integer"
            )

        query = str(query).strip()

        # Candidate count must be at least as large
        # as the requested final result count.
        candidate_k = max(
            candidate_k,
            top_k,
        )

        # ---------------------------------------------------------
        # Stage 1A: FAISS dense retrieval
        # ---------------------------------------------------------

        query_vector = embed_query(query)

        raw_hits = self.store.search(
            query_vector,
            top_k=candidate_k,
        )

        faiss_candidates = _map_hits(
            raw_hits,
            metadata_size=self.store.size,
        )

        print("\n--- FAISS CANDIDATES ---")

        for rank, candidate in enumerate(faiss_candidates, start=1):
            print(
                f"{rank:2d}. "
                f"page={candidate['page']} "
                f"score={candidate['score']:.4f} "
                f"section={candidate['section']}"
            )


        # ---------------------------------------------------------
        # Stage 1B: BM25 lexical retrieval
        # ---------------------------------------------------------

        bm25_candidates = self.bm25_store.search(
            query,
            top_k=candidate_k,
        )

        print("\n--- BM25 CANDIDATES ---")

        for rank, candidate in enumerate(bm25_candidates, start=1):
            print(
                f"{rank:2d}. "
                f"page={candidate['page']} "
                f"score={candidate['bm25_score']:.4f} "
                f"section={candidate['section']}"
            )


        # ---------------------------------------------------------
        # Stage 2: Reciprocal Rank Fusion
        # ---------------------------------------------------------

        fused_candidates = reciprocal_rank_fusion(
            faiss_candidates,
            bm25_candidates,
        )

        # Allow BM25-only candidates to survive into reranking.

        print("\n--- RRF FUSED CANDIDATES ---")

        for rank, candidate in enumerate(fused_candidates, start=1):
            print(
                f"{rank:2d}. "
                f"page={candidate['page']} "
                f"rrf={candidate.get('rrf_score', 0.0):.6f} "
                f"faiss_rank={candidate.get('faiss_rank')} "
                f"bm25_rank={candidate.get('bm25_rank')} "
                f"section={candidate['section']}"
            )

        # No CrossEncoder:
        # RRF score becomes the final ranking score.
        results = []

        for candidate in fused_candidates[:top_k]:
            result = dict(candidate)
            result["score"] = float(
                candidate.get("rrf_score", 0.0)
            )
            results.append(result)

        return results



def reciprocal_rank_fusion(
    faiss_results: list[RetrievalResult],
    bm25_results: list[dict[str, Any]],
    *,
    rrf_k: int = RRF_K,
) -> list[RetrievalResult]:
    """Fuse FAISS and BM25 rankings using Reciprocal Rank Fusion."""

    fused: dict[str, RetrievalResult] = {}

    # FAISS results
    for rank, item in enumerate(faiss_results, start=1):
        chunk_id = item["chunk_id"]

        result = dict(item)
        result["faiss_rank"] = rank
        result["faiss_score"] = float(item["score"])
        result["rrf_score"] = 1.0 / (rrf_k + rank)

        fused[chunk_id] = result

    # BM25 results
    for rank, item in enumerate(bm25_results, start=1):
        chunk_id = str(item["chunk_id"])

        if chunk_id in fused:
            result = fused[chunk_id]
        else:
            result = {
                "chunk_id": chunk_id,
                "text": str(item["text"]),
                "page": int(item["page"]),
                "section": str(item["section"]),
                "score": 0.0,
                "rrf_score": 0.0,
            }
            fused[chunk_id] = result

        result["bm25_rank"] = rank
        result["bm25_score"] = float(item["bm25_score"])

        result["rrf_score"] = (
            float(result.get("rrf_score", 0.0))
            + 1.0 / (rrf_k + rank)
        )

    results = list(fused.values())

    results.sort(
        key=lambda item: float(item.get("rrf_score", 0.0)),
        reverse=True,
    )

    return results


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
        faiss_score = item.get("faiss_score")
        bm25_score = item.get("bm25_score")
        lines.extend(
            [
                f"--- Rank {rank} ---",
                f"score:          {item['score']:.4f}",
                f"rrf_score:      {item.get('rrf_score', 0.0):.6f}",
                f"faiss_rank:     {item.get('faiss_rank', '-')}",
                f"faiss_score:    {faiss_score:.4f}" if faiss_score is not None
                    else "faiss_score:    -",
                f"bm25_rank:      {item.get('bm25_rank', '-')}",
                f"bm25_score:     {bm25_score:.4f}" if bm25_score is not None          
                    else "bm25_score:     -",
                f"page:           {item['page']}",
                f"section:        {item['section']}",
                f"chunk:          {item['chunk_id']}",
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