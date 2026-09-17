"""Interactive query CLI for an already-ingested vehicle service manual."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.embeddings import get_model
from src.pipeline import PipelineExtractionError, PipelineRetrievalError, RAGPipeline
from src.retriever import Retriever
from src.vector_store import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH, VectorStoreError

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query an existing vehicle-manual FAISS index with Gemini extraction."
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieved chunks per query")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    return parser


def load_pipeline(
    *,
    top_k: int = 5,
    index_path: Path = DEFAULT_INDEX_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> RAGPipeline:
    """Load query-time artifacts once and return a ready-to-use RAG pipeline.

    This is shared by the terminal CLI and the Streamlit UI. It intentionally
    loads only persisted retrieval artifacts and the embedding model; it never
    invokes ingestion or reparses the PDF.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    retriever = Retriever(index_path=index_path, metadata_path=metadata_path)
    get_model()
    return RAGPipeline(retriever, top_k=top_k)


def _run_interactive(pipeline: RAGPipeline) -> int:
    print("Vehicle specification RAG. Enter a query, or 'quit' to exit.")
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
            result = pipeline.run(query)
        except PipelineRetrievalError as exc:
            logger.error("Retrieval error: %s", exc)
            print(f"Retrieval error: {exc}", file=sys.stderr)
            continue
        except PipelineExtractionError as exc:
            logger.error("Extraction error: %s", exc)
            print(f"Extraction error: {exc}", file=sys.stderr)
            continue
        except ValueError as exc:
            print(f"Query error: {exc}", file=sys.stderr)
            continue
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    """Load persisted artifacts once, then serve interactive queries."""
    args = _build_arg_parser().parse_args(argv)
    if args.top_k <= 0:
        print("Configuration error: --top-k must be a positive integer", file=sys.stderr)
        return 2

    try:
        # Load these once at startup; individual queries only embed the query.
        pipeline = load_pipeline(
            top_k=args.top_k,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
        )
    except VectorStoreError as exc:
        logger.error("Unable to load persisted retrieval artifacts: %s", exc)
        print(f"Startup error: {exc}. Run `python ingest.py <manual.pdf>` first.", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unable to load embedding model")
        print(f"Startup error loading embedding model: {exc}", file=sys.stderr)
        return 1

    return _run_interactive(pipeline)


if __name__ == "__main__":
    raise SystemExit(main())
