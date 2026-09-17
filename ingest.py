"""One-time PDF ingestion: parse, clean, chunk, embed, and persist FAISS."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.chunker import chunk_pages
from src.embeddings import embed_documents
from src.pdf_parser import extract_pages
from src.text_cleaner import clean_pages
from src.vector_store import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH, VectorStore

logger = logging.getLogger(__name__)


def ingest_pdf(
    pdf_path: str | Path,
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    """Build and persist an index for one PDF.

    Query-time code never calls this function, so it cannot reparse or
    re-embed the source PDF.
    """
    source = Path(pdf_path)
    logger.info("Extracting text from %s", source)
    pages = extract_pages(source)
    cleaned_pages = clean_pages(pages)
    chunks = chunk_pages(cleaned_pages)
    if not chunks:
        raise ValueError("No indexable text chunks were produced from the PDF")

    logger.info("Embedding %d chunk(s)", len(chunks))
    embeddings = embed_documents(chunks, show_progress_bar=False)
    store = VectorStore(dimension=int(embeddings.shape[1]))
    store.add(embeddings, chunks)
    store.save(index_path=index_path, metadata_path=metadata_path)

    return {
        "pdf": str(source),
        "pages": len(pages),
        "pages_with_text": sum(bool(page["text"].strip()) for page in cleaned_pages),
        "chunks": len(chunks),
        "embedding_dimension": store.dimension,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a service-manual PDF into FAISS.")
    parser.add_argument("pdf_path", type=Path, help="Path to the service-manual PDF")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for one-time document ingestion."""
    args = _build_arg_parser().parse_args(argv)
    try:
        summary = ingest_pdf(
            args.pdf_path,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
        )
    except Exception as exc:
        logger.exception("Ingestion failed")
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Ingested {pdf}: {pages} pages ({pages_with_text} with text), "
        "{chunks} chunks, {embedding_dimension}D embeddings.\n"
        "Saved FAISS index to {index_path} and metadata to {metadata_path}.".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
