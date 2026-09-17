"""Query-time RAG orchestration: retrieve persisted chunks, then extract specs.

This module deliberately does not know about PDFs or indexing. Ingestion is
performed separately by :mod:`ingest`, so a query never reparses or re-embeds
the source document.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.extractor import ExtractionError, ExtractionResult, extract_specifications
from src.retriever import RetrievalResult, Retriever

logger = logging.getLogger(__name__)


class PipelineRetrievalError(RuntimeError):
    """Raised when embedding a query or searching the persisted store fails."""


class PipelineExtractionError(RuntimeError):
    """Raised when Gemini extraction or structured-output validation fails."""


Extractor = Callable[[str, list[RetrievalResult]], ExtractionResult]


class RAGPipeline:
    """A small composition of a loaded retriever and an extraction function."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        extractor: Extractor = extract_specifications,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self.retriever = retriever
        self.extractor = extractor
        self.top_k = top_k

    def run(self, query: str) -> ExtractionResult:
        """Retrieve relevant chunks and return Gemini's validated result."""
        if not query or not str(query).strip():
            raise ValueError("query must be a non-empty string")
        normalized_query = str(query).strip()

        try:
            chunks = self.retriever.retrieve(normalized_query, top_k=self.top_k)
        except Exception as exc:
            logger.exception("Retrieval failed for query: %r", normalized_query)
            raise PipelineRetrievalError(f"Retrieval failed: {exc}") from exc

        logger.info("Retrieved %d chunk(s) for query: %r", len(chunks), normalized_query)
        if not chunks:
            return ExtractionResult(
                found=False,
                specifications=[],
                reason="No relevant chunks were retrieved from the indexed manual.",
            )

        try:
            result = self.extractor(normalized_query, chunks)
        except ExtractionError as exc:
            logger.exception("Structured extraction failed for query: %r", normalized_query)
            raise PipelineExtractionError(f"Extraction failed: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected extraction failure for query: %r", normalized_query)
            raise PipelineExtractionError(f"Extraction failed: {exc}") from exc

        logger.info(
            "Extraction completed for query: %r (found=%s, specifications=%d)",
            normalized_query,
            result.found,
            len(result.specifications),
        )
        return result


def run_pipeline(
    query: str,
    *,
    retriever: Retriever | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run retrieval plus extraction and return a JSON-serializable result."""
    active_retriever = retriever if retriever is not None else Retriever()
    return RAGPipeline(active_retriever, top_k=top_k).run(query).to_dict()
