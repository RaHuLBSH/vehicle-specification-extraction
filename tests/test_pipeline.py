"""Tests for ingestion and query-time RAG composition."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ingest import ingest_pdf
from src.extractor import ExtractionError, ExtractionResult
from src.pipeline import PipelineExtractionError, PipelineRetrievalError, RAGPipeline


RETRIEVED_CHUNKS = [
    {
        "chunk_id": "chunk_0001",
        "text": "Front Brake caliper bolt torque: 28 Nm",
        "page": 42,
        "section": "Front Brake",
        "score": 0.91,
    }
]


def test_pipeline_retrieves_then_extracts() -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = RETRIEVED_CHUNKS
    extractor = MagicMock(
        return_value=ExtractionResult(found=False, specifications=[], reason="Not present")
    )

    result = RAGPipeline(retriever, extractor=extractor, top_k=3).run("caliper torque")

    assert result.found is False
    retriever.retrieve.assert_called_once_with("caliper torque", top_k=3)
    extractor.assert_called_once_with("caliper torque", RETRIEVED_CHUNKS)


def test_pipeline_returns_not_found_without_calling_gemini() -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    extractor = MagicMock()

    result = RAGPipeline(retriever, extractor=extractor).run("unknown specification")

    assert result.found is False
    assert "No relevant chunks" in result.reason
    extractor.assert_not_called()


def test_pipeline_distinguishes_retrieval_failure() -> None:
    retriever = MagicMock()
    retriever.retrieve.side_effect = RuntimeError("index unavailable")

    with pytest.raises(PipelineRetrievalError, match="Retrieval failed"):
        RAGPipeline(retriever).run("torque")


def test_pipeline_distinguishes_extraction_failure() -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = RETRIEVED_CHUNKS

    def failing_extractor(query: str, chunks: list[dict]) -> ExtractionResult:
        raise ExtractionError("Gemini unavailable")

    with pytest.raises(PipelineExtractionError, match="Extraction failed"):
        RAGPipeline(retriever, extractor=failing_extractor).run("torque")


def test_ingest_pdf_persists_the_completed_index(tmp_path: Path) -> None:
    chunks = [
        {
            "chunk_id": "chunk_0001",
            "text": "Brake torque: 28 Nm",
            "page": 1,
            "section": "Brakes",
            "start_page": 1,
            "end_page": 1,
        }
    ]
    embeddings = np.ones((1, 384), dtype=np.float32)
    index_path = tmp_path / "manual.faiss"
    metadata_path = tmp_path / "metadata.json"

    with (
        patch("ingest.extract_pages", return_value=[{"page": 1, "text": "raw"}]),
        patch("ingest.clean_pages", return_value=[{"page": 1, "text": "clean"}]),
        patch("ingest.chunk_pages", return_value=chunks),
        patch("ingest.embed_documents", return_value=embeddings),
    ):
        summary = ingest_pdf(
            "manual.pdf", index_path=index_path, metadata_path=metadata_path
        )

    assert summary["chunks"] == 1
    assert summary["embedding_dimension"] == 384
    assert index_path.is_file()
    assert metadata_path.is_file()
