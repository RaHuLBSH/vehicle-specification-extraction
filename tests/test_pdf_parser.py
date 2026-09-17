"""Tests for PDF text extraction."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from src.pdf_parser import extract_pages, extract_text, summarize_extraction


def _make_pdf(path: Path, pages: list[str]) -> Path:
    """Create a simple multi-page PDF with the given page texts."""
    doc = pymupdf.open()
    try:
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        doc.save(path)
    finally:
        doc.close()
    return path


def test_extract_pages_preserves_page_numbers_and_text(tmp_path: Path) -> None:
    pdf = _make_pdf(
        tmp_path / "manual.pdf",
        ["Torque specs for wheel nuts", "Engine oil capacity 4.5 L"],
    )

    pages = extract_pages(pdf)

    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert pages[1]["page"] == 2
    assert "Torque specs" in pages[0]["text"]
    assert "Engine oil capacity" in pages[1]["text"]


def test_extract_pages_handles_empty_pages(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "blank_middle.pdf", ["Page one text", "", "Page three text"])

    pages = extract_pages(pdf)

    assert len(pages) == 3
    assert pages[0]["page"] == 1
    assert pages[1]["page"] == 2
    assert pages[1]["text"] == ""
    assert "Page three" in pages[2]["text"]


def test_extract_pages_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        extract_pages(missing)


def test_extract_pages_rejects_non_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a file"):
        extract_pages(tmp_path)


def test_extract_pages_rejects_invalid_pdf(tmp_path: Path) -> None:
    bogus = tmp_path / "not_a_pdf.pdf"
    bogus.write_text("this is not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to open PDF"):
        extract_pages(bogus)


def test_extract_text_joins_pages(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "joined.pdf", ["Alpha", "Beta"])

    text = extract_text(pdf)

    assert "Alpha" in text
    assert "Beta" in text
    assert "\x0c" in text


def test_summarize_extraction_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pdf = _make_pdf(tmp_path / "summary.pdf", ["Hello world", "", "More text here"])
    pages = extract_pages(pdf)
    expected_chars = sum(len(page["text"]) for page in pages)

    summary = summarize_extraction(pdf, preview_pages=2, preview_chars=20)

    assert summary == {
        "total_pages": 3,
        "pages_with_text": 2,
        "total_characters": expected_chars,
    }
    captured = capsys.readouterr().out
    assert "Total pages: 3" in captured
    assert "Pages containing text: 2" in captured
    assert "Hello world" in captured
