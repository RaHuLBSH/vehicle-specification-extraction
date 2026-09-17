"""PDF text extraction using PyMuPDF.

Uses ``import pymupdf`` (preferred over the deprecated ``fitz`` alias).
Images and diagrams are ignored; only embedded text is extracted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TypedDict

import pymupdf


class PageText(TypedDict):
    """Text extracted from a single PDF page."""

    page: int
    text: str


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract plain text from every page of a PDF.

    Pages are returned in document order with 1-based page numbers.
    Pages that contain no extractable text (e.g. blank or image-only)
    are included with an empty ``text`` string so page numbering stays
    aligned with the source document.

    Args:
        pdf_path: Path to a PDF file on disk.

    Returns:
        A list of ``{"page": <int>, "text": <str>}`` dictionaries.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        ValueError: If the path is not a file, or PyMuPDF cannot open it
            as a PDF.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    pages: list[PageText] = []
    try:
        with pymupdf.open(path) as doc:
            if doc.page_count == 0:
                return pages
            for index, page in enumerate(doc):
                # "text" extracts plain text only; images/diagrams are skipped.
                raw = page.get_text("text") or ""
                pages.append({"page": index + 1, "text": raw})
    except pymupdf.FileDataError as exc:
        raise ValueError(f"Unable to open PDF: {path}") from exc

    return pages


def extract_text(pdf_path: str | Path) -> str:
    """Extract all page text joined with form-feed separators.

    Convenience wrapper around :func:`extract_pages` for callers that
    want a single string. Prefer :func:`extract_pages` when page numbers
    matter.

    Args:
        pdf_path: Path to a PDF file on disk.

    Returns:
        Full document text with pages separated by ``\\x0c``.
    """
    pages = extract_pages(pdf_path)
    return "\x0c".join(page["text"] for page in pages)


def summarize_extraction(
    pdf_path: str | Path,
    *,
    preview_pages: int = 3,
    preview_chars: int = 500,
) -> dict[str, int]:
    """Print a debug summary of PDF text extraction quality.

    Reports total pages, how many contain text, total character count,
    and a preview of the first few non-empty pages.

    Args:
        pdf_path: Path to a PDF file on disk.
        preview_pages: Max number of non-empty pages to preview.
        preview_chars: Max characters to show per previewed page.

    Returns:
        Summary counts: ``total_pages``, ``pages_with_text``,
        ``total_characters``.
    """
    pages = extract_pages(pdf_path)
    total_pages = len(pages)
    non_empty = [p for p in pages if p["text"].strip()]
    total_characters = sum(len(p["text"]) for p in pages)

    print(f"PDF: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print(f"Pages containing text: {len(non_empty)}")
    print(f"Total extracted characters: {total_characters}")
    print()

    for page in non_empty[:preview_pages]:
        snippet = page["text"][:preview_chars]
        print(f"--- Page {page['page']} (first {preview_chars} chars) ---")
        print(snippet)
        if len(page["text"]) > preview_chars:
            print("...")
        print()

    return {
        "total_pages": total_pages,
        "pages_with_text": len(non_empty),
        "total_characters": total_characters,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug PDF text extraction (page-by-page, text only)."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "--preview-pages",
        type=int,
        default=3,
        help="Number of non-empty pages to preview (default: 3)",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Characters to show per previewed page (default: 500)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for temporary extraction-quality checks."""
    args = _build_arg_parser().parse_args(argv)
    try:
        summarize_extraction(
            args.pdf_path,
            preview_pages=args.preview_pages,
            preview_chars=args.preview_chars,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
