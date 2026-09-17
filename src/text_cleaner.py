"""Conservative text cleaning for automotive service-manual pages.

Cleaning prioritizes specification fidelity over cosmetic polish.
Numeric values, units, part numbers, oil grades, and bolt sizes must
not be altered.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Iterable, TypedDict

# Re-export-compatible page shape (matches pdf_parser.PageText).
class PageText(TypedDict):
    """Page-level text payload."""

    page: int
    text: str


# Lines matching these are treated as specification-bearing and never
# stripped as headers/footers, and never altered beyond whitespace.
_SPEC_GUARD_RE = re.compile(
    r"""
    (?:
        \d+(?:[.,]\d+)?\s*(?:N[·.]?m|Nm|psi|kPa|bar|mm|cm|m|L|l|ml|mL|kg|g|°?C|F)
      | \d+\s*W\s*-\s*\d+                          # oil grade e.g. 5W-30
      | M\d+(?:\s*[x×]\s*[\d.]+)?                  # bolt size e.g. M10x1.25
      | [A-Z0-9]+(?:-[A-Z0-9]+)+                   # part / ID e.g. 90915-YZZD1
      | \d{4,}-\w+                                 # numeric-led part numbers
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Soft hyphenation: "...capac-\nity..." → "capacity"
# Only when the break looks like a mid-word wrap, not a real hyphenated token.
_SOFT_HYPHEN_BREAK_RE = re.compile(
    r"(?<=[A-Za-z])-\n(?=[a-z])"
)

_MULTI_SPACE_RE = re.compile(r"[^\S\n]{2,}")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[^\S\n]+$", re.MULTILINE)
_LEADING_SPACE_RE = re.compile(r"^[^\S\n]+", re.MULTILINE)

# High-confidence PDF / web-export chrome. Matched lines are dropped only when
# they do not also carry critical measurement/spec tokens.
_ARTIFACT_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "Page 7 sur 8", "Page 7 of 8", "Page 7/8"
    re.compile(r"^page\s+\d+\s*(?:sur|of|\/)\s*\d+$", re.IGNORECASE),
    # Local/cache export URLs
    re.compile(r"^file:///\S+$", re.IGNORECASE),
    re.compile(r"^https?://\S+$", re.IGNORECASE),
    # Standalone ISO dates from export headers
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    # Known export/site watermarks
    re.compile(r"^repair4less$", re.IGNORECASE),
    # e.g. "2014 F-150 Workshop Manual"
    re.compile(
        r"^\d{4}\s+.+\bworkshop\s+manual\b.*$",
        re.IGNORECASE,
    ),
    re.compile(r"^workshop\s+manual$", re.IGNORECASE),
)

# Stronger gate used when stripping export artifacts (vehicle codes like
# "F-150" must not block removal of workshop-manual title lines).
_CRITICAL_SPEC_RE = re.compile(
    r"""
    (?:
        \b\d+(?:[.,]\d+)?\s*(?:N[·.]?m|Nm|psi|kPa|bar|mm|cm|mL|ml|L|kg|°C)\b
      | \b\d+\s*W\s*-\s*\d+\b
      | \bM\d+(?:\s*[x×]\s*[\d.]+)?\b
      | \b\d{4,}[A-Z0-9]*-[A-Z0-9]*[A-Za-z][A-Z0-9]*\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _contains_spec_token(text: str) -> bool:
    """Return True if text appears to contain automotive specification tokens."""
    return bool(_SPEC_GUARD_RE.search(text))


def _contains_critical_spec(text: str) -> bool:
    """Return True for measurement / part tokens that must never be dropped."""
    return bool(_CRITICAL_SPEC_RE.search(text))


def _is_export_artifact_line(line: str) -> bool:
    """Return True for obvious PDF/web-export chrome lines."""
    stripped = line.strip()
    if not stripped:
        return False
    if _contains_critical_spec(stripped):
        return False
    return any(pattern.match(stripped) for pattern in _ARTIFACT_LINE_PATTERNS)


def _strip_export_artifact_lines(text: str) -> str:
    """Drop high-confidence export artifact lines; keep everything else."""
    if not text:
        return text
    kept = [
        line
        for line in text.splitlines()
        if not _is_export_artifact_line(line)
    ]
    return "\n".join(kept)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and repeated blank lines; trim line edges."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _LEADING_SPACE_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _is_unsafe_soft_hyphen_context(text: str, hyphen_index: int) -> bool:
    """Return True when a letter-hyphen linebreak may be a real spec token.

    Soft wraps look like ``capac-\\nity``. Real tokens (oil grades, part
    numbers) often have digits or uppercase on either side of the hyphen.
    """
    left = text[max(0, hyphen_index - 16) : hyphen_index]
    # Character after "-\\n"
    right_start = hyphen_index + 2
    right = text[right_start : right_start + 16]

    if re.search(r"\dW$", left, re.IGNORECASE):
        return True
    if re.search(r"\d$", left):
        return True
    if right and (right[0].isdigit() or right[0].isupper()):
        return True
    if _SPEC_GUARD_RE.search(left) or _SPEC_GUARD_RE.search(right):
        return True
    return False


def _join_safe_broken_lines(text: str) -> str:
    """Repair only clearly safe mid-word hyphenated line breaks.

    Example safe repair: ``capac-\\nity`` → ``capacity``.

    Does not invent spaces or merge unrelated lines. Avoids joining when
    the hyphen appears to belong to a specification token.
    """
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        if _is_unsafe_soft_hyphen_context(text, match.start()):
            return match.group(0)
        return ""

    return _SOFT_HYPHEN_BREAK_RE.sub(_replace, text)


def clean_text(text: str) -> str:
    """Clean a single page's raw text conservatively.

    Steps:
        1. Normalize newlines.
        2. Join safe soft-hyphen line breaks only.
        3. Drop obvious PDF/web-export artifact lines.
        4. Collapse excessive horizontal whitespace.
        5. Collapse repeated blank lines to at most one blank line.

    Does not remove or rewrite numbers, units, part numbers, oil grades,
    or bolt sizes.

    Args:
        text: Raw text from PDF extraction.

    Returns:
        Cleaned text suitable for later chunking.
    """
    if not text or not text.strip():
        return ""

    cleaned = _join_safe_broken_lines(text)
    cleaned = _strip_export_artifact_lines(cleaned)
    cleaned = _normalize_whitespace(cleaned)
    return cleaned


def clean_page(page: PageText) -> PageText:
    """Clean one page while preserving ``page`` metadata.

    Args:
        page: ``{"page": int, "text": str}`` from the PDF parser.

    Returns:
        A new dict with the same page number and cleaned text.
    """
    return {"page": page["page"], "text": clean_text(page.get("text", ""))}


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _candidate_edge_lines(text: str, *, edge: str, depth: int = 2) -> list[str]:
    """Return up to ``depth`` non-empty lines from the top or bottom of a page."""
    lines = _nonempty_lines(text)
    if not lines:
        return []
    if edge == "top":
        return lines[:depth]
    return list(reversed(lines[-depth:]))


def _is_safe_header_footer_line(line: str) -> bool:
    """Conservative gate: only short, non-spec, non-sentence-like lines."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 80:
        return False
    if _contains_spec_token(stripped):
        return False
    # Avoid removing dense content lines.
    if stripped.count(" ") > 12:
        return False
    return True


def _detect_repeated_edge_lines(
    pages: Iterable[PageText],
    *,
    min_occurrences: int = 3,
) -> set[str]:
    """Find short top/bottom lines that repeat across many pages.

    A line is only considered removable when it appears on at least
    ``min_occurrences`` pages (or on every page when there are fewer than
    that many pages) and passes :func:`_is_safe_header_footer_line`.
    """
    pages_list = list(pages)
    if len(pages_list) < 2:
        return set()

    threshold = min(min_occurrences, len(pages_list))
    counts: dict[str, int] = {}

    for page in pages_list:
        text = page.get("text", "")
        seen_on_page: set[str] = set()
        for edge in ("top", "bottom"):
            for line in _candidate_edge_lines(text, edge=edge):
                if line in seen_on_page:
                    continue
                if not _is_safe_header_footer_line(line):
                    continue
                seen_on_page.add(line)
                counts[line] = counts.get(line, 0) + 1

    return {line for line, count in counts.items() if count >= threshold}


def _strip_edge_lines(text: str, removable: set[str]) -> str:
    """Remove known header/footer lines only from page edges."""
    if not text or not removable:
        return text

    lines = text.splitlines()
    # Strip from top.
    while lines and lines[0].strip() in removable:
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    # Strip from bottom.
    while lines and lines[-1].strip() in removable:
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def clean_pages(
    pages: list[PageText],
    *,
    remove_repeated_headers_footers: bool = True,
) -> list[PageText]:
    """Clean a sequence of pages, optionally dropping repeated headers/footers.

    Header/footer removal runs only when the same short, non-specification
    edge line repeats across enough pages. Spec-bearing lines are never
    removed.

    Args:
        pages: Page-level payloads from :mod:`src.pdf_parser`.
        remove_repeated_headers_footers: When True, attempt safe removal.

    Returns:
        Cleaned pages with original page numbers preserved.
    """
    removable: set[str] = set()
    if remove_repeated_headers_footers:
        removable = _detect_repeated_edge_lines(pages)

    cleaned_pages: list[PageText] = []
    for page in pages:
        text = page.get("text", "")
        if removable:
            text = _strip_edge_lines(text, removable)
        cleaned_pages.append({"page": page["page"], "text": clean_text(text)})
    return cleaned_pages


def compare_raw_vs_cleaned(
    page: PageText,
    *,
    cleaned: PageText | None = None,
    max_chars: int = 500,
) -> dict[str, object]:
    """Print and return a raw-vs-cleaned comparison for one page.

    Useful for verifying that cleaning did not damage specifications.

    Args:
        page: Original page payload.
        cleaned: Optional precomputed cleaned page; computed if omitted.
        max_chars: Max characters to print from each side.

    Returns:
        Dict with page number, lengths, whether text changed, and snippets.
    """
    cleaned_page = cleaned if cleaned is not None else clean_page(page)
    raw = page.get("text", "")
    new = cleaned_page.get("text", "")

    result: dict[str, object] = {
        "page": page["page"],
        "raw_chars": len(raw),
        "cleaned_chars": len(new),
        "changed": raw != new,
        "raw_preview": raw[:max_chars],
        "cleaned_preview": new[:max_chars],
    }

    print(f"=== Page {page['page']} ===")
    print(f"Raw chars: {result['raw_chars']} | Cleaned chars: {result['cleaned_chars']}")
    print(f"Changed: {result['changed']}")
    print("--- RAW ---")
    print(result["raw_preview"])
    if len(raw) > max_chars:
        print("...")
    print("--- CLEANED ---")
    print(result["cleaned_preview"])
    if len(new) > max_chars:
        print("...")
    print()
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug raw vs cleaned text for pages extracted from a PDF."
    )
    parser.add_argument("pdf_path", help="Path to a PDF file")
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="1-based page number to compare (default: first non-empty page)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="Preview length for raw/cleaned output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI: extract one PDF page and compare raw vs cleaned text."""
    # Local import keeps the cleaner usable without requiring PDF I/O in tests.
    from src.pdf_parser import extract_pages

    args = _build_arg_parser().parse_args(argv)
    try:
        pages = extract_pages(args.pdf_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not pages:
        print("PDF has no pages.", file=sys.stderr)
        return 1

    if args.page is not None:
        target = next((p for p in pages if p["page"] == args.page), None)
        if target is None:
            print(f"Error: page {args.page} not found", file=sys.stderr)
            return 1
    else:
        target = next((p for p in pages if p["text"].strip()), pages[0])

    cleaned_pages = clean_pages(pages)
    cleaned = next(p for p in cleaned_pages if p["page"] == target["page"])
    compare_raw_vs_cleaned(target, cleaned=cleaned, max_chars=args.max_chars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
