"""Section-aware chunking for cleaned automotive service-manual pages.

Chunks prefer heading / paragraph boundaries and keep specification lines
with their nearby component context. Token counts are approximated as
whitespace-separated words (good enough for assignment-scale RAG).
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import TypedDict


class PageText(TypedDict):
    """Cleaned page-level payload."""

    page: int
    text: str


class Chunk(TypedDict):
    """One retrieval chunk with page and section metadata."""

    chunk_id: str
    text: str
    page: int
    section: str
    start_page: int
    end_page: int


class _Block(TypedDict):
    text: str
    page: int
    section: str
    is_heading: bool


# Defaults ≈ 500–800 tokens with ~100-token overlap (word ≈ token).
DEFAULT_TARGET_TOKENS = 650
DEFAULT_MAX_TOKENS = 800
DEFAULT_MIN_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 100

_HEADING_NUMBERED_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*|[A-Z])[.)]\s+\S+"
)
_ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 /&-]{2,60}$")
_SPEC_LINE_RE = re.compile(
    r"""
    (?:
        \d+(?:[.,]\d+)?\s*(?:N[·.]?m|Nm|psi|kPa|bar|mm|cm|L|ml|mL|kg|°?C)
      | \d+\s*W\s*-\s*\d+
      | M\d+(?:\s*[x×]\s*[\d.]+)?
      | \d{4,}[A-Z0-9]*-[A-Z0-9]+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def estimate_tokens(text: str) -> int:
    """Approximate token count as whitespace-separated word count."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def _looks_like_heading(line: str) -> bool:
    """Heuristic heading detector for workshop-manual structure."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if _SPEC_LINE_RE.search(stripped):
        return False
    if stripped.endswith((".", ";")):
        return False
    if _HEADING_NUMBERED_RE.match(stripped):
        return True
    if _ALL_CAPS_HEADING_RE.match(stripped) and " " in stripped:
        return True
    # Short Title Case / name-like headings without heavy punctuation.
    words = stripped.split()
    if 1 <= len(words) <= 8 and stripped[0].isupper() and not stripped.endswith("."):
        letter_words = [w for w in words if re.search(r"[A-Za-z]", w)]
        if letter_words and all(
            w[0].isupper() or w.lower() in {"and", "or", "of", "the", "for", "to"}
            for w in letter_words
        ):
            return True
    return False


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _pages_to_blocks(pages: list[PageText]) -> list[_Block]:
    """Turn cleaned pages into ordered heading/paragraph blocks."""
    blocks: list[_Block] = []
    current_section = "General"

    for page in pages:
        page_no = page["page"]
        text = (page.get("text") or "").strip()
        if not text:
            continue

        for paragraph in _split_paragraphs(text):
            lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
            if not lines:
                continue

            # Promote a leading heading line into its own block when present.
            if _looks_like_heading(lines[0]) and len(lines) > 1:
                current_section = lines[0].rstrip(":")
                blocks.append(
                    {
                        "text": current_section,
                        "page": page_no,
                        "section": current_section,
                        "is_heading": True,
                    }
                )
                body = "\n".join(lines[1:]).strip()
                if body:
                    blocks.append(
                        {
                            "text": body,
                            "page": page_no,
                            "section": current_section,
                            "is_heading": False,
                        }
                    )
            elif _looks_like_heading(paragraph) and "\n" not in paragraph:
                current_section = paragraph.rstrip(":")
                blocks.append(
                    {
                        "text": current_section,
                        "page": page_no,
                        "section": current_section,
                        "is_heading": True,
                    }
                )
            else:
                blocks.append(
                    {
                        "text": paragraph,
                        "page": page_no,
                        "section": current_section,
                        "is_heading": False,
                    }
                )

    return blocks


def _make_chunk_id(index: int) -> str:
    """Deterministic zero-padded chunk identifier."""
    return f"chunk_{index:04d}"


def _join_block_texts(blocks: list[_Block]) -> str:
    return "\n\n".join(block["text"] for block in blocks if block["text"].strip())


def _blocks_to_chunk(blocks: list[_Block], chunk_index: int) -> Chunk:
    text = _join_block_texts(blocks)
    pages = [b["page"] for b in blocks]
    start_page = min(pages)
    end_page = max(pages)
    section = blocks[0]["section"]
    # Prefer first heading section label in the group.
    for block in blocks:
        if block["is_heading"]:
            section = block["section"]
            break
    return {
        "chunk_id": _make_chunk_id(chunk_index),
        "text": text,
        "page": start_page,
        "section": section,
        "start_page": start_page,
        "end_page": end_page,
    }


def _split_oversized_text(
    text: str,
    *,
    page: int,
    section: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[_Block]:
    """Fallback: fixed-size word windows for a single oversized block."""
    words = text.split()
    if len(words) <= max_tokens:
        return [{"text": text, "page": page, "section": section, "is_heading": False}]

    step = max(max_tokens - overlap_tokens, 1)
    pieces: list[_Block] = []
    for start in range(0, len(words), step):
        window = words[start : start + max_tokens]
        if not window:
            break
        pieces.append(
            {
                "text": " ".join(window),
                "page": page,
                "section": section,
                "is_heading": False,
            }
        )
        if start + max_tokens >= len(words):
            break
    return pieces


def _expand_blocks_with_fallback(
    blocks: list[_Block],
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[_Block]:
    expanded: list[_Block] = []
    for block in blocks:
        if estimate_tokens(block["text"]) <= max_tokens:
            expanded.append(block)
            continue
        # Keep a heading glued to the start of the first split piece when possible.
        if block["is_heading"]:
            expanded.append(block)
            continue
        expanded.extend(
            _split_oversized_text(
                block["text"],
                page=block["page"],
                section=block["section"],
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return expanded


def _pack_blocks(
    blocks: list[_Block],
    *,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Pack blocks into overlapping chunks without orphaning headings."""
    if not blocks:
        return []

    chunks: list[Chunk] = []
    current: list[_Block] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunks.append(_blocks_to_chunk(current, len(chunks) + 1))
        # Overlap: keep a trailing word window from the previous chunk text.
        if overlap_tokens > 0 and current:
            overlap_text = _overlap_tail(_join_block_texts(current), overlap_tokens)
            last = current[-1]
            current = [
                {
                    "text": overlap_text,
                    "page": last["page"],
                    "section": last["section"],
                    "is_heading": False,
                }
            ] if overlap_text else []
            current_tokens = estimate_tokens(overlap_text)
        else:
            current = []
            current_tokens = 0

    i = 0
    while i < len(blocks):
        block = blocks[i]
        block_tokens = estimate_tokens(block["text"])

        # Keep heading + following body together when both fit.
        if block["is_heading"] and i + 1 < len(blocks) and not blocks[i + 1]["is_heading"]:
            paired = [block, blocks[i + 1]]
            paired_tokens = estimate_tokens(_join_block_texts(paired))
            if current and current_tokens + paired_tokens > max_tokens:
                flush()
            if current and current_tokens + paired_tokens > max_tokens:
                current = []
                current_tokens = 0
            if paired_tokens > max_tokens:
                # Heading alone, then fall back-split the body.
                if current and current_tokens + estimate_tokens(block["text"]) > max_tokens:
                    flush()
                if current and current_tokens + estimate_tokens(block["text"]) > max_tokens:
                    current = []
                    current_tokens = 0
                current.append(block)
                current_tokens += estimate_tokens(block["text"])
                i += 1
                continue

            current.extend(paired)
            current_tokens += paired_tokens
            if current_tokens >= target_tokens:
                flush()
            i += 2
            continue

        if current and current_tokens + block_tokens > max_tokens:
            flush()
        # Overlap seed may still make a legal block overflow the ceiling.
        if current and current_tokens + block_tokens > max_tokens:
            current = []
            current_tokens = 0

        current.append(block)
        current_tokens += block_tokens
        if current_tokens >= target_tokens:
            flush()
        i += 1

    if current:
        # If the only content is leftover overlap from a previous flush and
        # nothing new was added, skip emitting a near-duplicate tail chunk.
        if chunks and estimate_tokens(_join_block_texts(current)) <= overlap_tokens:
            pass
        else:
            chunks.append(_blocks_to_chunk(current, len(chunks) + 1))

    return chunks


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    words = text.split()
    if not words or overlap_tokens <= 0:
        return ""
    return " ".join(words[-overlap_tokens:])


def chunk_pages(
    pages: list[PageText],
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk cleaned page-level text into overlapping retrieval units.

    Args:
        pages: Cleaned ``{"page", "text"}`` records.
        target_tokens: Preferred chunk size in approximate tokens.
        max_tokens: Hard ceiling before flushing a chunk.
        overlap_tokens: Approximate tokens reused at the start of the next chunk.

    Returns:
        Deterministic list of chunk dicts with metadata.
    """
    if max_tokens < target_tokens:
        raise ValueError("max_tokens must be >= target_tokens")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < max_tokens")

    blocks = _pages_to_blocks(pages)
    blocks = _expand_blocks_with_fallback(
        blocks,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    return _pack_blocks(
        blocks,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_TARGET_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
    *,
    page: int = 1,
) -> list[str]:
    """Backward-compatible helper returning only chunk strings.

    ``chunk_size`` / ``overlap`` are interpreted as approximate tokens.
    """
    pages: list[PageText] = [{"page": page, "text": text}]
    chunks = chunk_pages(
        pages,
        target_tokens=chunk_size,
        max_tokens=max(chunk_size, DEFAULT_MAX_TOKENS),
        overlap_tokens=overlap,
    )
    return [chunk["text"] for chunk in chunks]


def summarize_chunks(
    chunks: list[Chunk],
    *,
    example_count: int = 5,
) -> dict[str, object]:
    """Print and return chunking quality stats for debugging."""
    lengths = [estimate_tokens(chunk["text"]) for chunk in chunks]
    total = len(chunks)
    avg = (sum(lengths) / total) if total else 0.0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    print(f"Total chunks: {total}")
    print(f"Average chunk length (tokens≈words): {avg:.1f}")
    print(f"Minimum chunk length: {min_len}")
    print(f"Maximum chunk length: {max_len}")
    print()

    examples = chunks[:example_count]
    for chunk in examples:
        print(f"--- {chunk['chunk_id']} ---")
        print(
            f"page={chunk['page']} start_page={chunk['start_page']} "
            f"end_page={chunk['end_page']} section={chunk['section']!r}"
        )
        preview = chunk["text"][:400]
        print(preview)
        if len(chunk["text"]) > 400:
            print("...")
        print()

    return {
        "total_chunks": total,
        "average_chunk_length": avg,
        "min_chunk_length": min_len,
        "max_chunk_length": max_len,
        "examples": examples,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug chunking over a cleaned PDF extraction."
    )
    parser.add_argument("pdf_path", help="Path to a PDF file")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    parser.add_argument("--examples", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI: parse → clean → chunk, then print summary stats."""
    from src.pdf_parser import extract_pages
    from src.text_cleaner import clean_pages

    args = _build_arg_parser().parse_args(argv)
    try:
        raw_pages = extract_pages(args.pdf_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    cleaned = clean_pages(raw_pages)
    chunks = chunk_pages(
        cleaned,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    summarize_chunks(chunks, example_count=args.examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
