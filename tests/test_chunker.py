"""Tests for section-aware automotive chunking."""

from __future__ import annotations

from src.chunker import (
    chunk_pages,
    chunk_text,
    estimate_tokens,
    summarize_chunks,
)


def _brake_pages() -> list[dict]:
    return [
        {
            "page": 42,
            "text": (
                "Front Brake\n\n"
                "Inspect the front brake assembly before service.\n"
                "Wheel nut torque: 108 Nm\n"
                "Bleed screw torque: 15 Nm\n"
                "Use bolt size M10x1.25 when reinstalling the caliper bracket."
            ),
        },
        {
            "page": 43,
            "text": (
                "Brake Fluid\n\n"
                "Reservoir capacity: 0.7 L\n"
                "Specification: DOT 4\n"
                "Replace fluid if contaminated."
            ),
        },
    ]


def test_chunk_preserves_page_and_section_metadata() -> None:
    chunks = chunk_pages(_brake_pages(), target_tokens=40, max_tokens=80, overlap_tokens=10)
    assert chunks
    assert chunks[0]["chunk_id"] == "chunk_0001"
    assert chunks[0]["page"] == chunks[0]["start_page"]
    assert "section" in chunks[0]
    assert chunks[0]["start_page"] <= chunks[0]["end_page"]


def test_heading_stays_with_specification_context() -> None:
    chunks = chunk_pages(_brake_pages(), target_tokens=50, max_tokens=100, overlap_tokens=10)
    joined = "\n".join(chunk["text"] for chunk in chunks)

    assert "108 Nm" in joined
    assert "M10x1.25" in joined
    assert "0.7 L" in joined

    # Spec lines should appear in a chunk that still carries the component heading.
    torque_chunk = next(c for c in chunks if "108 Nm" in c["text"])
    assert "Front Brake" in torque_chunk["text"] or torque_chunk["section"] == "Front Brake"


def test_deterministic_chunk_ids() -> None:
    a = chunk_pages(_brake_pages(), target_tokens=50, max_tokens=100, overlap_tokens=10)
    b = chunk_pages(_brake_pages(), target_tokens=50, max_tokens=100, overlap_tokens=10)
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]
    assert [c["chunk_id"] for c in a] == [f"chunk_{i:04d}" for i in range(1, len(a) + 1)]


def test_fallback_fixed_size_chunking_on_long_text() -> None:
    words = [f"word{i}" for i in range(1200)]
    # Plant a specification near the start with a heading.
    text = "Engine Oil\n\nOil capacity: 4.5 L grade 5W-30\n\n" + " ".join(words)
    chunks = chunk_pages(
        [{"page": 10, "text": text}],
        target_tokens=200,
        max_tokens=250,
        overlap_tokens=50,
    )
    assert len(chunks) > 1
    assert all(estimate_tokens(c["text"]) <= 250 for c in chunks)
    assert any("4.5 L" in c["text"] and "5W-30" in c["text"] for c in chunks)


def test_chunk_text_helper_returns_strings() -> None:
    text = "Front Brake\n\nTorque: 108 Nm\n\n" + ("procedure step\n" * 200)
    parts = chunk_text(text, chunk_size=80, overlap=20)
    assert parts
    assert all(isinstance(p, str) for p in parts)
    assert any("108 Nm" in p for p in parts)


def test_summarize_chunks_output(capsys) -> None:
    chunks = chunk_pages(_brake_pages(), target_tokens=40, max_tokens=80, overlap_tokens=10)
    summary = summarize_chunks(chunks, example_count=2)
    assert summary["total_chunks"] == len(chunks)
    out = capsys.readouterr().out
    assert "Total chunks:" in out
    assert "Average chunk length" in out
    assert "Minimum chunk length" in out
    assert "Maximum chunk length" in out


def test_does_not_drop_part_numbers_or_oil_grades() -> None:
    pages = [
        {
            "page": 5,
            "text": (
                "Lubrication\n\n"
                "Engine oil: 5W-30\n"
                "Oil filter: 90915-YZZD1\n"
                "Fill quantity: 4.5 L"
            ),
        }
    ]
    chunks = chunk_pages(pages, target_tokens=30, max_tokens=60, overlap_tokens=5)
    blob = "\n".join(c["text"] for c in chunks)
    assert "5W-30" in blob
    assert "90915-YZZD1" in blob
    assert "4.5 L" in blob
