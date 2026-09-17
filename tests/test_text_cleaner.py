"""Tests for conservative automotive text cleaning."""

from __future__ import annotations

from src.text_cleaner import (
    clean_page,
    clean_pages,
    clean_text,
    compare_raw_vs_cleaned,
)


def test_normalize_excessive_whitespace() -> None:
    raw = "Wheel   nut    torque:\t108  Nm"
    assert clean_text(raw) == "Wheel nut torque: 108 Nm"


def test_normalize_repeated_blank_lines() -> None:
    raw = "Section A\n\n\n\nTorque values"
    assert clean_text(raw) == "Section A\n\nTorque values"


def test_preserves_torque_values_and_units() -> None:
    raw = "Tighten to 108 Nm (80 lbf·ft)\nAlternate: 120 N·m"
    cleaned = clean_text(raw)
    assert "108 Nm" in cleaned
    assert "80 lbf·ft" in cleaned
    assert "120 N·m" in cleaned


def test_preserves_fluid_capacities() -> None:
    raw = "Engine oil capacity:  4.5 L\nCoolant: 7.2 L\nWasher fluid: 3.5 ml residual"
    cleaned = clean_text(raw)
    assert "4.5 L" in cleaned
    assert "7.2 L" in cleaned
    assert "3.5 ml" in cleaned


def test_preserves_part_numbers_with_hyphens() -> None:
    raw = "Oil filter: 90915-YZZD1\nGasket: 04152-31030"
    cleaned = clean_text(raw)
    assert "90915-YZZD1" in cleaned
    assert "04152-31030" in cleaned


def test_preserves_oil_grades() -> None:
    raw = "Recommended viscosity: 5W-30\nCold climate: 0W-20"
    cleaned = clean_text(raw)
    assert "5W-30" in cleaned
    assert "0W-20" in cleaned


def test_preserves_bolt_sizes_and_dimensions() -> None:
    raw = "Use bolt size M10x1.25\nClearance: 0.25 mm\nLength: 12.5 mm"
    cleaned = clean_text(raw)
    assert "M10x1.25" in cleaned
    assert "0.25 mm" in cleaned
    assert "12.5 mm" in cleaned


def test_preserves_psi_and_decimals() -> None:
    raw = "Tire pressure: 32 psi front / 35.0 psi rear"
    cleaned = clean_text(raw)
    assert "32 psi" in cleaned
    assert "35.0 psi" in cleaned


def test_safe_soft_hyphen_line_join() -> None:
    raw = "Check the capac-\nity before refill."
    cleaned = clean_text(raw)
    assert "capacity" in cleaned
    assert "capac-\nity" not in cleaned


def test_does_not_break_hyphenated_part_number_across_lines() -> None:
    # Soft-hyphen joiner only fires for letter-hyphen + lowercase continuation.
    # Part-number style tokens must remain intact.
    raw = "Install filter 90915-YZZD1\nthen torque to 18 Nm"
    cleaned = clean_text(raw)
    assert "90915-YZZD1" in cleaned
    assert "18 Nm" in cleaned


def test_clean_page_preserves_metadata() -> None:
    page = {"page": 42, "text": "Torque:   108 Nm"}
    cleaned = clean_page(page)
    assert cleaned["page"] == 42
    assert cleaned["text"] == "Torque: 108 Nm"


def test_empty_and_whitespace_pages() -> None:
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""
    assert clean_page({"page": 1, "text": "\n\n"}) == {"page": 1, "text": ""}


def test_removes_repeated_headers_footers_safely() -> None:
    pages = [
        {
            "page": 1,
            "text": "SERVICE MANUAL\nWheel nut torque 108 Nm\nConfidential",
        },
        {
            "page": 2,
            "text": "SERVICE MANUAL\nOil capacity 4.5 L\nConfidential",
        },
        {
            "page": 3,
            "text": "SERVICE MANUAL\nFilter 90915-YZZD1\nConfidential",
        },
    ]
    cleaned = clean_pages(pages)
    for page in cleaned:
        assert "SERVICE MANUAL" not in page["text"]
        assert "Confidential" not in page["text"]

    assert "108 Nm" in cleaned[0]["text"]
    assert "4.5 L" in cleaned[1]["text"]
    assert "90915-YZZD1" in cleaned[2]["text"]


def test_does_not_remove_spec_looking_edge_lines() -> None:
    # Identical torque lines on every page must NOT be treated as headers.
    pages = [
        {"page": 1, "text": "108 Nm\nBody content A"},
        {"page": 2, "text": "108 Nm\nBody content B"},
        {"page": 3, "text": "108 Nm\nBody content C"},
    ]
    cleaned = clean_pages(pages)
    for page in cleaned:
        assert "108 Nm" in page["text"]


def test_compare_raw_vs_cleaned(capsys) -> None:
    page = {"page": 7, "text": "Gap:   0.25 mm\n\n\nOil: 5W-30"}
    result = compare_raw_vs_cleaned(page, max_chars=200)
    assert result["page"] == 7
    assert result["changed"] is True
    assert "0.25 mm" in str(result["cleaned_preview"])
    assert "5W-30" in str(result["cleaned_preview"])
    out = capsys.readouterr().out
    assert "RAW" in out and "CLEANED" in out


def test_strips_pdf_web_export_artifacts() -> None:
    raw = "\n".join(
        [
            "Page 7 sur 8",
            "2014 F-150 Workshop Manual",
            "2014-03-01",
            "file:///C:/TSO/tsocache/manual/page.html",
            "repair4less",
            "Front Brake",
            "Wheel nut torque: 108 Nm",
            "Oil grade: 5W-30",
            "Filter: 90915-YZZD1",
        ]
    )
    cleaned = clean_text(raw)
    assert "Page 7 sur 8" not in cleaned
    assert "Workshop Manual" not in cleaned
    assert "2014-03-01" not in cleaned
    assert "file:///" not in cleaned
    assert "repair4less" not in cleaned
    assert "Front Brake" in cleaned
    assert "108 Nm" in cleaned
    assert "5W-30" in cleaned
    assert "90915-YZZD1" in cleaned


def test_page_of_artifact_variants() -> None:
    assert "Torque 18 Nm" in clean_text("Page 3 of 10\nTorque 18 Nm")
    assert "Page 3 of 10" not in clean_text("Page 3 of 10\nTorque 18 Nm")
    assert "Clearance 0.25 mm" in clean_text("Page 2/4\nClearance 0.25 mm")
