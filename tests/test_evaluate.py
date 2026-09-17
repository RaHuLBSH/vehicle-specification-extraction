"""Tests for offline evaluation metrics; no embedding model or Gemini calls."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.evaluate import evaluate_extraction, evaluate_retrieval, load_evaluation_cases
from src.extractor import ExtractionResult, VehicleSpecification


CASES = [
    {
        "category": "torque",
        "query": "wheel nut torque",
        "expected_component": "Wheel nuts",
        "expected_value": 204.0,
        "expected_unit": "Nm",
        "expected_page": 199,
    },
    {
        "category": "dimension",
        "query": "brake disc thickness",
        "expected_component": "Front brake disc minimum thickness",
        "expected_value": 32.0,
        "expected_unit": "mm",
        "expected_page": 599,
    },
]


def test_dataset_has_verified_cases_across_required_categories() -> None:
    cases = load_evaluation_cases()
    categories = {case["category"] for case in cases}

    assert 15 <= len(cases) <= 20
    assert {"torque", "fluid_capacity", "part_number", "dimension", "pressure", "oil_fluid_specification"} <= categories
    assert all(case["expected_page"] > 0 for case in cases)


def test_retrieval_metrics_report_top_1_top_3_and_top_5() -> None:
    retriever = MagicMock()
    retriever.retrieve.side_effect = [
        [{"page": 199}],
        [{"page": 100}, {"page": 599}],
    ]

    metrics = evaluate_retrieval(CASES, retriever)

    assert metrics["top_1_accuracy"] == 0.5
    assert metrics["top_3_accuracy"] == 1.0
    assert metrics["top_5_accuracy"] == 1.0


def test_extraction_metrics_are_scored_independently() -> None:
    def predict(query: str) -> ExtractionResult:
        if "wheel" in query:
            return ExtractionResult(
                found=True,
                specifications=[
                    VehicleSpecification(
                        component="wheel nuts",
                        spec_type="torque",
                        value="204 Nm (150 lb-ft)",
                        unit="",
                        page=199,
                        source_text="Tighten to 204 Nm (150 lb-ft).",
                    )
                ],
                reason="",
            )
        return ExtractionResult(
            found=True,
            specifications=[
                VehicleSpecification(
                    component="Front brake disc minimum thickness",
                    spec_type="minimum_thickness",
                    value=32,
                    unit="mm (1.259 in)",
                    page=599,
                    source_text="Front brake disc minimum thickness 32 mm (1.259 in)",
                )
            ],
            reason="",
        )

    metrics = evaluate_extraction(CASES, predict)

    assert metrics["component_accuracy"] == 1.0
    assert metrics["value_accuracy"] == 1.0
    assert metrics["unit_accuracy"] == 1.0
    assert metrics["exact_specification_accuracy"] == 1.0
