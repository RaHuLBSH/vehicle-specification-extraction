"""Offline-friendly evaluation for persisted retrieval and Gemini extraction.

The dataset is manually verified against the supplied service manual. Retrieval
can be measured without Gemini; extraction is deliberately opt-in because it
uses the configured external API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence, TypedDict

from src.extractor import ExtractionResult, extract_specifications
from src.retriever import RetrievalResult, Retriever

DEFAULT_DATASET_PATH = Path("data/evaluation.json")


class EvaluationCase(TypedDict):
    category: str
    query: str
    expected_component: str
    expected_value: float | str
    expected_unit: str
    expected_page: int


def load_evaluation_cases(path: str | Path = DEFAULT_DATASET_PATH) -> list[EvaluationCase]:
    """Load and minimally validate manually verified evaluation cases."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read evaluation dataset: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid evaluation JSON: {source}") from exc
    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must be a JSON list")

    required = {
        "category", "query", "expected_component", "expected_value", "expected_unit", "expected_page"
    }
    cases: list[EvaluationCase] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or required - item.keys():
            raise ValueError(f"Evaluation case {index} is missing required fields")
        if not isinstance(item["expected_page"], int):
            raise ValueError(f"Evaluation case {index} has an invalid expected_page")
        cases.append(item)  # type: ignore[arg-type]
    return cases


def evaluate_retrieval(
    cases: Sequence[EvaluationCase], retriever: Retriever
) -> dict[str, float | int]:
    """Measure whether a known supporting page appears in ranks 1, 3, and 5."""
    totals = {1: 0, 3: 0, 5: 0}
    for case in cases:
        hits = retriever.retrieve(case["query"], top_k=5)
        pages = [hit["page"] for hit in hits]
        for k in totals:
            if case["expected_page"] in pages[:k]:
                totals[k] += 1
    total = len(cases)
    return {
        "cases": total,
        "top_1_correct": totals[1],
        "top_1_accuracy": totals[1] / total if total else 0.0,
        "top_3_correct": totals[3],
        "top_3_accuracy": totals[3] / total if total else 0.0,
        "top_5_correct": totals[5],
        "top_5_accuracy": totals[5] / total if total else 0.0,
    }


def _normalized(value: Any) -> str:
    """Normalize benign text and numeric formatting differences for scoring."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.12g}"
    return " ".join(str(value).casefold().split())


def _matches(specification: Any, case: EvaluationCase) -> tuple[bool, bool, bool]:
    return (
        _normalized(specification.component) == _normalized(case["expected_component"]),
        _normalized(specification.value) == _normalized(case["expected_value"]),
        _normalized(specification.unit) == _normalized(case["expected_unit"]),
    )


def evaluate_extraction(
    cases: Sequence[EvaluationCase],
    predict: Callable[[str], ExtractionResult],
) -> dict[str, float | int]:
    """Score structured extraction independently of how ``predict`` is implemented."""
    component = value = unit = exact = 0
    for case in cases:
        result = predict(case["query"])
        comparisons = [_matches(spec, case) for spec in result.specifications]
        component += any(match[0] for match in comparisons)
        value += any(match[1] for match in comparisons)
        unit += any(match[2] for match in comparisons)
        exact += any(all(match) for match in comparisons)
    total = len(cases)
    return {
        "cases": total,
        "correct_component": component,
        "component_accuracy": component / total if total else 0.0,
        "correct_value": value,
        "value_accuracy": value / total if total else 0.0,
        "correct_unit": unit,
        "unit_accuracy": unit / total if total else 0.0,
        "exact_specification_matches": exact,
        "exact_specification_accuracy": exact / total if total else 0.0,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and optional Gemini extraction.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--with-extraction", action="store_true", help="Call Gemini after retrieval (uses API quota)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run retrieval evaluation; optionally run API-backed extraction evaluation."""
    args = _build_arg_parser().parse_args(argv)
    try:
        cases = load_evaluation_cases(args.dataset)
        retriever = Retriever()
        report: dict[str, Any] = {"retrieval": evaluate_retrieval(cases, retriever)}
        if args.with_extraction:
            def predict(query: str) -> ExtractionResult:
                return extract_specifications(query, retriever.retrieve(query, top_k=5))
            report["extraction"] = evaluate_extraction(cases, predict)
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
