"""Tests for Gemini structured extraction (mocked SDK calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.extractor import (
    ExtractionError,
    ExtractionResult,
    VehicleSpecification,
    extract_specifications,
    reset_client_cache,
    _finalize_result,
    _gemini_response_schema,
)


SAMPLE_CHUNKS = [
    {
        "chunk_id": "chunk_0001",
        "text": "Front Brake caliper bolt torque: 28 Nm",
        "page": 42,
        "section": "Front Brake",
        "score": 0.9,
    }
]


@pytest.fixture(autouse=True)
def _reset_client() -> None:
    reset_client_cache()
    yield
    reset_client_cache()


def test_vehicle_specification_preserves_part_numbers_and_units() -> None:
    spec = VehicleSpecification(
        component="Engine",
        spec_type="oil_filter",
        value="90915-YZZD1",
        unit="",
        page=11,
        source_text="Oil filter: 90915-YZZD1",
    )
    assert spec.value == "90915-YZZD1"
    assert spec.unit == ""


def test_vehicle_specification_parses_numeric_value() -> None:
    spec = VehicleSpecification(
        component="Front Brake",
        spec_type="torque",
        value="28",
        unit="Nm",
        page=42,
        source_text="caliper bolt torque: 28 Nm",
    )
    assert spec.value == 28.0


@pytest.mark.parametrize(
    ("combined_value", "expected_value", "expected_unit", "expected_alt_value", "expected_alt_unit"),
    [
        ("204 Nm (150 lb-ft)", 204.0, "Nm", 150.0, "lb-ft"),
        ("32 mm (1.259 in)", 32.0, "mm", 1.259, "in"),
        ("23.0 mm (0.906 in)", 23.0, "mm", 0.906, "in"),
        ("207-345 kPa (30-50 psi)", "207-345", "kPa", "30-50", "psi"),
    ],
)
def test_vehicle_specification_splits_combined_measurement_into_value_and_unit(
    combined_value: str,
    expected_value: float | str,
    expected_unit: str,
    expected_alt_value: float | str,
    expected_alt_unit: str,
) -> None:
    spec = VehicleSpecification(
        component="Brake disc",
        spec_type="minimum_thickness",
        value=combined_value,
        unit="",
        page=599,
        source_text=f"Minimum thickness: {combined_value}",
    )

    assert spec.value == expected_value
    assert spec.unit == expected_unit
    assert spec.alt_value == expected_alt_value
    assert spec.alt_unit == expected_alt_unit


def test_not_found_result_validation() -> None:
    result = _finalize_result(
        ExtractionResult(found=False, specifications=[], reason="")
    )
    assert result.found is False
    assert result.specifications == []
    assert "No supporting" in result.reason


def test_found_requires_specifications() -> None:
    with pytest.raises(ValidationError):
        ExtractionResult(found=True, specifications=[], reason="")


def test_extract_empty_chunks_returns_not_found() -> None:
    result = extract_specifications("torque?", [])
    assert result.found is False
    assert result.specifications == []


def test_extract_blank_query_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        extract_specifications("  ", SAMPLE_CHUNKS)


def test_extract_uses_structured_schema_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    payload = {
        "found": True,
        "specifications": [
            {
                "component": "Front Brake",
                "spec_type": "torque",
                "value": 28,
                "unit": "Nm",
                "page": 42,
                "source_text": "caliper bolt torque: 28 Nm",
            }
        ],
        "reason": "",
    }
    fake_response = MagicMock()
    fake_response.parsed = payload
    fake_response.text = None

    fake_models = MagicMock()
    fake_models.generate_content.return_value = fake_response
    fake_client = MagicMock()
    fake_client.models = fake_models

    result = extract_specifications(
        "Torque for brake caliper bolts",
        SAMPLE_CHUNKS,
        client=fake_client,
    )

    assert result.found is True
    assert result.specifications[0].value == 28.0
    assert result.specifications[0].unit == "Nm"
    assert result.specifications[0].alt_value is None
    assert result.specifications[0].alt_unit is None
    assert result.specifications[0].page == 42

    kwargs = fake_models.generate_content.call_args.kwargs
    assert kwargs["config"]["response_mime_type"] == "application/json"
    schema = kwargs["config"]["response_schema"]
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    assert "additionalProperties" not in str(schema)
    assert kwargs["config"]["automatic_function_calling"] == {"disable": True}
    assert "ONLY" in kwargs["contents"] or "only" in kwargs["contents"].lower()
    assert "28 Nm" in kwargs["contents"]
    assert "page=42" in kwargs["contents"]


def test_extract_retries_with_flash_fallback_for_unavailable_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    fake_response = MagicMock(parsed={"found": False, "specifications": [], "reason": "Missing"})
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        RuntimeError("404 model not found"),
        fake_response,
    ]

    result = extract_specifications(
        "q", SAMPLE_CHUNKS, model_name="gemini-unavailable", client=fake_client
    )

    assert result.found is False
    assert [call.kwargs["model"] for call in fake_client.models.generate_content.call_args_list] == [
        "gemini-unavailable",
        "gemini-3.8-flash",
    ]


def test_extract_retries_with_flash_fallback_for_service_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    fake_response = MagicMock(parsed={"found": False, "specifications": [], "reason": "Missing"})
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        RuntimeError("503 service unavailable: high demand"),
        fake_response,
    ]

    result = extract_specifications(
        "q", SAMPLE_CHUNKS, model_name="gemini-3.5-flash", client=fake_client
    )

    assert result.found is False
    assert [call.kwargs["model"] for call in fake_client.models.generate_content.call_args_list] == [
        "gemini-3.5-flash",
        "gemini-3.8-flash",
    ]


def test_gemini_response_schema_removes_unsupported_additional_properties() -> None:
    schema = _gemini_response_schema()

    def contains_unsupported_keyword(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                key in {"additionalProperties", "additional_properties"}
                or contains_unsupported_keyword(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(contains_unsupported_keyword(item) for item in value)
        return False

    assert not contains_unsupported_keyword(schema)
    assert schema["$defs"]["VehicleSpecification"]["type"] == "object"


def test_extract_falls_back_to_response_text_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    fake_response = MagicMock()
    fake_response.parsed = None
    fake_response.text = (
        '{"found": false, "specifications": [], '
        '"reason": "Context does not mention wheel bearing torque."}'
    )
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    result = extract_specifications("wheel bearing torque", SAMPLE_CHUNKS, client=fake_client)
    assert result.found is False
    assert result.specifications == []


def test_extract_raises_on_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    fake_response = MagicMock()
    fake_response.parsed = None
    fake_response.text = "   "
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    with pytest.raises(ExtractionError, match="empty"):
        extract_specifications("q", SAMPLE_CHUNKS, client=fake_client)


def test_extract_raises_on_invalid_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    fake_response = MagicMock()
    fake_response.parsed = {"found": True, "specifications": [], "reason": ""}
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    with pytest.raises(ExtractionError, match="validation"):
        extract_specifications("q", SAMPLE_CHUNKS, client=fake_client)


def test_extract_raises_on_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("quota exceeded")

    with pytest.raises(ExtractionError, match="API request failed"):
        extract_specifications("q", SAMPLE_CHUNKS, client=fake_client)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("src.extractor.load_dotenv"):
        reset_client_cache()
        with pytest.raises(ExtractionError, match="GEMINI_API_KEY"):
            extract_specifications("q", SAMPLE_CHUNKS)
