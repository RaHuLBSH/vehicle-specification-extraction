"""Structured vehicle-spec extraction via Google Gemini and Pydantic.

Uses the official ``google.genai`` SDK with JSON response schemas.
Gemini must extract ONLY from retrieved chunks — never from model memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Sequence, Union

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# --- Central Gemini configuration -------------------------------------------------
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
# Stable Flash fallbacks, ordered from newest to oldest. The configured model
# is always attempted first, regardless of whether it appears in this list.
GEMINI_FLASH_FALLBACK_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
)

_client: Any = None
logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    """Raised when Gemini extraction fails or returns invalid structured data."""


def _import_genai() -> tuple[Any, Any]:
    """Import the official Google Gen AI SDK (lazy, clear error if missing)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError(
            "The official Google Gen AI SDK is required. Install with: "
            "pip install 'google-genai>=1.0.0'"
        ) from exc
    return genai, types


class VehicleSpecification(BaseModel):
    """One automotive specification grounded in retrieved context."""

    model_config = ConfigDict(extra="forbid")

    component: str = Field(description="Component or system the specification applies to")
    spec_type: str = Field(description="Kind of specification, e.g. torque, capacity, part_number")
    value: Union[float, str] = Field(
        description="The value only, without its unit; numeric when the value is a measurement"
    )
    unit: str = Field(
        description="Primary measurement unit only; empty only for unitless values"
    )
    alt_value: Union[float, str, None] = Field(
        default=None,
        description="Optional alternate/conversion value; null when no alternate unit is supplied",
    )
    alt_unit: str | None = Field(
        default=None,
        description="Optional alternate/conversion unit; null when no alternate value is supplied",
    )
    page: int = Field(description="Source page number from the provided chunk metadata")
    source_text: str = Field(description="Short verbatim excerpt from the provided context")

    @field_validator("component", "spec_type", "unit", "source_text", mode="before")
    @classmethod
    def _coerce_str(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("alt_unit", mode="before")
    @classmethod
    def _coerce_optional_str(cls, value: Any) -> str | None:
        return None if value is None else str(value).strip()

    @field_validator("page", mode="before")
    @classmethod
    def _coerce_page(cls, value: Any) -> int:
        return int(value)

    @field_validator("value", "alt_value", mode="before")
    @classmethod
    def _coerce_value(cls, value: Any) -> Union[float, str, None]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("boolean values are not allowed for specification value")
        if isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, float):
            return value
        if isinstance(value, str):
            text = value.strip()
            # Keep part numbers / grades as strings; parse plain numerics.
            try:
                if text and all(ch.isdigit() or ch in ".-" for ch in text) and any(
                    ch.isdigit() for ch in text
                ):
                    return float(text)
            except ValueError:
                pass
            return text
        raise ValueError(f"unsupported value type: {type(value)!r}")

    @model_validator(mode="after")
    def _normalize_measurement_fields(self) -> VehicleSpecification:
        """Normalize combined and alternate measurement values from model output.

        This is deliberately narrow: it only changes an empty unit and a value
        made of one leading number followed by whitespace and a unit-like text.
        Part numbers and unitless strings remain untouched.
        """
        if not self.unit and isinstance(self.value, str):
            match = re.fullmatch(
                r"([+-]?\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)\s+([A-Za-z°µ][A-Za-z0-9°µ./·^() -]*)",
                self.value.strip(),
            )
            if match is not None:
                value_text = match.group(1)
                self.value = (
                    float(value_text)
                    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value_text)
                    else value_text
                )
                self.unit = match.group(2).strip()

        if self.unit and self.alt_value is None and self.alt_unit is None:
            alternate = re.fullmatch(
                r"(.+?)\s*\(\s*([+-]?\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)\s+([^)]+)\s*\)",
                self.unit,
            )
            if alternate is not None:
                alt_value_text = alternate.group(2)
                self.unit = alternate.group(1).strip()
                self.alt_value = (
                    float(alt_value_text)
                    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", alt_value_text)
                    else alt_value_text
                )
                self.alt_unit = alternate.group(3).strip()
        return self


class ExtractionResult(BaseModel):
    """Structured Gemini extraction outcome (found or clean not-found)."""

    model_config = ConfigDict(extra="forbid")

    found: bool = Field(description="True only when at least one grounded specification is found")
    specifications: list[VehicleSpecification] = Field(
        description="Extracted specifications; must be empty when found is false"
    )
    reason: str = Field(
        description="Empty when found; otherwise a short explanation that context is insufficient"
    )

    @model_validator(mode="after")
    def _normalize_found_state(self) -> ExtractionResult:
        if self.found and not self.specifications:
            raise ValueError("found=true requires at least one specification")
        if not self.found and self.specifications:
            raise ValueError("found=false must not include specifications")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return self.model_dump()


def get_gemini_config() -> dict[str, str]:
    """Return centralized Gemini configuration (no secrets)."""
    return {
        "model_name": GEMINI_MODEL_NAME,
        "api_key_env": GEMINI_API_KEY_ENV,
    }


def _load_api_key() -> str:
    """Read ``GEMINI_API_KEY`` from the environment (``.env`` supported)."""
    load_dotenv()
    api_key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ExtractionError(
            f"{GEMINI_API_KEY_ENV} is not set. Copy .env.example to .env and add your key."
        )
    return api_key


def get_client() -> Any:
    """Return a shared Gemini client loaded once per process."""
    global _client
    if _client is None:
        api_key = _load_api_key()
        genai, _types = _import_genai()
        _client = genai.Client(api_key=api_key)
    return _client


def reset_client_cache() -> None:
    """Drop the cached Gemini client (intended for tests)."""
    global _client
    _client = None


def _format_chunks_for_prompt(chunks: Sequence[dict[str, Any]]) -> str:
    """Render retrieved chunks with page/section metadata for the model."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", f"chunk_{index}")
        page = chunk.get("page", "?")
        section = chunk.get("section", "")
        score = chunk.get("score")
        text = chunk.get("text", "")
        header = f"[Chunk {index} | id={chunk_id} | page={page} | section={section}"
        if score is not None:
            header += f" | score={score}"
        header += "]"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def build_extraction_prompt(query: str, chunks: Sequence[dict[str, Any]]) -> str:
    """Build a strict grounding prompt for Gemini."""
    context = _format_chunks_for_prompt(chunks)
    return f"""You are a careful automotive service-manual extraction engine.

Extract vehicle specifications that answer the user query.
Use ONLY the retrieved context below. Do NOT use automotive knowledge from memory.
Do NOT guess. Do NOT invent values, units, part numbers, or pages.

Rules:
1. Use ONLY the provided context chunks.
2. Put only the primary value in value and only its primary unit in unit. When
   the source includes a parenthetical conversion, put its number in alt_value
   and its unit in alt_unit. For example, "204 Nm (150 lb-ft)" means
   value=204, unit="Nm", alt_value=150, alt_unit="lb-ft". Use null for both
   alternate fields when no alternate measurement is supplied.
3. Preserve units exactly (Nm, N·m, L, ml, mm, psi, etc.).
4. Preserve part numbers and identifiers exactly (including hyphens).
5. If multiple specifications in the context genuinely answer the query, return all of them.
6. If the context does not support an answer, set found=false, specifications=[], and explain in reason.
7. source_text must be a short supporting excerpt copied from the supplied context.
8. page must come from the chunk metadata that supplied the source_text.
9. unit may be an empty string when not applicable (e.g. some part numbers).

User query:
{query}

Retrieved context:
{context}
"""


def _finalize_result(result: ExtractionResult) -> ExtractionResult:
    """Normalize reason fields after Pydantic validation."""
    if result.found:
        if result.reason:
            return result.model_copy(update={"reason": ""})
        return result
    reason = (result.reason or "").strip()
    if not reason:
        reason = "No supporting specification found in the retrieved context."
    if reason != result.reason:
        return result.model_copy(update={"reason": reason})
    return result


def _not_found(reason: str) -> ExtractionResult:
    return _finalize_result(
        ExtractionResult(found=False, specifications=[], reason=reason)
    )


def _gemini_response_schema() -> dict[str, Any]:
    """Build a Gemini Developer API-compatible schema for the response.

    Pydantic emits ``additionalProperties: false`` for models configured with
    ``extra=\"forbid\"``.  The Gemini Developer API does not accept
    ``additionalProperties`` in a response schema, while we still want that
    strictness when validating the model's response locally.  Remove the
    unsupported keyword from every nested Pydantic schema node before passing
    it to the SDK.
    """
    def remove_unsupported_keywords(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: remove_unsupported_keywords(item)
                for key, item in value.items()
                if key not in {"additionalProperties", "additional_properties"}
            }
        if isinstance(value, list):
            return [remove_unsupported_keywords(item) for item in value]
        return value

    return remove_unsupported_keywords(ExtractionResult.model_json_schema())


def _parse_response_payload(payload: Any) -> ExtractionResult:
    """Validate Gemini structured output with Pydantic."""
    if payload is None:
        raise ExtractionError("Gemini returned an empty structured response")

    if isinstance(payload, ExtractionResult):
        return _finalize_result(payload)

    if isinstance(payload, BaseModel):
        payload = payload.model_dump()

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            raise ExtractionError("Gemini returned an empty text response")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExtractionError("Gemini returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise ExtractionError(
            f"Unexpected Gemini payload type: {type(payload)!r}"
        )

    try:
        return _finalize_result(ExtractionResult.model_validate(payload))
    except ValidationError as exc:
        raise ExtractionError(f"Gemini structured output failed validation: {exc}") from exc


def _model_candidates(primary_model: str) -> list[str]:
    """Return the configured model followed by unique supported fallbacks."""
    return list(dict.fromkeys((primary_model, *GEMINI_FLASH_FALLBACK_MODELS)))


def _generate_with_model_fallback(
    client: Any, *, prompt: str, config: dict[str, Any], primary_model: str
) -> Any:
    """Retry every failed generation request with the next Flash model.

    ``generate_content`` returns normally only for successful HTTP responses;
    failures such as an unavailable model (404) or high demand (503) arrive as
    exceptions. Parsing and validation failures are intentionally not retried.
    """
    last_error: Exception | None = None
    candidates = _model_candidates(primary_model)
    for position, candidate in enumerate(candidates):
        try:
            return client.models.generate_content(model=candidate, contents=prompt, config=config)
        except Exception as exc:
            last_error = exc
            if position == len(candidates) - 1:
                raise
            logger.warning(
                "Gemini request failed with model %s; retrying with %s: %s",
                candidate,
                candidates[position + 1],
                exc,
            )
    raise RuntimeError("No Gemini model candidates were available") from last_error


def extract_specifications(
    query: str,
    chunks: Sequence[dict[str, Any]] | None = None,
    *,
    context: str | None = None,
    model_name: str | None = None,
    client: Any | None = None,
) -> ExtractionResult:
    """Extract structured vehicle specifications from retrieved chunks.

    Args:
        query: User's natural-language specification question.
        chunks: Retrieved chunk dicts (``text``, ``page``, ``section``, …).
        context: Optional pre-joined context string (used only when ``chunks``
            is omitted; prefer passing ``chunks`` so page metadata is preserved).
        model_name: Override ``GEMINI_MODEL`` / default model id.
        client: Optional Gemini client (for tests / dependency injection).

    Returns:
        Validated :class:`ExtractionResult` (``found`` true/false).

    Raises:
        ExtractionError: On missing API key, API failures, empty/invalid output.
        ValueError: On blank query.
    """
    if not query or not str(query).strip():
        raise ValueError("query must be a non-empty string")

    chunk_list: list[dict[str, Any]]
    if chunks is not None:
        chunk_list = list(chunks)
    elif context is not None:
        chunk_list = [
            {
                "chunk_id": "chunk_0001",
                "text": context,
                "page": 0,
                "section": "Provided context",
                "score": 0.0,
            }
        ]
    else:
        raise ValueError("provide retrieved chunks (preferred) or a context string")

    if not chunk_list or not any(str(c.get("text", "")).strip() for c in chunk_list):
        return _not_found("No retrieved context was provided.")

    prompt = build_extraction_prompt(str(query).strip(), chunk_list)
    active_client = client or get_client()
    model = model_name or GEMINI_MODEL_NAME

    # Use a sanitized dict instead of passing the Pydantic model directly:
    # Gemini Developer API rejects Pydantic's ``additionalProperties: false``.
    # Local parsing below remains strict through ``extra=\"forbid\"``.
    config = {
        "response_mime_type": "application/json",
        "response_schema": _gemini_response_schema(),
        "temperature": 0.0,
        # This request has no tools. Disable the SDK's default AFC wrapper,
        # which otherwise emits an irrelevant Models.generate_content warning.
        "automatic_function_calling": {"disable": True},
    }

    try:
        response = _generate_with_model_fallback(
            active_client,
            prompt=prompt,
            config=config,
            primary_model=model,
        )
    except ExtractionError:
        raise
    except Exception as exc:  # SDK / network / auth errors
        raise ExtractionError(f"Gemini API request failed: {exc}") from exc

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return _parse_response_payload(parsed)

    text = getattr(response, "text", None)
    if text is None or not str(text).strip():
        raise ExtractionError("Gemini returned an empty response")
    return _parse_response_payload(text)


def extract_specifications_as_dict(
    query: str,
    chunks: Sequence[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper returning a plain dict."""
    return extract_specifications(query, chunks, **kwargs).to_dict()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug Gemini structured extraction from sample retrieved chunks."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Torque for brake caliper bolts",
        help="Natural-language specification query",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI smoke test with hard-coded sample chunks (no retrieval changes)."""
    args = _build_arg_parser().parse_args(argv)
    sample_chunks = [
        {
            "chunk_id": "chunk_0001",
            "text": "Front Brake\nCaliper bolt torque: 28 Nm\nBleed screw torque: 11 Nm",
            "page": 42,
            "section": "Front Brake",
            "score": 0.91,
        },
        {
            "chunk_id": "chunk_0002",
            "text": "Engine oil capacity: 4.5 L grade 5W-30\nOil filter: 90915-YZZD1",
            "page": 10,
            "section": "Engine Oil",
            "score": 0.55,
        },
    ]
    try:
        result = extract_specifications(args.query, sample_chunks)
    except (ExtractionError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
