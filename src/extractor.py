"""Structured vehicle-spec extraction via Google Gemini and Pydantic.

Uses the official ``google.genai`` SDK with JSON response schemas.

Gemini extracts specifications ONLY from retrieved chunks and must never
answer using model memory or unsupported automotive knowledge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Sequence, Union

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
)

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

_client: Any = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ExtractionError(RuntimeError):
    """Raised when Gemini extraction fails or returns invalid data."""


# ---------------------------------------------------------------------------
# Gemini SDK
# ---------------------------------------------------------------------------


def _import_genai() -> tuple[Any, Any]:
    """Import the official Google Gen AI SDK lazily."""

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ExtractionError(
            "The official Google Gen AI SDK is required. "
            "Install it with: pip install 'google-genai>=1.0.0'"
        ) from exc

    return genai, types


def _load_api_key() -> str:
    """Load GEMINI_API_KEY from environment or .env."""

    load_dotenv()

    api_key = os.getenv(
        GEMINI_API_KEY_ENV,
        "",
    ).strip()

    if (
        not api_key
        or api_key == "your_gemini_api_key_here"
    ):
        raise ExtractionError(
            f"{GEMINI_API_KEY_ENV} is not set. "
            "Copy .env.example to .env and add your Gemini API key."
        )

    return api_key


def get_client() -> Any:
    """Return one shared Gemini client per process."""

    global _client

    if _client is None:
        api_key = _load_api_key()
        genai, _types = _import_genai()

        _client = genai.Client(
            api_key=api_key
        )

    return _client


def reset_client_cache() -> None:
    """Reset cached Gemini client.

    Primarily useful for unit tests.
    """

    global _client
    _client = None


def get_gemini_config() -> dict[str, str]:
    """Return non-secret Gemini configuration."""

    return {
        "model_name": GEMINI_MODEL_NAME,
        "api_key_env": GEMINI_API_KEY_ENV,
    }


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class VehicleSpecification(BaseModel):
    """One automotive specification grounded in retrieved context."""

    model_config = ConfigDict(
        extra="forbid"
    )

    component: str = Field(
        description=(
            "Component or system the specification applies to"
        )
    )

    spec_type: str = Field(
        description=(
            "Specification type, such as torque, capacity, "
            "part_number, pressure, thickness, or length"
        )
    )

    value: Union[float, str] = Field(
        description=(
            "Primary value without the unit. "
            "Use numeric values for measurements when possible."
        )
    )

    unit: str = Field(
        description=(
            "Primary measurement unit. "
            "May be empty for unitless values or part numbers."
        )
    )

    alt_value: Union[float, str, None] = Field(
        default=None,
        description=(
            "Optional alternate/conversion value. "
            "Null when no alternate measurement is provided."
        ),
    )

    alt_unit: str | None = Field(
        default=None,
        description=(
            "Optional alternate/conversion unit. "
            "Null when no alternate measurement is provided."
        ),
    )

    page: int = Field(
        description=(
            "Page number from the retrieved chunk metadata"
        )
    )

    source_text: str = Field(
        description=(
            "Short verbatim supporting excerpt "
            "from the retrieved context"
        )
    )

    # ------------------------------------------------------------------
    # Field validation
    # ------------------------------------------------------------------

    @field_validator(
        "component",
        "spec_type",
        "unit",
        "source_text",
        mode="before",
    )
    @classmethod
    def _coerce_str(
        cls,
        value: Any,
    ) -> str:
        if value is None:
            return ""

        return str(value).strip()

    @field_validator(
        "alt_unit",
        mode="before",
    )
    @classmethod
    def _coerce_optional_str(
        cls,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return str(value).strip()

    @field_validator(
        "page",
        mode="before",
    )
    @classmethod
    def _coerce_page(
        cls,
        value: Any,
    ) -> int:
        return int(value)

    @field_validator(
        "value",
        "alt_value",
        mode="before",
    )
    @classmethod
    def _coerce_value(
        cls,
        value: Any,
    ) -> Union[float, str, None]:
        """Normalize plain numeric values while preserving identifiers."""

        if value is None:
            return None

        if isinstance(value, bool):
            raise ValueError(
                "boolean values are not allowed "
                "for specification values"
            )

        if isinstance(value, int):
            return float(value)

        if isinstance(value, float):
            return value

        if isinstance(value, str):
            text = value.strip()

            # Parse only plain numeric strings.
            # Preserve things such as:
            #   5W-30
            #   M10x1.25
            #   XY-75W140-QL
            #   90915-YZZD1

            try:
                if (
                    text
                    and all(
                        ch.isdigit() or ch in ".-"
                        for ch in text
                    )
                    and any(
                        ch.isdigit()
                        for ch in text
                    )
                ):
                    return float(text)

            except ValueError:
                pass

            return text

        raise ValueError(
            f"Unsupported value type: {type(value)!r}"
        )

    # ------------------------------------------------------------------
    # Model normalization
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _normalize_measurement_fields(
        self,
    ) -> VehicleSpecification:
        """Normalize narrowly formatted measurement responses."""

        # Example:
        #
        # value="35 Nm"
        # unit=""
        #
        # becomes:
        #
        # value=35
        # unit="Nm"

        if (
            not self.unit
            and isinstance(self.value, str)
        ):
            match = re.fullmatch(
                (
                    r"([+-]?\d+(?:\.\d+)?"
                    r"(?:\s*[-–]\s*\d+(?:\.\d+)?)?)"
                    r"\s+"
                    r"([A-Za-z°µ]"
                    r"[A-Za-z0-9°µ./·^() -]*)"
                ),
                self.value.strip(),
            )

            if match is not None:
                value_text = match.group(1)

                if re.fullmatch(
                    r"[+-]?\d+(?:\.\d+)?",
                    value_text,
                ):
                    self.value = float(
                        value_text
                    )
                else:
                    self.value = value_text

                self.unit = (
                    match.group(2).strip()
                )

        # Example:
        #
        # unit="Nm (150 lb-ft)"
        #
        # becomes:
        #
        # unit="Nm"
        # alt_value=150
        # alt_unit="lb-ft"

        if (
            self.unit
            and self.alt_value is None
            and self.alt_unit is None
        ):
            alternate = re.fullmatch(
                (
                    r"(.+?)\s*"
                    r"\(\s*"
                    r"([+-]?\d+(?:\.\d+)?"
                    r"(?:\s*[-–]\s*\d+(?:\.\d+)?)?)"
                    r"\s+([^)]+)\s*\)"
                ),
                self.unit,
            )

            if alternate is not None:
                alt_value_text = (
                    alternate.group(2)
                )

                self.unit = (
                    alternate.group(1).strip()
                )

                if re.fullmatch(
                    r"[+-]?\d+(?:\.\d+)?",
                    alt_value_text,
                ):
                    self.alt_value = float(
                        alt_value_text
                    )
                else:
                    self.alt_value = (
                        alt_value_text
                    )

                self.alt_unit = (
                    alternate.group(3).strip()
                )

        return self


class ExtractionResult(BaseModel):
    """Structured Gemini extraction result."""

    model_config = ConfigDict(
        extra="forbid"
    )

    found: bool = Field(
        description=(
            "True only when at least one grounded "
            "specification is found"
        )
    )

    specifications: list[VehicleSpecification] = Field(
        description=(
            "Extracted specifications. "
            "Must be empty when found is false."
        )
    )

    reason: str = Field(
        description=(
            "Empty when found=true. "
            "Otherwise explains why no supported answer was found."
        )
    )

    @model_validator(mode="after")
    def _normalize_found_state(
        self,
    ) -> ExtractionResult:

        if (
            self.found
            and not self.specifications
        ):
            raise ValueError(
                "found=true requires at least "
                "one specification"
            )

        if (
            not self.found
            and self.specifications
        ):
            raise ValueError(
                "found=false must not include "
                "specifications"
            )

        return self

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Serialize result to plain dictionary."""

        return self.model_dump()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_chunks_for_prompt(
    chunks: Sequence[dict[str, Any]],
) -> str:
    """Format retrieved chunks with their metadata."""

    blocks: list[str] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        chunk_id = chunk.get(
            "chunk_id",
            f"chunk_{index}",
        )

        page = chunk.get(
            "page",
            "?",
        )

        section = chunk.get(
            "section",
            "",
        )

        score = chunk.get("score")
        text = chunk.get("text", "")

        header = (
            f"[Chunk {index}"
            f" | id={chunk_id}"
            f" | page={page}"
            f" | section={section}"
        )

        if score is not None:
            header += (
                f" | score={score}"
            )

        header += "]"

        blocks.append(
            f"{header}\n{text}"
        )

    return "\n\n".join(blocks)


def build_extraction_prompt(
    query: str,
    chunks: Sequence[dict[str, Any]],
) -> str:
    """Build strict grounded extraction prompt."""

    context = _format_chunks_for_prompt(
        chunks
    )

    return f"""
You are a careful automotive service-manual specification extraction engine.

Your task is to extract vehicle specifications that directly answer the
user's query.

Use ONLY the retrieved context supplied below.

Do NOT use automotive knowledge from memory.
Do NOT guess.
Do NOT infer unsupported values.
Do NOT invent values, units, part numbers, components, or page numbers.

Rules:

1. Use ONLY information explicitly supported by the retrieved context.

2. Put only the primary value in "value" and only its primary measurement
   unit in "unit".

3. When the source provides a parenthetical alternate/conversion value,
   store it in "alt_value" and "alt_unit".

   Example:

       204 Nm (150 lb-ft)

   should become:

       value = 204
       unit = "Nm"
       alt_value = 150
       alt_unit = "lb-ft"

4. Use null for alt_value and alt_unit when no alternate measurement exists.

5. Preserve measurement units exactly as written in the source whenever
   possible, including:
   Nm, N·m, L, ml, mm, psi, lb-ft, etc.

6. Preserve part numbers, lubricant identifiers, grades, and other
   identifiers exactly, including hyphens.

7. If multiple specifications genuinely answer the query, return all of them.

8. If the retrieved context does NOT contain enough information to answer
   the query:

       found = false
       specifications = []

   and provide a short explanation in "reason".

9. "source_text" must contain a short verbatim excerpt from the supplied
   retrieved context that supports the extracted specification.

10. "page" must be copied from the metadata of the chunk containing the
    supporting source_text.

11. "unit" may be an empty string when a measurement unit is not applicable,
    such as for some part numbers.

User query:
{query}

Retrieved context:
{context}
""".strip()


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------


def _finalize_result(
    result: ExtractionResult,
) -> ExtractionResult:
    """Normalize reason field after validation."""

    if result.found:
        if result.reason:
            return result.model_copy(
                update={"reason": ""}
            )

        return result

    reason = (
        result.reason or ""
    ).strip()

    if not reason:
        reason = (
            "No supporting specification found "
            "in the retrieved context."
        )

    if reason != result.reason:
        return result.model_copy(
            update={"reason": reason}
        )

    return result


def _not_found(
    reason: str,
) -> ExtractionResult:
    """Return standardized not-found result."""

    return _finalize_result(
        ExtractionResult(
            found=False,
            specifications=[],
            reason=reason,
        )
    )


# ---------------------------------------------------------------------------
# Gemini response schema
# ---------------------------------------------------------------------------


def _gemini_response_schema() -> dict[str, Any]:
    """Create Gemini-compatible JSON response schema.

    Pydantic emits ``additionalProperties: false`` when models use
    ``extra="forbid"``.

    Gemini's response schema does not accept that field, so it is removed
    recursively before sending the schema to the API.

    Strict validation is still performed locally using Pydantic.
    """

    def remove_unsupported_keywords(
        value: Any,
    ) -> Any:

        if isinstance(value, dict):
            return {
                key: remove_unsupported_keywords(
                    item
                )
                for key, item in value.items()
                if key
                not in {
                    "additionalProperties",
                    "additional_properties",
                }
            }

        if isinstance(value, list):
            return [
                remove_unsupported_keywords(
                    item
                )
                for item in value
            ]

        return value

    schema = (
        ExtractionResult.model_json_schema()
    )

    return remove_unsupported_keywords(
        schema
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response_payload(
    payload: Any,
) -> ExtractionResult:
    """Validate Gemini structured output using Pydantic."""

    if payload is None:
        raise ExtractionError(
            "Gemini returned an empty "
            "structured response"
        )

    if isinstance(
        payload,
        ExtractionResult,
    ):
        return _finalize_result(
            payload
        )

    if isinstance(
        payload,
        BaseModel,
    ):
        payload = payload.model_dump()

    if isinstance(
        payload,
        str,
    ):
        text = payload.strip()

        if not text:
            raise ExtractionError(
                "Gemini returned an empty "
                "text response"
            )

        try:
            payload = json.loads(
                text
            )

        except json.JSONDecodeError as exc:
            raise ExtractionError(
                "Gemini returned invalid JSON"
            ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise ExtractionError(
            "Unexpected Gemini payload type: "
            f"{type(payload)!r}"
        )

    try:
        result = (
            ExtractionResult.model_validate(
                payload
            )
        )

    except ValidationError as exc:
        raise ExtractionError(
            "Gemini structured output "
            f"failed validation: {exc}"
        ) from exc

    return _finalize_result(
        result
    )


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_specifications(
    query: str,
    chunks: Sequence[dict[str, Any]] | None = None,
    *,
    context: str | None = None,
    model_name: str | None = None,
    client: Any | None = None,
) -> ExtractionResult:
    """Extract structured vehicle specifications using Gemini.

    The function performs ONE Gemini request.

    Retry/backoff behavior should be implemented by the caller when needed,
    such as the offline evaluation pipeline.

    Args:
        query:
            Natural-language specification question.

        chunks:
            Retrieved chunk dictionaries containing fields such as text,
            page, section, chunk_id, and retrieval score.

        context:
            Optional raw context string. Prefer ``chunks`` because chunk
            metadata provides page-level provenance.

        model_name:
            Optional Gemini model override.

        client:
            Optional Gemini client, primarily for dependency injection/tests.

    Returns:
        Validated ExtractionResult.

    Raises:
        ExtractionError:
            Gemini API failure or invalid structured output.

        ValueError:
            Invalid query or missing context.
    """

    # ------------------------------------------------------------------
    # Validate query
    # ------------------------------------------------------------------

    if (
        not query
        or not str(query).strip()
    ):
        raise ValueError(
            "query must be a non-empty string"
        )

    # ------------------------------------------------------------------
    # Prepare context
    # ------------------------------------------------------------------

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
        raise ValueError(
            "provide retrieved chunks "
            "(preferred) or a context string"
        )

    if (
        not chunk_list
        or not any(
            str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()
            for chunk in chunk_list
        )
    ):
        return _not_found(
            "No retrieved context was provided."
        )

    # ------------------------------------------------------------------
    # Build Gemini request
    # ------------------------------------------------------------------

    prompt = build_extraction_prompt(
        str(query).strip(),
        chunk_list,
    )

    active_client = (
        client or get_client()
    )

    model = (
        model_name
        or GEMINI_MODEL_NAME
    )

    config = {
        "response_mime_type": "application/json",
        "response_schema": _gemini_response_schema(),
        "temperature": 0.0,

        # No function calling is required for extraction.
        "automatic_function_calling": {
            "disable": True
        },
    }

    # ------------------------------------------------------------------
    # Gemini API call
    # ------------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # This function deliberately makes ONE API request.
    #
    # It does NOT:
    #   - cycle through fallback models
    #   - perform retries
    #   - sleep
    #   - implement exponential backoff
    #
    # Retry/backoff belongs to callers such as evaluate.py.
    # ------------------------------------------------------------------

    try:
        response = (
            active_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        )

    except Exception as exc:
        raise ExtractionError(
            "Gemini API request failed "
            f"with model {model}: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Structured response
    # ------------------------------------------------------------------

    parsed = getattr(
        response,
        "parsed",
        None,
    )

    if parsed is not None:
        return _parse_response_payload(
            parsed
        )

    # ------------------------------------------------------------------
    # Text fallback
    # ------------------------------------------------------------------

    text = getattr(
        response,
        "text",
        None,
    )

    if (
        text is None
        or not str(text).strip()
    ):
        raise ExtractionError(
            "Gemini returned an empty response"
        )

    return _parse_response_payload(
        text
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def extract_specifications_as_dict(
    query: str,
    chunks: Sequence[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Extract specifications and return plain dictionary."""

    return extract_specifications(
        query,
        chunks,
        **kwargs,
    ).to_dict()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Debug Gemini structured extraction "
            "using sample retrieved chunks."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "Torque for brake caliper bolts"
        ),
        help=(
            "Natural-language specification query"
        ),
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    """Run Gemini extraction against sample context."""

    args = (
        _build_arg_parser()
        .parse_args(argv)
    )

    sample_chunks = [
        {
            "chunk_id": "chunk_0001",
            "text": (
                "Front Brake\n"
                "Caliper bolt torque: 28 Nm\n"
                "Bleed screw torque: 11 Nm"
            ),
            "page": 42,
            "section": "Front Brake",
            "score": 0.91,
        },
        {
            "chunk_id": "chunk_0002",
            "text": (
                "Engine oil capacity: 4.5 L "
                "grade 5W-30\n"
                "Oil filter: 90915-YZZD1"
            ),
            "page": 10,
            "section": "Engine Oil",
            "score": 0.55,
        },
    ]

    try:
        result = extract_specifications(
            args.query,
            sample_chunks,
        )

    except (
        ExtractionError,
        ValueError,
    ) as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())