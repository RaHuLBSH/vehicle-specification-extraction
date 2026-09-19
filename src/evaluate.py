"""Evaluation for persisted retrieval and Gemini extraction.

Retrieval evaluation runs fully offline against the persisted FAISS index.
Gemini extraction evaluation is opt-in because it uses the external API.

Metrics
-------
Retrieval (answerable cases only)
    * page hit@k          - an expected page is among the top-k chunks
    * answer-in-context@k - some top-k chunk actually contains the expected
                            value (a fairer measure than page hit, because
                            the same spec often appears on several pages)

Extraction
    * core accuracy  - value AND unit AND page correct in the same
                       returned specification  (the headline metric)
    * full accuracy  - core AND the component label matches softly
    * value / unit / page / component accuracy individually
    * unanswerable accuracy - cases marked ``"expected_found": false`` must
                       come back ``found=false`` (hallucination check)

Component matching is "soft": the expected label's words must be covered
(>= --component-coverage, default 0.75) by the returned component and
spec_type. Manual row labels such as "Drive pinion flange runout" therefore
match ``component="Drive pinion flange", spec_type="runout"``.

Free-tier friendly features
---------------------------
* Every successful extraction is written to a JSONL cache immediately, so a
  rerun only spends API quota on cases that have not been evaluated yet.
* ``--cache-only`` re-scores cached results with NO API calls, which is
  handy after changing the scoring or dataset.
* Real API calls are paced (``--delay``); cached cases are never delayed.
* 503 / temporary 429 errors are retried, honouring the server's suggested
  wait. A daily-quota 429 (or ``limit: 0``) stops the run cleanly.

Dataset format (backward compatible)
------------------------------------
Required: ``category``, ``query``, ``expected_component``,
``expected_unit`` and a value and page in any of these forms:

    "expected_value": 204.0            or a list of acceptable values
    "expected_values": [204.0]         (alternative spelling)
    "expected_page": 199               or
    "expected_pages": [199, 446]       (any of these pages is accepted)

Optional: ``"expected_found": false`` for unanswerable queries (then only
``category`` and ``query`` are required).

A string value ending in a degree sign with an empty unit (e.g.
"0.5°-3.0°") is split into value "0.5-3.0" and unit "°". A string value
with ";"-separated identifiers, e.g. "PM-1-C (US); CPM-1-C (Canada)",
matches if the answer contains ANY of the listed identifiers.

Usage:
    python -m src.evaluate
    python -m src.evaluate --with-extraction
    python -m src.evaluate --with-extraction --delay 8
    python -m src.evaluate --with-extraction --skip-retrieval
    python -m src.evaluate --cache-only --skip-retrieval
    python -m src.evaluate --with-extraction --fresh

Exit codes:
    0    everything requested was evaluated
    1    unexpected error
    2    extraction evaluation incomplete (API failures or quota stop)
    130  interrupted (cached progress is kept)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence, TypedDict

from src.extractor import (
    GEMINI_MODEL_NAME,
    ExtractionResult,
    extract_specifications,
)
from src.retriever import Retriever


DEFAULT_DATASET_PATH = Path("data/evaluation.json")
DEFAULT_CACHE_PATH = Path("data/extraction_cache.jsonl")

# Free-tier RPM limits are low; ~7 requests/minute is a safe default.
DEFAULT_DELAY_SECONDS = 8.0
DEFAULT_MAX_RETRIES = 6
DEFAULT_INITIAL_BACKOFF = 2.0
MAX_BACKOFF_SECONDS = 90.0

DEFAULT_COMPONENT_COVERAGE = 0.75

RETRIEVAL_TOP_K = 5
RETRIEVAL_KS = (1, 3, 5)


class EvaluationCase(TypedDict):
    """Normalized evaluation case (see module docstring for file format)."""

    category: str
    query: str
    expected_found: bool
    expected_component: str
    expected_values: list[float | str]
    expected_unit: str
    expected_pages: list[int]


class DailyQuotaExhausted(RuntimeError):
    """Raised when retrying cannot help.

    Either the daily request quota is used up, or the model has no free
    quota at all (``limit: 0``).
    """


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers_equal(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _try_float(value: Any) -> float | None:
    if _is_number(value):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None

    return None


# ---------------------------------------------------------------------------
# Text / value normalization
# ---------------------------------------------------------------------------


def _norm_text(value: Any) -> str:
    """Normalize a value for comparison.

    Numbers become a canonical float string. Text is casefolded with all
    whitespace and degree signs removed and dash variants unified.
    """

    if _is_number(value):
        return f"{float(value):.12g}"

    text = str(value).casefold()
    text = text.replace("–", "-").replace("—", "-").replace("°", "")

    return re.sub(r"\s+", "", text)


def _string_parts(text: str) -> list[str]:
    """Split ';'-separated identifiers, dropping parenthetical notes."""

    if ";" not in text:
        return [_norm_text(text)]

    parts = []

    for part in text.split(";"):
        normalized = _norm_text(re.sub(r"\(.*?\)", "", part))

        if normalized:
            parts.append(normalized)

    return parts


_UNIT_ALIASES = {
    "degrees": "°",
    "degree": "°",
    "deg": "°",
    "liters": "l",
    "litres": "l",
    "liter": "l",
    "litre": "l",
}


def _canon_unit(unit: Any) -> str:
    """Canonical form so 'N·m' == 'Nm' and 'lb-ft' == 'lb ft'."""

    text = str(unit or "").casefold().strip()

    for junk in ("·", " ", "-", "."):
        text = text.replace(junk, "")

    return _UNIT_ALIASES.get(text, text)


_STOPWORDS = {"the", "of", "a", "an", "and", "for", "to", "in", "on"}


def _tokens(text: str) -> set[str]:
    """Word tokens with trivial plural folding (nuts -> nut)."""

    tokens: set[str] = set()

    for token in re.findall(r"[0-9a-z]+", text.casefold()):
        if token in _STOPWORDS:
            continue

        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]

        tokens.add(token)

    return tokens


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


def load_evaluation_cases(
    path: str | Path = DEFAULT_DATASET_PATH,
) -> list[EvaluationCase]:
    """Load, validate and normalize manually verified evaluation cases."""

    source = Path(path)

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"Unable to read evaluation dataset: {source}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid evaluation JSON: {source}"
        ) from exc

    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must be a JSON list")

    cases: list[EvaluationCase] = []

    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Evaluation case {index} must be a JSON object"
            )

        expected_found = item.get("expected_found", True)

        if not isinstance(expected_found, bool):
            raise ValueError(
                f"Evaluation case {index}: expected_found must be a boolean"
            )

        required = {"category", "query"}

        if expected_found:
            required |= {"expected_component", "expected_unit"}

        missing = required - item.keys()

        if missing:
            raise ValueError(
                f"Evaluation case {index} is missing fields: "
                f"{sorted(missing)}"
            )

        values: list[float | str] = []
        pages: list[int] = []
        component = str(item.get("expected_component", ""))
        unit = str(item.get("expected_unit", ""))

        if expected_found:
            if "expected_values" in item:
                raw_values = _as_list(item["expected_values"])
            elif "expected_value" in item:
                raw_values = _as_list(item["expected_value"])
            else:
                raise ValueError(
                    f"Evaluation case {index} needs expected_value "
                    "or expected_values"
                )

            for value in raw_values:
                if isinstance(value, bool) or not isinstance(
                    value, (int, float, str)
                ):
                    raise ValueError(
                        f"Evaluation case {index} has an invalid "
                        "expected value"
                    )

                values.append(
                    float(value) if _is_number(value) else str(value)
                )

            if not values:
                raise ValueError(
                    f"Evaluation case {index} has no expected values"
                )

            if "expected_pages" in item:
                raw_pages = _as_list(item["expected_pages"])
            elif "expected_page" in item:
                raw_pages = _as_list(item["expected_page"])
            else:
                raise ValueError(
                    f"Evaluation case {index} needs expected_page "
                    "or expected_pages"
                )

            for page in raw_pages:
                if isinstance(page, bool) or not isinstance(page, int):
                    raise ValueError(
                        f"Evaluation case {index} has an invalid page"
                    )

                pages.append(page)

            if not pages:
                raise ValueError(
                    f"Evaluation case {index} has no expected pages"
                )

            # "0.5°-3.0°" with an empty unit -> "0.5-3.0" with unit "°"
            if not unit and any(
                isinstance(v, str) and "°" in v for v in values
            ):
                values = [
                    v.replace("°", "") if isinstance(v, str) else v
                    for v in values
                ]
                unit = "°"

        cases.append(
            {
                "category": str(item["category"]),
                "query": str(item["query"]),
                "expected_found": expected_found,
                "expected_component": component,
                "expected_values": values,
                "expected_unit": unit,
                "expected_pages": pages,
            }
        )

    return cases


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _value_matches(predicted: Any, expected: float | str) -> bool:
    """Compare one predicted value with one expected value."""

    if _is_number(expected):
        number = _try_float(predicted)

        return number is not None and _numbers_equal(
            number, float(expected)
        )

    parts = _string_parts(str(expected))
    predicted_text = _norm_text(predicted)

    if ";" in str(expected):
        return any(part in predicted_text for part in parts)

    return predicted_text == parts[0]


def _value_matches_any(
    predicted: Any,
    expected_values: Sequence[float | str],
) -> bool:
    return any(_value_matches(predicted, exp) for exp in expected_values)


def _component_matches(
    specification: Any,
    expected_component: str,
    min_coverage: float,
) -> bool:
    """Soft component match on word coverage of component + spec_type."""

    expected = _tokens(expected_component)

    if not expected:
        return True

    got = _tokens(
        f"{specification.component} {specification.spec_type}"
    )

    return len(expected & got) / len(expected) >= min_coverage


_FLAG_NAMES = ("core", "full", "component", "value", "unit", "page")


def _score_result(
    result: ExtractionResult,
    case: EvaluationCase,
    min_coverage: float,
) -> dict[str, bool]:
    """Score an answerable case. Each flag is True if ANY returned
    specification satisfies it; core/full require the same specification
    to satisfy all of their parts."""

    flags = {name: False for name in _FLAG_NAMES}

    for spec in result.specifications:
        component = _component_matches(
            spec, case["expected_component"], min_coverage
        )
        value = _value_matches_any(spec.value, case["expected_values"])
        unit = _canon_unit(spec.unit) == _canon_unit(
            case["expected_unit"]
        )
        page = spec.page in case["expected_pages"]

        flags["component"] |= component
        flags["value"] |= value
        flags["unit"] |= unit
        flags["page"] |= page
        flags["core"] |= value and unit and page
        flags["full"] |= component and value and unit and page

    return flags


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------


_NUMBER_PATTERN = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?!\d)")


def _text_has_number(text: str, number: float) -> bool:
    cleaned = re.sub(r"(?<=\d),(?=\d{3})", "", text)

    return any(
        _numbers_equal(float(match.group(0)), number)
        for match in _NUMBER_PATTERN.finditer(cleaned)
    )


def _chunk_contains_answer(
    text: str,
    case: EvaluationCase,
) -> bool:
    normalized = _norm_text(text)
    squashed = _canon_unit(text)
    unit = _canon_unit(case["expected_unit"])

    # ---------------------------------------------------------
    # 1. Expected value/unit must exist
    # ---------------------------------------------------------

    value_found = False

    for expected in case["expected_values"]:

        if _is_number(expected):
            if not _text_has_number(
                text,
                float(expected),
            ):
                continue

            if (
                len(unit) >= 2
                and unit not in squashed
            ):
                continue

            value_found = True
            break

        parts = _string_parts(
            str(expected)
        )

        if any(
            part and part in normalized
            for part in parts
        ):
            value_found = True
            break

    if not value_found:
        return False

    # ---------------------------------------------------------
    # 2. Component terminology should also overlap
    # ---------------------------------------------------------

    expected_tokens = _tokens(
        case["expected_component"]
    )

    chunk_tokens = _tokens(text)

    if not expected_tokens:
        return True

    overlap = (
        len(expected_tokens & chunk_tokens)
        / len(expected_tokens)
    )

    return overlap >= 0.5

def evaluate_retrieval(
    cases: Sequence[EvaluationCase],
    retriever: Retriever,
) -> dict[str, Any]:
    """Evaluate retrieval and print Top-K diagnostics."""

    page_hits = {
        k: 0 for k in RETRIEVAL_KS
    }

    answer_hits = {
        k: 0 for k in RETRIEVAL_KS
    }

    scored = 0
    misses: list[dict[str, Any]] = []

    print("\n=== Retrieval Evaluation ===\n")

    for index, case in enumerate(cases, start=1):

        # Unanswerable cases do not have expected pages/values.
        if not case["expected_found"]:
            print(
                f"[{index}/{len(cases)}] "
                f"{case['query']} -- skipped "
                "(unanswerable)"
            )
            continue

        scored += 1

        query = case["query"]
        expected_pages = case["expected_pages"]

        hits = retriever.retrieve(
            query,
            top_k=RETRIEVAL_TOP_K,
        )

        # ---------------------------------------------------------
        # Find expected-page rank
        # ---------------------------------------------------------

        page_rank = None

        for rank, hit in enumerate(
            hits,
            start=1,
        ):
            page = hit.get("page")

            if page in expected_pages:
                page_rank = rank
                break

        if page_rank is not None:
            status = (
                f"PAGE FOUND @ rank {page_rank}"
            )
        else:
            status = "EXPECTED PAGE NOT IN TOP-5"

        print(
            f"\n[{index}/{len(cases)}] {query}"
        )

        print(
            f"Expected page(s): "
            f"{expected_pages} | {status}"
        )

        # ---------------------------------------------------------
        # Print retrieved chunks
        # ---------------------------------------------------------

        for rank, hit in enumerate(
            hits,
            start=1,
        ):
            page = hit.get("page")
            score = hit.get("score", 0.0)
            section = hit.get("section", "")
            text = str(
                hit.get("text", "")
            )

            snippet = (
                text
                .replace("\n", " ")
                .strip()[:250]
            )

            page_marker = (
                " <-- EXPECTED PAGE"
                if page in expected_pages
                else ""
            )

            answer_marker = (
                " <-- CONTAINS ANSWER"
                if _chunk_contains_answer(
                    text,
                    case,
                )
                else ""
            )

            print(
                f"  Rank {rank}: "
                f"page={page} | "
                f"score={score:.4f} | "
                f"section={section}"
                f"{page_marker}"
                f"{answer_marker}"
            )

            print(
                f"           {snippet}"
            )

        # ---------------------------------------------------------
        # Calculate Top-K metrics
        # ---------------------------------------------------------

        for k in RETRIEVAL_KS:

            top_k_hits = hits[:k]

            # Page hit:
            # Any expected page appears in Top-K.
            page_correct = any(
                hit.get("page")
                in expected_pages
                for hit in top_k_hits
            )

            if page_correct:
                page_hits[k] += 1

            # Answer-in-context:
            # Any Top-K chunk actually contains
            # the expected value/unit.
            answer_correct = any(
                _chunk_contains_answer(
                    str(
                        hit.get(
                            "text",
                            "",
                        )
                    ),
                    case,
                )
                for hit in top_k_hits
            )

            if answer_correct:
                answer_hits[k] += 1

        # ---------------------------------------------------------
        # Store complete Top-5 misses
        # ---------------------------------------------------------

        page_top5 = any(
            hit.get("page")
            in expected_pages
            for hit in hits[:5]
        )

        answer_top5 = any(
            _chunk_contains_answer(
                str(
                    hit.get(
                        "text",
                        "",
                    )
                ),
                case,
            )
            for hit in hits[:5]
        )

        if not page_top5:
            misses.append(
                {
                    "query": query,
                    "expected_pages": (
                        expected_pages
                    ),
                    "retrieved_pages": [
                        hit.get("page")
                        for hit in hits[:5]
                    ],
                    "answer_in_context": (
                        answer_top5
                    ),
                }
            )

    # -------------------------------------------------------------
    # Summary helper
    # -------------------------------------------------------------

    def summarize(
        counts: dict[int, int],
    ) -> dict[str, Any]:

        return {
            f"top_{k}": {
                "correct": counts[k],
                "accuracy": _ratio(
                    counts[k],
                    scored,
                ),
            }
            for k in RETRIEVAL_KS
        }

    # -------------------------------------------------------------
    # Return structure expected by _print_summary()
    # -------------------------------------------------------------

    return {
        "cases": len(cases),
        "answerable_cases": scored,

        "page_hit": summarize(
            page_hits
        ),

        "answer_in_context": summarize(
            answer_hits
        ),

        "page_misses": misses,
    }
# ---------------------------------------------------------------------------
# Result cache (JSONL, one line per successful extraction)
# ---------------------------------------------------------------------------


def _cache_key(query: str, model: str) -> str:
    """Cache per model, so switching GEMINI_MODEL never reuses old results.

    Prompt or schema changes are NOT detected; use --fresh after editing them.
    """

    return f"{model}::{query}"


def _load_cache(path: Path) -> dict[str, ExtractionResult]:
    """Load cached extraction results, skipping corrupt lines."""

    cache: dict[str, ExtractionResult] = {}

    if not path.exists():
        return cache

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            row = json.loads(line)
            cache[row["key"]] = ExtractionResult.model_validate(
                row["result"]
            )
        except Exception as exc:  # noqa: BLE001 - tolerate bad lines
            print(
                f"  Ignoring unreadable cache line {line_number}: {exc}"
            )

    return cache


def _append_cache(
    path: Path,
    key: str,
    query: str,
    result: ExtractionResult,
) -> None:
    """Append one result immediately so progress survives interruptions."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "key": key,
                    "query": query,
                    "result": result.to_dict(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


# ---------------------------------------------------------------------------
# Gemini error classification and retry
# ---------------------------------------------------------------------------


def _squash(text: str) -> str:
    """Lowercase and strip separators for loose substring matching."""

    return re.sub(r"[\s_\-]+", "", text.casefold())


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "quota" in message
    )


def _is_hard_quota_error(exc: Exception) -> bool:
    """True when waiting a minute will not help.

    Gemini 429 bodies name the quota that was hit, e.g.
    ``GenerateRequestsPerDayPerProjectPerModel-FreeTier``. A model with no
    free quota reports ``limit: 0``.
    """

    if not _is_quota_error(exc):
        return False

    message = str(exc).casefold()

    return "perday" in _squash(message) or "limit: 0" in message


def _is_retryable_error(exc: Exception) -> bool:
    """Return True for temporary Gemini/API failures."""

    message = str(exc).casefold()

    # Bad model output is not a transient API failure.
    if "failed validation" in message or "invalid json" in message:
        return False

    retryable_markers = (
        "503",
        "unavailable",
        "overloaded",
        "high demand",
        "429",
        "resource_exhausted",
        "rate limit",
        "temporarily unavailable",
        "deadline exceeded",
        "timeout",
        "timed out",
        "connection",
    )

    return any(marker in message for marker in retryable_markers)


def _server_retry_delay(exc: Exception) -> float | None:
    """Extract the wait time Gemini suggests, if the error contains one.

    Matches text such as ``Please retry in 34.5s`` and ``'retryDelay': '34s'``.
    """

    match = re.search(
        r"retry(?:\s*_?delay)?\D{0,12}?(\d+(?:\.\d+)?)\s*s\b",
        str(exc),
        re.IGNORECASE,
    )

    return float(match.group(1)) if match else None


def _predict_with_retry(
    query: str,
    predict: Callable[[str], ExtractionResult],
    *,
    max_retries: int,
    initial_backoff: float,
) -> ExtractionResult:
    """Call Gemini, retrying transient failures.

    Raises:
        DailyQuotaExhausted: retrying cannot help (daily quota / no quota).
        Exception: the last error once retries are exhausted, or any
            non-retryable error.
    """

    for attempt in range(1, max_retries + 1):
        try:
            return predict(query)

        except Exception as exc:  # noqa: BLE001
            if _is_hard_quota_error(exc):
                raise DailyQuotaExhausted(str(exc)) from exc

            if not _is_retryable_error(exc) or attempt >= max_retries:
                raise

            server_delay = _server_retry_delay(exc)

            if server_delay is not None:
                delay = server_delay + 1.0 + random.uniform(0, 1)
            else:
                delay = (
                    initial_backoff * (2 ** (attempt - 1))
                    + random.uniform(0, 1)
                )

            delay = min(delay, MAX_BACKOFF_SECONDS)

            print(
                f"    Gemini temporarily unavailable "
                f"(attempt {attempt}/{max_retries})."
            )
            print(f"    Retrying in {delay:.1f}s...")

            time.sleep(delay)

    # Defensive fallback; loop should always return or raise.
    raise RuntimeError("Gemini retry loop ended unexpectedly")


# ---------------------------------------------------------------------------
# Extraction evaluation
# ---------------------------------------------------------------------------


def _describe_expected(case: EvaluationCase) -> str:
    if not case["expected_found"]:
        return "NOT FOUND (query is unanswerable from the manual)"

    values = " | ".join(str(v) for v in case["expected_values"])

    return (
        f"{case['expected_component']} = {values} "
        f"{case['expected_unit']} (page {case['expected_pages']})"
    ).replace("  ", " ")


def _describe_result(result: ExtractionResult) -> list[str]:
    if not result.found:
        return [f"NOT FOUND: {result.reason[:200]}"]

    return [
        f"{spec.component} / {spec.spec_type} = {spec.value} "
        f"{spec.unit} (page {spec.page})"
        for spec in result.specifications
    ]


def _mark(flag: bool) -> str:
    return "✓" if flag else "✗"


def evaluate_extraction(
    cases: Sequence[EvaluationCase],
    predict: Callable[[str], ExtractionResult] | None,
    *,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    cache_path: Path = DEFAULT_CACHE_PATH,
    fresh: bool = False,
    cache_only: bool = False,
    model_name: str = GEMINI_MODEL_NAME,
    component_min_coverage: float = DEFAULT_COMPONENT_COVERAGE,
) -> dict[str, Any]:
    """Evaluate structured Gemini extraction.

    * Cached results are reused and cost no API quota.
    * With ``cache_only`` nothing is sent to Gemini; uncached cases are
      skipped, so cached results can be re-scored for free.
    * Real API calls are spaced at least ``delay_seconds`` apart.
    * Temporary failures are retried; a case that still fails is recorded and
      the run continues.
    * A daily-quota error stops the run cleanly; rerun later to continue.
    """

    if predict is None and not cache_only:
        raise ValueError("predict is required unless cache_only=True")

    if fresh and cache_path.exists():
        cache_path.unlink()

    cache = {} if fresh else _load_cache(cache_path)

    totals = {name: 0 for name in _FLAG_NAMES}

    evaluated = 0
    answerable_evaluated = 0
    unanswerable_evaluated = 0
    unanswerable_correct = 0
    from_cache = 0
    not_cached = 0
    failed_queries: list[str] = []
    misses: list[dict[str, Any]] = []
    not_run = 0
    stopped_on_quota = False

    last_call_time: float | None = None

    print("\n=== Gemini Extraction Evaluation ===\n")
    print(f"Model: {model_name}")
    print(
        f"Cache: {cache_path} ({len(cache)} stored result(s))"
        f"{'  [cache-only: no API calls]' if cache_only else ''}\n"
    )

    for index, case in enumerate(cases, start=1):
        query = case["query"]
        key = _cache_key(query, model_name)

        result = cache.get(key)
        used_cache = result is not None

        print(
            f"[{index}/{len(cases)}] {query}"
            f"{'  (cached)' if used_cache else ''}"
        )

        if result is None:
            if cache_only:
                not_cached += 1
                print("    (not cached: skipped)")
                continue

            assert predict is not None

            # Pace real API calls only.
            if last_call_time is not None and delay_seconds > 0:
                wait = delay_seconds - (time.monotonic() - last_call_time)

                if wait > 0:
                    time.sleep(wait)

            try:
                result = _predict_with_retry(
                    query,
                    predict,
                    max_retries=max_retries,
                    initial_backoff=initial_backoff,
                )

            except DailyQuotaExhausted as exc:
                stopped_on_quota = True
                not_run = len(cases) - index + 1

                print(f"\n    QUOTA EXHAUSTED: {exc}")
                print(
                    "    Stopping. Progress is saved; rerun the same "
                    "command after the quota resets (midnight Pacific "
                    "time), or switch model / enable billing."
                )
                break

            except Exception as exc:  # noqa: BLE001
                failed_queries.append(query)
                last_call_time = time.monotonic()

                print(f"    FAILED after retries: {exc}")
                continue

            last_call_time = time.monotonic()

            cache[key] = result
            _append_cache(cache_path, key, query, result)

        else:
            from_cache += 1

        evaluated += 1

        # -----------------------------------------------------------------
        # Unanswerable case: the correct behaviour is found=false.
        # -----------------------------------------------------------------

        if not case["expected_found"]:
            unanswerable_evaluated += 1
            correct = not result.found
            unanswerable_correct += int(correct)

            if correct:
                print("    correctly returned not-found ✓")
            else:
                print("    ✗ answered an unanswerable query")
                print(f"      expected: {_describe_expected(case)}")

                for line in _describe_result(result):
                    print(f"      got:      {line}")

                misses.append(
                    {
                        "query": query,
                        "expected": _describe_expected(case),
                        "got": _describe_result(result),
                    }
                )

            continue

        # -----------------------------------------------------------------
        # Answerable case
        # -----------------------------------------------------------------

        flags = _score_result(result, case, component_min_coverage)

        answerable_evaluated += 1

        for name in _FLAG_NAMES:
            totals[name] += int(flags[name])

        print(
            "    "
            f"core={_mark(flags['core'])} | "
            f"value={_mark(flags['value'])} | "
            f"unit={_mark(flags['unit'])} | "
            f"page={_mark(flags['page'])} | "
            f"component={_mark(flags['component'])}"
        )

        if not flags["full"]:
            print(f"      expected: {_describe_expected(case)}")

            for line in _describe_result(result):
                print(f"      got:      {line}")

        if not flags["core"]:
            misses.append(
                {
                    "query": query,
                    "expected": _describe_expected(case),
                    "got": _describe_result(result),
                }
            )

    total = len(cases)
    denominator = answerable_evaluated

    return {
        "model": model_name,
        "cases": total,
        "evaluated": evaluated,
        "answerable_evaluated": answerable_evaluated,
        "unanswerable_evaluated": unanswerable_evaluated,
        "from_cache": from_cache,
        "not_cached": not_cached,
        "failed_api_calls": len(failed_queries),
        "not_run": not_run,
        "stopped_on_quota": stopped_on_quota,
        # Accuracies use evaluated answerable cases as the denominator, so
        # API failures are not counted as extraction mistakes.
        "core_correct": totals["core"],
        "core_accuracy": _ratio(totals["core"], denominator),
        "full_correct": totals["full"],
        "full_accuracy": _ratio(totals["full"], denominator),
        "value_accuracy": _ratio(totals["value"], denominator),
        "unit_accuracy": _ratio(totals["unit"], denominator),
        "page_accuracy": _ratio(totals["page"], denominator),
        "component_accuracy": _ratio(totals["component"], denominator),
        "unanswerable_correct": unanswerable_correct,
        "unanswerable_accuracy": _ratio(
            unanswerable_correct, unanswerable_evaluated
        ),
        "failed_queries": failed_queries,
        "misses": misses,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FAISS retrieval and optional "
            "Gemini structured extraction."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to evaluation JSON dataset.",
    )

    parser.add_argument(
        "--with-extraction",
        action="store_true",
        help=(
            "Evaluate Gemini extraction. "
            "This uses external API quota."
        ),
    )

    parser.add_argument(
        "--cache-only",
        action="store_true",
        help=(
            "Score extraction from the cache only. Makes NO Gemini "
            "calls; uncached cases are skipped."
        ),
    )

    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help=(
            "Skip the retrieval evaluation "
            "(useful when only rerunning extraction)."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=(
            "Minimum seconds between real Gemini requests "
            f"(default: {DEFAULT_DELAY_SECONDS:g}). Cached cases are "
            "not delayed."
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Maximum Gemini attempts per query "
            f"(default: {DEFAULT_MAX_RETRIES})."
        ),
    )

    parser.add_argument(
        "--initial-backoff",
        type=float,
        default=DEFAULT_INITIAL_BACKOFF,
        help=(
            "Initial retry delay in seconds when the server does not "
            f"suggest one (default: {DEFAULT_INITIAL_BACKOFF:g})."
        ),
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=(
            "JSONL file storing successful extractions "
            f"(default: {DEFAULT_CACHE_PATH})."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore and delete the extraction cache. Use after changing "
            "the prompt or schema."
        ),
    )

    parser.add_argument(
        "--component-coverage",
        type=float,
        default=DEFAULT_COMPONENT_COVERAGE,
        help=(
            "Fraction of the expected component label's words that must "
            "appear in the returned component/spec_type "
            f"(default: {DEFAULT_COMPONENT_COVERAGE})."
        ),
    )

    return parser


def _print_summary(report: dict[str, Any]) -> None:
    """Print a compact human-readable summary."""

    print("\n=== Summary ===\n")

    retrieval = report.get("retrieval")

    if retrieval:
        page = retrieval["page_hit"]
        answer = retrieval["answer_in_context"]

        print(
            f"Retrieval ({retrieval['answerable_cases']} answerable cases)"
        )
        print(
            "  page hit          "
            + "  ".join(
                f"@{k}={_pct(page[f'top_{k}']['accuracy'])}"
                for k in RETRIEVAL_KS
            )
        )
        print(
            "  answer in context "
            + "  ".join(
                f"@{k}={_pct(answer[f'top_{k}']['accuracy'])}"
                for k in RETRIEVAL_KS
            )
        )

    extraction = report.get("extraction")

    if extraction:
        print(
            f"Extraction ({extraction['answerable_evaluated']} answerable "
            f"cases evaluated, {extraction['from_cache']} from cache)"
        )
        print(
            f"  core (value+unit+page)  "
            f"{_pct(extraction['core_accuracy'])}"
        )
        print(
            f"  full (core+component)   "
            f"{_pct(extraction['full_accuracy'])}"
        )
        print(
            f"  value {_pct(extraction['value_accuracy'])} | "
            f"unit {_pct(extraction['unit_accuracy'])} | "
            f"page {_pct(extraction['page_accuracy'])} | "
            f"component {_pct(extraction['component_accuracy'])}"
        )

        if extraction["unanswerable_evaluated"]:
            print(
                f"  unanswerable correctly refused: "
                f"{extraction['unanswerable_correct']}/"
                f"{extraction['unanswerable_evaluated']}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run retrieval and optional extraction evaluation."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    run_extraction = args.with_extraction or args.cache_only

    if args.skip_retrieval and not run_extraction:
        parser.error(
            "--skip-retrieval requires --with-extraction or --cache-only "
            "(nothing would be evaluated)"
        )

    if args.cache_only and args.fresh:
        parser.error(
            "--cache-only and --fresh cannot be combined "
            "(--fresh deletes the cache)"
        )

    # Make the check/cross marks safe on consoles with legacy encodings.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    report: dict[str, Any] = {}
    incomplete = False

    try:
        cases = load_evaluation_cases(args.dataset)

        print(f"Loaded {len(cases)} evaluation cases.")

        # The retriever (embedding model + index) is only loaded if needed.
        needs_retriever = (not args.skip_retrieval) or (
            run_extraction and not args.cache_only
        )

        retriever = Retriever() if needs_retriever else None

        # ---------------------------------------------------------------
        # Retrieval evaluation (offline, free)
        # ---------------------------------------------------------------

        if not args.skip_retrieval:
            assert retriever is not None
            report["retrieval"] = evaluate_retrieval(cases, retriever)

        # ---------------------------------------------------------------
        # Optional Gemini extraction evaluation
        # ---------------------------------------------------------------

        if run_extraction:
            predict: Callable[[str], ExtractionResult] | None = None

            if not args.cache_only:
                assert retriever is not None
                active_retriever = retriever

                def predict(query: str) -> ExtractionResult:
                    chunks: Sequence[dict[str, Any]] = (
                        active_retriever.retrieve(
                            query,
                            top_k=RETRIEVAL_TOP_K,
                        )
                    )

                    return extract_specifications(query, chunks)

            extraction_report = evaluate_extraction(
                cases,
                predict,
                delay_seconds=args.delay,
                max_retries=args.max_retries,
                initial_backoff=args.initial_backoff,
                cache_path=args.cache,
                fresh=args.fresh,
                cache_only=args.cache_only,
                component_min_coverage=args.component_coverage,
            )

            report["extraction"] = extraction_report

            incomplete = (
                extraction_report["failed_api_calls"] > 0
                or extraction_report["not_run"] > 0
            )

    except KeyboardInterrupt:
        print(
            "\nEvaluation interrupted. Cached extraction results "
            "are kept; rerun to continue."
        )
        return 130

    except Exception as exc:  # noqa: BLE001
        print(f"\nEvaluation failed: {exc}")
        return 1

    _print_summary(report)

    print("\n=== Final Evaluation Report ===\n")

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    extraction_report = report.get("extraction")

    if extraction_report and extraction_report["not_cached"]:
        print(
            f"\n{extraction_report['not_cached']} case(s) have no cached "
            "result. Run with --with-extraction to fetch them."
        )

    if incomplete:
        print(
            "\nExtraction evaluation is incomplete "
            "(see failed_queries / not_run). Rerun the same command "
            "to evaluate the remaining cases."
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())