"""Batch parse activity for Forge.

Classifies a stored batch result line (a serialized ``anthropic.types.Message``)
into a typed ``ParsedLLMResponse`` via the platform batch lane's
``classify_result_json``. Refusal, truncation, and schema-mismatch outcomes are
raised as non-retryable ``ApplicationError``s.

Design follows Function Core / Imperative Shell:
- Testable function: execute_parse_llm_response
- Imperative shell: the ``parse_llm_response`` bound method on ``BatchActivities``
  (forge.activities.roots), which resolves the S3 pointer through the
  composition-root blob store and injects the output-type registry.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from sax_platform.llm.models import Completion, MismatchOutcome, RefusedOutcome, TruncatedOutcome
from temporalio.exceptions import ApplicationError

from forge.models import ParsedLLMResponse
from forge.output_types import OUTPUT_TYPES, resolve_output_type

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


def execute_parse_llm_response(
    raw_json: str,
    output_type_name: str | None,
    output_types: Mapping[str, type[BaseModel]] = OUTPUT_TYPES,
) -> ParsedLLMResponse:
    """Classify a stored batch result line into a ``ParsedLLMResponse``.

    Anthropic-only: ``raw_json`` is a serialized ``anthropic.types.Message`` (as
    stored by ``fetch_batch_result_lines``), classified by ``classify_result_json``
    into exactly one value outcome.

    On a ``Completion`` the parsed output is serialized into ``parsed_json``:
    ``completion.output.model_dump_json()`` for a typed result (``output`` is the
    validated Pydantic instance), or ``json.dumps(output)`` for the text lane
    (``output_type_name is None``; ``output`` is the response text). ``stop_reason``
    comes straight off the classified completion.

    A refusal, truncation, or schema mismatch is raised as a **non-retryable**
    ``ApplicationError``. These outcomes are a deterministic property of the stored
    bytes — re-running the parse on the same line can never change the outcome — so
    retrying would only burn attempts. ``LLM_RETRY`` already lists these types as
    non-retryable; ``non_retryable=True`` on the error itself is the guarantee.
    Separated from the imperative shell so tests can call directly.
    """
    output_type = resolve_output_type(output_type_name, output_types) if output_type_name else None
    # Imported here, not at module level: sax_platform.llm.batch loads the
    # anthropic SDK, and forge.activities is chain-imported inside the Temporal
    # workflow sandbox (via workflow-bearing modules importing activity fns).
    from sax_platform.llm.batch import classify_result_json

    outcome = classify_result_json(raw_json, output_type=output_type)

    if isinstance(outcome, Completion):
        parsed_json = (
            json.dumps(outcome.output) if output_type is None else outcome.output.model_dump_json()
        )
        return ParsedLLMResponse(
            parsed_json=parsed_json,
            model_name=outcome.model,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            cache_creation_input_tokens=outcome.cache_creation_input_tokens,
            cache_read_input_tokens=outcome.cache_read_input_tokens,
            stop_reason=outcome.stop_reason,
        )

    # Refusal / truncation / schema mismatch: deterministic, non-retryable.
    raise _outcome_error(outcome)


def _outcome_error(
    outcome: RefusedOutcome | TruncatedOutcome | MismatchOutcome,
) -> ApplicationError:
    """Build the non-retryable ApplicationError for a failed classification outcome.

    The message names the ``stop_reason`` plus a short outcome-specific detail
    (refusal category / truncation cap + partial length / validation error); the
    ``type`` is ``LLMRefused`` / ``LLMTruncated`` / ``LLMSchemaMismatch``.
    """
    stop_reason = outcome.telemetry.stop_reason
    if isinstance(outcome, RefusedOutcome):
        return ApplicationError(
            f"LLM call refused (stop_reason={stop_reason!r}, category={outcome.category!r})",
            type="LLMRefused",
            non_retryable=True,
        )
    if isinstance(outcome, TruncatedOutcome):
        return ApplicationError(
            f"LLM output truncated (stop_reason={stop_reason!r}, "
            f"max_tokens={outcome.max_tokens}, {len(outcome.partial_text)} chars produced)",
            type="LLMTruncated",
            non_retryable=True,
        )
    return ApplicationError(
        f"LLM output did not match schema (stop_reason={stop_reason!r}): {outcome.error}",
        type="LLMSchemaMismatch",
        non_retryable=True,
    )
