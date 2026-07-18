"""Batch parse activity for Forge.

Classifies a stored batch result line (a serialized ``anthropic.types.Message``)
into a typed ``ParsedLLMResponse`` via the platform batch lane's
``classify_result_json``. Refusal, truncation, and schema-mismatch outcomes are
raised as non-retryable ``ApplicationError``s.

Design follows Function Core / Imperative Shell:
- Testable function: execute_parse_llm_response
- Imperative shell: parse_llm_response (activity with OTel tracing)
"""

from __future__ import annotations

import json
import logging

from sax_platform.llm.models import Completion, MismatchOutcome, RefusedOutcome, TruncatedOutcome
from temporalio import activity
from temporalio.exceptions import ApplicationError

from forge.message_log import write_message_log
from forge.models import ParsedLLMResponse, ParseResponseInput
from forge.output_types import resolve_output_type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


def execute_parse_llm_response(
    raw_json: str,
    output_type_name: str | None,
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
    output_type = resolve_output_type(output_type_name) if output_type_name else None
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


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


@activity.defn
async def parse_llm_response(input: ParseResponseInput) -> ParsedLLMResponse:
    """Activity wrapper with OTel tracing."""
    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.parse_llm_response") as span:
        logger.info(
            "Parse response: task_id=%s output_type=%s", input.task_id, input.output_type_name
        )
        # The body arrives inline or via an S3 pointer (a result envelope); fetch
        # and unwrap when only s3_key is set. The generic path ignores any images.
        if input.s3_key is not None:
            from sax_platform.contracts.models import BatchResult, resolve_batch_result

            raw_json, _images = resolve_batch_result(
                BatchResult(request_id="", batch_id="", s3_key=input.s3_key, result_type="")
            )
        else:
            raw_json = input.raw_response_json
        if raw_json is None:
            msg = "parse_llm_response: no body resolved (both raw_response_json and s3_key empty)"
            raise ValueError(msg)

        if input.log_messages and input.worktree_path:
            write_message_log(input.worktree_path, "response", raw_json)

        result = execute_parse_llm_response(raw_json, input.output_type_name)

        span.set_attributes(
            {
                "forge.batch.output_type": input.output_type_name or "",
                "forge.batch.task_id": input.task_id,
                "forge.batch.model_name": result.model_name,
            }
        )

        return result
