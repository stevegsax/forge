"""The batch lane (D90): a pure request-body builder plus thin submit/status/fetch helpers.

Forge is batch-first, so this is the primary lane, not a variant of the sync
one. `build_batch_request` does the one piece of real logic — assembling a
Messages API request body, including cache placement and structured-output
config — and is pure: no I/O, no client. `submit_batch`, `get_batch_status`,
and `fetch_batch_results` are thin shells that call the SDK and map its
response objects onto this module's value types; they carry no logic of
their own beyond that mapping.

`fetch_batch_results` never raises for a per-item classification failure
(refusal, truncation, schema mismatch, or a non-succeeded batch envelope) —
those are reported as values in `BatchItemResult.outcome`, mirroring the
value-shaped `*Outcome` types in `sax_platform.llm.models`. A batch fetch
resolves N independent request outcomes at once; one bad item raising would
abort classification of the other N-1, which is worse than a mixed-outcome
list. This is the opposite tradeoff from the sync lane, where a single
in-flight call has nothing else to preserve and can afford to raise.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, final

import anthropic.types
from anthropic import AsyncAnthropic
from anthropic.types.messages import MessageBatchSucceededResult
from pydantic import BaseModel, ConfigDict, ValidationError

from sax_platform.llm.cache import CacheSpec, apply_cache_control
from sax_platform.llm.models import (
    Completion,
    LLMOutcomeError,
    MismatchOutcome,
    RefusedOutcome,
    TruncatedOutcome,
    classify_message,
    outcome_from_error,
)
from sax_platform.llm.schema import to_json_schema, to_output_format

__all__ = [
    "BatchHandle",
    "BatchItemResult",
    "BatchRequestFailed",
    "BatchStatus",
    "build_batch_request",
    "fetch_batch_results",
    "get_batch_status",
    "submit_batch",
]

type _ItemOutcome = (
    Completion[Any] | RefusedOutcome | TruncatedOutcome | MismatchOutcome | BatchRequestFailed
)


def build_batch_request(
    custom_id: str,
    *,
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
    output_type: type[BaseModel] | None = None,
    output_schema: dict[str, Any] | None = None,
    cache: CacheSpec | None = None,
) -> dict[str, Any]:
    """Build one Message Batches API request entry: `{"custom_id", "params"}`.

    Pure: no I/O, no client — a plain dict transformation over its
    arguments, reusing `apply_cache_control` and the `schema` module exactly
    as the sync lane does, so both lanes derive the same wire shapes from
    the same core.

    `system`, when given as a plain `str`, is normalized to a single text
    block (`[{"type": "text", "text": system}]`) before `apply_cache_control`
    runs — a cache breakpoint attaches to a content block, so a bare string
    can't carry one. This runs regardless of whether `cache` is set; when
    `cache` is `None`, `apply_cache_control` is still a pure pass-through
    that returns an equivalent block list without a breakpoint.

    `output_type` and `output_schema` are mutually exclusive — passing both
    raises `ValueError`. `output_type` runs through `to_json_schema` (which
    closes every object node with `additionalProperties: false`) before
    `to_output_format`; `output_schema` is a raw JSON schema the caller
    already produced and is passed to `to_output_format` untouched — no
    closing applied. Passing neither omits `output_config` entirely (the
    text lane). `max_tokens` has no default: batch requests are billed and
    replay-executed up to 24 hours later, so a caller must state its cap
    explicitly rather than inherit one.
    """
    if output_type is not None and output_schema is not None:
        raise ValueError("pass at most one of output_type or output_schema, not both")

    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    if system is not None:
        system_blocks = [{"type": "text", "text": system}] if isinstance(system, str) else system
        params["system"] = apply_cache_control(system_blocks, model=model, spec=cache)

    if output_type is not None:
        params["output_config"] = {"format": to_output_format(to_json_schema(output_type))}
    elif output_schema is not None:
        params["output_config"] = {"format": to_output_format(output_schema)}

    return {"custom_id": custom_id, "params": params}


@final
class BatchHandle(BaseModel):
    """What `submit_batch` hands back: enough to poll and later fetch results."""

    model_config = ConfigDict(frozen=True)

    batch_id: str
    processing_status: str


@final
class BatchStatus(BaseModel):
    """A point-in-time snapshot of a batch's request-outcome tally.

    Counts mirror the SDK's `request_counts` on the retrieved batch object;
    their sum always equals the total number of requests submitted.
    """

    model_config = ConfigDict(frozen=True)

    batch_id: str
    processing_status: str
    succeeded: int
    errored: int
    canceled: int
    expired: int
    processing: int


@final
class BatchRequestFailed(BaseModel):
    """A batch item whose request never produced a `Message` at all.

    Distinct from the sync lane's `LLMRefused`/`LLMTruncated`/schema-mismatch
    failures (which all presuppose a completed `Message`): these three kinds
    are properties of the *request*, not of a completion — the item errored
    before or during processing, or the batch ended before it was reached.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["errored", "canceled", "expired"]
    detail: str


@final
class BatchItemResult(BaseModel):
    """One line of `fetch_batch_results`, keyed by `custom_id`.

    `outcome` is a value, never an exception — see the module docstring for
    why the batch lane reports failures this way instead of raising.
    """

    model_config = ConfigDict(frozen=True)

    custom_id: str
    outcome: _ItemOutcome


def _classify_succeeded_message(
    message: anthropic.types.Message,
    *,
    custom_id: str,
    output_types: Mapping[str, type[BaseModel]] | None,
) -> Completion[Any] | RefusedOutcome | TruncatedOutcome | MismatchOutcome:
    """Classify one succeeded batch item's `Message`. Pure: no I/O.

    Shares `classify_message`/`outcome_from_error` with the sync lane's
    classification core. `max_tokens` is passed as
    `message.usage.output_tokens` — the batch result envelope does not carry
    back the caller's originally-requested cap, only what the model actually
    produced — so a `TruncatedOutcome.max_tokens` here reports the tokens
    produced, which by construction equals the cap that was hit.

    When `custom_id` has an entry in `output_types`, the classified text is
    validated against that model (`MismatchOutcome` on a `ValidationError`,
    covering both malformed JSON and well-formed-but-wrong-shape JSON,
    including prose where JSON was expected). Otherwise the text is returned
    as-is as a `Completion[Any]` with a plain `str` output — the text lane.
    """
    try:
        classified = classify_message(message, max_tokens=message.usage.output_tokens)
    except LLMOutcomeError as err:
        return outcome_from_error(err)

    telemetry = classified.telemetry
    expected_type = output_types.get(custom_id) if output_types is not None else None
    output: Any = classified.text
    if expected_type is not None:
        try:
            output = expected_type.model_validate_json(classified.text)
        except ValidationError as exc:
            return MismatchOutcome(raw_text=classified.text, error=str(exc), telemetry=telemetry)

    return Completion[Any](
        output=output,
        model=telemetry.model,
        stop_reason=telemetry.stop_reason,
        input_tokens=telemetry.input_tokens,
        output_tokens=telemetry.output_tokens,
        cache_creation_input_tokens=telemetry.cache_creation_input_tokens,
        cache_read_input_tokens=telemetry.cache_read_input_tokens,
        request_id=telemetry.request_id,
    )


async def submit_batch(client: AsyncAnthropic, requests: Sequence[dict[str, Any]]) -> BatchHandle:
    """Submit pre-built request entries (see `build_batch_request`) as one batch.

    Thin: no logic beyond the SDK call and mapping its response onto
    `BatchHandle`.
    """
    batch = await client.messages.batches.create(requests=requests)  # type: ignore[arg-type]
    return BatchHandle(batch_id=batch.id, processing_status=batch.processing_status)


async def get_batch_status(client: AsyncAnthropic, batch_id: str) -> BatchStatus:
    """Poll a batch's processing status and request-outcome tally.

    Thin: no logic beyond the SDK call and mapping its response onto
    `BatchStatus`.
    """
    batch = await client.messages.batches.retrieve(batch_id)
    counts = batch.request_counts
    return BatchStatus(
        batch_id=batch.id,
        processing_status=batch.processing_status,
        succeeded=counts.succeeded,
        errored=counts.errored,
        canceled=counts.canceled,
        expired=counts.expired,
        processing=counts.processing,
    )


async def fetch_batch_results(
    client: AsyncAnthropic,
    batch_id: str,
    *,
    output_types: Mapping[str, type[BaseModel]] | None = None,
) -> list[BatchItemResult]:
    """Fetch and classify every result line of a finished batch.

    Streams the batch's `.jsonl` results (the SDK reads `results_url` off
    the retrieved batch and decodes it line by line) and classifies each
    line independently via `_classify_succeeded_message`, or maps a
    non-succeeded envelope to `BatchRequestFailed`. Results are not
    guaranteed to arrive in request order — callers must key off
    `BatchItemResult.custom_id`, not list position, exactly as the SDK's own
    docs specify for this endpoint.

    One item's failure never aborts the others: see the module docstring.
    """
    item_results: list[BatchItemResult] = []
    decoder = await client.messages.batches.results(batch_id)
    async for item in decoder:
        result = item.result
        outcome: _ItemOutcome
        if isinstance(result, MessageBatchSucceededResult):
            outcome = _classify_succeeded_message(
                result.message, custom_id=item.custom_id, output_types=output_types
            )
        else:
            outcome = BatchRequestFailed(kind=result.type, detail=str(result))
        item_results.append(BatchItemResult(custom_id=item.custom_id, outcome=outcome))
    return item_results
