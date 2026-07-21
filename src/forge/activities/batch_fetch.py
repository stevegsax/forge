"""Batch status/fetch activities for Forge's timer-loop transport (T4.1, D88).

The requester polls and fetches its own batch: a ``batch_status`` status poll
loop replaces the shared poller's signal delivery, and ``fetch_batch_result``
downloads this waiter's own result line once the batch has ended. Forge submits
anthropic only (T4.2 ST3) — a non-anthropic ``provider`` raises, since such a
batch is its owning app's concern — but the explicit provider threading stays
(honest transport).

Design follows Function Core / Imperative Shell:
- Testable functions: ``execute_batch_status`` / ``execute_fetch_batch_result``
  (take the AsyncAnthropic client and the blob-put callable as arguments, so
  tests inject fakes).
- Imperative shell: the ``batch_status`` / ``fetch_batch_result`` bound methods
  on ``BatchActivities`` (forge.activities.roots), which pass the
  composition-root client and blob store.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from sax_platform.contracts.models import dump_batch_result_payload

from forge.models import BatchFetchResult, BatchStatusResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from anthropic import AsyncAnthropic
    from sax_platform.llm.batch import BatchRequestFailed

    from forge.models import BatchStatusInput, FetchBatchResultInput

logger = logging.getLogger(__name__)

# Deliver the result inline when small; stash to S3 and return a pointer when the
# payload is large or carries images.
_INLINE_THRESHOLD_BYTES = 256 * 1024

type _BatchState = Literal["in_progress", "ended", "failed", "expired", "canceled"]


def _reject_non_anthropic(provider: str) -> NoReturn:
    """Raise for any non-anthropic provider: forge's transport is anthropic-only.

    Forge submits anthropic only (T4.2 ST3); a non-anthropic batch is polled and
    fetched by its owning app, never through forge's transport.
    """
    msg = (
        f"forge batch transport received provider {provider!r}: non-anthropic "
        "batches are their owning app's concern; forge submits anthropic only."
    )
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# batch_status core
# ---------------------------------------------------------------------------


async def execute_batch_status(
    input: BatchStatusInput,
    *,
    client: AsyncAnthropic,
) -> BatchStatusResult:
    """Poll one anthropic batch's normalized lifecycle state.

    Forge submits anthropic only (T4.2 ST3): a non-anthropic ``input.provider``
    raises before the client is touched. The injected *client* comes from the
    ``BatchActivities`` composition root.
    """
    if input.provider != "anthropic":
        _reject_non_anthropic(input.provider)
    return await _anthropic_status(client, input.batch_id)


async def _anthropic_status(client: AsyncAnthropic, batch_id: str) -> BatchStatusResult:
    """Normalize an Anthropic batch's processing_status to a BatchStatusResult.

    Anthropic's batch-level ``processing_status`` is only ever
    in_progress/canceling/ended — per-request errored/expired/canceled outcomes
    surface as failed *lines*, never a batch-level failure — so ``"ended"`` maps
    to ``ended`` and every other status maps to ``in_progress``; the
    failed/expired/canceled states are unreachable for anthropic.
    """
    # Imported here, not at module level: sax_platform.llm.batch loads the
    # anthropic SDK, and forge.activities is chain-imported inside the Temporal
    # workflow sandbox (via workflow-bearing modules importing activity fns).
    from sax_platform.llm.batch import get_batch_status

    status = await get_batch_status(client, batch_id)
    state: _BatchState = "ended" if status.processing_status == "ended" else "in_progress"
    return BatchStatusResult(batch_id=batch_id, state=state)


# ---------------------------------------------------------------------------
# fetch_batch_result core
# ---------------------------------------------------------------------------


async def execute_fetch_batch_result(
    input: FetchBatchResultInput,
    *,
    client: AsyncAnthropic,
    put_result_blob: Callable[[str, bytes], str],
    inline_threshold: int = _INLINE_THRESHOLD_BYTES,
) -> BatchFetchResult:
    """Fetch this waiter's anthropic result line and claim-check it.

    Selects the line whose ``custom_id`` equals ``input.request_id`` from the
    finished batch. A failed line becomes ``error``; a missing custom_id becomes
    ``error``. A succeeded line is delivered inline when its body is small, else
    stashed to a blob (via *put_result_blob*) and returned as an ``s3_key``
    pointer. Forge submits anthropic only (T4.2 ST3): a non-anthropic
    ``input.provider`` raises.
    """
    if input.provider != "anthropic":
        _reject_non_anthropic(input.provider)
    return await _fetch_anthropic(input, client, put_result_blob, inline_threshold)


async def _fetch_anthropic(
    input: FetchBatchResultInput,
    client: AsyncAnthropic,
    put_result_blob: Callable[[str, bytes], str],
    inline_threshold: int,
) -> BatchFetchResult:
    """Fetch one anthropic result line by custom_id and claim-check it."""
    # Local import for sandbox safety (see _anthropic_status).
    from sax_platform.llm.batch import BatchRequestFailed, fetch_batch_result_lines

    for custom_id, line in await fetch_batch_result_lines(client, input.batch_id):
        if custom_id != input.request_id:
            continue
        if isinstance(line, BatchRequestFailed):
            return BatchFetchResult(error=_format_request_failure(line))
        # A succeeded anthropic line is the verbatim serialized Message; no images.
        return _claim_check(input.request_id, line, [], put_result_blob, inline_threshold)
    return BatchFetchResult(error=_missing_custom_id(input))


def _claim_check(
    request_id: str,
    body: str | None,
    images: list[dict[str, Any]],
    put_result_blob: Callable[[str, bytes], str],
    inline_threshold: int,
) -> BatchFetchResult:
    """Choose inline vs pointer delivery for a succeeded result (pure).

    Images or a large body force pointer delivery — the verbatim body plus any
    provider-extracted images are wrapped in the result envelope and stashed;
    small image-free bodies travel inline.
    """
    too_big = body is not None and len(body.encode("utf-8")) > inline_threshold
    if images or too_big:
        envelope = dump_batch_result_payload(body, images)
        s3_key = put_result_blob(request_id, envelope.encode("utf-8"))
        return BatchFetchResult(s3_key=s3_key)
    return BatchFetchResult(raw_response_json=body)


def _missing_custom_id(input: FetchBatchResultInput) -> str:
    """Error string for a waiter whose custom_id was absent from the batch."""
    return (
        f"custom_id {input.request_id} not found in batch {input.batch_id} "
        f"(provider {input.provider})"
    )


def _format_request_failure(failure: BatchRequestFailed) -> str:
    """Human-readable error string for a failed anthropic batch line."""
    if failure.kind == "errored":
        return f"Batch error: {failure.detail}"
    if failure.kind == "expired":
        return "Batch request expired (24h limit)"
    return "Batch request was canceled"
