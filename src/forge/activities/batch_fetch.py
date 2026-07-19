"""Batch status/fetch activities for Forge's timer-loop transport (T4.1, D88).

The requester polls and fetches its own batch: a ``batch_status`` status poll
loop replaces the shared poller's signal delivery, and ``fetch_batch_result``
downloads this waiter's own result line once the batch has ended. Both are
provider-agnostic at the seam — anthropic and mistral answers are normalized to
the same value types (``BatchStatusResult`` / ``BatchFetchResult``).

Design follows Function Core / Imperative Shell:
- Testable functions: ``execute_batch_status`` / ``execute_fetch_batch_result``
  (take the AsyncAnthropic client / MistralOcr and the blob-put callable as
  arguments, so tests inject fakes).
- Imperative shell: the ``batch_status`` / ``fetch_batch_result`` bound methods
  on ``BatchActivities`` (forge.activities.roots), which pass the
  composition-root client, mistral client, and blob store.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from sax_platform.contracts.models import dump_batch_result_payload
from sax_platform.ocr import BatchPollStatus

from forge.models import BatchFetchResult, BatchStatusResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from anthropic import AsyncAnthropic
    from sax_platform.llm.batch import BatchRequestFailed
    from sax_platform.ocr import MistralOcr

    from forge.models import BatchStatusInput, FetchBatchResultInput

logger = logging.getLogger(__name__)

# Deliver the result inline when small; stash to S3 and return a pointer when the
# payload is large or carries images.
_INLINE_THRESHOLD_BYTES = 256 * 1024

type _BatchState = Literal["in_progress", "ended", "failed", "expired", "canceled"]

# Mistral has a native status-only call (`MistralOcr.get_batch_status`, one
# `jobs.get` with no download), so batch_status maps its normalized
# BatchPollStatus onto the provider-agnostic state directly. PENDING collapses to
# in_progress: the timer loop only distinguishes "keep waiting" from the terminal
# states. The result download happens later, once, in fetch_batch_result.
_MISTRAL_STATUS_TO_STATE: dict[BatchPollStatus, _BatchState] = {
    BatchPollStatus.PENDING: "in_progress",
    BatchPollStatus.IN_PROGRESS: "in_progress",
    BatchPollStatus.ENDED: "ended",
    BatchPollStatus.FAILED: "failed",
    BatchPollStatus.CANCELED: "canceled",
    BatchPollStatus.EXPIRED: "expired",
}


# ---------------------------------------------------------------------------
# batch_status core
# ---------------------------------------------------------------------------


async def execute_batch_status(
    input: BatchStatusInput,
    *,
    client: AsyncAnthropic | None,
    mistral_ocr: MistralOcr | None,
) -> BatchStatusResult:
    """Poll one batch's normalized lifecycle state through its provider.

    Provider dispatch: ``"mistral"`` routes through the injected ``MistralOcr``;
    every other provider is Anthropic's Message Batches API. The injected
    *client* / *mistral_ocr* come from the ``BatchActivities`` composition root;
    a missing one is a configuration error raised at point of use.
    """
    if input.provider == "mistral":
        if mistral_ocr is None:
            msg = (
                "mistral batch status requires MISTRAL_API_KEY to be set at worker "
                "startup (no MistralOcr was constructed)."
            )
            raise RuntimeError(msg)
        status = await mistral_ocr.get_batch_status(input.batch_id)
        return BatchStatusResult(
            batch_id=input.batch_id,
            state=_MISTRAL_STATUS_TO_STATE[status],
        )

    if client is None:
        msg = "anthropic batch status requires an AsyncAnthropic client (none was injected)."
        raise RuntimeError(msg)
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
    client: AsyncAnthropic | None,
    mistral_ocr: MistralOcr | None,
    put_result_blob: Callable[[str, bytes], str],
    inline_threshold: int = _INLINE_THRESHOLD_BYTES,
) -> BatchFetchResult:
    """Fetch this waiter's result line and claim-check it into a BatchFetchResult.

    Selects the line whose ``custom_id`` equals ``input.request_id`` from the
    finished batch. A failed line becomes ``error``; a missing custom_id becomes
    ``error``. A succeeded line is delivered inline when its body is small and
    image-free, else stashed to a blob (via *put_result_blob*) and returned as an
    ``s3_key`` pointer. Provider dispatch mirrors ``execute_batch_status``.
    """
    if input.provider == "mistral":
        if mistral_ocr is None:
            msg = (
                "mistral batch fetch requires MISTRAL_API_KEY to be set at worker "
                "startup (no MistralOcr was constructed)."
            )
            raise RuntimeError(msg)
        return await _fetch_mistral(input, mistral_ocr, put_result_blob, inline_threshold)

    if client is None:
        msg = "anthropic batch fetch requires an AsyncAnthropic client (none was injected)."
        raise RuntimeError(msg)
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


async def _fetch_mistral(
    input: FetchBatchResultInput,
    mistral_ocr: MistralOcr,
    put_result_blob: Callable[[str, bytes], str],
    inline_threshold: int,
) -> BatchFetchResult:
    """Fetch the mistral batch results, find this waiter's entry, and claim-check it."""
    entries = await mistral_ocr.fetch_batch_results(input.batch_id)
    for entry in entries:
        if entry.custom_id != input.request_id:
            continue
        if not entry.succeeded:
            return BatchFetchResult(error=entry.error or "Unknown error")
        images = [img.model_dump(mode="json") for img in entry.extracted_images]
        return _claim_check(
            input.request_id, entry.raw_response_json, images, put_result_blob, inline_threshold
        )
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
