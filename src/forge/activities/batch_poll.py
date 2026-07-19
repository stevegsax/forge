"""Batch poll activity for Forge.

Polls LLM providers for completed batch results and signals waiting
workflows via Temporal.

Design follows Function Core / Imperative Shell:
- Testable function: execute_poll_batch_results (takes all dependencies as args,
  including the per-provider ``poll_fn`` dispatch)
- Imperative shell: the ``poll_batch_results`` bound method on ``BatchActivities``
  (forge.activities.roots), which wires the composition-root store engine, blob
  store, Temporal client, and poll dispatch.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sax_platform.contracts.constants import BATCH_RESULT_SIGNAL
from sax_platform.contracts.models import dump_batch_result_payload
from sax_platform.ocr import BatchPollResult, BatchPollStatus, BatchResultEntry

from forge.models import (
    BatchJobStatus,
    BatchPollerResult,
    BatchResult,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from anthropic import AsyncAnthropic
    from sax_platform.llm.batch import BatchRequestFailed
    from sax_platform.ocr import MistralOcr
    from temporalio.client import Client

logger = logging.getLogger(__name__)

_MISSING_THRESHOLD = timedelta(hours=24)

# Deliver the result inline when small; stash to S3 and signal a pointer when the
# payload is large or carries images (well under Temporal's signal payload limit).
_INLINE_THRESHOLD_BYTES = 256 * 1024

# Provider terminal-failure statuses → the generic batch_jobs failure state. The
# coarse status needs no cancel/fail distinction, so CANCELED collapses to FAILED.
_POLL_TO_JOB_STATUS = {
    BatchPollStatus.FAILED: BatchJobStatus.FAILED,
    BatchPollStatus.EXPIRED: BatchJobStatus.EXPIRED,
    BatchPollStatus.CANCELED: BatchJobStatus.FAILED,
}
_TERMINAL_FAILURE_STATUSES = frozenset(_POLL_TO_JOB_STATUS)


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_poll_batch_results(
    pending_jobs: list[dict[str, Any]],
    temporal_client: Client,
    update_status_fn: Callable[..., None],
    put_result_blob: Callable[[str, bytes], str],
    *,
    inline_threshold: int = _INLINE_THRESHOLD_BYTES,
    poll_fn: Callable[[str, str], Awaitable[BatchPollResult]] | None = None,
) -> BatchPollerResult:
    """Poll LLM providers for batch results and signal waiting workflows.

    Domain-agnostic: the poller forwards the verbatim provider result (the body
    plus any images the provider extracted) to the waiting consumer and never
    stores or decodes images itself. Small image-free results travel inline in the
    signal; large or image-bearing results are stashed to a blob and delivered by
    pointer.

    Args:
        pending_jobs: Rows from get_pending_batch_jobs() (dicts with batch job fields).
        temporal_client: Temporal client for sending signals to workflows.
        update_status_fn: Callable to update batch job status in the store.
        put_result_blob: Uploads ``(custom_id, bytes) -> s3_key`` for pointer delivery.
        inline_threshold: Max inline payload size (bytes) before switching to a pointer.
        poll_fn: Per-provider poll dispatch ``(provider_name, batch_id) ->
            BatchPollResult``. The ``BatchActivities`` composition root passes a
            bound closure over its AsyncAnthropic client + optional MistralOcr;
            when ``None`` (the module-global ``_poll_batch_for``, which tests
            patch) is used.
    """
    poll = _poll_batch_for if poll_fn is None else poll_fn
    batches_checked = 0
    signals_sent = 0
    errors_found = 0

    for job in pending_jobs:
        batch_id = job["batch_id"]
        request_id = job["id"]
        workflow_id = job["workflow_id"]
        created_at = job["created_at"]
        provider_name = job.get("provider", "anthropic")

        batches_checked += 1

        try:
            poll_result = await poll(provider_name, batch_id)
        except Exception:
            logger.warning("Failed to poll batch %s", batch_id, exc_info=True)
            errors_found += 1
            age = datetime.now(UTC) - _ensure_utc(created_at)
            if age > _MISSING_THRESHOLD:
                error_msg = f"Batch {batch_id} unretrievable after 24h"
                logger.warning("Batch %s is >24h old and unretrievable, marking MISSING", batch_id)
                # Wake the waiter with an error so it fails fast; previously this
                # path sent no signal and the waiter burned the full 25h timeout.
                signal = BatchResult(
                    request_id=request_id,
                    batch_id=batch_id,
                    error=error_msg,
                    result_type="errored",
                )
                if await _deliver_signal(temporal_client, workflow_id, batch_id, signal):
                    signals_sent += 1
                update_status_fn(
                    request_id=request_id,
                    status=BatchJobStatus.MISSING,
                    error_message=error_msg,
                )
            continue

        # Handle terminal failure statuses — signal the waiting workflow
        # with an error and update the DB so the poller stops re-polling.
        if poll_result.status in _TERMINAL_FAILURE_STATUSES:
            error_msg = f"Batch {batch_id} terminated with status: {poll_result.status.value}"
            logger.warning(error_msg)
            signal = BatchResult(
                request_id=request_id,
                batch_id=batch_id,
                error=error_msg,
                result_type="errored",
            )
            if await _deliver_signal(temporal_client, workflow_id, batch_id, signal):
                signals_sent += 1
            else:
                errors_found += 1
            # The batch failed at the provider, so this state is terminal even if
            # the signal did not reach the waiter — update regardless.
            update_status_fn(
                request_id=request_id,
                status=_POLL_TO_JOB_STATUS[poll_result.status],
                error_message=error_msg,
            )
            continue

        if poll_result.status != BatchPollStatus.ENDED:
            continue  # PENDING or IN_PROGRESS — check again next cycle

        job_signals = 0
        delivery_failed = False
        for entry in poll_result.entries:
            if entry.succeeded:
                signal = _build_success_signal(entry, batch_id, put_result_blob, inline_threshold)
            else:
                signal = BatchResult(
                    request_id=entry.custom_id,
                    batch_id=batch_id,
                    error=entry.error or "Unknown error",
                    result_type="errored",
                )

            if await _deliver_signal(temporal_client, workflow_id, batch_id, signal):
                job_signals += 1
                signals_sent += 1
            else:
                errors_found += 1
                delivery_failed = True

        # A transient signal-delivery failure must not lose the paid result: leave
        # the row SUBMITTED so the next cycle re-polls and re-delivers. Duplicate
        # deliveries are no-ops under T1.2 (waiters key by request_id + setdefault).
        if delivery_failed:
            continue
        # Every entry delivered: the provider lifecycle is done from the platform's
        # view, so advance to PROCESSING (handed to the consumer) and stop
        # re-polling. The consumer tracks its own stored/failed lifecycle in its
        # own status table. No entry to deliver => terminal FAILED.
        final_status = BatchJobStatus.PROCESSING if job_signals > 0 else BatchJobStatus.FAILED
        update_status_fn(request_id=request_id, status=final_status)

    if errors_found > 0:
        # INTERIM: no longer raise on partial per-job errors (that wedged the
        # poller into an unbounded retry loop while overlap=SKIP starved every
        # later run). Report the count; the next scheduled run is the retry.
        logger.warning(
            "Batch poller completed with %d error(s) across %d job(s); next scheduled run retries",
            errors_found,
            batches_checked,
        )

    return BatchPollerResult(
        batches_checked=batches_checked,
        signals_sent=signals_sent,
        errors_found=errors_found,
    )


async def _poll_batch_for(
    provider_name: str,
    batch_id: str,
    *,
    client: AsyncAnthropic | None = None,
    mistral_ocr: MistralOcr | None = None,
) -> BatchPollResult:
    """Poll ``batch_id`` through the provider for ``provider_name``.

    The AsyncAnthropic *client* and optional *mistral_ocr* are supplied by the
    ``BatchActivities`` composition root (via the ``poll_fn`` closure passed to
    ``execute_poll_batch_results``), replacing the former module-global caches.
    Both are keyword-only with ``None`` defaults so this stays assignable to the
    ``(provider_name, batch_id)`` poll-dispatch shape; a missing one is a
    configuration error raised at point of use. Mistral routes through the
    injected ``MistralOcr``; every other provider is Anthropic's Message Batches
    API, adapted to the same ``sax_platform.ocr`` poll shape by
    ``_poll_anthropic_batch`` — so every branch in ``execute_poll_batch_results``
    stays provider-agnostic, seeing one poll-result type regardless of which
    provider answered.
    """
    if provider_name == "mistral":
        if mistral_ocr is None:
            msg = (
                "mistral batch polling requires MISTRAL_API_KEY to be set at worker "
                "startup (no MistralOcr was constructed)."
            )
            raise RuntimeError(msg)
        return await mistral_ocr.poll_batch(batch_id)

    if client is None:
        msg = "anthropic batch polling requires an AsyncAnthropic client (none was injected)."
        raise RuntimeError(msg)
    return await _poll_anthropic_batch(client, batch_id)


async def _poll_anthropic_batch(client: AsyncAnthropic, batch_id: str) -> BatchPollResult:
    """Poll an Anthropic Message Batch, normalized to the ``sax_platform.ocr`` shape.

    Faithful port of the retired Anthropic provider's ``poll_batch``: the batch is
    either still running (any ``processing_status`` other than ``"ended"`` →
    IN_PROGRESS) or finished (``"ended"`` → ENDED), in which case every result
    line is fetched as raw bytes and turned into a ``BatchResultEntry``.

    Anthropic's batch-level ``processing_status`` is only ever
    in_progress/canceling/ended — per-request errored/expired/canceled outcomes
    surface as failed *lines*, not a batch-level failure — so this adapter never
    emits FAILED/EXPIRED/CANCELED, exactly as the retired provider never did, and
    the terminal-failure branch downstream stays unreachable for anthropic.
    """
    # Imported here, not at module level: sax_platform.llm.batch loads the
    # anthropic SDK, and forge.activities is chain-imported inside the Temporal
    # workflow sandbox (via workflow-bearing modules importing activity fns).
    from sax_platform.llm.batch import (
        BatchRequestFailed,
        fetch_batch_result_lines,
        get_batch_status,
    )

    status = await get_batch_status(client, batch_id)
    if status.processing_status != "ended":
        return BatchPollResult(status=BatchPollStatus.IN_PROGRESS)

    entries: list[BatchResultEntry] = []
    for custom_id, line in await fetch_batch_result_lines(client, batch_id):
        if isinstance(line, BatchRequestFailed):
            entries.append(
                BatchResultEntry(
                    custom_id=custom_id,
                    succeeded=False,
                    error=_format_request_failure(line),
                )
            )
        else:
            entries.append(
                BatchResultEntry(
                    custom_id=custom_id,
                    succeeded=True,
                    raw_response_json=line,
                )
            )
    return BatchPollResult(status=BatchPollStatus.ENDED, entries=entries)


def _format_request_failure(failure: BatchRequestFailed) -> str:
    """Human-readable error string for a failed batch line.

    Mirrors the message strings from the retired Anthropic provider's ``_format_batch_error``
    so the error signals waiters receive are unchanged across the migration.
    """
    if failure.kind == "errored":
        return f"Batch error: {failure.detail}"
    if failure.kind == "expired":
        return "Batch request expired (24h limit)"
    return "Batch request was canceled"


async def _deliver_signal(
    temporal_client: Client,
    workflow_id: str,
    batch_id: str,
    signal: BatchResult,
) -> bool:
    """Deliver a batch-result signal to a waiting workflow.

    Returns True on delivery, False on failure (logged). Signal delivery is
    at-least-once; duplicate deliveries are no-ops under T1.2 (waiters key by
    request_id and setdefault), so a caller may safely retry on a later cycle.
    """
    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        await handle.signal(BATCH_RESULT_SIGNAL, signal)
    except Exception:
        logger.warning(
            "Failed to signal workflow %s for batch %s",
            workflow_id,
            batch_id,
            exc_info=True,
        )
        return False
    return True


def _build_success_signal(
    entry: Any,
    batch_id: str,
    put_result_blob: Callable[[str, bytes], str],
    inline_threshold: int,
) -> BatchResult:
    """Build a succeeded BatchResult, choosing inline vs pointer delivery.

    Images (or a large body) force pointer delivery: the verbatim body plus any
    provider-extracted images are wrapped in the result envelope and stashed; the
    platform never decodes or stores the images itself.
    """
    body = entry.raw_response_json
    images = [img.model_dump(mode="json") for img in entry.extracted_images]
    too_big = body is not None and len(body.encode("utf-8")) > inline_threshold
    if images or too_big:
        envelope = dump_batch_result_payload(body, images)
        s3_key = put_result_blob(entry.custom_id, envelope.encode("utf-8"))
        return BatchResult(
            request_id=entry.custom_id,
            batch_id=batch_id,
            s3_key=s3_key,
            result_type="succeeded",
        )
    return BatchResult(
        request_id=entry.custom_id,
        batch_id=batch_id,
        raw_response_json=body,
        result_type="succeeded",
    )


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (UTC). SQLite datetimes may be naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt
