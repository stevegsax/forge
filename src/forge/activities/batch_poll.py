"""Batch poll activity for Forge.

Polls LLM providers for completed batch results and signals waiting
workflows via Temporal.

Design follows Function Core / Imperative Shell:
- Testable function: execute_poll_batch_results (takes all dependencies as args)
- Imperative shell: poll_batch_results (activity decorator, wires up real deps)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from forge_contracts.constants import BATCH_RESULT_SIGNAL
from forge_contracts.models import dump_batch_result_payload
from sax_llm.models import BatchPollResult, BatchPollStatus, BatchResultEntry, ExtractedImage
from temporalio import activity

from forge.activities._heartbeat import heartbeat_during
from forge.models import (
    BatchJobStatus,
    BatchPollerInput,
    BatchPollerResult,
    BatchResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sax_platform.ocr import BatchPollResult as _SaxPlatformBatchPollResult
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
# Module-global Temporal client (set by worker.py before activities run)
# ---------------------------------------------------------------------------

_temporal_client: Client | None = None


def set_temporal_client(client: Client) -> None:
    """Called by worker startup to inject the Temporal client for signal delivery."""
    global _temporal_client
    _temporal_client = client


def get_temporal_client() -> Client:
    """Return the injected Temporal client. Raises if not set."""
    if _temporal_client is None:
        msg = "Temporal client not set. Call set_temporal_client() during worker startup."
        raise RuntimeError(msg)
    return _temporal_client


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
) -> BatchPollerResult:
    """Poll LLM providers for batch results and signal waiting workflows.

    Domain-agnostic: the poller forwards the verbatim provider result (the body
    plus any images sax-llm extracted) to the waiting consumer and never stores or
    decodes images itself. Small image-free results travel inline in the signal;
    large or image-bearing results are stashed to a blob and delivered by pointer.

    Args:
        pending_jobs: Rows from get_pending_batch_jobs() (dicts with batch job fields).
        temporal_client: Temporal client for sending signals to workflows.
        update_status_fn: Callable to update batch job status in the store.
        put_result_blob: Uploads ``(custom_id, bytes) -> s3_key`` for pointer delivery.
        inline_threshold: Max inline payload size (bytes) before switching to a pointer.
    """
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
            poll_result = await _poll_batch_for(provider_name, batch_id)
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


async def _poll_batch_for(provider_name: str, batch_id: str) -> BatchPollResult:
    """Poll ``batch_id`` through the provider for ``provider_name``.

    Mistral routes through the shared lazily-cached ``MistralOcr`` resolver
    (``forge.activities._mistral``) instead of sax_llm's registry (T3.3:
    mistral's OCR capability moved to the platform library; sax_llm carries no
    provider entry for it anymore) — one client is built once per worker
    process and reused across poll cycles, not rebuilt every cycle.
    MistralOcr's poll-result types are field-for-field identical to
    sax_llm.models' (see the sax_platform.ocr module docstring), so
    constructing sax_llm's BatchPollResult/entries directly from the
    sax_platform objects' attributes is lossless and keeps every branch in
    execute_poll_batch_results provider-agnostic — it only ever sees
    sax_llm.models.BatchPollResult, regardless of which provider answered.
    """
    if provider_name == "mistral":
        from forge.activities._mistral import get_mistral_ocr

        raw_result = await get_mistral_ocr().poll_batch(batch_id)
        return _sax_platform_poll_result_to_sax_llm(raw_result)

    from sax_llm import get_provider_by_name

    provider = get_provider_by_name(provider_name)
    return await provider.poll_batch(batch_id)


def _sax_platform_poll_result_to_sax_llm(
    raw_result: _SaxPlatformBatchPollResult,
) -> BatchPollResult:
    """Build sax_llm's ``BatchPollResult`` directly from a sax_platform.ocr one.

    Replaces a ``model_dump(mode="json")``/``model_validate`` round-trip, which
    serialized and reparsed every base64 image payload twice per poll cycle
    (2026-07 Phase 3 code review, item 5b). The two schemas are field-for-field
    identical (see the sax_platform.ocr module docstring and ``_poll_batch_for``'s
    own docstring), so this constructs sax_llm's models directly from the
    sax_platform objects' already-validated attributes — behavior identical,
    one copy of each payload instead of two.
    """
    return BatchPollResult(
        status=BatchPollStatus(raw_result.status.value),
        entries=[
            BatchResultEntry(
                custom_id=entry.custom_id,
                succeeded=entry.succeeded,
                raw_response_json=entry.raw_response_json,
                error=entry.error,
                extracted_images=[
                    ExtractedImage(
                        original_image_id=img.original_image_id,
                        page_index=img.page_index,
                        image_base64=img.image_base64,
                        mime_type=img.mime_type,
                        top_left_x=img.top_left_x,
                        top_left_y=img.top_left_y,
                        bottom_right_x=img.bottom_right_x,
                        bottom_right_y=img.bottom_right_y,
                    )
                    for img in entry.extracted_images
                ],
            )
            for entry in raw_result.entries
        ],
    )


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
    sax-llm-extracted images are wrapped in the result envelope and stashed; the
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


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


@activity.defn
async def poll_batch_results(_input: BatchPollerInput) -> BatchPollerResult:
    """Activity wrapper — wires up real dependencies and delegates."""
    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.poll_batch_results") as span:
        # Get pending jobs from store — let DB errors propagate so Temporal
        # retries on transient failures and surfaces persistent ones.
        from forge.store import get_pending_batch_jobs, get_store_engine

        engine = get_store_engine()
        pending_jobs = get_pending_batch_jobs(engine)

        if not pending_jobs:
            span.set_attribute("forge.poll.pending_count", 0)
            return BatchPollerResult()

        span.set_attribute("forge.poll.pending_count", len(pending_jobs))

        # Build update_status closure over the engine
        from forge.store import update_batch_status

        def update_status_fn(
            *,
            request_id: str,
            status: BatchJobStatus | str,
            error_message: str | None = None,
        ) -> None:
            update_batch_status(
                engine,
                request_id=request_id,
                status=status,
                error_message=error_message,
            )

        # Stash large/image-bearing results to S3 for pointer delivery. The key
        # lands in a reapable namespace (bucket TTL GC); the platform never opens
        # the blob — the consumer fetches and parses it.
        from forge_contracts import s3_blobs

        def put_result_blob(custom_id: str, data: bytes) -> str:
            key: str = s3_blobs.build_key(f"batch-result-{custom_id}")
            s3_blobs.put(key, data, "application/json")
            return key

        temporal_client = get_temporal_client()

        async with heartbeat_during():
            result = await execute_poll_batch_results(
                pending_jobs=pending_jobs,
                temporal_client=temporal_client,
                update_status_fn=update_status_fn,
                put_result_blob=put_result_blob,
            )

        span.set_attributes(
            {
                "forge.poll.batches_checked": result.batches_checked,
                "forge.poll.signals_sent": result.signals_sent,
                "forge.poll.errors_found": result.errors_found,
            }
        )

        return result
