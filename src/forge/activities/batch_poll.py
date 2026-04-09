"""Batch poll activity for Forge.

Polls LLM providers for completed batch results and signals waiting
workflows via Temporal.

Design follows Function Core / Imperative Shell:
- Testable function: execute_poll_batch_results (takes all dependencies as args)
- Imperative shell: poll_batch_results (activity decorator, wires up real deps)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sax_llm.models import BatchPollStatus
from temporalio import activity

from forge.activities._heartbeat import heartbeat_during
from forge.models import BatchPollerInput, BatchPollerResult, BatchResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from sax_llm.models import ExtractedImage
    from temporalio.client import Client

logger = logging.getLogger(__name__)

_MISSING_THRESHOLD = timedelta(hours=24)

_TERMINAL_FAILURE_STATUSES = frozenset({
    BatchPollStatus.FAILED,
    BatchPollStatus.EXPIRED,
    BatchPollStatus.CANCELED,
})

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
    store_images_fn: Callable[[list[ExtractedImage]], dict[str, str]] | None = None,
) -> BatchPollerResult:
    """Poll LLM providers for batch results and signal waiting workflows.

    Args:
        pending_jobs: Rows from get_pending_batch_jobs() (dicts with batch job fields).
        temporal_client: Temporal client for sending signals to workflows.
        update_status_fn: Callable to update batch job status in the store.
        store_images_fn: Optional callable to store extracted OCR images.
            Accepts a list of ExtractedImage, returns {original_image_id: uuid} mapping.
    """
    from sax_llm import get_provider_by_name

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

        provider = get_provider_by_name(provider_name)

        try:
            poll_result = await provider.poll_batch(batch_id)
        except Exception:
            logger.warning("Failed to poll batch %s", batch_id, exc_info=True)
            errors_found += 1
            age = datetime.now(UTC) - _ensure_utc(created_at)
            if age > _MISSING_THRESHOLD:
                logger.warning("Batch %s is >24h old and unretrievable, marking MISSING", batch_id)
                _safe_update_status(
                    update_status_fn,
                    request_id=request_id,
                    status="missing",
                    error_message="Batch unretrievable after 24h",
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
            try:
                handle = temporal_client.get_workflow_handle(workflow_id)
                await handle.signal("batch_result_received", signal)
                signals_sent += 1
            except Exception:
                logger.warning(
                    "Failed to signal workflow %s for failed batch %s",
                    workflow_id,
                    batch_id,
                    exc_info=True,
                )
                errors_found += 1

            _safe_update_status(
                update_status_fn,
                request_id=request_id,
                status=poll_result.status.value,
                error_message=error_msg,
            )
            continue

        if poll_result.status != BatchPollStatus.ENDED:
            continue  # PENDING or IN_PROGRESS — check again next cycle

        job_signals = 0
        for entry in poll_result.entries:
            if entry.succeeded:
                raw_json = entry.raw_response_json
                # Store extracted images before signaling (Temporal payload limit)
                if entry.extracted_images and store_images_fn is not None:
                    try:
                        image_mapping = store_images_fn(entry.extracted_images)
                        # Embed mapping in raw_response_json for parse activity
                        if image_mapping and raw_json:
                            data = json.loads(raw_json)
                            data["_image_mapping"] = image_mapping
                            raw_json = json.dumps(data)
                    except Exception:
                        logger.warning(
                            "Failed to store images for batch %s entry %s",
                            batch_id,
                            entry.custom_id,
                            exc_info=True,
                        )

                signal = BatchResult(
                    request_id=entry.custom_id,
                    batch_id=batch_id,
                    raw_response_json=raw_json,
                    result_type="succeeded",
                )
            else:
                signal = BatchResult(
                    request_id=entry.custom_id,
                    batch_id=batch_id,
                    error=entry.error or "Unknown error",
                    result_type="errored",
                )

            try:
                handle = temporal_client.get_workflow_handle(workflow_id)
                await handle.signal("batch_result_received", signal)
                job_signals += 1
                signals_sent += 1
            except Exception:
                logger.warning(
                    "Failed to signal workflow %s for batch %s",
                    workflow_id,
                    batch_id,
                    exc_info=True,
                )
                errors_found += 1

        final_status = "succeeded" if job_signals > 0 else "errored"
        _safe_update_status(
            update_status_fn,
            request_id=request_id,
            status=final_status,
        )

    result = BatchPollerResult(
        batches_checked=batches_checked,
        signals_sent=signals_sent,
        errors_found=errors_found,
    )

    if errors_found > 0:
        msg = f"Batch poller completed with {errors_found} error(s) across {batches_checked} job(s)"
        raise RuntimeError(msg)

    return result


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (UTC). SQLite datetimes may be naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _safe_update_status(
    update_fn: Callable[..., None],
    *,
    request_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    """Best-effort status update. Never raises (D42)."""
    try:
        update_fn(request_id=request_id, status=status, error_message=error_message)
    except Exception:
        logger.warning("Failed to update batch job status for %s", request_id, exc_info=True)


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
        from forge.store import get_db_path, get_engine, get_pending_batch_jobs

        db_path = get_db_path()
        if db_path is None:
            span.set_attribute("forge.poll.skipped", True)
            return BatchPollerResult()

        engine = get_engine(db_path)
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
            status: str,
            error_message: str | None = None,
        ) -> None:
            update_batch_status(
                engine,
                request_id=request_id,
                status=status,
                error_message=error_message,
            )

        # Build store_images closure over the engine
        from forge.store import save_ocr_image

        def store_images_fn(images: list[ExtractedImage]) -> dict[str, str]:
            """Decode base64, store each image, return {original_id: uuid}."""
            import base64
            import uuid

            mapping: dict[str, str] = {}
            for img in images:
                image_id = str(uuid.uuid4())
                raw_b64 = img.image_base64
                mime_type = img.mime_type
                # Strip data URI prefix if present (e.g. "data:image/png;base64,")
                # and extract the real MIME type from the header
                if raw_b64.startswith("data:"):
                    header, raw_b64 = raw_b64.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                data = base64.b64decode(raw_b64)
                save_ocr_image(
                    engine,
                    image_id=image_id,
                    page_index=img.page_index,
                    original_image_id=img.original_image_id,
                    data=data,
                    mime_type=mime_type,
                    file_size_bytes=len(data),
                    top_left_x=img.top_left_x,
                    top_left_y=img.top_left_y,
                    bottom_right_x=img.bottom_right_x,
                    bottom_right_y=img.bottom_right_y,
                )
                mapping[img.original_image_id] = image_id
            return mapping

        temporal_client = get_temporal_client()

        async with heartbeat_during():
            result = await execute_poll_batch_results(
                pending_jobs=pending_jobs,
                temporal_client=temporal_client,
                update_status_fn=update_status_fn,
                store_images_fn=store_images_fn,
            )

        span.set_attributes(
            {
                "forge.poll.batches_checked": result.batches_checked,
                "forge.poll.signals_sent": result.signals_sent,
                "forge.poll.errors_found": result.errors_found,
            }
        )

        return result
