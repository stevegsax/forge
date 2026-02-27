"""Batch poll activity for Forge.

Polls the Anthropic API for completed batch results and signals waiting
workflows via Temporal.

Design follows Function Core / Imperative Shell:
- Testable function: execute_poll_batch_results (takes all dependencies as args)
- Imperative shell: poll_batch_results (activity decorator, wires up real deps)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from temporalio import activity

from forge.activities._heartbeat import heartbeat_during
from forge.models import BatchPollerInput, BatchPollerResult, BatchResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from anthropic import AsyncAnthropic
    from temporalio.client import Client

logger = logging.getLogger(__name__)

_MISSING_THRESHOLD = timedelta(hours=24)

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
    client: AsyncAnthropic,
    temporal_client: Client,
    update_status_fn: Callable[..., None],
) -> BatchPollerResult:
    """Poll Anthropic for batch results and signal waiting workflows.

    Args:
        pending_jobs: Rows from get_pending_batch_jobs() (dicts with batch job fields).
        client: Anthropic async client.
        temporal_client: Temporal client for sending signals to workflows.
        update_status_fn: Callable to update batch job status in the store.
    """
    batches_checked = 0
    signals_sent = 0
    errors_found = 0

    for job in pending_jobs:
        batch_id = job["batch_id"]
        request_id = job["id"]
        workflow_id = job["workflow_id"]
        created_at = job["created_at"]

        batches_checked += 1

        # Retrieve batch status from Anthropic
        try:
            batch = await client.messages.batches.retrieve(batch_id)
        except Exception:
            logger.warning("Failed to retrieve batch %s", batch_id, exc_info=True)
            # If the job is old, mark as MISSING
            age = datetime.now(UTC) - _ensure_utc(created_at)
            if age > _MISSING_THRESHOLD:
                logger.warning("Batch %s is >24h old and unretrievable, marking MISSING", batch_id)
                _safe_update_status(
                    update_status_fn,
                    request_id=request_id,
                    status="missing",
                    error_message="Batch unretrievable after 24h",
                )
                errors_found += 1
            continue

        # Skip batches that are still processing
        if batch.processing_status != "ended":
            continue

        # Iterate results for ended batches
        try:
            results_iter = await client.messages.batches.results(batch_id)
        except Exception:
            logger.warning("Failed to retrieve results for batch %s", batch_id, exc_info=True)
            errors_found += 1
            continue

        job_signals = 0
        async for entry in results_iter:
            signal = _build_signal_from_entry(entry, batch_id)

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

        # Update batch job status (best-effort, D42)
        final_status = "succeeded" if job_signals > 0 else "errored"
        _safe_update_status(
            update_status_fn,
            request_id=request_id,
            status=final_status,
        )

    return BatchPollerResult(
        batches_checked=batches_checked,
        signals_sent=signals_sent,
        errors_found=errors_found,
    )


def _build_signal_from_entry(entry: Any, batch_id: str) -> BatchResult:
    """Build a BatchResult signal from a batch results entry."""
    request_id = entry.custom_id
    result_type = entry.result.type

    if result_type == "succeeded":
        raw_json = entry.result.message.model_dump_json()
        return BatchResult(
            request_id=request_id,
            batch_id=batch_id,
            raw_response_json=raw_json,
            result_type="succeeded",
        )

    if result_type == "errored":
        error_msg = str(entry.result.error)
        return BatchResult(
            request_id=request_id,
            batch_id=batch_id,
            error=f"Batch error: {error_msg}",
            result_type="errored",
        )

    if result_type == "expired":
        return BatchResult(
            request_id=request_id,
            batch_id=batch_id,
            error="Batch request expired (24h limit)",
            result_type="expired",
        )

    if result_type == "canceled":
        return BatchResult(
            request_id=request_id,
            batch_id=batch_id,
            error="Batch request was canceled",
            result_type="canceled",
        )

    # Unknown result type — treat as error
    return BatchResult(
        request_id=request_id,
        batch_id=batch_id,
        error=f"Unknown result type: {result_type}",
        result_type=result_type,
    )


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
    from forge.llm_client import get_anthropic_client
    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.poll_batch_results") as span:
        # Get pending jobs from store
        try:
            from forge.store import get_db_path, get_engine, get_pending_batch_jobs

            db_path = get_db_path()
            if db_path is None:
                span.set_attribute("forge.poll.skipped", True)
                return BatchPollerResult()

            engine = get_engine(db_path)
            pending_jobs = get_pending_batch_jobs(engine)
        except Exception:
            logger.warning("Failed to query pending batch jobs", exc_info=True)
            return BatchPollerResult()

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

        anthropic_client = get_anthropic_client()
        temporal_client = get_temporal_client()

        async with heartbeat_during():
            result = await execute_poll_batch_results(
                pending_jobs=pending_jobs,
                client=anthropic_client,
                temporal_client=temporal_client,
                update_status_fn=update_status_fn,
            )

        span.set_attributes(
            {
                "forge.poll.batches_checked": result.batches_checked,
                "forge.poll.signals_sent": result.signals_sent,
                "forge.poll.errors_found": result.errors_found,
            }
        )

        return result
