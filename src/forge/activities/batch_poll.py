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

from temporalio import activity

from forge.activities._heartbeat import heartbeat_during
from forge.models import BatchPollerInput, BatchPollerResult, BatchResult

if TYPE_CHECKING:
    from collections.abc import Callable

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
    temporal_client: Client,
    update_status_fn: Callable[..., None],
) -> BatchPollerResult:
    """Poll LLM providers for batch results and signal waiting workflows.

    Args:
        pending_jobs: Rows from get_pending_batch_jobs() (dicts with batch job fields).
        temporal_client: Temporal client for sending signals to workflows.
        update_status_fn: Callable to update batch job status in the store.
    """
    from forge.llm_providers import get_provider
    from forge.llm_providers.models import BatchPollStatus

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

        provider = get_provider(provider_name)

        try:
            poll_result = await provider.poll_batch(batch_id)
        except Exception:
            logger.warning("Failed to poll batch %s", batch_id, exc_info=True)
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

        if poll_result.status != BatchPollStatus.ENDED:
            continue

        job_signals = 0
        for entry in poll_result.entries:
            if entry.succeeded:
                signal = BatchResult(
                    request_id=entry.custom_id,
                    batch_id=batch_id,
                    raw_response_json=entry.raw_response_json,
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

    return BatchPollerResult(
        batches_checked=batches_checked,
        signals_sent=signals_sent,
        errors_found=errors_found,
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

        temporal_client = get_temporal_client()

        async with heartbeat_during():
            result = await execute_poll_batch_results(
                pending_jobs=pending_jobs,
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
