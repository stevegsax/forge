"""Survivable store-write primitive shared by the platform and consumer apps.

``persist_block`` funnels a store write through the ``persist_to_store`` activity
(targeted by string name) with a generous-but-finite retry policy: a transient DB
outage retries only this cheap write — the expensive LLM/OCR/batch call already
returned to the workflow and is never re-run. A prolonged outage exhausts the
schedule-to-close cap and fails the workflow loudly.

Each app registers its OWN ``persist_to_store`` activity on its OWN task queue
writing its OWN store; ``persist_block`` dispatches to the impl on the caller's
queue by default, or to another queue via ``task_queue`` (e.g. a consumer recording
a ``batch_jobs`` row on the platform queue). This module imports only temporalio +
pydantic, so it is safe under ``workflow.unsafe.imports_passed_through()``.

The batch persist-request models live here (not in the platform) because a consumer
records its submission on the platform store cross-queue, so both apps must import
the request shape. They carry no domain fields — ``batch_jobs`` is generic.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel
from temporalio import workflow
from temporalio.common import RetryPolicy

# Survivable store writes: backoff 1,2,4,8,16,32,60,60… fits ~18-20 tries in the
# 20-minute schedule_to_close governor, after which the activity fails loudly.
# ValueError is validation (never succeeds on retry); idempotency-key collisions
# are absorbed by insert_or_ignore and never raise.
PERSIST_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
    maximum_attempts=20,
    non_retryable_error_types=["ValueError"],
)
PERSIST_START_TO_CLOSE = timedelta(seconds=30)
PERSIST_SCHEDULE_TO_CLOSE = timedelta(minutes=20)


class PersistResult(BaseModel):
    """Outcome of a persist: which kind ran and whether a new row was written."""

    kind: str
    applied: bool


class PersistBatchSubmission(BaseModel):
    """A submitted batch job (batch_jobs table), keyed by request_id.

    Generic — carries no domain fields (file_path/document_id live in the
    consumer's own status table, keyed by the same request_id).
    """

    kind: Literal["batch_submission"] = "batch_submission"
    request_id: str
    batch_id: str
    workflow_id: str
    provider: str = "anthropic"


class PersistBatchFailure(BaseModel):
    """A failed batch submission (batch_jobs table), keyed by request_id."""

    kind: Literal["batch_failure"] = "batch_failure"
    request_id: str
    workflow_id: str
    error_message: str
    provider: str = "anthropic"


async def persist_block(req: BaseModel, *, task_queue: str | None = None) -> PersistResult:
    """Survivable store write: invoke ``persist_to_store`` with the persist retry.

    Dispatches to the ``persist_to_store`` activity on the caller's own task queue,
    or on ``task_queue`` when set (cross-queue, e.g. recording ``batch_jobs`` on the
    platform queue). A transient DB outage retries only this cheap write; the
    expensive call that produced ``req`` already returned and is never re-run.
    """
    kwargs: dict[str, Any] = {
        "start_to_close_timeout": PERSIST_START_TO_CLOSE,
        "schedule_to_close_timeout": PERSIST_SCHEDULE_TO_CLOSE,
        "retry_policy": PERSIST_RETRY,
        "result_type": PersistResult,
    }
    if task_queue is not None:
        kwargs["task_queue"] = task_queue
    # The string-name overload of execute_activity returns Any; result_type above
    # only steers runtime deserialization, so narrow via an annotated local.
    result: PersistResult = await workflow.execute_activity("persist_to_store", req, **kwargs)
    return result
