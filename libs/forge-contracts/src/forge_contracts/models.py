"""Wire models exchanged between the platform and consumer apps over Temporal.

These cross task-queue boundaries (e.g. the batch-result signal), so they must
be importable by both the platform and consumer apps and serialize identically
via ``pydantic_data_converter``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class BatchJobStatus(StrEnum):
    """Lifecycle state of a batch_jobs row.

    Happy path: SUBMITTED -> STORING -> SUCCEEDED.

    Failure paths:
    - FAILED: provider API refused the submission before returning a batch_id.
    - ERRORED: per-entry failure from the provider, or the parse/store step
      raised after signal delivery.
    - EXPIRED / CANCELED: terminal provider states propagated from
      BatchPollStatus.
    - MISSING: batch unretrievable after 24h; the poller gave up.
    """

    SUBMITTED = "submitted"
    """Row created; batch in flight at the provider, waiting for completion."""

    STORING = "storing"
    """Provider reported the batch complete and the poller delivered the
    result signal to the waiting workflow. Parse + write to ``ocr_results``
    (or the equivalent consumer for non-OCR batches) is in progress. A row
    stuck in this state past the store workflow's retry budget is a genuine
    failure."""

    SUCCEEDED = "succeeded"
    """End-to-end complete: the downstream consumer (e.g. OcrStoreWorkflow)
    has committed its output. Only the consumer writes this value, and only
    after its write succeeds."""

    ERRORED = "errored"
    """Per-entry failure from the provider, or the parse/store step raised
    after signal delivery."""

    FAILED = "failed"
    """Provider API refused the submission before returning a batch_id.
    Written by ``record_batch_failure`` when the submit activity raises."""

    EXPIRED = "expired"
    """Provider marked the batch as expired. Written from
    ``poll_result.status.value`` in the poller's terminal-failure branch."""

    CANCELED = "canceled"
    """Provider marked the batch as canceled. Written from
    ``poll_result.status.value`` in the poller's terminal-failure branch."""

    MISSING = "missing"
    """Batch unretrievable after 24h. The poller gave up and marked the row
    so it stops being re-polled."""


class BatchResult(BaseModel):
    """Signal payload delivering a batch result to a waiting workflow.

    The result travels either inline (``raw_response_json``) for small payloads
    or by reference (``s3_key``) for large ones that would exceed Temporal's
    signal payload limit; the platform poller chooses by size. Exactly one of
    the two is set on success; on failure both are ``None`` and ``error`` is set.
    """

    request_id: str
    batch_id: str
    raw_response_json: str | None = None
    s3_key: str | None = None
    error: str | None = None
    result_type: str = Field(description="Output type name for deserialization.")
