"""Wire models exchanged between the platform and consumer apps over Temporal.

These cross task-queue boundaries (e.g. the batch-result signal), so they must
be importable by both the platform and consumer apps and serialize identically
via ``pydantic_data_converter``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sax_platform.contracts.s3_blobs import S3Blobs


class BatchJobStatus(StrEnum):
    """Provider-batch lifecycle state of a ``batch_jobs`` row (platform single-writer).

    This tracks ONLY the provider batch lifecycle. The downstream consumer's
    processing/terminal lifecycle (stored/failed) lives in the consumer's own
    status table, joined to this on ``request_id``; the user-facing status is the
    join of the two.

    Happy path: SUBMITTED -> PROCESSING (handed to the consumer).
    Failure paths: FAILED / EXPIRED (terminal provider states), MISSING (gave up).
    """

    SUBMITTED = "submitted"
    """Row created; batch in flight at the provider, waiting for completion.
    Only rows in this state are polled."""

    PROCESSING = "processing"
    """Provider reported the batch complete and the poller delivered the result
    signal to the waiting consumer workflow. From the platform's view this is the
    terminal provider state — the poller stops re-polling it. The consumer tracks
    its own processing/terminal lifecycle in its own status table."""

    FAILED = "failed"
    """Terminal provider failure: the API refused the submission before returning
    a batch_id (``record_batch_failure``), reported the batch FAILED/CANCELED, or a
    per-entry error came back. CANCELED maps here (the coarse status needs no
    cancel/fail distinction)."""

    EXPIRED = "expired"
    """Provider marked the batch as expired (TIMEOUT_EXCEEDED). Terminal."""

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


class BatchSubmitResult(BaseModel):
    """Outcome of a batch submission (the opaque-blob submit SPI and the
    platform's own generic submit both return this shape).

    Crosses the queue boundary: a consumer app submits via the platform SPI
    cross-queue and receives this back, so it lives in the shared contract.
    """

    request_id: str = Field(description="Provider custom_id == batch_jobs PK, minted once.")
    batch_id: str = Field(description="Provider batch ID.")
    provider: str = Field(
        default="anthropic",
        description="Provider name, threaded back so the caller can record the submission.",
    )


class BatchSubmitSpiInput(BaseModel):
    """Input to the platform opaque-blob submit activity (Option 1).

    A consumer app builds its provider request body, writes it to S3 as an opaque
    blob, and hands the platform only this pointer. The platform fetches the blob
    and calls ``provider.submit_batch`` verbatim — it never parses the body, so it
    stays domain-agnostic. The submit activity writes nothing; a separate persist
    records ``batch_jobs`` (so a provider submit and a DB write never share a
    re-runnable activity — the double-submit-safety invariant).
    """

    s3_key: str = Field(description="S3 key of the pre-built request blob (JSON list of requests).")
    model: str = Field(description="Provider model id (no provider prefix).")
    endpoint: str = Field(default="", description="Provider endpoint, e.g. '/v1/ocr'. '' default.")
    provider: str = Field(default="anthropic", description="Provider name for get_provider_by_name")
    custom_id: str = Field(
        description="Correlation id minted once by the consumer: request_id == custom_id == PK."
    )


# ---------------------------------------------------------------------------
# Batch-result delivery payload envelope
# ---------------------------------------------------------------------------
#
# The platform poller delivers a verbatim provider result to the waiting consumer.
# It travels inline (``BatchResult.raw_response_json``) for small image-free
# payloads, or by reference (``BatchResult.s3_key``) for large ones. The blob at
# ``s3_key`` is an *envelope* so the consumer can recover both the provider body
# and any images sax-llm extracted from it — the platform forwards both opaquely
# (it never decodes/stores images itself).


def dump_batch_result_payload(
    raw_response_json: str | None,
    extracted_images: list[dict[str, Any]],
) -> str:
    """Serialize the result envelope stashed to S3 (pure)."""
    import json

    return json.dumps(
        {"raw_response_json": raw_response_json, "extracted_images": extracted_images}
    )


def parse_batch_result_payload(envelope_json: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Parse the S3 result envelope back into (body, images) (pure)."""
    import json

    data = json.loads(envelope_json)
    return data.get("raw_response_json"), data.get("extracted_images", [])


def resolve_batch_result(
    result: BatchResult, blobs: S3Blobs
) -> tuple[str | None, list[dict[str, Any]]]:
    """Return ``(raw_response_json, extracted_images)`` for a delivered result.

    Fetches the S3 envelope through the injected :class:`S3Blobs` when the result
    was delivered by reference, else uses the inline body (which never carries
    images). Performs S3 I/O — call only from an activity, never inside a
    workflow.
    """
    if result.s3_key:
        raw = blobs.get(result.s3_key)
        return parse_batch_result_payload(raw.decode("utf-8"))
    return result.raw_response_json, []
