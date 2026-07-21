"""Wire models exchanged between the platform and consumer apps over Temporal.

``BatchJobStatus`` crosses task-queue boundaries — a consumer app records the
lifecycle of its own batch on the platform ``batch_jobs`` ledger cross-queue — so
it must be importable by both the platform and consumer apps and serialize
identically via ``pydantic_data_converter``. The result-payload envelope helpers
are forge's claim-check for stashing a large or image-bearing batch body to S3.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class BatchJobStatus(StrEnum):
    """Provider-batch lifecycle state of a ``batch_jobs`` row (audit/spend ledger).

    Under the timer-loop batch transport (D88, T4.1) the waiting workflow is the
    recipient of its own batch result, so ``batch_jobs`` is a forge-internal
    audit/spend ledger written at two points only — submit and terminal outcome.
    This still tracks ONLY the provider batch lifecycle; a consumer's own
    processing/terminal lifecycle lives in the consumer's status table, joined on
    ``request_id``.

    Happy path: SUBMITTED -> ENDED (the waiter fetched the finished batch's
    result). Failure paths: FAILED / EXPIRED (terminal provider states) and
    MISSING (the waiter gave up at its 25h ceiling). SUBMITTED is the only
    non-terminal state; every terminal status absorbs all later updates (the
    monotonic ``WHERE status = 'submitted'`` guard in ``update_batch_status``).
    """

    SUBMITTED = "submitted"
    """Row created at submit; batch in flight at the provider, waiting for
    completion. The only non-terminal state — every terminal update transitions
    from here."""

    ENDED = "ended"
    """Terminal success: the waiter fetched the finished batch's result. Written at
    final fetch by the timer-loop transport (T4.1)."""

    PROCESSING = "processing"
    """Legacy member. Written only by the retired shared poller (pre-T4.1), which
    set it after delivering a result signal to the waiting consumer. Kept so old
    rows remain readable; the timer-loop transport never writes it."""

    FAILED = "failed"
    """Terminal provider failure: the API refused the submission before returning
    a batch_id (``record_batch_failure``), or the provider reported the batch
    FAILED/CANCELED. CANCELED maps here (the coarse status needs no cancel/fail
    distinction)."""

    EXPIRED = "expired"
    """Terminal: the provider marked the batch expired (TIMEOUT_EXCEEDED)."""

    MISSING = "missing"
    """Terminal: the waiter gave up at its 25h ceiling — it stopped waiting for a
    batch it never saw complete."""


# ---------------------------------------------------------------------------
# Batch-result delivery payload envelope
# ---------------------------------------------------------------------------
#
# Forge's batch transport claim-checks a large or image-bearing result body to S3
# as an *envelope* carrying both the provider body and any images the OCR lane
# extracted. The fetch activity stashes it; the parse activity fetches and unwraps
# it. Small image-free bodies travel inline and never touch this path.


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
