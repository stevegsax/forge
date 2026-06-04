"""Survivable-write retry/timeout presets for OCR workflows.

A transient DB outage retries only the cheap persist activity — the expensive
provider call already returned to the workflow and is never re-run. The
``persist_block`` helper and the OCR-side ``persist_to_store`` activity it targets
are added in the cross-queue increment; for now this module provides the presets
the store workflow uses on its own ``execute_activity`` calls.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio.common import RetryPolicy

_LOCAL_RETRY = RetryPolicy(maximum_attempts=2)

# Survivable store writes: backoff 1,2,4,8,16,32,60,60… fits ~18-20 tries in the
# 20-minute schedule_to_close governor, after which the activity fails loudly.
# ValueError is validation (never succeeds on retry); idempotency-key collisions
# are absorbed by insert_or_ignore and never raise.
_PERSIST_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
    maximum_attempts=20,
    non_retryable_error_types=["ValueError"],
)
_PERSIST_START_TO_CLOSE = timedelta(seconds=30)
_PERSIST_SCHEDULE_TO_CLOSE = timedelta(minutes=20)
