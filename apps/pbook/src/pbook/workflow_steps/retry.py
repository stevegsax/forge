"""Shared Temporal retry policy for LLM-backed activities.

Temporal's *default* activity retry policy has ``maximum_attempts=0``
(unlimited). For LLM/embedding activities that turns a permanent failure
(e.g. a worker started without an API key) into a workflow that retries
forever — the ingestion session then hangs at ``running`` instead of
flipping to ``error``. Applying :data:`LLM_RETRY_POLICY` bounds the
attempts so the activity error eventually propagates and the workflow
fails. Genuinely permanent faults (auth/config) are additionally raised
non-retryable inside the activity (see
:func:`pbook.workflow_steps._errors.is_nonretryable_auth_error`), so they
fail on the first attempt rather than climbing this ladder.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Final

from temporalio.common import RetryPolicy

__all__ = ["LLM_RETRY_POLICY"]

LLM_RETRY_POLICY: Final = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=4,
)
