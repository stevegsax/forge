"""OcrListJobsWorkflow — list OCR job submissions.

Returns one entry per ``ocr_job_status`` row (OCR's single-writer status
projection), left-joined to the platform ``batch_jobs`` ledger on
``request_id``. The displayed status is derived from the pair (OCR processing
status by provider batch status) by ``_derive_status``.

Composable as a child workflow for other workflows that need to
inspect OCR job state.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ocr.models import OcrListJobsInput, OcrListJobsResult

_QUERY_TIMEOUT = timedelta(seconds=30)
_QUERY_RETRY = RetryPolicy(maximum_attempts=2)


@workflow.defn
class OcrListJobsWorkflow:
    """List OCR job submissions with aggregate status."""

    @workflow.run
    async def run(self, input: OcrListJobsInput) -> OcrListJobsResult:
        workflow.logger.info(
            "OcrListJobs started: limit=%d status_filter=%s",
            input.limit,
            input.status_filter or "(none)",
        )

        result: OcrListJobsResult = await workflow.execute_activity(
            "list_ocr_jobs",
            input,
            start_to_close_timeout=_QUERY_TIMEOUT,
            retry_policy=_QUERY_RETRY,
            result_type=OcrListJobsResult,
        )

        workflow.logger.info(
            "OcrListJobs done: %d job(s) returned",
            len(result.jobs),
        )

        return result
