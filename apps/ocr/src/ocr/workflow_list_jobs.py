"""OcrListJobsWorkflow — list OCR job submissions.

Returns one entry per user submission (grouped by file_path), with
aggregate status derived from the underlying batch_jobs rows and
document_id from ocr_results when available.

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
            input.model_dump_json(),
            start_to_close_timeout=_QUERY_TIMEOUT,
            retry_policy=_QUERY_RETRY,
            result_type=OcrListJobsResult,
        )

        workflow.logger.info(
            "OcrListJobs done: %d job(s) returned",
            len(result.jobs),
        )

        return result
