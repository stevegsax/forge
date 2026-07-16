"""OCR mark/clear removal workflows.

OcrMarkForRemovalWorkflow sets marked_for_removal=True on a document.
OcrClearRemovalMarkWorkflow sets marked_for_removal=False on a document.

Both are composable as child workflows and operate on one document
per invocation. A separate periodic workflow handles actual deletion
of marked documents.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ocr.models import OcrMarkInput, OcrMarkResult

_MARK_TIMEOUT = timedelta(seconds=30)
_MARK_RETRY = RetryPolicy(maximum_attempts=2)


@workflow.defn
class OcrMarkForRemovalWorkflow:
    """Mark a single OCR document for removal."""

    @workflow.run
    async def run(self, input: OcrMarkInput) -> OcrMarkResult:
        workflow.logger.info(
            "OcrMarkForRemoval started: document_id=%s",
            input.document_id,
        )

        result: OcrMarkResult = await workflow.execute_activity(
            "mark_ocr_for_removal",
            input.document_id,
            start_to_close_timeout=_MARK_TIMEOUT,
            retry_policy=_MARK_RETRY,
            result_type=OcrMarkResult,
        )

        workflow.logger.info(
            "OcrMarkForRemoval done: document_id=%s found=%s",
            result.document_id,
            result.found,
        )

        return result


@workflow.defn
class OcrClearRemovalMarkWorkflow:
    """Clear the removal mark on a single OCR document."""

    @workflow.run
    async def run(self, input: OcrMarkInput) -> OcrMarkResult:
        workflow.logger.info(
            "OcrClearRemovalMark started: document_id=%s",
            input.document_id,
        )

        result: OcrMarkResult = await workflow.execute_activity(
            "clear_ocr_removal_mark",
            input.document_id,
            start_to_close_timeout=_MARK_TIMEOUT,
            retry_policy=_MARK_RETRY,
            result_type=OcrMarkResult,
        )

        workflow.logger.info(
            "OcrClearRemovalMark done: document_id=%s found=%s",
            result.document_id,
            result.found,
        )

        return result
