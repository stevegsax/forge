"""OcrExportWorkflow — export OCR text and images to the filesystem.

Exports the markdown text and all associated images for a document
to a directory on disk. Designed to be composable as a child workflow.

Steps:
1. Run export_ocr_document activity (reads DB, writes files)
2. Return OcrExportResult with paths and counts
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ocr.models import OcrExportInput, OcrExportResult

_EXPORT_TIMEOUT = timedelta(seconds=60)
_EXPORT_RETRY = RetryPolicy(maximum_attempts=2)


@workflow.defn
class OcrExportWorkflow:
    """Export OCR document text and images to the filesystem."""

    @workflow.run
    async def run(self, input: OcrExportInput) -> OcrExportResult:
        workflow.logger.info(
            "OcrExport started: document_id=%s",
            input.document_id,
        )

        result = await workflow.execute_activity(
            "export_ocr_document",
            input.model_dump_json(),
            start_to_close_timeout=_EXPORT_TIMEOUT,
            retry_policy=_EXPORT_RETRY,
            result_type=OcrExportResult,
        )

        workflow.logger.info(
            "OcrExport done: document_id=%s export_dir=%s image_count=%d",
            result.document_id,
            result.export_dir,
            result.image_count,
        )

        return result
