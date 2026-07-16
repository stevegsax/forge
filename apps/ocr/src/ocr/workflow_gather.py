"""OcrGatherWorkflow — await chunk completions and reassemble.

Started as a child of OcrSubmitWorkflow when a document is split into
multiple chunks.  Each OcrStoreWorkflow signals this workflow upon
completion.  Once all chunks have reported in, the reassemble activity
combines results into one.
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ocr.models import (
        OcrGatherInput,
        OcrStoreResult,
    )

_REASSEMBLE_TIMEOUT = timedelta(seconds=60)
_GATHER_WAIT_TIMEOUT = timedelta(hours=26)
_IO_RETRY = RetryPolicy(maximum_attempts=2)


@workflow.defn
class OcrGatherWorkflow:
    """Await chunk store completions, then reassemble into a single OCR result.

    Each OcrStoreWorkflow signals ``chunk_completed`` when it finishes.
    This workflow waits until all expected chunks have reported, then
    runs the reassemble activity.
    """

    def __init__(self) -> None:
        self._completed_chunks: set[str] = set()

    @workflow.signal
    async def chunk_completed(self, chunk_document_id: str) -> None:
        """Signal from an OcrStoreWorkflow that a chunk is done."""
        self._completed_chunks.add(chunk_document_id)

    @workflow.run
    async def run(self, input: OcrGatherInput) -> OcrStoreResult:
        expected = set(input.chunk_document_ids)
        workflow.logger.info(
            "OcrGather started: document_id=%s expecting %d chunks",
            input.document_id,
            len(expected),
        )

        # Wait for all chunks to report completion
        await workflow.wait_condition(
            lambda: expected.issubset(self._completed_chunks),
            timeout=_GATHER_WAIT_TIMEOUT,
        )

        workflow.logger.info(
            "All %d chunks completed, reassembling",
            len(expected),
        )

        # Reassemble chunk results into a single OCR result
        reassemble_data = json.dumps(
            {
                "document_id": input.document_id,
                "chunk_document_ids": input.chunk_document_ids,
                "file_path": input.file_path,
                "total_pages": input.total_pages,
            }
        )
        result: OcrStoreResult = await workflow.execute_activity(
            "reassemble_ocr_chunks",
            reassemble_data,
            start_to_close_timeout=_REASSEMBLE_TIMEOUT,
            retry_policy=_IO_RETRY,
            result_type=OcrStoreResult,
        )

        workflow.logger.info(
            "OcrGather done: document_id=%s text_length=%d",
            result.document_id,
            result.text_length,
        )

        return result
