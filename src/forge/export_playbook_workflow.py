"""Temporal workflow for exporting playbook entries.

Fans out one activity per playbook row for parallel conversion,
then gathers results into a single ExportPlaybookResult.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from forge.models import (
        ExportPlaybookInput,
        ExportPlaybookResult,
        ExportSinglePlaybookInput,
        FetchPlaybookIdsInput,
        PlaybookEntry,
    )

_FETCH_TIMEOUT = timedelta(seconds=30)
_EXPORT_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class ExportPlaybookWorkflow:
    """Fetch matching playbook IDs, fan-out export per row, gather results."""

    @workflow.run
    async def run(self, input: ExportPlaybookInput) -> ExportPlaybookResult:
        # Step 1: Fetch matching IDs
        ids = await workflow.execute_activity(
            "fetch_playbook_ids",
            FetchPlaybookIdsInput(
                tags=input.tags,
                source_task_id=input.source_task_id,
                limit=input.limit,
            ),
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )

        if not ids:
            return ExportPlaybookResult(entries=[], count=0)

        # Step 2: Fan-out — start one activity per row
        handles = []
        for playbook_id in ids:
            handle = workflow.start_activity(
                "export_single_playbook",
                ExportSinglePlaybookInput(playbook_id=playbook_id),
                start_to_close_timeout=_EXPORT_TIMEOUT,
                result_type=PlaybookEntry,
            )
            handles.append(handle)

        # Step 3: Gather
        entries = []
        for handle in handles:
            entry = await handle
            entries.append(entry)

        return ExportPlaybookResult(entries=entries, count=len(entries))
