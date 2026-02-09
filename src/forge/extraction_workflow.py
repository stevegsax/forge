"""Temporal workflow for knowledge extraction.

Extracts structured lessons from completed task results and stores them
as playbook entries. Runs independently from task execution (D13).
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from forge.models import (
        ExtractionCallResult,
        ExtractionInput,
        ExtractionWorkflowInput,
        ExtractionWorkflowResult,
        FetchExtractionInput,
        SaveExtractionInput,
    )

_FETCH_TIMEOUT = timedelta(seconds=30)
_LLM_TIMEOUT = timedelta(minutes=5)
_SAVE_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class ForgeExtractionWorkflow:
    """Extract knowledge from completed task results.

    1. Fetch unprocessed runs from the store.
    2. Call extraction LLM to produce playbook entries.
    3. Save entries to the playbooks table.
    """

    @workflow.run
    async def run(self, input: ExtractionWorkflowInput) -> ExtractionWorkflowResult:
        extraction_input = await workflow.execute_activity(
            "fetch_extraction_input",
            FetchExtractionInput(
                limit=input.limit,
                since_hours=input.since_hours,
            ),
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=ExtractionInput,
        )

        if not extraction_input.source_workflow_ids:
            return ExtractionWorkflowResult(
                entries_created=0,
                source_workflow_ids=[],
            )

        call_result = await workflow.execute_activity(
            "call_extraction_llm",
            extraction_input,
            start_to_close_timeout=_LLM_TIMEOUT,
            result_type=ExtractionCallResult,
        )

        if call_result.result.entries:
            await workflow.execute_activity(
                "save_extraction_results",
                SaveExtractionInput(
                    entries=call_result.result.entries,
                    source_workflow_ids=call_result.source_workflow_ids,
                    extraction_workflow_id=workflow.info().workflow_id,
                ),
                start_to_close_timeout=_SAVE_TIMEOUT,
                result_type=type(None),
            )

        return ExtractionWorkflowResult(
            entries_created=len(call_result.result.entries),
            source_workflow_ids=call_result.source_workflow_ids,
        )
