"""Temporal workflow for knowledge extraction.

Extracts structured lessons from completed task results and stores them
as playbook entries. Runs independently from task execution (D13).
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from forge.models import (
        CapabilityTier,
        ExtractionCallResult,
        ExtractionInput,
        ExtractionWorkflowInput,
        ExtractionWorkflowResult,
        FetchExtractionInput,
        ValidatePlaybookInput,
        ValidatePlaybookResult,
        resolve_model,
    )
    from forge.persist_models import PersistPlaybooks, build_persist_interaction
    from forge.workflow_blocks import persist_block as _persist_block

_FETCH_TIMEOUT = timedelta(seconds=30)
_LLM_TIMEOUT = timedelta(minutes=5)
_LLM_HEARTBEAT = timedelta(seconds=60)


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

        # Resolve model for extraction
        summarization_model = resolve_model(CapabilityTier.SUMMARIZATION, input.model_routing)
        extraction_input = extraction_input.model_copy(
            update={"model_name": summarization_model},
        )

        call_result = await workflow.execute_activity(
            "call_extraction_llm",
            extraction_input,
            start_to_close_timeout=_LLM_TIMEOUT,
            heartbeat_timeout=_LLM_HEARTBEAT,
            result_type=ExtractionCallResult,
        )

        # Survivably persist the extraction interaction (one per run).
        await _persist_block(
            build_persist_interaction(
                idempotency_key=f"{workflow.info().workflow_id}:extraction",
                role="extraction",
                task_id="__extraction__",
                system_prompt=extraction_input.system_prompt,
                user_prompt=extraction_input.user_prompt,
                result=call_result,
            )
        )

        # Validate each extracted entry through the shared activity
        validated_entries = []
        for entry in call_result.result.entries:
            v = await workflow.execute_activity(
                "validate_playbook_entry",
                ValidatePlaybookInput(raw_json=entry.model_dump_json()),
                start_to_close_timeout=_FETCH_TIMEOUT,
                result_type=ValidatePlaybookResult,
            )
            if v.valid and v.entry is not None:
                validated_entries.append(v.entry)

        if validated_entries:
            # Survivable, idempotent playbook write (replaces save_extraction_results).
            await _persist_block(
                PersistPlaybooks(
                    extraction_workflow_id=workflow.info().workflow_id,
                    entries=validated_entries,
                )
            )

        return ExtractionWorkflowResult(
            entries_created=len(validated_entries),
            source_workflow_ids=call_result.source_workflow_ids,
        )
