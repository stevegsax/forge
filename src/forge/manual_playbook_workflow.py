"""Temporal workflow for manually submitted playbook entries.

Validates, reviews via LLM, and saves a manually submitted playbook entry.
The CLI submits raw JSON; all processing happens inside Temporal activities.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from forge.models import (
        CapabilityTier,
        FetchExistingPlaybooksInput,
        ManualPlaybookInput,
        ManualPlaybookResult,
        ReviewManualPlaybookInput,
        ReviewManualPlaybookResult,
        SaveExtractionInput,
        ValidatePlaybookInput,
        ValidatePlaybookResult,
        resolve_model,
    )

_VALIDATE_TIMEOUT = timedelta(seconds=30)
_FETCH_TIMEOUT = timedelta(seconds=30)
_REVIEW_TIMEOUT = timedelta(minutes=2)
_REVIEW_HEARTBEAT = timedelta(seconds=60)
_SAVE_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class ManualPlaybookWorkflow:
    """Validate, review, and save a manually submitted playbook entry."""

    @workflow.run
    async def run(self, input: ManualPlaybookInput) -> ManualPlaybookResult:
        # Step 1: Validate raw JSON
        validation = await workflow.execute_activity(
            "validate_playbook_entry",
            ValidatePlaybookInput(raw_json=input.raw_json),
            start_to_close_timeout=_VALIDATE_TIMEOUT,
            result_type=ValidatePlaybookResult,
        )
        if not validation.valid:
            return ManualPlaybookResult(
                approved=False,
                validation_error=validation.error,
            )

        assert validation.entry is not None  # guaranteed when valid=True

        # Step 2: Fetch existing playbooks for duplication context
        existing = await workflow.execute_activity(
            "fetch_existing_playbooks",
            FetchExistingPlaybooksInput(limit=50),
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )

        # Step 3: LLM review
        classification_model = resolve_model(CapabilityTier.CLASSIFICATION, input.model_routing)
        review_result = await workflow.execute_activity(
            "review_manual_playbook",
            ReviewManualPlaybookInput(
                entry=validation.entry,
                existing_playbooks=existing,
                model_name=classification_model,
            ),
            start_to_close_timeout=_REVIEW_TIMEOUT,
            heartbeat_timeout=_REVIEW_HEARTBEAT,
            result_type=ReviewManualPlaybookResult,
        )
        if not review_result.approved:
            return ManualPlaybookResult(
                approved=False,
                rejection_reason=review_result.rejection_reason,
            )

        # Step 4: Save (reuse existing extraction save activity)
        await workflow.execute_activity(
            "save_extraction_results",
            SaveExtractionInput(
                entries=[review_result.final_entry],
                source_workflow_ids=[],
                extraction_workflow_id=workflow.info().workflow_id,
            ),
            start_to_close_timeout=_SAVE_TIMEOUT,
            result_type=type(None),
        )

        return ManualPlaybookResult(
            approved=True,
            entry=review_result.final_entry,
        )
