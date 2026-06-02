"""Temporal workflows for the worker-only CLI surface.

Each workflow is a thin wrapper over a single activity in
``pbook.activities.cli_ops``. The CLI submits one of these for every
direct-DB command (``pbook get``, ``pbook approve``, etc.), and the
worker runs the activity against its configured DB. The worker's
``PBOOK_DATABASE_URL`` is the single source of truth for which DB any
operation hits.

These workflows do not call LLMs or do anything time-skipping; they
exist solely to route CLI ops through the worker process.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pbook.models import (
        AddEntryInput,
        ApproveEntryInput,
        CheckDuplicateInput,
        FilterAlreadyIngestedInput,
        GetEntryInput,
        GetSessionTextInput,
        ListEntriesInput,
        ListSessionsInput,
        ListSourcesInput,
        ListTagsInput,
        PruneInput,
        RecordFeedbackInput,
        RecordStartedSessionsInput,
        RejectEntryInput,
        ReviewQueueInput,
        UpdateEntryInput,
    )

_DB_TIMEOUT = timedelta(seconds=30)
_TRANSCRIPT_TIMEOUT = timedelta(seconds=60)


# ---------------------------------------------------------------------------
# Read workflows
# ---------------------------------------------------------------------------


@workflow.defn
class GetEntryWorkflow:
    """Fetch a single entry by id."""

    @workflow.run
    async def run(self, input: GetEntryInput) -> dict | None:
        return await workflow.execute_activity(
            "get_entry_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
        )


@workflow.defn
class ListEntriesWorkflow:
    """List entries with tag/type/project/needs-review filters."""

    @workflow.run
    async def run(self, input: ListEntriesInput) -> list:
        return await workflow.execute_activity(
            "list_entries_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=list,
        )


@workflow.defn
class ListSourcesWorkflow:
    """List entry_sources rows for an entry. Returns ``{found, rows}``
    so the workflow caller can surface a not_found error vs an empty
    sources list."""

    @workflow.run
    async def run(self, input: ListSourcesInput) -> dict:
        return await workflow.execute_activity(
            "list_sources_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=dict,
        )


@workflow.defn
class ListTagsWorkflow:
    """Return canonical tag namespaces and values currently in use."""

    @workflow.run
    async def run(self, _input: ListTagsInput) -> dict:
        return await workflow.execute_activity(
            "list_tags_activity",
            {},
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=dict,
        )


@workflow.defn
class ReviewQueueWorkflow:
    """List entries in the review queue (flat or clustered by experience)."""

    @workflow.run
    async def run(self, input: ReviewQueueInput) -> dict:
        return await workflow.execute_activity(
            "review_queue_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=dict,
        )


@workflow.defn
class ListSessionsWorkflow:
    """List ingested_sessions rows."""

    @workflow.run
    async def run(self, input: ListSessionsInput) -> list:
        return await workflow.execute_activity(
            "list_sessions_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=list,
        )


@workflow.defn
class GetSessionTextWorkflow:
    """Render a Claude Code session transcript by id."""

    @workflow.run
    async def run(self, input: GetSessionTextInput) -> dict:
        return await workflow.execute_activity(
            "get_session_text_activity",
            input.model_dump(),
            start_to_close_timeout=_TRANSCRIPT_TIMEOUT,
            result_type=dict,
        )


@workflow.defn
class CheckDuplicateWorkflow:
    """Find entries with similar titles for duplicate detection."""

    @workflow.run
    async def run(self, input: CheckDuplicateInput) -> list:
        return await workflow.execute_activity(
            "check_duplicate_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=list,
        )


# ---------------------------------------------------------------------------
# Write workflows
# ---------------------------------------------------------------------------


@workflow.defn
class AddEntryWorkflow:
    """Insert a new playbook entry."""

    @workflow.run
    async def run(self, input: AddEntryInput) -> dict:
        return await workflow.execute_activity(
            "add_entry_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=dict,
        )


@workflow.defn
class ApproveEntryWorkflow:
    """Clear ``needs_review`` on an entry."""

    @workflow.run
    async def run(self, input: ApproveEntryInput) -> dict:
        return await workflow.execute_activity(
            "approve_entry_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=dict,
        )


@workflow.defn
class RejectEntryWorkflow:
    """Soft-mark an entry as rejected with an optional reason."""

    @workflow.run
    async def run(self, input: RejectEntryInput) -> dict:
        return await workflow.execute_activity(
            "reject_entry_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=dict,
        )


@workflow.defn
class UpdateEntryWorkflow:
    """Update arbitrary entry columns."""

    @workflow.run
    async def run(self, input: UpdateEntryInput) -> dict:
        return await workflow.execute_activity(
            "update_entry_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=dict,
        )


@workflow.defn
class RecordFeedbackWorkflow:
    """Record helpful/harmful feedback on a retrieved entry."""

    @workflow.run
    async def run(self, input: RecordFeedbackInput) -> dict:
        return await workflow.execute_activity(
            "record_feedback_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=dict,
        )


@workflow.defn
class PruneWorkflow:
    """Identify (and optionally apply) prune candidates."""

    @workflow.run
    async def run(self, input: PruneInput) -> dict:
        return await workflow.execute_activity(
            "prune_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=dict,
        )


# ---------------------------------------------------------------------------
# Ingest helper workflows (called by `pbook ingest` so the CLI doesn't
# open the DB directly for filter/seed bookkeeping).
# ---------------------------------------------------------------------------


@workflow.defn
class FilterAlreadyIngestedWorkflow:
    """Filter session_ids against the worker's ingested_sessions table."""

    @workflow.run
    async def run(self, input: FilterAlreadyIngestedInput) -> dict:
        return await workflow.execute_activity(
            "filter_already_ingested_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=dict,
        )


@workflow.defn
class RecordStartedSessionsWorkflow:
    """Seed ``ingested_sessions`` running rows after submitting batch ingest."""

    @workflow.run
    async def run(self, input: RecordStartedSessionsInput) -> dict:
        return await workflow.execute_activity(
            "record_started_sessions_activity",
            input.model_dump(),
            start_to_close_timeout=_DB_TIMEOUT,
            result_type=dict,
        )
