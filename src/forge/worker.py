"""Temporal worker entry point for Forge.

Connects to the Temporal server, registers all activities and workflows,
and runs the worker until interrupted.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleIntervalSpec,
    ScheduleSpec,
    ScheduleState,
    ScheduleUpdate,
    ScheduleUpdateInput,
)
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from forge.activities import (
    assemble_conflict_resolution_context,
    assemble_context,
    assemble_exploration_context,
    assemble_planner_context,
    assemble_sanity_check_context,
    assemble_step_context,
    assemble_sub_task_context,
    call_conflict_resolution,
    call_exploration_llm,
    call_extraction_llm,
    call_llm,
    call_planner,
    call_sanity_check,
    commit_changes_activity,
    create_worktree_activity,
    detect_file_conflicts_activity,
    evaluate_transition,
    export_single_playbook,
    fetch_existing_playbooks,
    fetch_extraction_input,
    fetch_playbook_ids,
    fulfill_context_requests,
    parse_llm_response,
    poll_batch_results,
    remove_worktree_activity,
    reset_worktree_activity,
    review_manual_playbook,
    save_extraction_results,
    submit_batch_request,
    validate_output,
    validate_playbook_entry,
    write_files,
    write_output,
)
from forge.activities.batch_poll import set_temporal_client
from forge.batch_poller_workflow import BatchPollerWorkflow
from forge.export_playbook_workflow import ExportPlaybookWorkflow
from forge.extraction_workflow import ForgeExtractionWorkflow

try:
    from forge.activities.ingestion import prepare_transcript
    from forge.ingestion_workflow import BatchIngestionWorkflow, TranscriptIngestionWorkflow

    _INGESTION_AVAILABLE = True
except ImportError:
    _INGESTION_AVAILABLE = False
    prepare_transcript = None  # type: ignore[assignment]
    BatchIngestionWorkflow = None  # type: ignore[assignment,misc]
    TranscriptIngestionWorkflow = None  # type: ignore[assignment,misc]
from forge.manual_playbook_workflow import ManualPlaybookWorkflow
from forge.models import BatchPollerInput, ExtractionWorkflowInput
from forge.ocr.activities import (
    call_ocr_sync,
    check_ocr_duplicate,
    clear_ocr_removal_mark,
    export_ocr_document,
    list_ocr_jobs,
    mark_ocr_for_removal,
    parse_ocr_result,
    read_and_store_file_content,
    reassemble_ocr_chunks,
    split_file_into_chunks,
    store_ocr_result,
    submit_ocr_batch,
    update_batch_job_status,
)
from forge.ocr.workflow_export import OcrExportWorkflow
from forge.ocr.workflow_gather import OcrGatherWorkflow
from forge.ocr.workflow_list_jobs import OcrListJobsWorkflow
from forge.ocr.workflow_mark_removal import (
    OcrClearRemovalMarkWorkflow,
    OcrMarkForRemovalWorkflow,
)
from forge.ocr.workflow_store import OcrStoreWorkflow
from forge.ocr.workflow_submit import OcrSubmitWorkflow
from forge.ocr.workflow_sync import OcrSyncWorkflow
from forge.workflows import FORGE_TASK_QUEUE, ForgeSubTaskWorkflow, ForgeTaskWorkflow

DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"

logger = logging.getLogger(__name__)


def _init_store() -> None:
    """Run database migrations on startup.

    The store is mandatory: ``FORGE_DB_URL`` must be set (a ``sqlite:///`` URL
    for dev/tests, a ``postgresql+psycopg2://`` URL for production). An unset
    URL or an unreachable database raises, and the worker refuses to start.
    """
    from sqlalchemy.engine import make_url

    from forge.store import get_store_url, run_migrations

    url = get_store_url()
    run_migrations(url)
    logger.info(
        "Database migrations complete: %s",
        make_url(url).render_as_string(hide_password=True),
    )


def _register_output_types() -> None:
    """Register Forge's Pydantic models with the shared LLM output type registry.

    Required for batch response parsing — the shared sax-llm package uses a
    plugin pattern instead of hardcoded imports.
    """
    from sax_llm import register_output_type

    from forge.eval.models import JudgeVerdict
    from forge.models import (
        ConflictResolutionResponse,
        ExplorationResponse,
        ExtractionResult,
        LLMResponse,
        Plan,
        SanityCheckResponse,
    )

    register_output_type("LLMResponse", LLMResponse)
    register_output_type("Plan", Plan)
    register_output_type("ExplorationResponse", ExplorationResponse)
    register_output_type("SanityCheckResponse", SanityCheckResponse)
    register_output_type("ConflictResolutionResponse", ConflictResolutionResponse)
    register_output_type("ExtractionResult", ExtractionResult)
    register_output_type("JudgeVerdict", JudgeVerdict)

    try:
        from pbook.ingestion_prompts import TranscriptAnalysisResult

        register_output_type("TranscriptAnalysisResult", TranscriptAnalysisResult)
    except ImportError:
        logger.warning(
            "pbook not installed — TranscriptAnalysisResult output type not registered. "
            "Ingestion workflows will not be available."
        )


async def _ensure_schedule(
    client: Client,
    schedule_id: str,
    workflow_name: str,
    workflow_arg: object,
    interval: timedelta,
) -> None:
    """Create or update a Temporal schedule (idempotent).

    On first run, creates the schedule. On subsequent runs, updates the interval
    if it has changed. Handles the "already exists" case gracefully.
    """
    schedule = Schedule(
        action=ScheduleActionStartWorkflow(
            workflow_name,
            workflow_arg,
            id=f"{schedule_id}-run",
            task_queue=FORGE_TASK_QUEUE,
        ),
        spec=ScheduleSpec(
            intervals=[ScheduleIntervalSpec(every=interval)],
        ),
        state=ScheduleState(
            note=f"Forge schedule: {schedule_id}",
        ),
    )

    try:
        await client.create_schedule(schedule_id, schedule)
        logger.info("Created schedule %s (interval=%s)", schedule_id, interval)
    except ScheduleAlreadyRunningError:
        # Update the existing schedule with the new interval
        handle = client.get_schedule_handle(schedule_id)

        async def _updater(input: ScheduleUpdateInput) -> ScheduleUpdate:
            input.description.schedule.spec = schedule.spec
            return ScheduleUpdate(schedule=input.description.schedule)

        await handle.update(_updater)
        logger.info("Updated schedule %s (interval=%s)", schedule_id, interval)


async def run_worker(
    address: str | None = None,
    *,
    batch_poll_interval: int = 600,
    extraction_interval: int = 14400,
    identity: str | None = None,
) -> None:
    """Connect to Temporal and run the Forge worker."""
    from forge.tracing import init_tracing, shutdown_tracing

    if address is None:
        address = os.environ.get("FORGE_TEMPORAL_ADDRESS", DEFAULT_TEMPORAL_ADDRESS)

    from forge.logging_config import silence_noisy_loggers

    _init_store()
    _register_output_types()
    init_tracing()
    silence_noisy_loggers()

    connect_kwargs: dict[str, object] = {
        "data_converter": pydantic_data_converter,
    }
    if identity is not None:
        connect_kwargs["identity"] = identity

    client = await Client.connect(
        address,
        **connect_kwargs,
    )

    # Inject Temporal client for poll activity signal delivery
    set_temporal_client(client)

    # Create/update schedules — if these fail, the worker is useless
    await _ensure_schedule(
        client,
        schedule_id="forge-batch-poller",
        workflow_name="BatchPollerWorkflow",
        workflow_arg=BatchPollerInput(),
        interval=timedelta(seconds=batch_poll_interval),
    )

    await _ensure_schedule(
        client,
        schedule_id="forge-extraction-schedule",
        workflow_name="ForgeExtractionWorkflow",
        workflow_arg=ExtractionWorkflowInput(),
        interval=timedelta(seconds=extraction_interval),
    )

    workflows: list[type] = [
        ForgeTaskWorkflow,
        ForgeSubTaskWorkflow,
        ForgeExtractionWorkflow,
        ExportPlaybookWorkflow,
        ManualPlaybookWorkflow,
        BatchPollerWorkflow,
        OcrSubmitWorkflow,
        OcrSyncWorkflow,
        OcrStoreWorkflow,
        OcrGatherWorkflow,
        OcrExportWorkflow,
        OcrListJobsWorkflow,
        OcrMarkForRemovalWorkflow,
        OcrClearRemovalMarkWorkflow,
    ]
    if _INGESTION_AVAILABLE:
        assert TranscriptIngestionWorkflow is not None
        assert BatchIngestionWorkflow is not None
        workflows.extend([TranscriptIngestionWorkflow, BatchIngestionWorkflow])
    else:
        logger.warning("pbook not installed — ingestion workflows skipped at worker registration")

    activities: list = [
        assemble_conflict_resolution_context,
        assemble_context,
        assemble_exploration_context,
        assemble_planner_context,
        assemble_sanity_check_context,
        assemble_step_context,
        assemble_sub_task_context,
        call_conflict_resolution,
        call_exploration_llm,
        call_extraction_llm,
        call_llm,
        call_planner,
        call_sanity_check,
        commit_changes_activity,
        create_worktree_activity,
        detect_file_conflicts_activity,
        evaluate_transition,
        export_single_playbook,
        fetch_existing_playbooks,
        fetch_extraction_input,
        fetch_playbook_ids,
        fulfill_context_requests,
        parse_llm_response,
        poll_batch_results,
        remove_worktree_activity,
        reset_worktree_activity,
        review_manual_playbook,
        save_extraction_results,
        submit_batch_request,
        validate_output,
        validate_playbook_entry,
        write_files,
        write_output,
        # OCR activities
        list_ocr_jobs,
        call_ocr_sync,
        check_ocr_duplicate,
        clear_ocr_removal_mark,
        export_ocr_document,
        mark_ocr_for_removal,
        read_and_store_file_content,
        split_file_into_chunks,
        submit_ocr_batch,
        parse_ocr_result,
        store_ocr_result,
        update_batch_job_status,
        reassemble_ocr_chunks,
    ]
    if _INGESTION_AVAILABLE:
        assert prepare_transcript is not None
        activities.append(prepare_transcript)

    worker = Worker(
        client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=workflows,
        activities=activities,
        graceful_shutdown_timeout=timedelta(seconds=30),
    )

    try:
        await worker.run()
    finally:
        shutdown_tracing()
