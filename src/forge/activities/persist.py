"""Survivable store-write activity (Phase C).

Every store write a workflow needs is funneled through ``persist_to_store``, a
dedicated, idempotent activity invoked with a generous-but-finite retry policy
(see ``_PERSIST_RETRY`` in ``workflow_blocks``). A transient DB outage retries only
this cheap write — the expensive LLM/OCR/batch call already returned to the workflow
and is never re-run. A prolonged outage exhausts the schedule-to-close cap and fails
the workflow loudly. Duplicate re-applies are absorbed by ``insert_or_ignore``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, assert_never

from forge.persist_models import (
    PersistBatchFailure,
    PersistBatchOutcome,
    PersistBatchStatus,
    PersistBatchSubmission,
    PersistInteraction,
    PersistPlaybooks,
    PersistRequest,
    PersistResult,
    PersistRun,
)

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)


async def execute_persist(req: PersistRequest, engine: Engine) -> PersistResult:
    """Apply one idempotent store write, dispatched on ``req.kind``.

    Core of the ``persist_to_store`` activity: the store *engine* is injected by
    the ``StoreActivities`` composition root (the bound method wraps this). An
    unreachable DB raises, which is exactly what triggers the Temporal retry that
    makes the write survivable; duplicate re-applies are absorbed by
    ``insert_or_ignore``.
    """
    from forge.store import (
        InteractionRow,
        build_playbook_dict,
        record_batch_failure,
        record_batch_submission,
        save_interaction,
        save_playbooks,
        save_run,
        update_batch_status,
    )

    match req:
        case PersistInteraction():
            applied = save_interaction(
                engine,
                InteractionRow(
                    idempotency_key=req.idempotency_key,
                    task_id=req.task_id,
                    step_id=req.step_id,
                    sub_task_id=req.sub_task_id,
                    role=req.role,
                    system_prompt=req.system_prompt,
                    user_prompt=req.user_prompt,
                    model_name=req.model_name,
                    input_tokens=req.input_tokens,
                    output_tokens=req.output_tokens,
                    latency_ms=req.latency_ms,
                    explanation=req.explanation,
                    context_stats_json=req.context_stats_json,
                    cache_creation_input_tokens=req.cache_creation_input_tokens,
                    cache_read_input_tokens=req.cache_read_input_tokens,
                    stop_reason=req.stop_reason,
                ),
            )
        case PersistRun():
            applied = save_run(engine, req.task_result, req.workflow_id, req.run_id)
        case PersistBatchSubmission():
            applied = record_batch_submission(
                engine,
                request_id=req.request_id,
                batch_id=req.batch_id,
                workflow_id=req.workflow_id,
                provider=req.provider,
            )
        case PersistBatchFailure():
            applied = record_batch_failure(
                engine,
                request_id=req.request_id,
                workflow_id=req.workflow_id,
                error_message=req.error_message,
                provider=req.provider,
            )
        case PersistBatchStatus():
            # A status transition is a plain UPDATE (no dedupe); always "applied".
            update_batch_status(
                engine,
                request_id=req.request_id,
                status=req.status,
                error_message=req.error_message,
            )
            applied = True
        case PersistBatchOutcome():
            # Terminal provider-lifecycle outcome: a monotonic UPDATE guarded to
            # SUBMITTED rows, so a stale/duplicate re-apply is a silent no-op that
            # never regresses a terminal status. Always reported as "applied".
            update_batch_status(
                engine,
                request_id=req.request_id,
                status=req.status,
                error_message=req.error_message,
            )
            applied = True
        case PersistPlaybooks():
            dicts = [build_playbook_dict(e, req.extraction_workflow_id) for e in req.entries]
            applied = save_playbooks(engine, dicts)
        case _ as unreachable:  # pragma: no cover - exhaustiveness guard
            assert_never(unreachable)

    logger.debug("persist_to_store kind=%s applied=%s", req.kind, applied)
    return PersistResult(kind=req.kind, applied=applied)
