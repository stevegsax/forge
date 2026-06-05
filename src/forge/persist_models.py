"""Pure request/result models for the survivable ``persist_to_store`` activity.

These models cross the workflow boundary, so this module imports **only** pure
pydantic models (``forge.models``) — never ``sqlalchemy`` or ``forge.store``. That
keeps it safe to import inside the Temporal workflow sandbox under
``workflow.unsafe.imports_passed_through()``.

Each store write is expressed as a discriminated ``PersistRequest`` variant (tagged
by ``kind``). The ``persist_to_store`` activity dispatches on ``kind`` and applies
an idempotent write, returning a ``PersistResult`` whose ``applied`` flag reports
whether a new row was written (vs. a duplicate absorbed on retry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from forge_contracts.persist import (
    PersistBatchFailure as PersistBatchFailure,
)
from forge_contracts.persist import (
    PersistBatchSubmission as PersistBatchSubmission,
)
from forge_contracts.persist import (
    PersistResult as PersistResult,
)
from pydantic import BaseModel, Field

from forge.models import (
    ConflictResolutionCallResult,
    ExtractionCallResult,
    LLMCallResult,
    PlanCallResult,
    PlaybookEntry,
    SanityCheckCallResult,
    TaskResult,
)

if TYPE_CHECKING:
    from forge.models import ContextStats

# The LLM-family result types that an interaction row can be built from. They all
# carry model_name/input_tokens/output_tokens/latency_ms and the cache token fields.
_LLMResult = (
    LLMCallResult
    | PlanCallResult
    | SanityCheckCallResult
    | ConflictResolutionCallResult
    | ExtractionCallResult
)


class PersistInteraction(BaseModel):
    """An LLM interaction row (interactions table), keyed by idempotency_key."""

    kind: Literal["interaction"] = "interaction"
    idempotency_key: str
    task_id: str
    role: str
    system_prompt: str
    user_prompt: str
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    step_id: str | None = None
    sub_task_id: str | None = None
    explanation: str = ""
    context_stats_json: str | None = None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class PersistRun(BaseModel):
    """A workflow run result (runs table), keyed by workflow_id."""

    kind: Literal["run"] = "run"
    workflow_id: str
    task_result: TaskResult


class PersistBatchStatus(BaseModel):
    """A batch job status transition (plain UPDATE on batch_jobs.id)."""

    kind: Literal["batch_status"] = "batch_status"
    request_id: str
    status: str
    error_message: str | None = None


class PersistPlaybooks(BaseModel):
    """Extracted playbook entries (playbooks table), keyed per-entry by uuid5."""

    kind: Literal["playbooks"] = "playbooks"
    extraction_workflow_id: str
    entries: list[PlaybookEntry]


PersistRequest = Annotated[
    PersistInteraction
    | PersistRun
    | PersistBatchSubmission
    | PersistBatchFailure
    | PersistBatchStatus
    | PersistPlaybooks,
    Field(discriminator="kind"),
]


def build_persist_interaction(
    *,
    idempotency_key: str,
    role: str,
    task_id: str,
    system_prompt: str,
    user_prompt: str,
    result: _LLMResult,
    step_id: str | None = None,
    sub_task_id: str | None = None,
    context_stats: ContextStats | None = None,
) -> PersistInteraction:
    """Build a ``PersistInteraction`` from an LLM-family result (pure; sandbox-safe).

    Mirrors the historical ``build_interaction_dict`` explanation rule exactly:
    explanation comes from ``result.response.explanation`` (LLM/sanity), else
    ``result.plan.explanation`` (planner), else "" (conflict-resolution/extraction).
    """
    explanation = ""
    response = getattr(result, "response", None)
    if response is not None:
        explanation = response.explanation
    else:
        plan = getattr(result, "plan", None)
        if plan is not None:
            explanation = plan.explanation

    context_stats_json = (
        context_stats.model_dump_json() if context_stats is not None else None
    )
    return PersistInteraction(
        idempotency_key=idempotency_key,
        task_id=task_id,
        role=role,
        step_id=step_id,
        sub_task_id=sub_task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=result.model_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        explanation=explanation,
        context_stats_json=context_stats_json,
        cache_creation_input_tokens=getattr(result, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(result, "cache_read_input_tokens", 0) or 0,
    )
