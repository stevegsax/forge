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

from pydantic import BaseModel, Field
from sax_platform.contracts.persist import (
    PersistBatchFailure as PersistBatchFailure,
)
from sax_platform.contracts.persist import (
    PersistBatchSubmission as PersistBatchSubmission,
)
from sax_platform.contracts.persist import (
    PersistResult as PersistResult,
)

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
    stop_reason: str | None = None


class PersistRun(BaseModel):
    """A workflow run result (runs table), keyed by ``(workflow_id, run_id)``.

    ``run_id`` is the Temporal execution run id (``workflow.info().run_id``). It
    distinguishes reruns of the same ``workflow_id`` — the deterministic
    ``forge-task-{task_id}`` id is reused on every rerun, so without ``run_id`` the
    second run's rows would be swallowed by ``insert_or_ignore`` (T1.6a).
    """

    kind: Literal["run"] = "run"
    workflow_id: str
    run_id: str
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


def build_interaction_idempotency_key(
    *,
    workflow_id: str,
    run_id: str,
    role: str,
    occurrence: int,
) -> str:
    """Deterministic per-interaction idempotency key (pure; sandbox-safe).

    Shape: ``{workflow_id}:{run_id}:{role}:{occurrence}``. ``run_id`` isolates
    reruns of the same (reused) ``workflow_id``; ``occurrence`` is a per-role
    counter held in workflow state, so a repeated same-role call (e.g. a second
    sanity check) gets a distinct key instead of colliding (T1.6a).
    """
    return f"{workflow_id}:{run_id}:{role}:{occurrence}"


def reshape_legacy_interaction_key(old_key: str, *, run_id: str) -> str:
    """Insert a sentinel ``run_id`` into a legacy ``{wf}:{role}:{seq}`` key.

    Migration-only (002). Legacy keys (pre-T1.6a) have exactly three trailing
    colon-segments ending in a numeric positional seq; the new shape carries a
    ``run_id`` between ``wf`` and ``role``. The transform inserts the sentinel and
    reuses the old seq as the occurrence — injective, so distinct legacy keys map
    to distinct new keys with no transient uniqueness violation. Keys that don't
    match the legacy shape (e.g. the bespoke ``{wf}:extraction`` key, or ``None``)
    are returned unchanged.
    """
    wf, sep_role, seq = _split_legacy(old_key)
    if seq is None:
        return old_key
    return f"{wf}:{run_id}:{sep_role}:{seq}"


def restore_legacy_interaction_key(new_key: str, *, run_id: str) -> str:
    """Inverse of :func:`reshape_legacy_interaction_key` (migration 002 downgrade).

    Only strips the sentinel for keys whose run_id segment equals ``run_id``; keys
    minted by real reruns (a genuine Temporal run_id) are left untouched.
    """
    parts = new_key.rsplit(":", 3)
    if len(parts) == 4 and parts[1] == run_id and parts[3].isdigit():
        wf, _sentinel, role, seq = parts
        return f"{wf}:{role}:{seq}"
    return new_key


def _split_legacy(key: str) -> tuple[str, str, str | None]:
    """Split a legacy ``{wf}:{role}:{seq}`` key; ``seq`` is ``None`` if no match."""
    parts = key.rsplit(":", 2)
    if len(parts) == 3 and parts[2].isdigit():
        return parts[0], parts[1], parts[2]
    return key, "", None


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

    context_stats_json = context_stats.model_dump_json() if context_stats is not None else None
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
        stop_reason=getattr(result, "stop_reason", None),
    )
