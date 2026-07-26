"""The single fan-out gather block (T5.3) — one copy of start → await → merge.

A fan-out gather always does the same six moves: reject duplicate sub-task ids,
start one child workflow per sub-task, await them all, detect (and optionally
resolve) file conflicts, write and validate the merged output, and hand back a
neutral outcome. That sequence lived twice in ``forge.workflows`` — once for a
plan's fan-out step and once for a fan-out nested inside a sub-task — and the
nested copy is exactly where the T1.5 propagation bug was bred: it silently
dropped ``resolve_conflicts``, ``thinking``, and ``model_routing`` that the
parent copy honored. Child-input construction now happens in one place
(:func:`build_child_input`), so those fields cannot diverge again.

Two things differ between the callers, and they live in one pure table
(:data:`GATHER_POLICIES`): who owns the worktree, and whether a successful
gather commits. Everything else that differs is a *value* the caller puts in its
:class:`GatherSpec`.

Worktree ownership decides cleanup, exactly as in ``blocks/step.py``. The
*borrowed* row runs in the plan's worktree, which ``_run_planned`` owns and
already wraps. The *owned* row creates its own worktree and removes it on every
exit — success, failure outcome, or exception — which closes the leak the nested
copy had: it created a worktree with no exception wrap and removed it in six
hand-repeated result paths, so any raise in between left the worktree *and* its
``forge/<id>`` branch behind, and the next run of that id failed on
``worktree add``. Batch-wait failures are the one deliberate exception: the
workflow's ``run()`` catches those (T1.6b), cleans this same worktree, and
records a terminal result, so they are re-raised untouched.

**Per-child failure isolation.** Each child await is wrapped: a child that
*raises* — its execution timeout expiring, an activity failure escaping, a
workflow bug — becomes a failed ``SubTaskResult`` (``failure_kind`` =
``child_crashed``) while the remaining children are still awaited to completion.
Before T5.3 both gathers bare-awaited, so one crashed child propagated a
``ChildWorkflowError`` out of ``run()``: no ``TaskResult``, no run record, no
worktree cleanup, and every in-flight sibling terminated with it — orphaning the
paid batches they were waiting on. ``CancelledError`` is a ``BaseException`` and
deliberately still propagates: a cancelled parent must not keep gathering.

**ParentClosePolicy: explicit TERMINATE** (owner-approved 2026-07-26). It stamps
the same proto value the SDK already defaults to, so the committed fan-out
histories replay unregenerated. With isolation in place the parent no longer
closes when a child fails, so TERMINATE now fires only when the parent itself
dies (its own execution timeout, or an operator terminate) — and there the
children are bounded by their own derived execution timeouts anyway. ABANDON was
considered and rejected: sub-tasks never commit and their results are read by
nobody once the parent is gone, so abandoned children would burn up to 25h of
worker occupancy finishing work into a void.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from sax_platform.contracts.constants import FORGE_TASK_QUEUE
    from sax_platform.temporal.retries import IO_RETRY
    from temporalio.workflow import ParentClosePolicy

    from forge.blocks.dispatch import dispatch_conflict_resolution
    from forge.blocks.step import (
        _CONTEXT_TIMEOUT,
        _GIT_RETRY,
        _GIT_TIMEOUT,
        _VALIDATE_HEARTBEAT,
        _VALIDATE_TIMEOUT,
        _WRITE_RETRY,
        _WRITE_TIMEOUT,
        cleanup_worktree_after_exception,
    )
    from forge.models import (
        CapabilityTier,
        CommitChangesInput,
        CommitChangesOutput,
        ConflictResolutionCallInput,
        ConflictResolutionCallResult,
        ConflictResolutionInput,
        CreateWorktreeInput,
        CreateWorktreeOutput,
        DetectFileConflictsInput,
        DetectFileConflictsOutput,
        FileConflict,
        ModelConfig,
        SubTask,
        SubTaskInput,
        SubTaskResult,
        TaskDomain,
        ThinkingPolicy,
        TransitionSignal,
        ValidateOutputInput,
        ValidationConfig,
        ValidationResult,
        WriteFilesInput,
        WriteResult,
        resolve_model,
    )
    from forge.step_logic import (
        MissingResolutions,
        child_timeout,
        compound_sub_task_id,
        crashed_child,
        determine_transition,
        failure_summary,
        merge_resolution,
        sub_task_workflow_id,
        subtask_failure_summary,
    )
    from forge.workflow_blocks import BATCH_WAIT_FAILURES, remove_worktree

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from temporalio.workflow import ChildWorkflowHandle

    from forge.blocks.dispatch import DispatchHost
    from forge.models import FailureKind

__all__ = [
    "GATHER_POLICIES",
    "GatherFailure",
    "GatherMode",
    "GatherOutcome",
    "GatherPolicy",
    "GatherSpec",
    "GatherSuccess",
    "build_child_input",
    "duplicate_sub_task_ids",
    "gather_commit_message",
    "run_fan_out_gather",
]

# The child workflow started for every sub-task. Referenced by name so this
# block does not import ``forge.workflows`` (which imports this module);
# tests/test_gather_block.py pins the name against the class.
SUB_TASK_WORKFLOW = "ForgeSubTaskWorkflow"

# Activity timeout/retry presets are imported from blocks/step.py rather than
# copied a fourth time; ST8 gives them one home.


# ---------------------------------------------------------------------------
# The policy table (pure)
# ---------------------------------------------------------------------------

type GatherMode = Literal["fan_out_step", "nested_fan_out"]

type WorktreeOwnership = Literal["borrowed", "owned"]
"""Who owns the worktree the merged output is written into.

- ``borrowed`` — the plan's worktree, created and cleaned by
  ``ForgeTaskWorkflow._run_planned``; the block only writes in it.
- ``owned`` — the block creates it from the parent branch and removes it on
  every exit, success included (sub-tasks never commit, D16).
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class GatherPolicy:
    """The two things that actually differ between the gather callers."""

    worktree: WorktreeOwnership
    commit: bool


GATHER_POLICIES: Mapping[GatherMode, GatherPolicy] = MappingProxyType(
    {
        "fan_out_step": GatherPolicy(worktree="borrowed", commit=True),
        "nested_fan_out": GatherPolicy(worktree="owned", commit=False),
    }
)


# ---------------------------------------------------------------------------
# Spec and outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class GatherSpec:
    """Everything one fan-out gather needs, independent of which workflow drives it.

    Lives only inside workflow code — never serialized — so it may carry a
    borrowed worktree handle and the plan's own models directly.
    """

    mode: GatherMode
    task_id: str
    """The git/write/validate identity, and the ``parent_task_id`` the children
    are told: a task id at the top level, a compound sub-task id when nested."""

    step_id: str
    """The fan-out unit's id (a plan step id, or the nesting sub-task's id) —
    used in log lines, the commit message, and conflict-resolution context."""

    repo_root: str
    sub_tasks: Sequence[SubTask]
    task_description: str
    """Description of the whole task; also the children's ``parent_description``."""

    step_description: str
    """Description of the fan-out unit, for conflict-resolution context only."""

    validation: ValidationConfig
    domain: TaskDomain
    child_depth: int
    """Depth stamped on the spawned children (0 at the top level)."""

    max_depth: int
    child_max_attempts: int
    child_model_name: str
    resolve_conflicts: bool
    model_routing: ModelConfig
    thinking: ThinkingPolicy
    sync_mode: bool
    log_messages: bool
    batch_poll_interval_seconds: int
    base_branch: str = ""
    """Branch an *owned* worktree is created from (the parent's branch)."""

    borrowed_worktree: CreateWorktreeOutput | None = None
    """The caller's worktree — required by ``borrowed`` mode, unused otherwise."""

    def __post_init__(self) -> None:
        """Reject a spec whose worktree ownership and handle disagree.

        An owned mode handed a borrowed worktree would remove the caller's
        worktree at the end of the gather; a borrowed mode without a handle
        would create one over the plan's branch (``worktree add -B``), throwing
        away committed step work.
        """
        borrowed = GATHER_POLICIES[self.mode].worktree == "borrowed"
        if borrowed and self.borrowed_worktree is None:
            msg = f"mode {self.mode!r} borrows its worktree but no borrowed_worktree was given"
            raise ValueError(msg)
        if not borrowed and self.borrowed_worktree is not None:
            msg = f"mode {self.mode!r} creates its own worktree; borrowed_worktree must be None"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class GatherSuccess:
    """Every child succeeded and the merged output validated."""

    output_files: dict[str, str]
    validation_results: list[ValidationResult]
    sub_task_results: list[SubTaskResult]
    conflict_resolution: ConflictResolutionCallResult | None = None
    commit_sha: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class GatherFailure:
    """The gather ended terminally; the caller stamps it onto its own result type."""

    failure_kind: FailureKind
    error: str
    sub_task_results: list[SubTaskResult] = field(default_factory=list)
    output_files: dict[str, str] = field(default_factory=dict)
    validation_results: list[ValidationResult] = field(default_factory=list)
    conflict_resolution: ConflictResolutionCallResult | None = None


type GatherOutcome = GatherSuccess | GatherFailure
"""Neutral gather result: the callers turn it into a ``StepResult`` (fan-out
step) or a ``SubTaskResult`` (nested) with the T5.1 ``step_logic`` builders."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def duplicate_sub_task_ids(sub_tasks: Sequence[SubTask]) -> bool:
    """True when two sub-tasks share an id (their worktrees would collide)."""
    ids = [st.sub_task_id for st in sub_tasks]
    return len(ids) != len(set(ids))


def gather_commit_message(spec: GatherSpec) -> str:
    """The commit message a committing gather uses (pure)."""
    return f"forge({spec.task_id}): step {spec.step_id} fan-out gather"


def build_child_input(spec: GatherSpec, sub_task: SubTask, parent_branch: str) -> SubTaskInput:
    """Build one child's input from the gather spec (pure).

    The single place fan-out children are constructed. ``resolve_conflicts``,
    ``thinking``, and ``model_routing`` are propagated here — the nested copy of
    this construction dropped all three before T1.5, and one construction site
    is what stops that from recurring.
    """
    return SubTaskInput(
        parent_task_id=spec.task_id,
        parent_description=spec.task_description,
        sub_task=sub_task,
        repo_root=spec.repo_root,
        parent_branch=parent_branch,
        validation=spec.validation,
        max_attempts=spec.child_max_attempts,
        model_name=spec.child_model_name,
        domain=spec.domain,
        depth=spec.child_depth,
        max_depth=spec.max_depth,
        resolve_conflicts=spec.resolve_conflicts,
        model_routing=spec.model_routing,
        thinking=spec.thinking,
        sync_mode=spec.sync_mode,
        log_messages=spec.log_messages,
        batch_poll_interval_seconds=spec.batch_poll_interval_seconds,
    )


def _duplicate_ids_failure(spec: GatherSpec) -> GatherFailure:
    """The terminal outcome for colliding sub-task ids (wording per mode)."""
    noun = "nested sub-task" if spec.mode == "nested_fan_out" else "sub-task"
    return GatherFailure(
        failure_kind="duplicate_sub_task_ids",
        error=f"Duplicate {noun} IDs detected",
    )


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


async def run_fan_out_gather(spec: GatherSpec, host: DispatchHost) -> GatherOutcome:
    """Run one fan-out gather to a neutral outcome.

    Raises:
        Exception: whatever the underlying activities or children raise, after
            removing a worktree the block owns. A child that raises does *not*
            reach here — it becomes a failed ``SubTaskResult`` instead.
    """
    policy = GATHER_POLICIES[spec.mode]
    if policy.worktree == "borrowed":
        assert spec.borrowed_worktree is not None  # GatherSpec.__post_init__ guarantees this
        return await _gather(spec, policy, host, spec.borrowed_worktree)

    wt_output = await _create_worktree(spec)
    try:
        outcome = await _gather(spec, policy, host, wt_output)
    except Exception as exc:
        # A batch wait that failed is already handled by the workflow's run():
        # it cleans this same worktree and records a terminal result, so
        # cleaning here would only duplicate the removal.
        if not isinstance(exc, BATCH_WAIT_FAILURES):
            await cleanup_worktree_after_exception(spec.repo_root, spec.task_id, exc)
        raise
    # Every non-exception exit removes the owned worktree exactly once (D16):
    # sub-tasks never commit, so nothing here is worth keeping.
    await remove_worktree(spec.repo_root, spec.task_id)
    return outcome


async def _gather(
    spec: GatherSpec,
    policy: GatherPolicy,
    host: DispatchHost,
    wt_output: CreateWorktreeOutput,
) -> GatherOutcome:
    """Start → await → merge, inside a worktree somebody else has decided about."""
    if duplicate_sub_task_ids(spec.sub_tasks):
        return _duplicate_ids_failure(spec)

    workflow.logger.info(
        "Fan-out: step_id=%s sub_tasks=%d depth=%d",
        spec.step_id,
        len(spec.sub_tasks),
        spec.child_depth,
    )
    handles = await _start_children(spec, wt_output)
    sub_task_results = await _await_children(handles)

    failures = [r for r in sub_task_results if r.status != TransitionSignal.SUCCESS]
    workflow.logger.info(
        "Fan-out gather: step_id=%s successes=%d failures=%d",
        spec.step_id,
        len(sub_task_results) - len(failures),
        len(failures),
    )
    if failures:
        return GatherFailure(
            failure_kind="sub_task_failed",
            error=subtask_failure_summary(failures),
            sub_task_results=sub_task_results,
        )

    merged = await _merge_children(spec, host, wt_output, sub_task_results)
    if isinstance(merged, GatherFailure):
        return merged
    merged_files, conflict_resolution = merged

    validation_results = await _write_and_validate(spec, wt_output, merged_files)
    # One attempt only: merged fan-out output is never re-generated. An empty
    # merge validated nothing, which determine_transition reads as SUCCESS —
    # the same behavior both inline copies had.
    signal = determine_transition(validation_results, attempt=1, max_attempts=1)
    if signal != TransitionSignal.SUCCESS:
        return GatherFailure(
            failure_kind="merged_validation",
            error=f"Merged output validation failed: {failure_summary(validation_results)}",
            output_files=merged_files,
            validation_results=validation_results,
            sub_task_results=sub_task_results,
        )

    commit_sha = await _commit(spec) if policy.commit else ""
    return GatherSuccess(
        output_files=merged_files,
        validation_results=validation_results,
        sub_task_results=sub_task_results,
        conflict_resolution=conflict_resolution,
        commit_sha=commit_sha,
    )


# ---------------------------------------------------------------------------
# Children
# ---------------------------------------------------------------------------


async def _start_children(
    spec: GatherSpec, wt_output: CreateWorktreeOutput
) -> list[tuple[str, ChildWorkflowHandle[object, SubTaskResult]]]:
    """Start one child workflow per sub-task, in order."""
    exec_timeout = child_timeout(
        spec.child_depth,
        spec.max_depth,
        sync_mode=spec.sync_mode,
        max_attempts=spec.child_max_attempts,
    )
    handles: list[tuple[str, ChildWorkflowHandle[object, SubTaskResult]]] = []
    for sub_task in spec.sub_tasks:
        compound_id = compound_sub_task_id(spec.task_id, sub_task.sub_task_id)
        handle = await workflow.start_child_workflow(
            SUB_TASK_WORKFLOW,
            build_child_input(spec, sub_task, wt_output.branch_name),
            id=sub_task_workflow_id(compound_id),
            task_queue=FORGE_TASK_QUEUE,
            result_type=SubTaskResult,
            execution_timeout=exec_timeout,
            # Explicit (T5.3): the same value the SDK defaults to, chosen rather
            # than inherited. See the module docstring for why not ABANDON.
            parent_close_policy=ParentClosePolicy.TERMINATE,
        )
        handles.append((sub_task.sub_task_id, handle))
    return handles


async def _await_children(
    handles: Sequence[tuple[str, ChildWorkflowHandle[object, SubTaskResult]]],
) -> list[SubTaskResult]:
    """Await every child, converting a raising child into a failed result.

    Isolation is the point: one crashed child must not take the run — or its
    in-flight siblings — down with it (the pattern this mirrors is
    ``ingestion_workflow.BatchIngestionWorkflow``). ``CancelledError`` derives
    from ``BaseException``, so a cancelled parent still stops here.
    """
    results: list[SubTaskResult] = []
    for sub_task_id, handle in handles:
        try:
            # ChildWorkflowHandle subclasses asyncio.Task; await the handle
            # itself (``handle.result()`` is Task.result() and raises early).
            result: SubTaskResult = await handle
        except Exception as exc:
            workflow.logger.warning("Fan-out child failed: sub_task_id=%s: %r", sub_task_id, exc)
            result = crashed_child(sub_task_id=sub_task_id, error=f"{type(exc).__name__}: {exc}")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Merge, write, validate, commit
# ---------------------------------------------------------------------------


async def _merge_children(
    spec: GatherSpec,
    host: DispatchHost,
    wt_output: CreateWorktreeOutput,
    sub_task_results: list[SubTaskResult],
) -> tuple[dict[str, str], ConflictResolutionCallResult | None] | GatherFailure:
    """Detect file conflicts and resolve them, or fail the gather (D27)."""
    detect_result: DetectFileConflictsOutput = await workflow.execute_activity(
        "detect_file_conflicts_activity",
        DetectFileConflictsInput(
            sub_task_results=sub_task_results,
            worktree_path=wt_output.worktree_path,
        ),
        start_to_close_timeout=_GIT_TIMEOUT,
        retry_policy=IO_RETRY,
        result_type=DetectFileConflictsOutput,
    )
    non_conflicting = detect_result.non_conflicting_files
    conflicts = detect_result.conflicts

    if not conflicts:
        return non_conflicting, None

    workflow.logger.info(
        "Conflict resolution: step_id=%s conflicts=%d", spec.step_id, len(conflicts)
    )
    if not spec.resolve_conflicts:
        # D27: without a resolver, a conflict is terminal.
        conflict_paths_str = ", ".join(c.file_path for c in conflicts)
        return GatherFailure(
            failure_kind="conflict_unresolved",
            error=f"File conflict: {conflict_paths_str} produced by multiple sub-tasks",
            sub_task_results=sub_task_results,
        )

    call_input = await _assemble_conflict_resolution(spec, wt_output, conflicts, non_conflicting)
    resolution = await dispatch_conflict_resolution(host, call_input)
    merged = merge_resolution(conflicts, resolution.resolved_files, non_conflicting)
    if isinstance(merged, MissingResolutions):
        return GatherFailure(
            failure_kind="conflict_incomplete",
            error=merged.message,
            sub_task_results=sub_task_results,
            conflict_resolution=resolution,
        )
    return merged.files, resolution


async def _assemble_conflict_resolution(
    spec: GatherSpec,
    wt_output: CreateWorktreeOutput,
    conflicts: list[FileConflict],
    non_conflicting: dict[str, str],
) -> ConflictResolutionCallInput:
    """Assemble the conflict-resolution prompts for this gather's conflicts."""
    call_input: ConflictResolutionCallInput = await workflow.execute_activity(
        "assemble_conflict_resolution_context",
        ConflictResolutionInput(
            task_id=spec.task_id,
            step_id=spec.step_id,
            conflicts=conflicts,
            non_conflicting_files=non_conflicting,
            task_description=spec.task_description,
            step_description=spec.step_description,
            repo_root=spec.repo_root,
            worktree_path=wt_output.worktree_path,
            domain=spec.domain,
            model_name=resolve_model(CapabilityTier.REASONING, spec.model_routing),
            thinking=spec.thinking,
        ),
        start_to_close_timeout=_CONTEXT_TIMEOUT,
        retry_policy=IO_RETRY,
        result_type=ConflictResolutionCallInput,
    )
    return call_input.model_copy(
        update={"log_messages": spec.log_messages, "worktree_path": wt_output.worktree_path}
    )


async def _write_and_validate(
    spec: GatherSpec, wt_output: CreateWorktreeOutput, merged_files: dict[str, str]
) -> list[ValidationResult]:
    """Write the merged files into the worktree and run the deterministic checks.

    An empty merge writes nothing and validates nothing — there is no output to
    check, and the gather still succeeds.
    """
    if not merged_files:
        return []

    write_result: WriteResult = await workflow.execute_activity(
        "write_files",
        WriteFilesInput(
            task_id=spec.task_id,
            worktree_path=wt_output.worktree_path,
            files=merged_files,
        ),
        start_to_close_timeout=_WRITE_TIMEOUT,
        retry_policy=_WRITE_RETRY,
        result_type=WriteResult,
    )
    results: list[ValidationResult] = await workflow.execute_activity(
        "validate_output",
        ValidateOutputInput(
            task_id=spec.task_id,
            worktree_path=wt_output.worktree_path,
            files=write_result.files_written,
            validation=spec.validation,
        ),
        start_to_close_timeout=_VALIDATE_TIMEOUT,
        heartbeat_timeout=_VALIDATE_HEARTBEAT,
        retry_policy=IO_RETRY,
        result_type=list[ValidationResult],
    )
    return results


async def _create_worktree(spec: GatherSpec) -> CreateWorktreeOutput:
    """Create the worktree an *owned* gather runs in."""
    wt_output: CreateWorktreeOutput = await workflow.execute_activity(
        "create_worktree_activity",
        CreateWorktreeInput(
            repo_root=spec.repo_root,
            task_id=spec.task_id,
            base_branch=spec.base_branch,
        ),
        start_to_close_timeout=_GIT_TIMEOUT,
        retry_policy=_GIT_RETRY,
        result_type=CreateWorktreeOutput,
    )
    return wt_output


async def _commit(spec: GatherSpec) -> str:
    """Commit the merged output; returns the commit SHA."""
    commit_output: CommitChangesOutput = await workflow.execute_activity(
        "commit_changes_activity",
        CommitChangesInput(
            repo_root=spec.repo_root,
            task_id=spec.task_id,
            status="success",
            message=gather_commit_message(spec),
        ),
        start_to_close_timeout=_GIT_TIMEOUT,
        retry_policy=_GIT_RETRY,
        result_type=CommitChangesOutput,
    )
    return commit_output.commit_sha
