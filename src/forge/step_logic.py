"""Pure step logic for Forge workflows (T5.1, D95).

This module holds the deterministic "what happens next?" decisions that used to
be scattered inside ``forge.workflows`` — transition evaluation (formerly the
``evaluate_transition`` Temporal activity), failure-summary assembly, result
builders, merge-resolution handling, output slimming, run-total aggregation, and
the child-id/timeout helpers.

It has **zero** ``temporalio`` imports by design: every function here is pure and
deterministic, so a workflow can call it inline (Temporal replay reproduces the
same answer) and the whole decision surface gets exhaustive microsecond unit
tests without booting a Temporal server. ``imports_passed_through`` in
``forge.workflows`` pulls this module into the workflow sandbox.

Slimming (T5.1, the 2026-07-08 "contents once" sweep): file *contents* travel at
most once — successful steps' output is folded into the top-level
``TaskResult.output_files``; every embedded Step/SubTaskResult keeps paths +
sha256 digests (``output_digests``) instead, so a large fan-out's final result
cannot multiply content 3-4x and blow Temporal's ~2MB payload cap at the finish
line. The fan-out builders slim the children and conflict resolution they embed;
:func:`slim_result` slims a whole node at embed time and is a cheap no-op on
anything already slim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from forge.models import (
    BATCH_WAIT_CEILING,
    ConflictResolutionCallResult,
    ContextStats,
    FailureKind,
    FileConflict,
    LLMRunTotals,
    LLMStats,
    Plan,
    StepResult,
    SubTaskResult,
    TaskResult,
    TransitionSignal,
    ValidationResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

# ---------------------------------------------------------------------------
# Transition evaluation (moved verbatim from activities/transition.py)
# ---------------------------------------------------------------------------


def determine_transition(
    results: list[ValidationResult],
    attempt: int,
    max_attempts: int = 2,
) -> TransitionSignal:
    """Decide the workflow transition based on validation outcomes.

    - All passed (or empty) → SUCCESS
    - Any failed + attempt < max_attempts → FAILURE_RETRYABLE
    - Any failed + attempt >= max_attempts → FAILURE_TERMINAL
    """
    all_passed = all(r.passed for r in results)
    if all_passed:
        return TransitionSignal.SUCCESS
    if attempt < max_attempts:
        return TransitionSignal.FAILURE_RETRYABLE
    return TransitionSignal.FAILURE_TERMINAL


# ---------------------------------------------------------------------------
# Failure-summary helpers
# ---------------------------------------------------------------------------


def failure_summary(validation_results: Sequence[ValidationResult]) -> str:
    """Join the summaries of the failing validation results."""
    return "; ".join(r.summary for r in validation_results if not r.passed)


def subtask_failure_summary(failed: Sequence[SubTaskResult]) -> str:
    """Join ``{sub_task_id}: {error}`` for failed children of a fan-out gather."""
    return "; ".join(f"{r.sub_task_id}: {r.error}" for r in failed)


# ---------------------------------------------------------------------------
# Merge-resolution union (replaces the two inline conflict-merge blocks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class MergedFiles:
    """All conflicting paths were resolved; ``files`` is the merged file map."""

    files: dict[str, str]


@dataclass(frozen=True, slots=True, kw_only=True)
class MissingResolutions:
    """Resolution left some conflicting paths unresolved (sorted)."""

    missing: tuple[str, ...]

    @property
    def message(self) -> str:
        """The terminal-error wording for an incomplete resolution."""
        return f"Conflict resolution incomplete: missing resolved files: {', '.join(self.missing)}"


def merge_resolution(
    conflicts: Sequence[FileConflict],
    resolved: Mapping[str, str],
    non_conflicting: Mapping[str, str],
) -> MergedFiles | MissingResolutions:
    """Fold resolved conflict output over the non-conflicting files.

    Returns :class:`MissingResolutions` (sorted paths) when the resolver did not
    return every conflicting path, otherwise :class:`MergedFiles` with the
    non-conflicting files overlaid by the resolved ones.
    """
    conflict_paths = {c.file_path for c in conflicts}
    missing = conflict_paths - set(resolved)
    if missing:
        return MissingResolutions(missing=tuple(sorted(missing)))
    return MergedFiles(files={**non_conflicting, **resolved})


# ---------------------------------------------------------------------------
# Output slimming — file contents travel at most once
# ---------------------------------------------------------------------------


def file_digests(files: Mapping[str, str]) -> dict[str, str]:
    """Map each path to the sha256 hex of its content (deterministic)."""
    return {
        path: hashlib.sha256(content.encode("utf-8")).hexdigest() for path, content in files.items()
    }


def _slim_conflict_resolution(
    conflict_resolution: ConflictResolutionCallResult | None,
) -> ConflictResolutionCallResult | None:
    """Drop a conflict resolution's ``resolved_files`` contents, keep its LLM stats."""
    if conflict_resolution is None:
        return None
    return conflict_resolution.model_copy(update={"resolved_files": {}})


def _is_slim(result: StepResult | SubTaskResult) -> bool:
    """True when the node (and its whole subtree) carries no file contents."""
    return (
        not result.output_files
        and (result.conflict_resolution is None or not result.conflict_resolution.resolved_files)
        and all(_is_slim(child) for child in result.sub_task_results)
    )


def slim_result[R: (StepResult, SubTaskResult)](result: R) -> R:
    """Return a copy carrying digests instead of content (idempotent, recursive).

    Empties ``output_files`` into ``output_digests``, recurses into embedded
    children, and drops ``conflict_resolution.resolved_files`` (keeping its
    stats). A node that is already slim — including children the fan-out
    builders slimmed at embed time — is returned unchanged, not re-copied.
    """
    if _is_slim(result):
        return result
    digests = file_digests(result.output_files) if result.output_files else result.output_digests
    return result.model_copy(
        update={
            "output_files": {},
            "output_digests": digests,
            "sub_task_results": _slim_children(result.sub_task_results),
            "conflict_resolution": _slim_conflict_resolution(result.conflict_resolution),
        }
    )


def _slim_children(children: Sequence[SubTaskResult]) -> list[SubTaskResult]:
    """Slim every embedded child (the shared fan-out embed recipe)."""
    return [slim_result(child) for child in children]


# ---------------------------------------------------------------------------
# Run-total aggregation (D97: run-level LLM spend on TaskResult)
# ---------------------------------------------------------------------------


def _iter_node_stats(node: StepResult | SubTaskResult) -> Iterator[LLMStats]:
    if node.llm_stats is not None:
        yield node.llm_stats
    if node.conflict_resolution is not None:
        yield node.conflict_resolution
    for child in node.sub_task_results:
        yield from _iter_node_stats(child)


def _iter_task_stats(task_result: TaskResult) -> Iterator[LLMStats]:
    if task_result.planner_stats is not None:
        yield task_result.planner_stats
    if task_result.llm_stats is not None:
        yield task_result.llm_stats
    for step in task_result.step_results:
        yield from _iter_node_stats(step)


def llm_totals(task_result: TaskResult) -> LLMRunTotals:
    """Aggregate every surviving LLM call in the finished result tree (D97).

    Walks planner + single-step + per-step + per-sub-task + conflict-resolution
    stats. Slimming keeps every node's stats, so this is stable whether computed
    before or after slimming. See :class:`LLMRunTotals` for the known limitation
    (retried attempts are not in the tree).
    """
    return LLMRunTotals.from_stats(_iter_task_stats(task_result))


# ---------------------------------------------------------------------------
# Child-workflow id and timeout helpers
# ---------------------------------------------------------------------------


def compound_sub_task_id(parent_task_id: str, sub_task_id: str) -> str:
    """The dotted compound id for a sub-task worktree/child (``{parent}.sub.{id}``)."""
    return f"{parent_task_id}.sub.{sub_task_id}"


def sub_task_workflow_id(compound_id: str) -> str:
    """The Temporal child-workflow id for a compound sub-task id."""
    return f"forge-subtask-{compound_id}"


_CHILD_BASE_MINUTES = 15
_CHILD_OVERHEAD_MINUTES_PER_LEVEL = 5


def child_timeout(depth: int, max_depth: int, *, sync_mode: bool, max_attempts: int) -> timedelta:
    """Execution timeout for a spawned child workflow, derived from its wait budget.

    ``remaining = max_depth - depth`` is the number of nesting levels below the child.
    The orchestration margin (non-batch git/context/write/validate activity time
    accumulated across the child's subtree) is the pre-T4.1 sync formula:
    ``15 + 5*remaining`` minutes.

    - **sync mode** → the orchestration margin alone; no batch waits. Unchanged from
      pre-T4.1 behavior.
    - **batch mode** → the child performs up to ``max_attempts`` sequential generation
      waits at a leaf, and each nesting level below it adds one conflict-resolution wait
      after its children gather, so the child's own sequential batch-wait budget is
      ``max_attempts + remaining`` waits, each bounded by the 25h ``BATCH_WAIT_CEILING``
      (T4.1 ST3c — closes the timeout tree so a slow batch no longer kills the child)::

          (max_attempts + remaining) * BATCH_WAIT_CEILING  +  (15 + 5*remaining) min
    """
    remaining = max_depth - depth
    orchestration = timedelta(
        minutes=_CHILD_BASE_MINUTES + _CHILD_OVERHEAD_MINUTES_PER_LEVEL * remaining
    )
    if sync_mode:
        return orchestration
    waits = max_attempts + remaining
    return waits * BATCH_WAIT_CEILING + orchestration


# ---------------------------------------------------------------------------
# TaskResult builders
#
# Only constructions that compute something live here (failure_kind stamping,
# failure-summary wording, embed-time slimming). Success results with no logic
# are constructed inline at the workflow call sites.
# ---------------------------------------------------------------------------


def single_step_terminal(
    *,
    task_id: str,
    output_files: dict[str, str],
    validation_results: list[ValidationResult],
    worktree_path: str,
    worktree_branch: str,
    llm_stats: LLMStats,
    context_stats: ContextStats | None,
) -> TaskResult:
    """Terminal validation-failure TaskResult for the single-step path."""
    return TaskResult(
        task_id=task_id,
        status=TransitionSignal.FAILURE_TERMINAL,
        output_files=output_files,
        validation_results=validation_results,
        error=failure_summary(validation_results),
        failure_kind="validation",
        worktree_path=worktree_path,
        worktree_branch=worktree_branch,
        llm_stats=llm_stats,
        context_stats=context_stats,
    )


def task_batch_wait_failure(*, task_id: str, exc: BaseException) -> TaskResult:
    """Terminal TaskResult when the batch wait failed (25h ceiling / provider / fetch)."""
    return TaskResult(
        task_id=task_id,
        status=TransitionSignal.FAILURE_TERMINAL,
        error=f"Batch wait failed: {type(exc).__name__}: {exc}",
        failure_kind="batch_wait",
    )


def planned_failure(
    *,
    task_id: str,
    failure_kind: FailureKind,
    error: str,
    output_files: dict[str, str],
    worktree_path: str,
    worktree_branch: str,
    step_results: list[StepResult],
    plan: Plan,
    planner_stats: LLMStats,
    sanity_check_count: int,
) -> TaskResult:
    """Terminal TaskResult for the planned driver (step failure or sanity abort).

    The caller supplies the ``failure_kind`` and the already-worded ``error``,
    matching the fan-out builders.
    """
    return TaskResult(
        task_id=task_id,
        status=TransitionSignal.FAILURE_TERMINAL,
        output_files=output_files,
        error=error,
        failure_kind=failure_kind,
        worktree_path=worktree_path,
        worktree_branch=worktree_branch,
        step_results=step_results,
        plan=plan,
        planner_stats=planner_stats,
        sanity_check_count=sanity_check_count,
    )


# ---------------------------------------------------------------------------
# StepResult builders
# ---------------------------------------------------------------------------


def step_terminal(
    *,
    step_id: str,
    output_files: dict[str, str],
    validation_results: list[ValidationResult],
    llm_stats: LLMStats,
) -> StepResult:
    """Terminal validation-failure StepResult for a regular planned step."""
    return StepResult(
        step_id=step_id,
        status=TransitionSignal.FAILURE_TERMINAL,
        output_files=output_files,
        validation_results=validation_results,
        error=failure_summary(validation_results),
        failure_kind="validation",
        llm_stats=llm_stats,
    )


def fan_out_success(
    *,
    step_id: str,
    output_files: dict[str, str],
    validation_results: list[ValidationResult],
    commit_sha: str,
    sub_task_results: Sequence[SubTaskResult],
    conflict_resolution: ConflictResolutionCallResult | None,
) -> StepResult:
    """SUCCESS StepResult for a fan-out step.

    Embedded children and conflict resolution are slimmed; the step keeps its
    own merged output for the driver to fold into the top-level map.
    """
    return StepResult(
        step_id=step_id,
        status=TransitionSignal.SUCCESS,
        output_files=output_files,
        validation_results=validation_results,
        commit_sha=commit_sha,
        sub_task_results=_slim_children(sub_task_results),
        conflict_resolution=_slim_conflict_resolution(conflict_resolution),
    )


def fan_out_step_failure(
    *,
    step_id: str,
    failure_kind: FailureKind,
    error: str,
    sub_task_results: Sequence[SubTaskResult] = (),
    output_files: dict[str, str] | None = None,
    validation_results: list[ValidationResult] | None = None,
    conflict_resolution: ConflictResolutionCallResult | None = None,
) -> StepResult:
    """Terminal StepResult for any fan-out failure (embedded children slimmed).

    Covers duplicate ids, failed children, incomplete/unresolved conflicts, and
    merged-validation failure — the caller supplies the ``failure_kind`` and the
    already-worded ``error``.
    """
    return StepResult(
        step_id=step_id,
        status=TransitionSignal.FAILURE_TERMINAL,
        output_files=output_files or {},
        validation_results=validation_results or [],
        error=error,
        failure_kind=failure_kind,
        sub_task_results=_slim_children(sub_task_results),
        conflict_resolution=_slim_conflict_resolution(conflict_resolution),
    )


# ---------------------------------------------------------------------------
# SubTaskResult builders
# ---------------------------------------------------------------------------


def sub_task_terminal(
    *,
    sub_task_id: str,
    validation_results: list[ValidationResult],
    llm_stats: LLMStats,
) -> SubTaskResult:
    """Terminal validation-failure SubTaskResult for a leaf sub-task."""
    return SubTaskResult(
        sub_task_id=sub_task_id,
        status=TransitionSignal.FAILURE_TERMINAL,
        validation_results=validation_results,
        error=failure_summary(validation_results),
        failure_kind="validation",
        llm_stats=llm_stats,
    )


def sub_task_batch_wait_failure(*, sub_task_id: str, exc: BaseException) -> SubTaskResult:
    """Terminal SubTaskResult when this node's batch wait failed."""
    return SubTaskResult(
        sub_task_id=sub_task_id,
        status=TransitionSignal.FAILURE_TERMINAL,
        error=f"Batch wait failed: {type(exc).__name__}: {exc}",
        failure_kind="batch_wait",
    )


def nested_fan_out_success(
    *,
    sub_task_id: str,
    output_files: dict[str, str],
    validation_results: list[ValidationResult],
    sub_task_results: Sequence[SubTaskResult],
    conflict_resolution: ConflictResolutionCallResult | None,
) -> SubTaskResult:
    """SUCCESS SubTaskResult for a nested fan-out node.

    Embedded children and conflict resolution are slimmed *before* the result
    crosses the child→parent workflow boundary — resolved contents already
    travel in the merged ``output_files``, so keeping them in
    ``conflict_resolution`` would double them in both workflows' histories.
    """
    return SubTaskResult(
        sub_task_id=sub_task_id,
        status=TransitionSignal.SUCCESS,
        output_files=output_files,
        validation_results=validation_results,
        sub_task_results=_slim_children(sub_task_results),
        conflict_resolution=_slim_conflict_resolution(conflict_resolution),
    )


def nested_fan_out_failure(
    *,
    sub_task_id: str,
    failure_kind: FailureKind,
    error: str,
    sub_task_results: Sequence[SubTaskResult] = (),
    output_files: dict[str, str] | None = None,
    validation_results: list[ValidationResult] | None = None,
    conflict_resolution: ConflictResolutionCallResult | None = None,
) -> SubTaskResult:
    """Terminal SubTaskResult for any nested fan-out failure (children slimmed)."""
    return SubTaskResult(
        sub_task_id=sub_task_id,
        status=TransitionSignal.FAILURE_TERMINAL,
        output_files=output_files or {},
        validation_results=validation_results or [],
        error=error,
        failure_kind=failure_kind,
        sub_task_results=_slim_children(sub_task_results),
        conflict_resolution=_slim_conflict_resolution(conflict_resolution),
    )
