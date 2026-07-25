"""The single step block (T5.2) — one copy of the universal step pipeline.

Every forge step runs the same five moves: assemble the context, call the LLM,
write the result, validate it, act on the transition. This module owns the one
implementation. ``forge.workflows`` drives it three times with different
:class:`StepSpec` values instead of maintaining three hand-synchronized copies
(the T1.5 nested fan-out bug was divergence bred by exactly that duplication).

Only three things differ between the modes, and they live in one pure table
(:data:`MODE_POLICIES`): which assemble activity runs, who owns the worktree,
and what a successful attempt commits. Three legal rows, not twenty-seven
implicit combinations.

Worktree ownership also decides cleanup. The two *fresh* modes create the
worktree they run in, so the block wraps their attempt loop: any exception
removes the worktree **and** its ``forge/<task_id>`` branch before re-raising.
That is what stops an ordinary transient from leaving debris behind — before
T5.2 nothing in ``workflows.py`` had a ``finally``, and the leftovers made the
next run of the same task id fail permanently. The *borrowed* mode's worktree
belongs to the caller (``ForgeTaskWorkflow._run_planned``), which carries the
same wrap around the whole plan. The git seams are idempotent since T5.2's
first three sub-tasks, so a repeated cleanup is a no-op rather than an error.

Batch-wait failures are the one deliberate exception: both workflow ``run()``
methods already catch them (T1.6b), clean the same worktree, and record a
terminal result, so the block re-raises them untouched rather than cleaning the
same worktree twice — which also keeps the committed replay histories
(``single_step_batch_ceiling``) valid.

Transition evaluation stays pure and inlined (``step_logic.determine_transition``,
D95); the block only sequences activities around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from sax_platform.temporal.retries import IO_RETRY

    from forge.models import (
        AssembleContextInput,
        AssembledContext,
        AssembleStepContextInput,
        AssembleSubTaskContextInput,
        CommitChangesInput,
        CommitChangesOutput,
        ContextStats,
        CreateWorktreeInput,
        CreateWorktreeOutput,
        LLMCallResult,
        ResetWorktreeInput,
        TransitionSignal,
        ValidateOutputInput,
        ValidationConfig,
        ValidationResult,
        WriteOutputInput,
        WriteResult,
    )
    from forge.step_logic import determine_transition
    from forge.workflow_blocks import BATCH_WAIT_FAILURES
    from forge.workflow_blocks import remove_worktree as _remove_worktree

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

__all__ = [
    "MODE_POLICIES",
    "ExplorationHook",
    "ModePolicy",
    "StepAttemptOutcome",
    "StepHost",
    "StepMode",
    "StepSpec",
    "cleanup_worktree_after_exception",
    "run_step_attempts",
    "stamp_context",
]


# ---------------------------------------------------------------------------
# Activity timeout and retry presets
#
# Duplicated from workflows.py on purpose while the gather paths still live
# there (T5.3 moves them into blocks/gather.py; T5.4 splits the monolith).
# Both copies must stay identical until one of those tasks retires the other.
# ---------------------------------------------------------------------------

_GIT_TIMEOUT = timedelta(seconds=30)
_CONTEXT_TIMEOUT = timedelta(seconds=30)
_WRITE_TIMEOUT = timedelta(seconds=30)
_VALIDATE_TIMEOUT = timedelta(minutes=2)
_VALIDATE_HEARTBEAT = timedelta(seconds=60)

_GIT_RETRY = RetryPolicy(
    maximum_attempts=2,
    non_retryable_error_types=["CommitError", "RepoDiscoveryError"],
)
_WRITE_RETRY = RetryPolicy(
    maximum_attempts=2,
    non_retryable_error_types=["OutputWriteError", "EditApplicationError"],
)


# ---------------------------------------------------------------------------
# The mode table (pure)
# ---------------------------------------------------------------------------

type StepMode = Literal["single_step", "planned_step", "sub_task"]

type WorktreePolicy = Literal["fresh-keep", "borrowed", "fresh-dispose"]
"""Who owns the worktree an attempt runs in.

- ``fresh-keep`` — the block creates one per attempt, removes it on retry, and
  leaves the last one in place for inspection (single-step tasks).
- ``borrowed`` — the caller created it and keeps it; the block only resets it
  between attempts (planned steps).
- ``fresh-dispose`` — the block creates one per attempt and removes it at the
  end of every attempt, success included (sub-tasks never commit, D16).
"""

type CommitPolicy = Literal["task", "step", "never"]
"""What a finished attempt commits: a task-level commit (success *and* the
terminal-failure autopsy commit), a step-scoped commit on success only, or
nothing at all."""

type AssembleInput = AssembleContextInput | AssembleStepContextInput | AssembleSubTaskContextInput

type ExplorationHook = Callable[[AssembledContext, str], Awaitable[AssembledContext]]
"""``(context, worktree_path) -> context``, run per attempt against that
attempt's own worktree."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModePolicy:
    """The three things that actually differ between the step modes."""

    assemble_activity: str
    worktree: WorktreePolicy
    commit: CommitPolicy


MODE_POLICIES: Mapping[StepMode, ModePolicy] = MappingProxyType(
    {
        "single_step": ModePolicy(
            assemble_activity="assemble_context",
            worktree="fresh-keep",
            commit="task",
        ),
        "planned_step": ModePolicy(
            assemble_activity="assemble_step_context",
            worktree="borrowed",
            commit="step",
        ),
        "sub_task": ModePolicy(
            assemble_activity="assemble_sub_task_context",
            worktree="fresh-dispose",
            commit="never",
        ),
    }
)


# ---------------------------------------------------------------------------
# Spec, host, and outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class StepSpec:
    """Everything one step needs, independent of which workflow drives it.

    Lives only inside workflow code — it is never serialized, so it may carry
    activity-input models and a borrowed worktree handle directly.
    """

    mode: StepMode
    task_id: str
    """The git/validate identity: a task id, or a sub-task's compound id."""

    repo_root: str
    base_branch: str
    assemble_input: AssembleInput
    """The attempt-1 assemble input; each attempt copies it with that attempt's
    ``prior_errors`` / ``attempt`` / ``worktree_path``."""

    max_attempts: int
    validation: ValidationConfig
    model_name: str = ""
    """Model to stamp on the assembled context; ``""`` means don't stamp."""

    log_messages: bool = False
    commit_message: str | None = None
    """Commit-message override (planned steps); ``None`` uses the default."""

    exploration_rounds: int = 0
    borrowed_worktree: CreateWorktreeOutput | None = None
    """The caller's worktree — required by ``borrowed`` mode, unused otherwise."""

    def __post_init__(self) -> None:
        """Reject a spec whose worktree ownership and handle disagree.

        A borrowed mode without a handle would make the block create a worktree
        the caller believes it owns, and ``create_worktree`` resets the branch
        (``worktree add -B``), discarding committed step work.
        """
        borrowed = MODE_POLICIES[self.mode].worktree == "borrowed"
        if borrowed and self.borrowed_worktree is None:
            msg = f"mode {self.mode!r} borrows its worktree but no borrowed_worktree was given"
            raise ValueError(msg)
        if not borrowed and self.borrowed_worktree is not None:
            msg = f"mode {self.mode!r} creates its own worktree; borrowed_worktree must be None"
            raise ValueError(msg)


class StepHost(Protocol):
    """The workflow instance's persisting LLM dispatch.

    Both workflow classes expose this method identically; the block stays free
    of the per-workflow state (sync mode, poll interval, persist counters) it
    hides.
    """

    async def call_generation(self, context: AssembledContext) -> LLMCallResult: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class StepAttemptOutcome:
    """The neutral result of running a step to a terminal signal.

    Mode-specific results (TaskResult / StepResult / SubTaskResult) are built by
    the callers from this.
    """

    signal: TransitionSignal
    output_files: dict[str, str]
    validation_results: list[ValidationResult]
    llm_result: LLMCallResult
    context_stats: ContextStats | None
    worktree_path: str
    worktree_branch: str
    commit_sha: str = ""


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


async def run_step_attempts(
    spec: StepSpec,
    host: StepHost,
    explore: ExplorationHook | None = None,
) -> StepAttemptOutcome:
    """Run one step's attempt loop to a terminal signal.

    Retries rebuild the assemble input with the previous attempt's validation
    errors; exploration (when the caller supplies a hook) runs inside the loop
    so it sees that attempt's own worktree.

    Raises:
        Exception: whatever the underlying activities raise, after cleaning up a
            worktree the block owns.
    """
    policy = MODE_POLICIES[spec.mode]
    if policy.worktree == "borrowed":
        # The caller owns this worktree and carries the cleanup wrap.
        return await _attempt_loop(spec, policy, host, explore)

    try:
        return await _attempt_loop(spec, policy, host, explore)
    except Exception as exc:
        # A batch wait that failed is already handled by the workflow's run():
        # it cleans this same worktree and records a terminal result, so
        # cleaning here would only duplicate the removal.
        if not isinstance(exc, BATCH_WAIT_FAILURES):
            await cleanup_worktree_after_exception(spec.repo_root, spec.task_id, exc)
        raise


async def _attempt_loop(
    spec: StepSpec,
    policy: ModePolicy,
    host: StepHost,
    explore: ExplorationHook | None,
) -> StepAttemptOutcome:
    """Assemble → LLM → write → validate → act, until a terminal signal."""
    prior_errors: list[ValidationResult] = []

    for attempt in range(1, spec.max_attempts + 1):
        workflow.logger.info(
            "Step attempt %d/%d: mode=%s id=%s", attempt, spec.max_attempts, spec.mode, spec.task_id
        )
        wt_output = await _acquire_worktree(spec, policy)

        context = await _assemble(
            spec,
            policy,
            attempt=attempt,
            prior_errors=prior_errors,
            worktree_path=wt_output.worktree_path,
        )

        if explore is not None and spec.exploration_rounds > 0:
            context = await explore(context, wt_output.worktree_path)

        context = stamp_context(spec, context, wt_output.worktree_path)
        llm_result = await host.call_generation(context)
        write_result = await _write_output(llm_result, wt_output.worktree_path)
        validation_results = await _validate_output(spec, write_result, wt_output.worktree_path)

        # --- Evaluate transition inline (deterministic; D95) ---
        signal = determine_transition(validation_results, attempt, spec.max_attempts)
        workflow.logger.info(
            "Step transition: mode=%s id=%s signal=%s attempt=%d/%d",
            spec.mode,
            spec.task_id,
            signal.value,
            attempt,
            spec.max_attempts,
        )

        commit_sha = await _act_on_signal(spec, policy, signal)

        if signal == TransitionSignal.FAILURE_RETRYABLE:
            prior_errors = validation_results
            continue

        return StepAttemptOutcome(
            signal=signal,
            output_files=write_result.output_files,
            validation_results=validation_results,
            llm_result=llm_result,
            context_stats=context.context_stats,
            worktree_path=wt_output.worktree_path,
            worktree_branch=wt_output.branch_name,
            commit_sha=commit_sha,
        )

    # Should not be reachable, but satisfy the type checker.
    msg = f"Step {spec.task_id} exhausted all attempts without a terminal transition"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


async def _acquire_worktree(spec: StepSpec, policy: ModePolicy) -> CreateWorktreeOutput:
    """Return the attempt's worktree: the borrowed one, or a freshly created one."""
    if policy.worktree == "borrowed":
        assert spec.borrowed_worktree is not None  # StepSpec.__post_init__ guarantees this
        return spec.borrowed_worktree

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


async def _assemble(
    spec: StepSpec,
    policy: ModePolicy,
    *,
    attempt: int,
    prior_errors: list[ValidationResult],
    worktree_path: str,
) -> AssembledContext:
    """Run this mode's assemble activity for one attempt."""
    assemble_input = spec.assemble_input.model_copy(
        update={
            "prior_errors": prior_errors,
            "attempt": attempt,
            "worktree_path": worktree_path,
        }
    )
    context: AssembledContext = await workflow.execute_activity(
        policy.assemble_activity,
        assemble_input,
        start_to_close_timeout=_CONTEXT_TIMEOUT,
        retry_policy=IO_RETRY,
        result_type=AssembledContext,
    )
    return context


def stamp_context(
    spec: StepSpec, context: AssembledContext, worktree_path: str
) -> AssembledContext:
    """Stamp the dispatch fields the assemble activities do not set (pure).

    An empty ``model_name`` leaves the context's own value alone — a sub-task
    inherits the parent's model only when the parent chose one.
    """
    update: dict[str, object] = {
        "log_messages": spec.log_messages,
        "worktree_path": worktree_path,
    }
    if spec.model_name:
        update["model_name"] = spec.model_name
    return context.model_copy(update=update)


async def _write_output(llm_result: LLMCallResult, worktree_path: str) -> WriteResult:
    """Write the LLM's files into the attempt's worktree."""
    write_result: WriteResult = await workflow.execute_activity(
        "write_output",
        WriteOutputInput(llm_result=llm_result, worktree_path=worktree_path),
        start_to_close_timeout=_WRITE_TIMEOUT,
        retry_policy=_WRITE_RETRY,
        result_type=WriteResult,
    )
    return write_result


async def _validate_output(
    spec: StepSpec, write_result: WriteResult, worktree_path: str
) -> list[ValidationResult]:
    """Run the deterministic checks over what was just written."""
    results: list[ValidationResult] = await workflow.execute_activity(
        "validate_output",
        ValidateOutputInput(
            task_id=spec.task_id,
            worktree_path=worktree_path,
            files=write_result.files_written,
            validation=spec.validation,
        ),
        start_to_close_timeout=_VALIDATE_TIMEOUT,
        heartbeat_timeout=_VALIDATE_HEARTBEAT,
        retry_policy=IO_RETRY,
        result_type=list[ValidationResult],
    )
    return results


async def _act_on_signal(spec: StepSpec, policy: ModePolicy, signal: TransitionSignal) -> str:
    """Apply the mode's worktree and commit policy to this attempt's signal.

    Returns the commit SHA when the attempt committed, otherwise ``""``.
    """
    if policy.worktree == "fresh-dispose":
        # Sub-tasks never commit and dispose their worktree at the end of every
        # attempt, success included (D16).
        await _remove_worktree(spec.repo_root, spec.task_id)
        return ""

    if signal == TransitionSignal.SUCCESS:
        return await _commit(spec, status="success")

    if signal == TransitionSignal.FAILURE_RETRYABLE:
        if policy.worktree == "fresh-keep":
            await _remove_worktree(spec.repo_root, spec.task_id)
        else:
            await _reset_worktree(spec)
        return ""

    # FAILURE_TERMINAL: a task-level commit records the failed attempt so the
    # worktree it leaves behind is inspectable; a step-scoped one commits
    # nothing and lets the driver fail the task.
    if policy.commit == "task":
        await _commit(spec, status="failure")
    return ""


async def _commit(spec: StepSpec, *, status: str) -> str:
    """Commit the worktree's changes; returns the commit SHA."""
    commit_output: CommitChangesOutput = await workflow.execute_activity(
        "commit_changes_activity",
        CommitChangesInput(
            repo_root=spec.repo_root,
            task_id=spec.task_id,
            status=status,
            message=spec.commit_message,
        ),
        start_to_close_timeout=_GIT_TIMEOUT,
        retry_policy=_GIT_RETRY,
        result_type=CommitChangesOutput,
    )
    return commit_output.commit_sha


async def _reset_worktree(spec: StepSpec) -> None:
    """Discard a failed attempt's uncommitted changes in a borrowed worktree."""
    await workflow.execute_activity(
        "reset_worktree_activity",
        ResetWorktreeInput(repo_root=spec.repo_root, task_id=spec.task_id),
        start_to_close_timeout=_GIT_TIMEOUT,
        retry_policy=_GIT_RETRY,
        result_type=type(None),
    )


# ---------------------------------------------------------------------------
# Ownership cleanup
# ---------------------------------------------------------------------------


async def cleanup_worktree_after_exception(
    repo_root: str, task_id: str, exc: BaseException
) -> None:
    """Remove a worktree and its branch after an exception; never masks *exc*.

    Removal is forced, so the activity deletes the ``forge/<task_id>`` branch
    too — the leak that used to make the next run of the same task id fail on
    ``worktree add``. Only ``ActivityError`` is swallowed (``CancelledError``
    must still propagate), so a cleanup blip cannot replace the real failure.
    """
    workflow.logger.warning("Step failed; cleaning worktree task_id=%s: %r", task_id, exc)
    try:
        await _remove_worktree(repo_root, task_id)
    except ActivityError as cleanup_exc:
        workflow.logger.warning(
            "Worktree cleanup after step failure did not complete: task_id=%s: %r",
            task_id,
            cleanup_exc,
        )
