"""Temporal workflow for Forge task execution.

Orchestrates the five core activities and three git activities into a retry
loop that creates worktrees, calls the LLM, validates output, and transitions.

Temporal workflows must be deterministic — all I/O happens in activities.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from forge.models import (
        AssembleContextInput,
        AssembledContext,
        CommitChangesInput,
        CommitChangesOutput,
        CreateWorktreeInput,
        CreateWorktreeOutput,
        ForgeTaskInput,
        LLMCallResult,
        RemoveWorktreeInput,
        TaskResult,
        TransitionInput,
        TransitionSignal,
        ValidateOutputInput,
        ValidationResult,
        WriteOutputInput,
        WriteResult,
    )

FORGE_TASK_QUEUE = "forge-task-queue"

# ---------------------------------------------------------------------------
# Activity timeout presets
# ---------------------------------------------------------------------------

_GIT_TIMEOUT = timedelta(seconds=30)
_CONTEXT_TIMEOUT = timedelta(seconds=30)
_LLM_TIMEOUT = timedelta(minutes=5)
_WRITE_TIMEOUT = timedelta(seconds=30)
_VALIDATE_TIMEOUT = timedelta(minutes=2)
_TRANSITION_TIMEOUT = timedelta(seconds=10)


@workflow.defn
class ForgeTaskWorkflow:
    """Execute a single Forge task with retry support.

    Control flow:
        for attempt in 1..max_attempts:
            create_worktree
            assemble_context → call_llm → write_output → validate_output → evaluate_transition
            match signal:
                SUCCESS         → commit("success"), return TaskResult(SUCCESS)
                FAILURE_RETRYABLE → remove_worktree, continue loop
                FAILURE_TERMINAL  → commit("failure"), return TaskResult(FAILURE_TERMINAL)
    """

    @workflow.run
    async def run(self, input: ForgeTaskInput) -> TaskResult:
        task = input.task
        max_attempts = input.max_attempts

        for attempt in range(1, max_attempts + 1):
            # --- Create worktree ---
            wt_output = await workflow.execute_activity(
                "create_worktree_activity",
                CreateWorktreeInput(
                    repo_root=input.repo_root,
                    task_id=task.task_id,
                    base_branch=task.base_branch,
                ),
                start_to_close_timeout=_GIT_TIMEOUT,
                result_type=CreateWorktreeOutput,
            )

            # --- Assemble context ---
            context = await workflow.execute_activity(
                "assemble_context",
                AssembleContextInput(
                    task=task,
                    repo_root=input.repo_root,
                    worktree_path=wt_output.worktree_path,
                ),
                start_to_close_timeout=_CONTEXT_TIMEOUT,
                result_type=AssembledContext,
            )

            # --- Call LLM ---
            llm_result = await workflow.execute_activity(
                "call_llm",
                context,
                start_to_close_timeout=_LLM_TIMEOUT,
                result_type=LLMCallResult,
            )

            # --- Write output ---
            write_result = await workflow.execute_activity(
                "write_output",
                WriteOutputInput(
                    llm_result=llm_result,
                    worktree_path=wt_output.worktree_path,
                ),
                start_to_close_timeout=_WRITE_TIMEOUT,
                result_type=WriteResult,
            )

            # --- Validate output ---
            validation_results = await workflow.execute_activity(
                "validate_output",
                ValidateOutputInput(
                    task_id=task.task_id,
                    worktree_path=wt_output.worktree_path,
                    files=write_result.files_written,
                    validation=task.validation,
                ),
                start_to_close_timeout=_VALIDATE_TIMEOUT,
                result_type=list[ValidationResult],
            )

            # --- Evaluate transition ---
            signal_value = await workflow.execute_activity(
                "evaluate_transition",
                TransitionInput(
                    validation_results=validation_results,
                    attempt=attempt,
                    max_attempts=max_attempts,
                ),
                start_to_close_timeout=_TRANSITION_TIMEOUT,
                result_type=str,
            )
            signal = TransitionSignal(signal_value)

            # --- Collect output files ---
            output_files = {f.file_path: f.content for f in llm_result.response.files}

            # --- Act on signal ---
            if signal == TransitionSignal.SUCCESS:
                await workflow.execute_activity(
                    "commit_changes_activity",
                    CommitChangesInput(
                        repo_root=input.repo_root,
                        task_id=task.task_id,
                        status="success",
                    ),
                    start_to_close_timeout=_GIT_TIMEOUT,
                    result_type=CommitChangesOutput,
                )
                return TaskResult(
                    task_id=task.task_id,
                    status=TransitionSignal.SUCCESS,
                    output_files=output_files,
                    validation_results=validation_results,
                    worktree_path=wt_output.worktree_path,
                    worktree_branch=wt_output.branch_name,
                )

            if signal == TransitionSignal.FAILURE_RETRYABLE:
                await workflow.execute_activity(
                    "remove_worktree_activity",
                    RemoveWorktreeInput(
                        repo_root=input.repo_root,
                        task_id=task.task_id,
                        force=True,
                    ),
                    start_to_close_timeout=_GIT_TIMEOUT,
                    result_type=type(None),
                )
                continue

            # FAILURE_TERMINAL
            await workflow.execute_activity(
                "commit_changes_activity",
                CommitChangesInput(
                    repo_root=input.repo_root,
                    task_id=task.task_id,
                    status="failure",
                ),
                start_to_close_timeout=_GIT_TIMEOUT,
                result_type=CommitChangesOutput,
            )
            error = "; ".join(r.summary for r in validation_results if not r.passed)
            return TaskResult(
                task_id=task.task_id,
                status=TransitionSignal.FAILURE_TERMINAL,
                output_files=output_files,
                validation_results=validation_results,
                error=error,
                worktree_path=wt_output.worktree_path,
                worktree_branch=wt_output.branch_name,
            )

        # Should not be reachable, but satisfy the type checker.
        msg = "Exhausted all attempts without a terminal transition"
        raise RuntimeError(msg)
