"""Temporal workflow for Forge task execution.

Orchestrates the core activities and git activities into retry loops.

Phase 1 (plan=False): Single-step execution with task-level retry.
Phase 2 (plan=True): Planning + multi-step execution with step-level retry.

Temporal workflows must be deterministic — all I/O happens in activities.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from forge.models import (
        AssembleContextInput,
        AssembledContext,
        AssembleStepContextInput,
        CommitChangesInput,
        CommitChangesOutput,
        CreateWorktreeInput,
        CreateWorktreeOutput,
        ForgeTaskInput,
        LLMCallResult,
        Plan,
        PlanCallResult,
        PlannerInput,
        RemoveWorktreeInput,
        ResetWorktreeInput,
        StepResult,
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
    """Execute a Forge task with optional planning and multi-step execution.

    Dispatches between two paths:

    plan=False (Phase 1):
        for attempt in 1..max_attempts:
            create_worktree
            assemble_context → call_llm → write_output → validate_output → evaluate_transition
            match signal:
                SUCCESS         → commit("success"), return TaskResult(SUCCESS)
                FAILURE_RETRYABLE → remove_worktree, continue loop
                FAILURE_TERMINAL  → commit("failure"), return TaskResult(FAILURE_TERMINAL)

    plan=True (Phase 2):
        create_worktree (once)
        assemble_planner_context → call_planner → Plan
        for step in plan.steps:
            for attempt in 1..max_step_attempts:
                assemble_step_context → call_llm → write_output → validate → transition
                SUCCESS → commit(step), break to next step
                FAILURE_RETRYABLE → reset_worktree, continue retry
                FAILURE_TERMINAL → return TaskResult(FAILURE_TERMINAL)
        All steps done → return TaskResult(SUCCESS)
    """

    @workflow.run
    async def run(self, input: ForgeTaskInput) -> TaskResult:
        if input.plan:
            return await self._run_planned(input)
        return await self._run_single_step(input)

    # ------------------------------------------------------------------
    # Phase 1: Single-step execution (unchanged from Phase 1)
    # ------------------------------------------------------------------

    async def _run_single_step(self, input: ForgeTaskInput) -> TaskResult:
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

    # ------------------------------------------------------------------
    # Phase 2: Planned multi-step execution
    # ------------------------------------------------------------------

    async def _run_planned(self, input: ForgeTaskInput) -> TaskResult:
        task = input.task
        max_step_attempts = input.max_step_attempts

        # --- Create worktree (once) ---
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

        # --- Assemble planner context ---
        planner_input = await workflow.execute_activity(
            "assemble_planner_context",
            AssembleContextInput(
                task=task,
                repo_root=input.repo_root,
                worktree_path=wt_output.worktree_path,
            ),
            start_to_close_timeout=_CONTEXT_TIMEOUT,
            result_type=PlannerInput,
        )

        # --- Call planner ---
        planner_result = await workflow.execute_activity(
            "call_planner",
            planner_input,
            start_to_close_timeout=_LLM_TIMEOUT,
            result_type=PlanCallResult,
        )
        plan: Plan = planner_result.plan
        step_results: list[StepResult] = []
        all_output_files: dict[str, str] = {}

        # --- Execute steps sequentially ---
        for step_index, step in enumerate(plan.steps):
            step_succeeded = False

            for attempt in range(1, max_step_attempts + 1):
                # --- Assemble step context ---
                context = await workflow.execute_activity(
                    "assemble_step_context",
                    AssembleStepContextInput(
                        task=task,
                        step=step,
                        step_index=step_index,
                        total_steps=len(plan.steps),
                        completed_steps=step_results,
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
                        max_attempts=max_step_attempts,
                    ),
                    start_to_close_timeout=_TRANSITION_TIMEOUT,
                    result_type=str,
                )
                signal = TransitionSignal(signal_value)

                output_files = {f.file_path: f.content for f in llm_result.response.files}

                if signal == TransitionSignal.SUCCESS:
                    # Commit this step's changes
                    commit_msg = f"forge({task.task_id}): step {step.step_id} success"
                    commit_output = await workflow.execute_activity(
                        "commit_changes_activity",
                        CommitChangesInput(
                            repo_root=input.repo_root,
                            task_id=task.task_id,
                            status="success",
                            message=commit_msg,
                        ),
                        start_to_close_timeout=_GIT_TIMEOUT,
                        result_type=CommitChangesOutput,
                    )
                    step_results.append(
                        StepResult(
                            step_id=step.step_id,
                            status=TransitionSignal.SUCCESS,
                            output_files=output_files,
                            validation_results=validation_results,
                            commit_sha=commit_output.commit_sha,
                        )
                    )
                    all_output_files.update(output_files)
                    step_succeeded = True
                    break

                if signal == TransitionSignal.FAILURE_RETRYABLE:
                    # Reset worktree (discard uncommitted changes) and retry
                    await workflow.execute_activity(
                        "reset_worktree_activity",
                        ResetWorktreeInput(
                            repo_root=input.repo_root,
                            task_id=task.task_id,
                        ),
                        start_to_close_timeout=_GIT_TIMEOUT,
                        result_type=type(None),
                    )
                    continue

                # FAILURE_TERMINAL — step failed, task fails
                error = "; ".join(r.summary for r in validation_results if not r.passed)
                step_results.append(
                    StepResult(
                        step_id=step.step_id,
                        status=TransitionSignal.FAILURE_TERMINAL,
                        output_files=output_files,
                        validation_results=validation_results,
                        error=error,
                    )
                )
                return TaskResult(
                    task_id=task.task_id,
                    status=TransitionSignal.FAILURE_TERMINAL,
                    output_files=all_output_files,
                    validation_results=validation_results,
                    error=f"Step {step.step_id} failed: {error}",
                    worktree_path=wt_output.worktree_path,
                    worktree_branch=wt_output.branch_name,
                    step_results=step_results,
                    plan=plan,
                )

            if not step_succeeded:
                # Exhausted retries for this step — should have hit TERMINAL above
                msg = f"Step {step.step_id} exhausted all attempts without a terminal transition"
                raise RuntimeError(msg)

        # --- All steps succeeded ---
        return TaskResult(
            task_id=task.task_id,
            status=TransitionSignal.SUCCESS,
            output_files=all_output_files,
            worktree_path=wt_output.worktree_path,
            worktree_branch=wt_output.branch_name,
            step_results=step_results,
            plan=plan,
        )
