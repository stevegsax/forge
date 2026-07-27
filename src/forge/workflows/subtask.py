"""The sub-task child workflow — one node of a fan-out tree.

``ForgeSubTaskWorkflow`` is the counterpart driver to ``ForgeTaskWorkflow``: it
routes between the two shapes a fan-out node can take (a leaf that generates, or
a nested fan-out that gathers its own children) and returns a ``SubTaskResult``
to whichever gather started it. Both shapes run the same shared blocks the root
workflow uses; a sub-task never plans, never runs a sanity check, and never
explores.

Temporal workflows must be deterministic — all I/O happens in activities.
"""

from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from forge.blocks.gather import GatherFailure, GatherSpec, run_fan_out_gather
    from forge.blocks.host import DispatchHostBase, RunSettings
    from forge.blocks.step import StepSpec, run_step_attempts
    from forge.blocks.transport import BATCH_WAIT_FAILURES
    from forge.blocks.worktree import cleanup_worktree_after_exception
    from forge.models import (
        AssembleSubTaskContextInput,
        SubTaskInput,
        SubTaskResult,
        TransitionSignal,
        build_llm_stats,
    )
    from forge.step_logic import (
        compound_sub_task_id,
        nested_fan_out_failure,
        nested_fan_out_success,
        sub_task_batch_wait_failure,
        sub_task_terminal,
    )


@workflow.defn
class ForgeSubTaskWorkflow(DispatchHostBase):
    """Execute a single sub-task within a fan-out step.

    Routes between two execution paths:

    Single-step (leaf or depth >= max_depth):
        run_step_attempts(mode="sub_task") — the shared step block (T5.2) in
        fresh-dispose mode: a worktree per attempt (compound ID, branched from
        the parent branch), removed at the end of every attempt including
        success, and no commit at any point (D16). The output travels home in
        the returned SubTaskResult.

    Nested fan-out (has sub_tasks and depth < max_depth):
        run_fan_out_gather(mode="nested_fan_out") — the shared gather block
        (T5.3) in owned-worktree mode: it creates this node's worktree from the
        parent branch, starts one child per nested sub-task at depth+1, awaits
        them with per-child failure isolation, merges (resolving conflicts when
        asked), validates the merged output, and removes the worktree on every
        exit without committing (D16). The merged output travels home in the
        returned SubTaskResult.

    Its LLM dispatch (generation, and conflict resolution on the nested path) is
    the one inherited from :class:`DispatchHostBase`, shared with
    ``ForgeTaskWorkflow`` — T5.3 retired the verbatim second copy that lived here.
    """

    @workflow.run
    async def run(self, input: SubTaskInput) -> SubTaskResult:
        self.configure(RunSettings.from_input(input))
        workflow.logger.info(
            "Sub-task started: sub_task_id=%s depth=%d/%d",
            input.sub_task.sub_task_id,
            input.depth,
            input.max_depth,
        )
        try:
            if input.sub_task.sub_tasks and input.depth < input.max_depth:
                return await self._run_nested_fan_out(input)
            return await self._run_single_step(input)
        except BATCH_WAIT_FAILURES as exc:
            # Same batch-wait failure symmetry as the parent (T1.6b): clean this
            # node's own worktree and return a terminal SubTaskResult instead of
            # crashing out. Sub-tasks write no run row of their own — returning a
            # normal failure lets the parent's failure handling record the run row.
            compound_id = compound_sub_task_id(input.parent_task_id, input.sub_task.sub_task_id)
            await cleanup_worktree_after_exception(input.repo_root, compound_id, exc)
            return sub_task_batch_wait_failure(sub_task_id=input.sub_task.sub_task_id, exc=exc)

    async def _run_single_step(self, input: SubTaskInput) -> SubTaskResult:
        """Execute a leaf sub-task through the shared step block.

        Fresh-dispose mode: each attempt gets its own worktree and gives it back
        at the end of the attempt, success included — sub-tasks never commit
        (D16); their output travels home in the result.
        """
        compound_id = compound_sub_task_id(input.parent_task_id, input.sub_task.sub_task_id)
        spec = StepSpec(
            mode="sub_task",
            task_id=compound_id,
            repo_root=input.repo_root,
            base_branch=input.parent_branch,
            assemble_input=AssembleSubTaskContextInput(
                parent_task_id=input.parent_task_id,
                parent_description=input.parent_description,
                sub_task=input.sub_task,
                worktree_path="",
                repo_root=input.repo_root,
                max_attempts=input.max_attempts,
                domain=input.domain,
            ),
            max_attempts=input.max_attempts,
            validation=input.validation,
            model_name=input.model_name,
            log_messages=self.log_messages,
        )
        outcome = await run_step_attempts(spec, self)

        if outcome.signal == TransitionSignal.SUCCESS:
            return SubTaskResult(
                sub_task_id=input.sub_task.sub_task_id,
                status=TransitionSignal.SUCCESS,
                output_files=outcome.output_files,
                validation_results=outcome.validation_results,
                digest=outcome.llm_result.response.explanation,
                llm_stats=build_llm_stats(outcome.llm_result),
            )
        return sub_task_terminal(
            sub_task_id=input.sub_task.sub_task_id,
            validation_results=outcome.validation_results,
            llm_stats=build_llm_stats(outcome.llm_result),
        )

    async def _run_nested_fan_out(self, input: SubTaskInput) -> SubTaskResult:
        """Execute a sub-task that itself contains nested sub-tasks.

        Owned-worktree mode: the block creates this node's worktree from the
        parent branch and removes it (with its branch) on every exit — success,
        failure, or exception. Nothing is committed here (D16); the merged
        output travels home in the returned SubTaskResult.
        """
        nested_sub_tasks = input.sub_task.sub_tasks
        assert nested_sub_tasks  # Caller guarantees this

        spec = GatherSpec(
            mode="nested_fan_out",
            task_id=compound_sub_task_id(input.parent_task_id, input.sub_task.sub_task_id),
            step_id=input.sub_task.sub_task_id,
            repo_root=input.repo_root,
            base_branch=input.parent_branch,
            sub_tasks=nested_sub_tasks,
            task_description=input.parent_description,
            step_description=input.sub_task.description,
            validation=input.validation,
            domain=input.domain,
            child_depth=input.depth + 1,
            max_depth=input.max_depth,
            child_max_attempts=input.max_attempts,
            child_model_name=input.model_name,
            resolve_conflicts=input.resolve_conflicts,
            model_routing=input.model_routing,
            thinking=input.thinking,
            sync_mode=input.sync_mode,
            log_messages=self.log_messages,
            batch_poll_interval_seconds=input.batch_poll_interval_seconds,
        )
        outcome = await run_fan_out_gather(spec, self)

        if isinstance(outcome, GatherFailure):
            return nested_fan_out_failure(
                sub_task_id=input.sub_task.sub_task_id,
                failure_kind=outcome.failure_kind,
                error=outcome.error,
                sub_task_results=outcome.sub_task_results,
                output_files=outcome.output_files,
                validation_results=outcome.validation_results,
                conflict_resolution=outcome.conflict_resolution,
            )
        return nested_fan_out_success(
            sub_task_id=input.sub_task.sub_task_id,
            output_files=outcome.output_files,
            validation_results=outcome.validation_results,
            sub_task_results=outcome.sub_task_results,
            conflict_resolution=outcome.conflict_resolution,
        )
