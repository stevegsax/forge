"""Temporal workflow for Forge task execution.

Orchestrates the core activities and git activities into retry loops.

Phase 1 (plan=False): Single-step execution with task-level retry.
Phase 2 (plan=True): Planning + multi-step execution with step-level retry.
Phase 3 (fan-out): Steps with sub_tasks spawn child workflows in parallel.
Phase 7 (exploration): LLM-guided context exploration loop before generation.

Temporal workflows must be deterministic — all I/O happens in activities.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from sax_platform.temporal.retries import IO_RETRY

    from forge.models import (
        AssembleContextInput,
        AssembledContext,
        AssembleSanityCheckContextInput,
        AssembleStepContextInput,
        AssembleSubTaskContextInput,
        CapabilityTier,
        ContextResult,
        CreateWorktreeInput,
        CreateWorktreeOutput,
        ExplorationInput,
        ForgeTaskInput,
        FulfillContextInput,
        LLMStats,
        Plan,
        PlannerInput,
        PlanStep,
        SanityCheckCallResult,
        SanityCheckInput,
        SanityCheckVerdict,
        StepResult,
        SubTaskInput,
        SubTaskResult,
        TaskDefinition,
        TaskResult,
        TransitionSignal,
        build_llm_stats,
        resolve_model,
    )
    from forge.providers import PROVIDER_SPECS
    from forge.step_logic import (
        compound_sub_task_id,
        fan_out_step_failure,
        fan_out_success,
        llm_totals,
        nested_fan_out_failure,
        nested_fan_out_success,
        planned_failure,
        single_step_terminal,
        slim_result,
        step_terminal,
        sub_task_batch_wait_failure,
        sub_task_terminal,
        task_batch_wait_failure,
    )

# ---------------------------------------------------------------------------
# Activity timeout and retry presets
#
# What is left here belongs to the plan driver: the LLM-call timeouts moved
# into blocks/dispatch.py with their arms, and the write/validate ones into
# blocks/gather.py with the gather (T5.3). IO_RETRY is the shared preset from
# sax_platform.temporal.retries (T3.4). ST8 gives the remainder one home.
# ---------------------------------------------------------------------------

_GIT_TIMEOUT = timedelta(seconds=30)
_CONTEXT_TIMEOUT = timedelta(seconds=30)
_EXPLORATION_FULFILL_TIMEOUT = timedelta(minutes=2)

_GIT_RETRY = RetryPolicy(
    maximum_attempts=2,
    non_retryable_error_types=["CommitError", "RepoDiscoveryError"],
)


# ---------------------------------------------------------------------------
# Shared blocks — the step pipeline, the gather, the LLM dispatch arms
# ---------------------------------------------------------------------------

with workflow.unsafe.imports_passed_through():
    from forge.blocks.dispatch import (
        dispatch_exploration,
        dispatch_planner,
        dispatch_sanity_check,
    )
    from forge.blocks.gather import GatherFailure, GatherSpec, run_fan_out_gather
    from forge.blocks.host import DispatchHostBase
    from forge.blocks.step import (
        StepSpec,
        cleanup_worktree_after_exception,
        run_step_attempts,
    )
    from forge.persist_models import PersistRun
    from forge.workflow_blocks import BATCH_WAIT_FAILURES
    from forge.workflow_blocks import (
        cleanup_worktree_after_failure as _cleanup_worktree_after_failure,
    )
    from forge.workflow_blocks import (
        persist_block as _persist_block,
    )


@workflow.defn
class ForgeTaskWorkflow(DispatchHostBase):
    """Execute a Forge task with optional planning and multi-step execution.

    Every step — single-step, planned step, sub-task — runs through the one
    pipeline in ``forge.blocks.step`` (T5.2); this class supplies the mode's
    :class:`StepSpec` and turns the neutral outcome into a result. The
    transition decision inside that block is pure and inlined
    (``step_logic.determine_transition``, D95): validation results plus the
    attempt number determine the signal — there is no ``evaluate_transition``
    activity.

    Every LLM call goes through ``forge.blocks.dispatch`` (T5.3) over the
    lane/cadence/persist state inherited from :class:`DispatchHostBase`, which
    ``ForgeSubTaskWorkflow`` inherits too — the dispatch shape and the
    interaction persist exist once for both classes.

    plan=False (Phase 1):
        run_step_attempts(mode="single_step") — fresh worktree per attempt,
        task-level commit on success and on terminal failure → TaskResult

    plan=True (Phase 2):
        create_worktree (once)
        assemble_planner_context → call_planner → Plan
        for step in plan.steps:
            fan-out steps → run_fan_out_gather(mode="fan_out_step") in the
            borrowed worktree — children gathered with per-child failure
            isolation, merged, validated, and committed by the block (T5.3)
            regular steps → run_step_attempts(mode="planned_step") in the
            borrowed worktree; a terminal step failure fails the task
        All steps done → return TaskResult(SUCCESS)
    """

    @workflow.run
    async def run(self, input: ForgeTaskInput) -> TaskResult:
        self.configure(
            sync_mode=input.sync_mode,
            log_messages=input.log_messages,
            batch_poll_interval_seconds=input.batch_poll_interval_seconds,
        )
        workflow.logger.info(
            "Workflow started: task_id=%s plan=%s sync=%s",
            input.task.task_id,
            input.plan,
            input.sync_mode,
        )
        try:
            if input.plan:
                result = await self._run_planned(input)
            else:
                result = await self._run_single_step(input)
        except BATCH_WAIT_FAILURES as exc:
            # The batch wait gave up at the 25h ceiling, or the provider reported a
            # terminal status, or the fetch carried an error (T4.1). Clean the
            # worktree and record a terminal failure so the run never crashes out
            # leaving no row and an orphaned worktree (T1.6b).
            await _cleanup_worktree_after_failure(input.repo_root, input.task.task_id, exc)
            result = task_batch_wait_failure(task_id=input.task.task_id, exc=exc)
        # Aggregate run-level LLM spend across the finished result tree (D97) once,
        # covering success, terminal-failure, and batch-wait paths alike.
        result = result.model_copy(update={"llm_totals": llm_totals(result)})
        # Survivably persist the run result (idempotent on (workflow_id, run_id)) so
        # every execution records a row — including a batch-wait failure and
        # fire-and-forget submissions, which the old CLI-side _persist_run never
        # covered. The failure path above reuses this exact persist, so its row is
        # keyed and written identically to the success path.
        await _persist_block(
            PersistRun(
                workflow_id=workflow.info().workflow_id,
                run_id=workflow.info().run_id,
                task_result=result,
            )
        )
        return result

    # ------------------------------------------------------------------
    # Phase 7: LLM-guided context exploration
    # ------------------------------------------------------------------

    async def _run_exploration_loop(
        self,
        task: TaskDefinition,
        repo_root: str,
        worktree_path: str,
        max_rounds: int,
        model_name: str = "",
    ) -> list[ContextResult]:
        """LLM-guided context exploration loop.

        The exploration LLM requests context from providers until it signals
        readiness (empty requests list) or the round limit is reached.
        """
        accumulated: list[ContextResult] = []
        round_num = 0

        for round_num in range(1, max_rounds + 1):
            workflow.logger.debug(
                "Exploration round %d/%d: task_id=%s", round_num, max_rounds, task.task_id
            )
            exploration_input = ExplorationInput(
                task_id=task.task_id,
                task_description=task.description,
                target_files=task.target_files,
                context_files=task.context_files,
                context_config=task.context,
                available_providers=PROVIDER_SPECS,
                accumulated_context=accumulated,
                round_number=round_num,
                max_rounds=max_rounds,
                repo_root=repo_root,
                model_name=model_name,
                log_messages=self._log_messages,
                worktree_path=worktree_path,
            )
            exploration_call = await dispatch_exploration(self, exploration_input)
            requests = exploration_call.response.requests
            workflow.logger.debug(
                "Exploration round %d: %d provider requests",
                round_num,
                len(requests),
            )

            if not requests:
                break  # LLM is ready to generate

            context_results = await workflow.execute_activity(
                "fulfill_context_requests",
                FulfillContextInput(
                    requests=requests,
                    repo_root=repo_root,
                    worktree_path=worktree_path,
                ),
                start_to_close_timeout=_EXPLORATION_FULFILL_TIMEOUT,
                retry_policy=IO_RETRY,
                result_type=list[ContextResult],
            )
            accumulated.extend(context_results)

        workflow.logger.info(
            "Exploration complete: task_id=%s rounds_used=%d results=%d",
            task.task_id,
            min(round_num, max_rounds),
            len(accumulated),
        )
        return accumulated

    @staticmethod
    def _format_exploration_context(results: list[ContextResult]) -> str:
        """Format exploration results as a prompt section."""
        if not results:
            return ""

        parts = ["", "## Exploration Results"]
        for ctx in results:
            parts.append("")
            parts.append(f"### From: {ctx.provider}")
            content = ctx.content
            if len(content) > 8000:
                content = content[:8000] + "\n... (truncated)"
            parts.append(content)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Phase 1: Single-step execution (the shared step block, fresh-keep mode)
    # ------------------------------------------------------------------

    async def _run_single_step(self, input: ForgeTaskInput) -> TaskResult:
        """Run the task as one step through the shared step block."""
        task = input.task
        generation_model = resolve_model(CapabilityTier.GENERATION, input.model_routing)
        exploration_model = resolve_model(CapabilityTier.CLASSIFICATION, input.model_routing)

        async def explore(context: AssembledContext, worktree_path: str) -> AssembledContext:
            """Phase 7 exploration for one attempt, against that attempt's own worktree."""
            exploration_results = await self._run_exploration_loop(
                task=task,
                repo_root=input.repo_root,
                worktree_path=worktree_path,
                max_rounds=input.max_exploration_rounds,
                model_name=exploration_model,
            )
            exploration_section = self._format_exploration_context(exploration_results)
            if not exploration_section:
                return context
            return AssembledContext(
                task_id=context.task_id,
                system_prompt=context.system_prompt + exploration_section,
                user_prompt=context.user_prompt,
                context_stats=context.context_stats,
                step_id=context.step_id,
                sub_task_id=context.sub_task_id,
            )

        spec = StepSpec(
            mode="single_step",
            task_id=task.task_id,
            repo_root=input.repo_root,
            base_branch=task.base_branch,
            assemble_input=AssembleContextInput(
                task_id=task.task_id,
                description=task.description,
                target_files=task.target_files,
                context_files=task.context_files,
                context_config=task.context,
                repo_root=input.repo_root,
                worktree_path="",
                max_attempts=input.max_attempts,
            ),
            max_attempts=input.max_attempts,
            validation=task.validation,
            model_name=generation_model,
            log_messages=self._log_messages,
            exploration_rounds=input.max_exploration_rounds,
        )
        outcome = await run_step_attempts(spec, self, explore)

        if outcome.signal == TransitionSignal.SUCCESS:
            return TaskResult(
                task_id=task.task_id,
                status=TransitionSignal.SUCCESS,
                output_files=outcome.output_files,
                validation_results=outcome.validation_results,
                worktree_path=outcome.worktree_path,
                worktree_branch=outcome.worktree_branch,
                llm_stats=build_llm_stats(outcome.llm_result),
                context_stats=outcome.context_stats,
            )
        return single_step_terminal(
            task_id=task.task_id,
            output_files=outcome.output_files,
            validation_results=outcome.validation_results,
            worktree_path=outcome.worktree_path,
            worktree_branch=outcome.worktree_branch,
            llm_stats=build_llm_stats(outcome.llm_result),
            context_stats=outcome.context_stats,
        )

    # ------------------------------------------------------------------
    # Phase 2: Planned multi-step execution
    # ------------------------------------------------------------------

    async def _plan_task(
        self,
        input: ForgeTaskInput,
        wt_output: CreateWorktreeOutput,
    ) -> tuple[Plan, LLMStats]:
        """Assemble planner context, run exploration, and call planner LLM."""
        task = input.task
        planner_model = resolve_model(CapabilityTier.REASONING, input.model_routing)
        exploration_model = resolve_model(CapabilityTier.CLASSIFICATION, input.model_routing)

        # --- Assemble planner context ---
        planner_input = await workflow.execute_activity(
            "assemble_planner_context",
            AssembleContextInput(
                task_id=task.task_id,
                description=task.description,
                target_files=task.target_files,
                context_files=task.context_files,
                context_config=task.context,
                repo_root=input.repo_root,
                worktree_path=wt_output.worktree_path,
            ),
            start_to_close_timeout=_CONTEXT_TIMEOUT,
            retry_policy=IO_RETRY,
            result_type=PlannerInput,
        )

        # --- Exploration loop for planner (Phase 7) ---
        if input.max_exploration_rounds > 0:
            exploration_results = await self._run_exploration_loop(
                task=task,
                repo_root=input.repo_root,
                worktree_path=wt_output.worktree_path,
                max_rounds=input.max_exploration_rounds,
                model_name=exploration_model,
            )
            exploration_section = self._format_exploration_context(exploration_results)
            if exploration_section:
                planner_input = PlannerInput(
                    task_id=planner_input.task_id,
                    system_prompt=planner_input.system_prompt + exploration_section,
                    user_prompt=planner_input.user_prompt,
                )

        # --- Set model_name, thinking config, and log_messages on planner input ---
        planner_update: dict[str, object] = {
            "model_name": planner_model,
            "thinking": input.thinking,
            "log_messages": self._log_messages,
            "worktree_path": wt_output.worktree_path,
        }
        planner_input = planner_input.model_copy(update=planner_update)

        # --- Call planner ---
        planner_result = await dispatch_planner(self, planner_input)
        plan: Plan = planner_result.plan
        workflow.logger.info("Plan created: task_id=%s steps=%d", task.task_id, len(plan.steps))
        return plan, build_llm_stats(planner_result)

    async def _run_planned(self, input: ForgeTaskInput) -> TaskResult:
        """Create the plan's worktree, drive the plan in it, and own its cleanup.

        The planned worktree is created once and borrowed by every step, so this
        method — not the step block — carries the cleanup wrap: any exception
        from the planner, a step, or a sanity check removes the worktree and its
        branch before re-raising. A leaked worktree/branch used to make the next
        run of the same task id fail permanently.
        """
        task = input.task

        # --- Create worktree (once) ---
        wt_output = await workflow.execute_activity(
            "create_worktree_activity",
            CreateWorktreeInput(
                repo_root=input.repo_root,
                task_id=task.task_id,
                base_branch=task.base_branch,
            ),
            start_to_close_timeout=_GIT_TIMEOUT,
            retry_policy=_GIT_RETRY,
            result_type=CreateWorktreeOutput,
        )

        try:
            return await self._drive_plan(input, wt_output)
        except Exception as exc:
            # run() already cleans this worktree for the batch-wait shapes
            # (T1.6b) and records a terminal result, so cleaning here too would
            # only duplicate the removal.
            if not isinstance(exc, BATCH_WAIT_FAILURES):
                await cleanup_worktree_after_exception(input.repo_root, task.task_id, exc)
            raise

    async def _drive_plan(
        self, input: ForgeTaskInput, wt_output: CreateWorktreeOutput
    ) -> TaskResult:
        """Plan the task, then run every step in the borrowed worktree."""
        task = input.task

        plan, p_stats = await self._plan_task(input, wt_output)
        step_results: list[StepResult] = []
        all_output_files: dict[str, str] = {}
        sanity_check_count = 0

        # --- Execute steps sequentially (while loop enables plan mutation on revise) ---
        step_index = 0
        while step_index < len(plan.steps):
            step = plan.steps[step_index]
            workflow.logger.info(
                "Step %d/%d: step_id=%s has_sub_tasks=%s",
                step_index + 1,
                len(plan.steps),
                step.step_id,
                bool(step.sub_tasks),
            )

            # Resolve model for this step (per-step tier override or default)
            step_tier = step.capability_tier or CapabilityTier.GENERATION
            step_model = resolve_model(step_tier, input.model_routing)

            if step.sub_tasks:
                # Phase 3: fan-out step
                step_result = await self._run_fan_out_step(
                    input,
                    step,
                    wt_output,
                    step_model=step_model,
                )
                succeeded = step_result.status == TransitionSignal.SUCCESS
                if succeeded:
                    # Fold contents into the top-level map, then embed the slim copy.
                    all_output_files.update(step_result.output_files)
                step_results.append(slim_result(step_result))
                if not succeeded:
                    return planned_failure(
                        task_id=task.task_id,
                        failure_kind="step_failed",
                        error=f"Step {step.step_id} fan-out failed: {step_result.error}",
                        output_files=all_output_files,
                        worktree_path=wt_output.worktree_path,
                        worktree_branch=wt_output.branch_name,
                        step_results=step_results,
                        plan=plan,
                        planner_stats=p_stats,
                        sanity_check_count=sanity_check_count,
                    )
                step_index += 1
                continue

            step_result = await self._execute_step_with_retries(
                input=input,
                step=step,
                step_index=step_index,
                total_steps=len(plan.steps),
                step_model=step_model,
                wt_output=wt_output,
                step_results=step_results,
            )
            succeeded = step_result.status == TransitionSignal.SUCCESS
            if succeeded:
                all_output_files.update(step_result.output_files)
            step_results.append(slim_result(step_result))
            if not succeeded:
                return planned_failure(
                    task_id=task.task_id,
                    failure_kind="step_failed",
                    error=f"Step {step.step_id} failed: {step_result.error}",
                    output_files=all_output_files,
                    worktree_path=wt_output.worktree_path,
                    worktree_branch=wt_output.branch_name,
                    step_results=step_results,
                    plan=plan,
                    planner_stats=p_stats,
                    sanity_check_count=sanity_check_count,
                )

            # --- Sanity check trigger ---
            if (
                input.sanity_check_interval > 0
                and len(step_results) % input.sanity_check_interval == 0
                and step_index < len(plan.steps) - 1  # skip after last step
            ):
                sanity_result = await self._run_sanity_check(
                    input, plan, step_results, plan.steps[step_index + 1 :], wt_output
                )
                sanity_check_count += 1
                workflow.logger.info(
                    "Sanity check #%d: verdict=%s",
                    sanity_check_count,
                    sanity_result.response.verdict.value,
                )

                if sanity_result.response.verdict == SanityCheckVerdict.ABORT:
                    return planned_failure(
                        task_id=task.task_id,
                        failure_kind="sanity_abort",
                        error=f"Sanity check aborted: {sanity_result.response.explanation}",
                        output_files=all_output_files,
                        worktree_path=wt_output.worktree_path,
                        worktree_branch=wt_output.branch_name,
                        step_results=step_results,
                        plan=plan,
                        planner_stats=p_stats,
                        sanity_check_count=sanity_check_count,
                    )

                if sanity_result.response.verdict == SanityCheckVerdict.REVISE:
                    revised = sanity_result.response.revised_steps or []
                    old_remaining = len(plan.steps) - step_index - 1
                    plan = Plan(
                        task_id=plan.task_id,
                        steps=plan.steps[: step_index + 1] + revised,
                        explanation=plan.explanation,
                    )
                    workflow.logger.info(
                        "Plan revised: remaining steps %d → %d",
                        old_remaining,
                        len(revised),
                    )

            step_index += 1

        # --- All steps succeeded ---
        workflow.logger.info("All %d steps completed: task_id=%s", len(plan.steps), task.task_id)
        return TaskResult(
            task_id=task.task_id,
            status=TransitionSignal.SUCCESS,
            output_files=all_output_files,
            worktree_path=wt_output.worktree_path,
            worktree_branch=wt_output.branch_name,
            step_results=step_results,
            plan=plan,
            planner_stats=p_stats,
            sanity_check_count=sanity_check_count,
        )

    # ------------------------------------------------------------------
    # Step execution helper
    # ------------------------------------------------------------------

    async def _execute_step_with_retries(
        self,
        input: ForgeTaskInput,
        step: PlanStep,
        step_index: int,
        total_steps: int,
        step_model: str,
        wt_output: CreateWorktreeOutput,
        step_results: list[StepResult],
    ) -> StepResult:
        """Execute a single planned step through the shared step block.

        Borrowed-worktree mode: the plan's worktree belongs to ``_run_planned``,
        so the block resets it between attempts and never creates or removes it.

        Returns StepResult with SUCCESS or FAILURE_TERMINAL status.
        """
        task = input.task
        spec = StepSpec(
            mode="planned_step",
            task_id=task.task_id,
            repo_root=input.repo_root,
            base_branch=task.base_branch,
            borrowed_worktree=wt_output,
            assemble_input=AssembleStepContextInput(
                task_id=task.task_id,
                task_description=task.description,
                context_config=task.context,
                step=step,
                step_index=step_index,
                total_steps=total_steps,
                completed_steps=step_results,
                repo_root=input.repo_root,
                worktree_path=wt_output.worktree_path,
                max_attempts=input.max_step_attempts,
            ),
            max_attempts=input.max_step_attempts,
            validation=task.validation,
            model_name=step_model,
            log_messages=self._log_messages,
            commit_message=f"forge({task.task_id}): step {step.step_id} success",
        )
        outcome = await run_step_attempts(spec, self)

        if outcome.signal == TransitionSignal.SUCCESS:
            return StepResult(
                step_id=step.step_id,
                status=TransitionSignal.SUCCESS,
                output_files=outcome.output_files,
                validation_results=outcome.validation_results,
                commit_sha=outcome.commit_sha,
                llm_stats=build_llm_stats(outcome.llm_result),
            )
        return step_terminal(
            step_id=step.step_id,
            output_files=outcome.output_files,
            validation_results=outcome.validation_results,
            llm_stats=build_llm_stats(outcome.llm_result),
        )

    # ------------------------------------------------------------------
    # Sanity check helper
    # ------------------------------------------------------------------

    async def _run_sanity_check(
        self,
        input: ForgeTaskInput,
        plan: Plan,
        step_results: list[StepResult],
        remaining_steps: list[PlanStep],
        wt_output: CreateWorktreeOutput,
    ) -> SanityCheckCallResult:
        """Run a sanity check: assemble context, call LLM, return result."""
        reasoning_model = resolve_model(CapabilityTier.REASONING, input.model_routing)

        sanity_input = await workflow.execute_activity(
            "assemble_sanity_check_context",
            AssembleSanityCheckContextInput(
                task_id=input.task.task_id,
                task_description=input.task.description,
                plan=plan,
                completed_steps=step_results,
                remaining_steps=remaining_steps,
                repo_root=input.repo_root,
                worktree_path=wt_output.worktree_path,
            ),
            start_to_close_timeout=_CONTEXT_TIMEOUT,
            retry_policy=IO_RETRY,
            result_type=SanityCheckInput,
        )

        # Set model, thinking config, and log_messages
        update: dict[str, object] = {
            "model_name": reasoning_model,
            "thinking": input.thinking,
            "log_messages": self._log_messages,
            "worktree_path": wt_output.worktree_path,
        }
        sanity_input = sanity_input.model_copy(update=update)

        return await dispatch_sanity_check(self, sanity_input)

    # ------------------------------------------------------------------
    # Phase 3: Fan-out step execution
    # ------------------------------------------------------------------

    async def _run_fan_out_step(
        self,
        input: ForgeTaskInput,
        step: PlanStep,
        wt_output: CreateWorktreeOutput,
        step_model: str = "",
    ) -> StepResult:
        """Execute a fan-out step through the shared gather block.

        Borrowed-worktree mode: the plan's worktree belongs to ``_run_planned``,
        so the block writes the merged output in it and commits, but never
        creates or removes it.
        """
        task = input.task
        sub_tasks = step.sub_tasks
        assert sub_tasks  # Caller guarantees this

        spec = GatherSpec(
            mode="fan_out_step",
            task_id=task.task_id,
            step_id=step.step_id,
            repo_root=input.repo_root,
            borrowed_worktree=wt_output,
            sub_tasks=sub_tasks,
            task_description=task.description,
            step_description=step.description,
            validation=task.validation,
            domain=task.domain,
            child_depth=0,
            max_depth=input.max_fan_out_depth,
            child_max_attempts=input.max_sub_task_attempts,
            child_model_name=step_model,
            resolve_conflicts=input.resolve_conflicts,
            model_routing=input.model_routing,
            thinking=input.thinking,
            sync_mode=input.sync_mode,
            log_messages=self._log_messages,
            batch_poll_interval_seconds=input.batch_poll_interval_seconds,
        )
        outcome = await run_fan_out_gather(spec, self)

        if isinstance(outcome, GatherFailure):
            return fan_out_step_failure(
                step_id=step.step_id,
                failure_kind=outcome.failure_kind,
                error=outcome.error,
                sub_task_results=outcome.sub_task_results,
                output_files=outcome.output_files,
                validation_results=outcome.validation_results,
                conflict_resolution=outcome.conflict_resolution,
            )
        return fan_out_success(
            step_id=step.step_id,
            output_files=outcome.output_files,
            validation_results=outcome.validation_results,
            commit_sha=outcome.commit_sha,
            sub_task_results=outcome.sub_task_results,
            conflict_resolution=outcome.conflict_resolution,
        )


# ===========================================================================
# Phase 3: Sub-task child workflow
# ===========================================================================


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
        self.configure(
            sync_mode=input.sync_mode,
            log_messages=input.log_messages,
            batch_poll_interval_seconds=input.batch_poll_interval_seconds,
        )
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
            await _cleanup_worktree_after_failure(input.repo_root, compound_id, exc)
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
            log_messages=self._log_messages,
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
            log_messages=self._log_messages,
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
