"""The root Forge task workflow — single-step and planned execution.

``ForgeTaskWorkflow`` is a driver: it decides *which* shared block runs next and
turns each block's neutral outcome into the run's ``TaskResult``. The work
itself lives under ``forge.blocks`` — the step pipeline (T5.2), the fan-out
gather (T5.3), the typed LLM dispatch arms (T5.3), and the exploration loop
(T5.4).

Phase 1 (plan=False): one step, task-level retry.
Phase 2 (plan=True): planning, then a step per plan step, with a periodic
sanity check that may continue, revise the remaining steps, or abort.
Phase 3 (fan-out): a plan step carrying sub_tasks gathers child workflows.
Phase 7 (exploration): an LLM-guided context loop before planning and before
each generation attempt.

Temporal workflows must be deterministic — all I/O happens in activities.
"""

from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from sax_platform.contracts.persist import persist_block
    from sax_platform.temporal.retries import IO_RETRY

    from forge.blocks.dispatch import PlanPreflightHalt, dispatch_planner, dispatch_sanity_check
    from forge.blocks.exploration import format_exploration_context, run_exploration_loop
    from forge.blocks.gather import GatherFailure, GatherSpec, run_fan_out_gather
    from forge.blocks.host import DispatchHostBase, RunSettings
    from forge.blocks.step import StepSpec, run_step_attempts
    from forge.blocks.transport import BATCH_WAIT_FAILURES
    from forge.blocks.worktree import cleanup_worktree_after_exception
    from forge.models import (
        AssembleContextInput,
        AssembledContext,
        AssembleSanityCheckContextInput,
        AssembleStepContextInput,
        CapabilityTier,
        CreateWorktreeInput,
        CreateWorktreeOutput,
        ForgeTaskInput,
        Plan,
        PlanCallResult,
        PlannerInput,
        PlanStep,
        SanityCheckCallResult,
        SanityCheckInput,
        SanityCheckVerdict,
        StepResult,
        TaskResult,
        TransitionSignal,
        build_llm_stats,
        resolve_model,
    )
    from forge.persist_models import PersistRun
    from forge.plan_checks import RevisionRejected, splice_revision
    from forge.presets import (
        CONTEXT_TIMEOUT,
        GIT_RETRY,
        GIT_TIMEOUT,
    )
    from forge.step_logic import (
        fan_out_step_failure,
        fan_out_success,
        llm_totals,
        plan_preflight_failure,
        planned_failure,
        single_step_terminal,
        slim_result,
        step_terminal,
        task_batch_wait_failure,
    )


@workflow.defn
class ForgeTaskWorkflow(DispatchHostBase):
    """Execute a Forge task with optional planning and multi-step execution.

    Every step this class runs — the single step, and each planned step — goes
    through the one pipeline in ``forge.blocks.step`` (T5.2, which also serves
    the sub-task mode ``ForgeSubTaskWorkflow`` drives); this class supplies the
    mode's :class:`StepSpec` and turns the neutral outcome into a result. The
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
        self.configure(RunSettings.from_input(input))
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
            await cleanup_worktree_after_exception(input.repo_root, input.task.task_id, exc)
            result = task_batch_wait_failure(task_id=input.task.task_id, exc=exc)
        # Aggregate run-level LLM spend across the finished result tree (D97) once,
        # covering success, terminal-failure, and batch-wait paths alike.
        result = result.model_copy(update={"llm_totals": llm_totals(result)})
        # Survivably persist the run result (idempotent on (workflow_id, run_id)) so
        # every execution records a row — including a batch-wait failure and
        # fire-and-forget submissions, which the old CLI-side _persist_run never
        # covered. The failure path above reuses this exact persist, so its row is
        # keyed and written identically to the success path.
        await persist_block(
            PersistRun(
                workflow_id=workflow.info().workflow_id,
                run_id=workflow.info().run_id,
                task_result=result,
            )
        )
        return result

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
            exploration_results = await run_exploration_loop(
                self,
                task=task,
                repo_root=input.repo_root,
                worktree_path=worktree_path,
                max_rounds=input.max_exploration_rounds,
                model_name=exploration_model,
                log_messages=self.log_messages,
            )
            exploration_section = format_exploration_context(exploration_results)
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
            log_messages=self.log_messages,
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
    ) -> PlanCallResult | PlanPreflightHalt:
        """Assemble planner context, run exploration, and call the planner LLM.

        The planner call goes through the preflight gate (T5.6), so this returns
        either an accepted plan or the halt the gate reached after
        ``MAX_PLANNER_ATTEMPTS`` structurally invalid ones.
        """
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
            start_to_close_timeout=CONTEXT_TIMEOUT,
            retry_policy=IO_RETRY,
            result_type=PlannerInput,
        )

        # --- Exploration loop for planner (Phase 7) ---
        if input.max_exploration_rounds > 0:
            exploration_results = await run_exploration_loop(
                self,
                task=task,
                repo_root=input.repo_root,
                worktree_path=wt_output.worktree_path,
                max_rounds=input.max_exploration_rounds,
                model_name=exploration_model,
                log_messages=self.log_messages,
            )
            exploration_section = format_exploration_context(exploration_results)
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
            "log_messages": self.log_messages,
            "worktree_path": wt_output.worktree_path,
        }
        planner_input = planner_input.model_copy(update=planner_update)

        # --- Call planner (through the preflight gate) ---
        planner_result = await dispatch_planner(self, planner_input)
        if isinstance(planner_result, PlanPreflightHalt):
            workflow.logger.error(
                "Plan preflight halted: task_id=%s %s", task.task_id, planner_result.error
            )
            return planner_result
        workflow.logger.info(
            "Plan created: task_id=%s steps=%d", task.task_id, len(planner_result.plan.steps)
        )
        return planner_result

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
            start_to_close_timeout=GIT_TIMEOUT,
            retry_policy=GIT_RETRY,
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

        planned = await self._plan_task(input, wt_output)
        if isinstance(planned, PlanPreflightHalt):
            return plan_preflight_failure(
                task_id=task.task_id,
                error=planned.error,
                worktree_path=wt_output.worktree_path,
                worktree_branch=wt_output.branch_name,
                planner_stats=build_llm_stats(planned.last_result),
            )
        plan: Plan = planned.plan
        p_stats = build_llm_stats(planned)
        step_results: list[StepResult] = []
        all_output_files: dict[str, str] = {}
        sanity_check_count = 0
        revision_count = 0

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
                    # The splice is checked, not attempted (T5.6): building
                    # Plan(steps=...) over the cap raises a pydantic
                    # ValidationError *inside workflow code*, which Temporal
                    # retries as a workflow task forever — a hung workflow, not a
                    # failed one. splice_revision catches that case, the
                    # revision-count cap, and a structurally invalid revision.
                    spliced = splice_revision(
                        plan,
                        completed_through=step_index,
                        revised_steps=revised,
                        revision_count=revision_count,
                    )
                    if isinstance(spliced, RevisionRejected):
                        return planned_failure(
                            task_id=task.task_id,
                            failure_kind="plan_revision",
                            error=spliced.reason,
                            output_files=all_output_files,
                            worktree_path=wt_output.worktree_path,
                            worktree_branch=wt_output.branch_name,
                            step_results=step_results,
                            plan=plan,
                            planner_stats=p_stats,
                            sanity_check_count=sanity_check_count,
                        )
                    plan = spliced.plan
                    revision_count += 1
                    workflow.logger.info(
                        "Plan revised: remaining steps %d → %d (revision %d)",
                        old_remaining,
                        len(revised),
                        revision_count,
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
            log_messages=self.log_messages,
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
            start_to_close_timeout=CONTEXT_TIMEOUT,
            retry_policy=IO_RETRY,
            result_type=SanityCheckInput,
        )

        # Set model, thinking config, and log_messages
        update: dict[str, object] = {
            "model_name": reasoning_model,
            "thinking": input.thinking,
            "log_messages": self.log_messages,
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
            log_messages=self.log_messages,
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
