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
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from forge.models import (
        AssembleContextInput,
        AssembledContext,
        AssembleSanityCheckContextInput,
        AssembleStepContextInput,
        AssembleSubTaskContextInput,
        BatchResult,
        BatchSubmitInput,
        BatchSubmitResult,
        CapabilityTier,
        CommitChangesInput,
        CommitChangesOutput,
        ConflictResolutionCallInput,
        ConflictResolutionCallResult,
        ConflictResolutionInput,
        ConflictResolutionResponse,
        ContextResult,
        CreateWorktreeInput,
        CreateWorktreeOutput,
        DetectFileConflictsInput,
        DetectFileConflictsOutput,
        ExplorationInput,
        ExplorationResponse,
        FileConflict,
        ForgeTaskInput,
        FulfillContextInput,
        LLMCallResult,
        LLMResponse,
        LLMStats,
        ModelConfig,
        ParsedLLMResponse,
        ParseResponseInput,
        Plan,
        PlanCallResult,
        PlannerInput,
        PlanStep,
        RemoveWorktreeInput,
        ResetWorktreeInput,
        SanityCheckCallResult,
        SanityCheckInput,
        SanityCheckResponse,
        SanityCheckVerdict,
        StepResult,
        SubTaskInput,
        SubTaskResult,
        TaskDefinition,
        TaskDomain,
        TaskResult,
        ThinkingConfig,
        TransitionInput,
        TransitionSignal,
        ValidateOutputInput,
        ValidationResult,
        WriteFilesInput,
        WriteOutputInput,
        WriteResult,
        build_llm_stats,
        build_planner_stats,
        resolve_model,
    )
    from forge.providers import PROVIDER_SPECS

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
_EXPLORATION_LLM_TIMEOUT = timedelta(minutes=5)
_EXPLORATION_FULFILL_TIMEOUT = timedelta(minutes=2)
_SANITY_CHECK_TIMEOUT = timedelta(minutes=5)
_CONFLICT_RESOLUTION_TIMEOUT = timedelta(minutes=5)
_SUBMIT_TIMEOUT = timedelta(seconds=60)
_PARSE_TIMEOUT = timedelta(seconds=30)
_BATCH_WAIT_TIMEOUT = timedelta(hours=25)  # Anthropic batch API expires at 24h

# ---------------------------------------------------------------------------
# Heartbeat timeouts — detect worker crashes during long-running activities
# ---------------------------------------------------------------------------

_LLM_HEARTBEAT = timedelta(seconds=60)
_VALIDATE_HEARTBEAT = timedelta(seconds=120)  # subprocess.run blocks event loop

# ---------------------------------------------------------------------------
# Activity retry policies
# ---------------------------------------------------------------------------

_LLM_RETRY = RetryPolicy(
    maximum_attempts=3,
    non_retryable_error_types=[
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
    ],
)
_LOCAL_RETRY = RetryPolicy(maximum_attempts=2)
_GIT_RETRY = RetryPolicy(
    maximum_attempts=2,
    non_retryable_error_types=["CommitError", "RepoDiscoveryError"],
)
_WRITE_RETRY = RetryPolicy(
    maximum_attempts=2,
    non_retryable_error_types=["OutputWriteError", "EditApplicationError"],
)

_CHILD_BASE_MINUTES = 15
_CHILD_OVERHEAD_MINUTES_PER_LEVEL = 5


def _child_timeout(depth: int, max_depth: int) -> timedelta:
    """Scale child workflow timeout by remaining depth.

    Base 15 min for leaves, +5 min per nesting level for orchestration overhead.
    """
    remaining = max_depth - depth
    return timedelta(minutes=_CHILD_BASE_MINUTES + _CHILD_OVERHEAD_MINUTES_PER_LEVEL * remaining)


async def _assemble_conflict_resolution(
    task_id: str,
    step_id: str,
    conflicts: list[FileConflict],
    non_conflicting: dict[str, str],
    task_description: str,
    step_description: str,
    repo_root: str,
    worktree_path: str,
    domain: TaskDomain,
    model_routing: ModelConfig,
    thinking: ThinkingConfig,
    *,
    log_messages: bool = False,
) -> ConflictResolutionCallInput:
    """Assemble conflict resolution context via activity. Module-level, called from any workflow."""
    reasoning_model = resolve_model(CapabilityTier.REASONING, model_routing)

    resolution_input = ConflictResolutionInput(
        task_id=task_id,
        step_id=step_id,
        conflicts=conflicts,
        non_conflicting_files=non_conflicting,
        task_description=task_description,
        step_description=step_description,
        repo_root=repo_root,
        worktree_path=worktree_path,
        domain=domain,
        model_name=reasoning_model,
        thinking_budget_tokens=thinking.budget_tokens if thinking.enabled else 0,
        thinking_effort=thinking.effort,
    )

    call_input = await workflow.execute_activity(
        "assemble_conflict_resolution_context",
        resolution_input,
        start_to_close_timeout=_CONTEXT_TIMEOUT,
        retry_policy=_LOCAL_RETRY,
        result_type=ConflictResolutionCallInput,
    )
    return call_input.model_copy(
        update={"log_messages": log_messages, "worktree_path": worktree_path}
    )


# ---------------------------------------------------------------------------
# Shared dispatch helpers — used by both workflow classes
# ---------------------------------------------------------------------------


async def _call_llm_batch_dispatch(
    batch_results: list[BatchResult],
    context: AssembledContext,
    output_type_name: str,
    *,
    thinking_budget_tokens: int = 0,
    thinking_effort: str = "high",
    max_tokens: int = 4096,
) -> ParsedLLMResponse:
    """Submit to batch API, wait for signal, then parse the response."""
    await workflow.execute_activity(
        "submit_batch_request",
        BatchSubmitInput(
            context=context,
            output_type_name=output_type_name,
            workflow_id=workflow.info().workflow_id,
            thinking_budget_tokens=thinking_budget_tokens,
            thinking_effort=thinking_effort,
            max_tokens=max_tokens,
        ),
        start_to_close_timeout=_SUBMIT_TIMEOUT,
        retry_policy=_LLM_RETRY,
        result_type=BatchSubmitResult,
    )
    await workflow.wait_condition(
        lambda: len(batch_results) > 0,
        timeout=_BATCH_WAIT_TIMEOUT,
    )
    result = batch_results.pop(0)
    if result.error:
        raise ApplicationError(f"Batch error: {result.error}")
    assert result.raw_response_json is not None
    return await workflow.execute_activity(
        "parse_llm_response",
        ParseResponseInput(
            raw_response_json=result.raw_response_json,
            output_type_name=output_type_name,
            task_id=context.task_id,
            log_messages=context.log_messages,
            worktree_path=context.worktree_path,
        ),
        start_to_close_timeout=_PARSE_TIMEOUT,
        retry_policy=_LOCAL_RETRY,
        result_type=ParsedLLMResponse,
    )


async def _call_generation_dispatch(
    batch_results: list[BatchResult],
    sync_mode: bool,
    context: AssembledContext,
) -> LLMCallResult:
    """Dispatch generation LLM call via sync or batch path."""
    if sync_mode:
        return await workflow.execute_activity(
            "call_llm",
            context,
            start_to_close_timeout=_LLM_TIMEOUT,
            heartbeat_timeout=_LLM_HEARTBEAT,
            retry_policy=_LLM_RETRY,
            result_type=LLMCallResult,
        )
    parsed = await _call_llm_batch_dispatch(batch_results, context, "LLMResponse")
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse.model_validate_json(parsed.parsed_json),
        model_name=parsed.model_name,
        input_tokens=parsed.input_tokens,
        output_tokens=parsed.output_tokens,
        latency_ms=parsed.latency_ms,
        cache_creation_input_tokens=parsed.cache_creation_input_tokens,
        cache_read_input_tokens=parsed.cache_read_input_tokens,
    )


async def _call_conflict_resolution_dispatch(
    batch_results: list[BatchResult],
    sync_mode: bool,
    call_input: ConflictResolutionCallInput,
) -> ConflictResolutionCallResult:
    """Dispatch conflict resolution LLM call via sync or batch path."""
    if sync_mode:
        return await workflow.execute_activity(
            "call_conflict_resolution",
            call_input,
            start_to_close_timeout=_CONFLICT_RESOLUTION_TIMEOUT,
            heartbeat_timeout=_LLM_HEARTBEAT,
            retry_policy=_LLM_RETRY,
            result_type=ConflictResolutionCallResult,
        )
    context = AssembledContext(
        task_id=call_input.task_id,
        system_prompt=call_input.system_prompt,
        user_prompt=call_input.user_prompt,
        model_name=call_input.model_name,
        log_messages=call_input.log_messages,
        worktree_path=call_input.worktree_path,
    )
    parsed = await _call_llm_batch_dispatch(
        batch_results,
        context,
        "ConflictResolutionResponse",
        thinking_budget_tokens=call_input.thinking_budget_tokens,
        thinking_effort=call_input.thinking_effort,
    )
    response = ConflictResolutionResponse.model_validate_json(parsed.parsed_json)
    return ConflictResolutionCallResult(
        task_id=call_input.task_id,
        resolved_files={f.file_path: f.content for f in response.resolved_files},
        explanation=response.explanation,
        model_name=parsed.model_name,
        input_tokens=parsed.input_tokens,
        output_tokens=parsed.output_tokens,
        latency_ms=parsed.latency_ms,
        cache_creation_input_tokens=parsed.cache_creation_input_tokens,
        cache_read_input_tokens=parsed.cache_read_input_tokens,
    )


async def _remove_worktree(repo_root: str, task_id: str) -> None:
    """Remove a worktree via activity. Shared by both workflow classes."""
    await workflow.execute_activity(
        "remove_worktree_activity",
        RemoveWorktreeInput(
            repo_root=repo_root,
            task_id=task_id,
            force=True,
        ),
        start_to_close_timeout=_GIT_TIMEOUT,
        retry_policy=_LOCAL_RETRY,
        result_type=type(None),
    )


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

    def __init__(self) -> None:
        self._batch_results: list[BatchResult] = []
        self._sync_mode: bool = True
        self._log_messages: bool = False

    @workflow.signal
    async def batch_result_received(self, result: BatchResult) -> None:
        self._batch_results.append(result)

    @workflow.run
    async def run(self, input: ForgeTaskInput) -> TaskResult:
        self._sync_mode = input.sync_mode
        self._log_messages = input.log_messages
        workflow.logger.info(
            "Workflow started: task_id=%s plan=%s sync=%s",
            input.task.task_id,
            input.plan,
            input.sync_mode,
        )
        if input.plan:
            return await self._run_planned(input)
        return await self._run_single_step(input)

    # ------------------------------------------------------------------
    # LLM dispatch methods (delegating to module-level shared functions)
    # ------------------------------------------------------------------

    async def _call_llm_batch(
        self,
        context: AssembledContext,
        output_type_name: str,
        *,
        thinking_budget_tokens: int = 0,
        thinking_effort: str = "high",
        max_tokens: int = 4096,
    ) -> ParsedLLMResponse:
        return await _call_llm_batch_dispatch(
            self._batch_results,
            context,
            output_type_name,
            thinking_budget_tokens=thinking_budget_tokens,
            thinking_effort=thinking_effort,
            max_tokens=max_tokens,
        )

    async def _call_generation(self, context: AssembledContext) -> LLMCallResult:
        return await _call_generation_dispatch(
            self._batch_results, self._sync_mode, context
        )

    async def _call_planner_llm(self, planner_input: PlannerInput) -> PlanCallResult:
        """Dispatch planner LLM call via sync or batch path."""
        if self._sync_mode:
            return await workflow.execute_activity(
                "call_planner",
                planner_input,
                start_to_close_timeout=_LLM_TIMEOUT,
                heartbeat_timeout=_LLM_HEARTBEAT,
                retry_policy=_LLM_RETRY,
                result_type=PlanCallResult,
            )
        context = AssembledContext(
            task_id=planner_input.task_id,
            system_prompt=planner_input.system_prompt,
            user_prompt=planner_input.user_prompt,
            model_name=planner_input.model_name,
            log_messages=planner_input.log_messages,
            worktree_path=planner_input.worktree_path,
        )
        parsed = await self._call_llm_batch(
            context,
            "Plan",
            thinking_budget_tokens=planner_input.thinking_budget_tokens,
            thinking_effort=planner_input.thinking_effort,
        )
        return PlanCallResult(
            task_id=planner_input.task_id,
            plan=Plan.model_validate_json(parsed.parsed_json),
            model_name=parsed.model_name,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            latency_ms=parsed.latency_ms,
            cache_creation_input_tokens=parsed.cache_creation_input_tokens,
            cache_read_input_tokens=parsed.cache_read_input_tokens,
        )

    async def _call_exploration(self, exploration_input: ExplorationInput) -> ExplorationResponse:
        """Dispatch exploration LLM call via sync or batch path."""
        if self._sync_mode:
            return await workflow.execute_activity(
                "call_exploration_llm",
                exploration_input,
                start_to_close_timeout=_EXPLORATION_LLM_TIMEOUT,
                heartbeat_timeout=_LLM_HEARTBEAT,
                retry_policy=_LLM_RETRY,
                result_type=ExplorationResponse,
            )
        context: AssembledContext = await workflow.execute_activity(
            "assemble_exploration_context",
            exploration_input,
            start_to_close_timeout=_CONTEXT_TIMEOUT,
            retry_policy=_LOCAL_RETRY,
            result_type=AssembledContext,
        )
        parsed = await self._call_llm_batch(context, "ExplorationResponse")
        return ExplorationResponse.model_validate_json(parsed.parsed_json)

    async def _call_sanity_check_llm(self, sanity_input: SanityCheckInput) -> SanityCheckCallResult:
        """Dispatch sanity check LLM call via sync or batch path."""
        if self._sync_mode:
            return await workflow.execute_activity(
                "call_sanity_check",
                sanity_input,
                start_to_close_timeout=_SANITY_CHECK_TIMEOUT,
                heartbeat_timeout=_LLM_HEARTBEAT,
                retry_policy=_LLM_RETRY,
                result_type=SanityCheckCallResult,
            )
        context = AssembledContext(
            task_id=sanity_input.task_id,
            system_prompt=sanity_input.system_prompt,
            user_prompt=sanity_input.user_prompt,
            model_name=sanity_input.model_name,
            log_messages=sanity_input.log_messages,
            worktree_path=sanity_input.worktree_path,
        )
        parsed = await self._call_llm_batch(
            context,
            "SanityCheckResponse",
            thinking_budget_tokens=sanity_input.thinking_budget_tokens,
            thinking_effort=sanity_input.thinking_effort,
        )
        response = SanityCheckResponse.model_validate_json(parsed.parsed_json)
        return SanityCheckCallResult(
            task_id=sanity_input.task_id,
            response=response,
            model_name=parsed.model_name,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            latency_ms=parsed.latency_ms,
            cache_creation_input_tokens=parsed.cache_creation_input_tokens,
            cache_read_input_tokens=parsed.cache_read_input_tokens,
        )

    async def _call_conflict_resolution(
        self, call_input: ConflictResolutionCallInput
    ) -> ConflictResolutionCallResult:
        return await _call_conflict_resolution_dispatch(
            self._batch_results, self._sync_mode, call_input
        )

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
            exploration_result = await self._call_exploration(exploration_input)
            workflow.logger.debug(
                "Exploration round %d: %d provider requests",
                round_num,
                len(exploration_result.requests),
            )

            if not exploration_result.requests:
                break  # LLM is ready to generate

            context_results = await workflow.execute_activity(
                "fulfill_context_requests",
                FulfillContextInput(
                    requests=exploration_result.requests,
                    repo_root=repo_root,
                    worktree_path=worktree_path,
                ),
                start_to_close_timeout=_EXPLORATION_FULFILL_TIMEOUT,
                retry_policy=_LOCAL_RETRY,
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
    # Phase 1: Single-step execution (unchanged from Phase 1)
    # ------------------------------------------------------------------

    async def _run_single_step(self, input: ForgeTaskInput) -> TaskResult:
        task = input.task
        max_attempts = input.max_attempts
        prior_errors: list[ValidationResult] = []

        # Resolve models for this workflow
        generation_model = resolve_model(CapabilityTier.GENERATION, input.model_routing)
        exploration_model = resolve_model(CapabilityTier.CLASSIFICATION, input.model_routing)

        for attempt in range(1, max_attempts + 1):
            workflow.logger.info(
                "Single-step attempt %d/%d: task_id=%s", attempt, max_attempts, task.task_id
            )

            # --- Create worktree ---
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

            # --- Assemble context ---
            context = await workflow.execute_activity(
                "assemble_context",
                AssembleContextInput(
                    task_id=task.task_id,
                    description=task.description,
                    target_files=task.target_files,
                    context_files=task.context_files,
                    context_config=task.context,
                    repo_root=input.repo_root,
                    worktree_path=wt_output.worktree_path,
                    prior_errors=prior_errors,
                    attempt=attempt,
                    max_attempts=max_attempts,
                ),
                start_to_close_timeout=_CONTEXT_TIMEOUT,
                retry_policy=_LOCAL_RETRY,
                result_type=AssembledContext,
            )

            # --- Exploration loop (Phase 7) ---
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
                    context = AssembledContext(
                        task_id=context.task_id,
                        system_prompt=context.system_prompt + exploration_section,
                        user_prompt=context.user_prompt,
                        context_stats=context.context_stats,
                        step_id=context.step_id,
                        sub_task_id=context.sub_task_id,
                    )

            # --- Set model_name and log_messages on context ---
            context = context.model_copy(
                update={
                    "model_name": generation_model,
                    "log_messages": self._log_messages,
                    "worktree_path": wt_output.worktree_path,
                }
            )

            # --- Call LLM ---
            llm_result = await self._call_generation(context)
            workflow.logger.info(
                "Generation complete: task_id=%s model=%s tokens=%din/%dout latency=%.0fms",
                task.task_id,
                llm_result.model_name,
                llm_result.input_tokens,
                llm_result.output_tokens,
                llm_result.latency_ms,
            )

            # --- Write output ---
            write_result = await workflow.execute_activity(
                "write_output",
                WriteOutputInput(
                    llm_result=llm_result,
                    worktree_path=wt_output.worktree_path,
                ),
                start_to_close_timeout=_WRITE_TIMEOUT,
                retry_policy=_WRITE_RETRY,
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
                heartbeat_timeout=_VALIDATE_HEARTBEAT,
                retry_policy=_LOCAL_RETRY,
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
                retry_policy=_LOCAL_RETRY,
                result_type=str,
            )
            signal = TransitionSignal(signal_value)
            workflow.logger.info(
                "Transition: task_id=%s signal=%s attempt=%d/%d",
                task.task_id,
                signal.value,
                attempt,
                max_attempts,
            )

            # --- Collect output files ---
            output_files = write_result.output_files

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
                    retry_policy=_GIT_RETRY,
                    result_type=CommitChangesOutput,
                )
                return TaskResult(
                    task_id=task.task_id,
                    status=TransitionSignal.SUCCESS,
                    output_files=output_files,
                    validation_results=validation_results,
                    worktree_path=wt_output.worktree_path,
                    worktree_branch=wt_output.branch_name,
                    llm_stats=build_llm_stats(llm_result),
                    context_stats=context.context_stats,
                )

            if signal == TransitionSignal.FAILURE_RETRYABLE:
                prior_errors = validation_results
                await _remove_worktree(input.repo_root, task.task_id)
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
                retry_policy=_GIT_RETRY,
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
                llm_stats=build_llm_stats(llm_result),
                context_stats=context.context_stats,
            )

        # Should not be reachable, but satisfy the type checker.
        msg = "Exhausted all attempts without a terminal transition"
        raise RuntimeError(msg)

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
            retry_policy=_LOCAL_RETRY,
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
            "log_messages": self._log_messages,
            "worktree_path": wt_output.worktree_path,
        }
        if input.thinking.enabled:
            planner_update["thinking_budget_tokens"] = input.thinking.budget_tokens
            planner_update["thinking_effort"] = input.thinking.effort
        planner_input = planner_input.model_copy(update=planner_update)

        # --- Call planner ---
        planner_result = await self._call_planner_llm(planner_input)
        plan: Plan = planner_result.plan
        workflow.logger.info("Plan created: task_id=%s steps=%d", task.task_id, len(plan.steps))
        return plan, build_planner_stats(planner_result)

    async def _run_planned(self, input: ForgeTaskInput) -> TaskResult:
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
                    step_results,
                    step_model=step_model,
                )
                step_results.append(step_result)
                if step_result.status != TransitionSignal.SUCCESS:
                    return TaskResult(
                        task_id=task.task_id,
                        status=TransitionSignal.FAILURE_TERMINAL,
                        output_files=all_output_files,
                        error=f"Step {step.step_id} fan-out failed: {step_result.error}",
                        worktree_path=wt_output.worktree_path,
                        worktree_branch=wt_output.branch_name,
                        step_results=step_results,
                        plan=plan,
                        planner_stats=p_stats,
                        sanity_check_count=sanity_check_count,
                    )
                all_output_files.update(step_result.output_files)
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
            step_results.append(step_result)
            if step_result.status != TransitionSignal.SUCCESS:
                return TaskResult(
                    task_id=task.task_id,
                    status=TransitionSignal.FAILURE_TERMINAL,
                    output_files=all_output_files,
                    error=f"Step {step.step_id} failed: {step_result.error}",
                    worktree_path=wt_output.worktree_path,
                    worktree_branch=wt_output.branch_name,
                    step_results=step_results,
                    plan=plan,
                    planner_stats=p_stats,
                    sanity_check_count=sanity_check_count,
                )
            all_output_files.update(step_result.output_files)

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
                    return TaskResult(
                        task_id=task.task_id,
                        status=TransitionSignal.FAILURE_TERMINAL,
                        output_files=all_output_files,
                        error=f"Sanity check aborted: {sanity_result.response.explanation}",
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
        """Execute a single step through its retry loop.

        Returns StepResult with SUCCESS or FAILURE_TERMINAL status.
        Raises RuntimeError if retries exhaust without a terminal signal.
        """
        task = input.task
        max_step_attempts = input.max_step_attempts
        prior_errors: list[ValidationResult] = []

        for attempt in range(1, max_step_attempts + 1):
            # --- Assemble step context ---
            context = await workflow.execute_activity(
                "assemble_step_context",
                AssembleStepContextInput(
                    task_id=task.task_id,
                    task_description=task.description,
                    context_config=task.context,
                    step=step,
                    step_index=step_index,
                    total_steps=total_steps,
                    completed_steps=step_results,
                    repo_root=input.repo_root,
                    worktree_path=wt_output.worktree_path,
                    prior_errors=prior_errors,
                    attempt=attempt,
                    max_attempts=max_step_attempts,
                ),
                start_to_close_timeout=_CONTEXT_TIMEOUT,
                retry_policy=_LOCAL_RETRY,
                result_type=AssembledContext,
            )

            # --- Set model_name and log_messages on context ---
            context = context.model_copy(
                update={
                    "model_name": step_model,
                    "log_messages": self._log_messages,
                    "worktree_path": wt_output.worktree_path,
                }
            )

            # --- Call LLM ---
            llm_result = await self._call_generation(context)

            # --- Write output ---
            write_result = await workflow.execute_activity(
                "write_output",
                WriteOutputInput(
                    llm_result=llm_result,
                    worktree_path=wt_output.worktree_path,
                ),
                start_to_close_timeout=_WRITE_TIMEOUT,
                retry_policy=_WRITE_RETRY,
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
                heartbeat_timeout=_VALIDATE_HEARTBEAT,
                retry_policy=_LOCAL_RETRY,
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
                retry_policy=_LOCAL_RETRY,
                result_type=str,
            )
            signal = TransitionSignal(signal_value)
            workflow.logger.info(
                "Step transition: step_id=%s signal=%s attempt=%d/%d",
                step.step_id,
                signal.value,
                attempt,
                max_step_attempts,
            )

            output_files = write_result.output_files

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
                    retry_policy=_GIT_RETRY,
                    result_type=CommitChangesOutput,
                )
                return StepResult(
                    step_id=step.step_id,
                    status=TransitionSignal.SUCCESS,
                    output_files=output_files,
                    validation_results=validation_results,
                    commit_sha=commit_output.commit_sha,
                    llm_stats=build_llm_stats(llm_result),
                )

            if signal == TransitionSignal.FAILURE_RETRYABLE:
                prior_errors = validation_results
                # Reset worktree (discard uncommitted changes) and retry
                await workflow.execute_activity(
                    "reset_worktree_activity",
                    ResetWorktreeInput(
                        repo_root=input.repo_root,
                        task_id=task.task_id,
                    ),
                    start_to_close_timeout=_GIT_TIMEOUT,
                    retry_policy=_GIT_RETRY,
                    result_type=type(None),
                )
                continue

            # FAILURE_TERMINAL — step failed
            error = "; ".join(r.summary for r in validation_results if not r.passed)
            return StepResult(
                step_id=step.step_id,
                status=TransitionSignal.FAILURE_TERMINAL,
                output_files=output_files,
                validation_results=validation_results,
                error=error,
                llm_stats=build_llm_stats(llm_result),
            )

        # Exhausted retries for this step — should have hit TERMINAL above
        msg = f"Step {step.step_id} exhausted all attempts without a terminal transition"
        raise RuntimeError(msg)

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
            retry_policy=_LOCAL_RETRY,
            result_type=SanityCheckInput,
        )

        # Set model, thinking config, and log_messages
        update: dict[str, object] = {
            "model_name": reasoning_model,
            "log_messages": self._log_messages,
            "worktree_path": wt_output.worktree_path,
        }
        if input.thinking.enabled:
            update["thinking_budget_tokens"] = input.thinking.budget_tokens
            update["thinking_effort"] = input.thinking.effort
        sanity_input = sanity_input.model_copy(update=update)

        return await self._call_sanity_check_llm(sanity_input)

    # ------------------------------------------------------------------
    # Phase 3: Fan-out step execution
    # ------------------------------------------------------------------

    async def _run_fan_out_step(
        self,
        input: ForgeTaskInput,
        step: PlanStep,
        wt_output: CreateWorktreeOutput,
        prior_step_results: list[StepResult],
        step_model: str = "",
    ) -> StepResult:
        """Execute a fan-out step by spawning child workflows in parallel.

        1. Validate sub-task ID uniqueness
        2. Start child workflows in parallel
        3. Await all children
        4. Check for file conflicts
        5. Write merged files to parent worktree
        6. Validate and commit
        """
        task = input.task
        sub_tasks = step.sub_tasks
        assert sub_tasks  # Caller guarantees this

        # --- Validate unique sub-task IDs ---
        sub_task_ids = [st.sub_task_id for st in sub_tasks]
        if len(sub_task_ids) != len(set(sub_task_ids)):
            return StepResult(
                step_id=step.step_id,
                status=TransitionSignal.FAILURE_TERMINAL,
                error="Duplicate sub-task IDs detected",
            )

        # --- Start child workflows in parallel ---
        workflow.logger.info("Fan-out: step_id=%s sub_tasks=%d", step.step_id, len(sub_tasks))
        child_timeout = _child_timeout(0, input.max_fan_out_depth)
        handles = []
        for st in sub_tasks:
            child_input = SubTaskInput(
                parent_task_id=task.task_id,
                parent_description=task.description,
                sub_task=st,
                repo_root=input.repo_root,
                parent_branch=wt_output.branch_name,
                validation=task.validation,
                max_attempts=input.max_sub_task_attempts,
                model_name=step_model,
                domain=task.domain,
                depth=0,
                max_depth=input.max_fan_out_depth,
                sync_mode=input.sync_mode,
                log_messages=self._log_messages,
            )
            compound_id = f"{task.task_id}.sub.{st.sub_task_id}"
            handle = await workflow.start_child_workflow(
                ForgeSubTaskWorkflow.run,
                child_input,
                id=f"forge-subtask-{compound_id}",
                task_queue=FORGE_TASK_QUEUE,
                execution_timeout=child_timeout,
            )
            handles.append(handle)

        # --- Await all children ---
        sub_task_results: list[SubTaskResult] = []
        for handle in handles:
            result: SubTaskResult = await handle
            sub_task_results.append(result)

        # --- Check for failures ---
        failures = [r for r in sub_task_results if r.status != TransitionSignal.SUCCESS]
        successes = len(sub_task_results) - len(failures)
        workflow.logger.info(
            "Fan-out gather: step_id=%s successes=%d failures=%d",
            step.step_id,
            successes,
            len(failures),
        )
        if failures:
            error_parts = [f"{r.sub_task_id}: {r.error}" for r in failures]
            return StepResult(
                step_id=step.step_id,
                status=TransitionSignal.FAILURE_TERMINAL,
                sub_task_results=sub_task_results,
                error="; ".join(error_parts),
            )

        # --- Detect and resolve file conflicts ---
        detect_result = await workflow.execute_activity(
            "detect_file_conflicts_activity",
            DetectFileConflictsInput(
                sub_task_results=sub_task_results,
                worktree_path=wt_output.worktree_path,
            ),
            start_to_close_timeout=_GIT_TIMEOUT,
            retry_policy=_LOCAL_RETRY,
            result_type=DetectFileConflictsOutput,
        )
        non_conflicting = detect_result.non_conflicting_files
        conflicts = detect_result.conflicts
        conflict_resolution_result: ConflictResolutionCallResult | None = None

        if conflicts:
            workflow.logger.info(
                "Conflict resolution: step_id=%s conflicts=%d", step.step_id, len(conflicts)
            )

        if conflicts and input.resolve_conflicts:
            call_input = await _assemble_conflict_resolution(
                task_id=task.task_id,
                step_id=step.step_id,
                conflicts=conflicts,
                non_conflicting=non_conflicting,
                task_description=task.description,
                step_description=step.description,
                repo_root=input.repo_root,
                worktree_path=wt_output.worktree_path,
                domain=task.domain,
                model_routing=input.model_routing,
                thinking=input.thinking,
                log_messages=self._log_messages,
            )
            conflict_resolution_result = await self._call_conflict_resolution(call_input)

            conflict_paths = {c.file_path for c in conflicts}
            resolved_paths = set(conflict_resolution_result.resolved_files.keys())
            missing = conflict_paths - resolved_paths
            if missing:
                return StepResult(
                    step_id=step.step_id,
                    status=TransitionSignal.FAILURE_TERMINAL,
                    sub_task_results=sub_task_results,
                    conflict_resolution=conflict_resolution_result,
                    error=(
                        f"Conflict resolution incomplete: "
                        f"missing resolved files: {', '.join(sorted(missing))}"
                    ),
                )

            merged_files = {**non_conflicting, **conflict_resolution_result.resolved_files}
        elif conflicts:
            # resolve_conflicts=False: fall back to D27 terminal error
            conflict_paths_str = ", ".join(c.file_path for c in conflicts)
            return StepResult(
                step_id=step.step_id,
                status=TransitionSignal.FAILURE_TERMINAL,
                sub_task_results=sub_task_results,
                error=f"File conflict: {conflict_paths_str} produced by multiple sub-tasks",
            )
        else:
            merged_files = non_conflicting

        # --- Write merged files to parent worktree ---
        if merged_files:
            write_result = await workflow.execute_activity(
                "write_files",
                WriteFilesInput(
                    task_id=task.task_id,
                    worktree_path=wt_output.worktree_path,
                    files=merged_files,
                ),
                start_to_close_timeout=_WRITE_TIMEOUT,
                retry_policy=_WRITE_RETRY,
                result_type=WriteResult,
            )

            # --- Validate merged output ---
            validation_results = await workflow.execute_activity(
                "validate_output",
                ValidateOutputInput(
                    task_id=task.task_id,
                    worktree_path=wt_output.worktree_path,
                    files=write_result.files_written,
                    validation=task.validation,
                ),
                start_to_close_timeout=_VALIDATE_TIMEOUT,
                heartbeat_timeout=_VALIDATE_HEARTBEAT,
                retry_policy=_LOCAL_RETRY,
                result_type=list[ValidationResult],
            )

            # --- Evaluate transition (single attempt for merged output) ---
            signal_value = await workflow.execute_activity(
                "evaluate_transition",
                TransitionInput(
                    validation_results=validation_results,
                    attempt=1,
                    max_attempts=1,
                ),
                start_to_close_timeout=_TRANSITION_TIMEOUT,
                retry_policy=_LOCAL_RETRY,
                result_type=str,
            )
            signal = TransitionSignal(signal_value)

            if signal != TransitionSignal.SUCCESS:
                error = "; ".join(r.summary for r in validation_results if not r.passed)
                return StepResult(
                    step_id=step.step_id,
                    status=TransitionSignal.FAILURE_TERMINAL,
                    output_files=merged_files,
                    validation_results=validation_results,
                    sub_task_results=sub_task_results,
                    error=f"Merged output validation failed: {error}",
                )
        else:
            validation_results = []

        # --- Commit ---
        commit_msg = f"forge({task.task_id}): step {step.step_id} fan-out gather"
        commit_output = await workflow.execute_activity(
            "commit_changes_activity",
            CommitChangesInput(
                repo_root=input.repo_root,
                task_id=task.task_id,
                status="success",
                message=commit_msg,
            ),
            start_to_close_timeout=_GIT_TIMEOUT,
            retry_policy=_GIT_RETRY,
            result_type=CommitChangesOutput,
        )

        return StepResult(
            step_id=step.step_id,
            status=TransitionSignal.SUCCESS,
            output_files=merged_files,
            validation_results=validation_results,
            commit_sha=commit_output.commit_sha,
            sub_task_results=sub_task_results,
            conflict_resolution=conflict_resolution_result,
        )


# ===========================================================================
# Phase 3: Sub-task child workflow
# ===========================================================================


@workflow.defn
class ForgeSubTaskWorkflow:
    """Execute a single sub-task within a fan-out step.

    Routes between two execution paths:

    Single-step (leaf or depth >= max_depth):
        1. Create worktree (compound ID, branched from parent branch)
        2. Retry loop:
           - assemble_sub_task_context → call_llm → write_output → validate → transition
           - SUCCESS: collect output_files, remove worktree, return SubTaskResult
           - FAILURE_RETRYABLE: remove worktree, recreate on next iteration
           - FAILURE_TERMINAL: remove worktree, return failure SubTaskResult

    Nested fan-out (has sub_tasks and depth < max_depth):
        1. Create worktree (compound ID, branched from parent branch)
        2. Validate nested sub-task ID uniqueness
        3. Start child ForgeSubTaskWorkflow instances in parallel (depth+1)
        4. Await all children, check failures / file conflicts
        5. Write merged files to worktree, validate merged output
        6. Remove worktree (sub-tasks never commit, D16)
        7. Return SubTaskResult with merged output_files and nested sub_task_results
    """

    def __init__(self) -> None:
        self._batch_results: list[BatchResult] = []
        self._sync_mode: bool = True
        self._log_messages: bool = False

    @workflow.signal
    async def batch_result_received(self, result: BatchResult) -> None:
        self._batch_results.append(result)

    @workflow.run
    async def run(self, input: SubTaskInput) -> SubTaskResult:
        self._sync_mode = input.sync_mode
        self._log_messages = input.log_messages
        workflow.logger.info(
            "Sub-task started: sub_task_id=%s depth=%d/%d",
            input.sub_task.sub_task_id,
            input.depth,
            input.max_depth,
        )
        if input.sub_task.sub_tasks and input.depth < input.max_depth:
            return await self._run_nested_fan_out(input)
        return await self._run_single_step(input)

    # ------------------------------------------------------------------
    # LLM dispatch methods (delegating to module-level shared functions)
    # ------------------------------------------------------------------

    async def _call_llm_batch(
        self,
        context: AssembledContext,
        output_type_name: str,
        *,
        thinking_budget_tokens: int = 0,
        thinking_effort: str = "high",
        max_tokens: int = 4096,
    ) -> ParsedLLMResponse:
        return await _call_llm_batch_dispatch(
            self._batch_results,
            context,
            output_type_name,
            thinking_budget_tokens=thinking_budget_tokens,
            thinking_effort=thinking_effort,
            max_tokens=max_tokens,
        )

    async def _call_generation(self, context: AssembledContext) -> LLMCallResult:
        return await _call_generation_dispatch(
            self._batch_results, self._sync_mode, context
        )

    async def _call_conflict_resolution(
        self, call_input: ConflictResolutionCallInput
    ) -> ConflictResolutionCallResult:
        return await _call_conflict_resolution_dispatch(
            self._batch_results, self._sync_mode, call_input
        )

    async def _run_single_step(self, input: SubTaskInput) -> SubTaskResult:
        """Execute a leaf sub-task: LLM call with retry loop."""
        compound_id = f"{input.parent_task_id}.sub.{input.sub_task.sub_task_id}"
        prior_errors: list[ValidationResult] = []

        for attempt in range(1, input.max_attempts + 1):
            # --- Create worktree ---
            wt_output = await workflow.execute_activity(
                "create_worktree_activity",
                CreateWorktreeInput(
                    repo_root=input.repo_root,
                    task_id=compound_id,
                    base_branch=input.parent_branch,
                ),
                start_to_close_timeout=_GIT_TIMEOUT,
                retry_policy=_GIT_RETRY,
                result_type=CreateWorktreeOutput,
            )

            # --- Assemble sub-task context ---
            context = await workflow.execute_activity(
                "assemble_sub_task_context",
                AssembleSubTaskContextInput(
                    parent_task_id=input.parent_task_id,
                    parent_description=input.parent_description,
                    sub_task=input.sub_task,
                    worktree_path=wt_output.worktree_path,
                    repo_root=input.repo_root,
                    prior_errors=prior_errors,
                    attempt=attempt,
                    max_attempts=input.max_attempts,
                    domain=input.domain,
                ),
                start_to_close_timeout=_CONTEXT_TIMEOUT,
                retry_policy=_LOCAL_RETRY,
                result_type=AssembledContext,
            )

            # --- Set model_name, log_messages from parent ---
            ctx_update: dict[str, object] = {
                "log_messages": self._log_messages,
                "worktree_path": wt_output.worktree_path,
            }
            if input.model_name:
                ctx_update["model_name"] = input.model_name
            context = context.model_copy(update=ctx_update)

            # --- Call LLM ---
            llm_result = await self._call_generation(context)

            # --- Write output ---
            write_result = await workflow.execute_activity(
                "write_output",
                WriteOutputInput(
                    llm_result=llm_result,
                    worktree_path=wt_output.worktree_path,
                ),
                start_to_close_timeout=_WRITE_TIMEOUT,
                retry_policy=_WRITE_RETRY,
                result_type=WriteResult,
            )

            # --- Validate output ---
            validation_results = await workflow.execute_activity(
                "validate_output",
                ValidateOutputInput(
                    task_id=compound_id,
                    worktree_path=wt_output.worktree_path,
                    files=write_result.files_written,
                    validation=input.validation,
                ),
                start_to_close_timeout=_VALIDATE_TIMEOUT,
                heartbeat_timeout=_VALIDATE_HEARTBEAT,
                retry_policy=_LOCAL_RETRY,
                result_type=list[ValidationResult],
            )

            # --- Evaluate transition ---
            signal_value = await workflow.execute_activity(
                "evaluate_transition",
                TransitionInput(
                    validation_results=validation_results,
                    attempt=attempt,
                    max_attempts=input.max_attempts,
                ),
                start_to_close_timeout=_TRANSITION_TIMEOUT,
                retry_policy=_LOCAL_RETRY,
                result_type=str,
            )
            signal = TransitionSignal(signal_value)
            workflow.logger.info(
                "Sub-task transition: sub_task_id=%s signal=%s attempt=%d/%d",
                input.sub_task.sub_task_id,
                signal.value,
                attempt,
                input.max_attempts,
            )

            # --- Collect output files before cleanup ---
            output_files = write_result.output_files
            digest = llm_result.response.explanation

            # --- Remove worktree (always — sub-tasks don't commit) ---
            await _remove_worktree(input.repo_root, compound_id)

            if signal == TransitionSignal.SUCCESS:
                return SubTaskResult(
                    sub_task_id=input.sub_task.sub_task_id,
                    status=TransitionSignal.SUCCESS,
                    output_files=output_files,
                    validation_results=validation_results,
                    digest=digest,
                    llm_stats=build_llm_stats(llm_result),
                )

            if signal == TransitionSignal.FAILURE_TERMINAL:
                error = "; ".join(r.summary for r in validation_results if not r.passed)
                return SubTaskResult(
                    sub_task_id=input.sub_task.sub_task_id,
                    status=TransitionSignal.FAILURE_TERMINAL,
                    validation_results=validation_results,
                    error=error,
                    llm_stats=build_llm_stats(llm_result),
                )

            # FAILURE_RETRYABLE — worktree already removed, loop will recreate
            prior_errors = validation_results

        # Should not be reachable, but satisfy the type checker.
        msg = f"Sub-task {input.sub_task.sub_task_id} exhausted all attempts"
        raise RuntimeError(msg)

    async def _run_nested_fan_out(self, input: SubTaskInput) -> SubTaskResult:
        """Execute a sub-task that itself contains nested sub-tasks."""
        compound_id = f"{input.parent_task_id}.sub.{input.sub_task.sub_task_id}"
        nested_sub_tasks = input.sub_task.sub_tasks
        assert nested_sub_tasks  # Caller guarantees this

        # --- Create worktree ---
        wt_output = await workflow.execute_activity(
            "create_worktree_activity",
            CreateWorktreeInput(
                repo_root=input.repo_root,
                task_id=compound_id,
                base_branch=input.parent_branch,
            ),
            start_to_close_timeout=_GIT_TIMEOUT,
            retry_policy=_GIT_RETRY,
            result_type=CreateWorktreeOutput,
        )

        # --- Validate unique sub-task IDs ---
        nested_ids = [st.sub_task_id for st in nested_sub_tasks]
        if len(nested_ids) != len(set(nested_ids)):
            await _remove_worktree(input.repo_root, compound_id)
            return SubTaskResult(
                sub_task_id=input.sub_task.sub_task_id,
                status=TransitionSignal.FAILURE_TERMINAL,
                error="Duplicate nested sub-task IDs detected",
            )

        # --- Start child workflows in parallel ---
        child_timeout = _child_timeout(input.depth + 1, input.max_depth)
        handles = []
        for st in nested_sub_tasks:
            child_input = SubTaskInput(
                parent_task_id=compound_id,
                parent_description=input.parent_description,
                sub_task=st,
                repo_root=input.repo_root,
                parent_branch=wt_output.branch_name,
                validation=input.validation,
                max_attempts=input.max_attempts,
                model_name=input.model_name,
                domain=input.domain,
                depth=input.depth + 1,
                max_depth=input.max_depth,
                sync_mode=input.sync_mode,
                log_messages=self._log_messages,
            )
            child_compound_id = f"{compound_id}.sub.{st.sub_task_id}"
            handle = await workflow.start_child_workflow(
                ForgeSubTaskWorkflow.run,
                child_input,
                id=f"forge-subtask-{child_compound_id}",
                task_queue=FORGE_TASK_QUEUE,
                execution_timeout=child_timeout,
            )
            handles.append(handle)

        # --- Await all children ---
        sub_task_results: list[SubTaskResult] = []
        for handle in handles:
            result: SubTaskResult = await handle
            sub_task_results.append(result)

        # --- Check for failures ---
        failures = [r for r in sub_task_results if r.status != TransitionSignal.SUCCESS]
        if failures:
            await _remove_worktree(input.repo_root, compound_id)
            error_parts = [f"{r.sub_task_id}: {r.error}" for r in failures]
            return SubTaskResult(
                sub_task_id=input.sub_task.sub_task_id,
                status=TransitionSignal.FAILURE_TERMINAL,
                sub_task_results=sub_task_results,
                error="; ".join(error_parts),
            )

        # --- Detect and resolve file conflicts ---
        detect_result = await workflow.execute_activity(
            "detect_file_conflicts_activity",
            DetectFileConflictsInput(
                sub_task_results=sub_task_results,
                worktree_path=wt_output.worktree_path,
            ),
            start_to_close_timeout=_GIT_TIMEOUT,
            retry_policy=_LOCAL_RETRY,
            result_type=DetectFileConflictsOutput,
        )
        non_conflicting = detect_result.non_conflicting_files
        conflicts = detect_result.conflicts
        conflict_resolution_result: ConflictResolutionCallResult | None = None

        if conflicts:
            # Build a ModelConfig that uses the inherited model_name for reasoning
            resolution_model_routing = ModelConfig()
            if input.model_name:
                resolution_model_routing = resolution_model_routing.model_copy(
                    update={"reasoning": input.model_name}
                )

            cr_call_input = await _assemble_conflict_resolution(
                task_id=compound_id,
                step_id=input.sub_task.sub_task_id,
                conflicts=conflicts,
                non_conflicting=non_conflicting,
                task_description=input.parent_description,
                step_description=input.sub_task.description,
                repo_root=input.repo_root,
                worktree_path=wt_output.worktree_path,
                domain=input.domain,
                model_routing=resolution_model_routing,
                thinking=ThinkingConfig(),
                log_messages=self._log_messages,
            )
            conflict_resolution_result = await self._call_conflict_resolution(cr_call_input)

            conflict_paths = {c.file_path for c in conflicts}
            resolved_paths = set(conflict_resolution_result.resolved_files.keys())
            missing = conflict_paths - resolved_paths
            if missing:
                await _remove_worktree(input.repo_root, compound_id)
                return SubTaskResult(
                    sub_task_id=input.sub_task.sub_task_id,
                    status=TransitionSignal.FAILURE_TERMINAL,
                    sub_task_results=sub_task_results,
                    conflict_resolution=conflict_resolution_result,
                    error=(
                        f"Conflict resolution incomplete: "
                        f"missing resolved files: {', '.join(sorted(missing))}"
                    ),
                )

            merged_files = {**non_conflicting, **conflict_resolution_result.resolved_files}
        else:
            merged_files = non_conflicting

        # --- Write merged files to worktree and validate ---
        validation_results: list[ValidationResult] = []
        if merged_files:
            write_result = await workflow.execute_activity(
                "write_files",
                WriteFilesInput(
                    task_id=compound_id,
                    worktree_path=wt_output.worktree_path,
                    files=merged_files,
                ),
                start_to_close_timeout=_WRITE_TIMEOUT,
                retry_policy=_WRITE_RETRY,
                result_type=WriteResult,
            )

            validation_results = await workflow.execute_activity(
                "validate_output",
                ValidateOutputInput(
                    task_id=compound_id,
                    worktree_path=wt_output.worktree_path,
                    files=write_result.files_written,
                    validation=input.validation,
                ),
                start_to_close_timeout=_VALIDATE_TIMEOUT,
                heartbeat_timeout=_VALIDATE_HEARTBEAT,
                retry_policy=_LOCAL_RETRY,
                result_type=list[ValidationResult],
            )

            signal_value = await workflow.execute_activity(
                "evaluate_transition",
                TransitionInput(
                    validation_results=validation_results,
                    attempt=1,
                    max_attempts=1,
                ),
                start_to_close_timeout=_TRANSITION_TIMEOUT,
                retry_policy=_LOCAL_RETRY,
                result_type=str,
            )
            signal = TransitionSignal(signal_value)

            if signal != TransitionSignal.SUCCESS:
                await _remove_worktree(input.repo_root, compound_id)
                error = "; ".join(r.summary for r in validation_results if not r.passed)
                return SubTaskResult(
                    sub_task_id=input.sub_task.sub_task_id,
                    status=TransitionSignal.FAILURE_TERMINAL,
                    output_files=merged_files,
                    validation_results=validation_results,
                    sub_task_results=sub_task_results,
                    error=f"Merged output validation failed: {error}",
                )

        # --- Remove worktree (sub-tasks never commit, D16) ---
        await _remove_worktree(input.repo_root, compound_id)

        return SubTaskResult(
            sub_task_id=input.sub_task.sub_task_id,
            status=TransitionSignal.SUCCESS,
            output_files=merged_files,
            validation_results=validation_results,
            sub_task_results=sub_task_results,
            conflict_resolution=conflict_resolution_result,
        )
