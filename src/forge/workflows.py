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
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from forge.models import (
        ActionRequest,
        ActionResult,
        ActionWorkflowInput,
        ActionWorkflowResult,
        AssembleContextInput,
        AssembledContext,
        AssembleSanityCheckContextInput,
        AssembleStepContextInput,
        AssembleSubTaskContextInput,
        BatchResult,
        BatchSubmitInput,
        BatchSubmitResult,
        CapabilityTier,
        CapabilityType,
        CommitChangesInput,
        CommitChangesOutput,
        ConflictResolutionCallInput,
        ConflictResolutionCallResult,
        ConflictResolutionInput,
        ConflictResolutionResponse,
        ContextProviderSpec,
        ContextResult,
        CreateWorktreeInput,
        CreateWorktreeOutput,
        ExecuteMCPToolInput,
        ExecuteMCPToolResult,
        ExecuteSkillInput,
        ExecuteSkillResult,
        ExplorationInput,
        ExplorationResponse,
        FileConflict,
        ForgeTaskInput,
        FulfillContextInput,
        LLMCallResult,
        LLMResponse,
        MCPConfig,
        MCPToolInfo,
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
        SkillDefinition,
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
_MCP_TOOL_TIMEOUT = timedelta(minutes=2)
_SKILL_EXECUTION_TIMEOUT = timedelta(minutes=5)
_ACTION_WORKFLOW_TIMEOUT = timedelta(minutes=5)
_CAPABILITY_DISCOVERY_TIMEOUT = timedelta(minutes=2)

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

    return await workflow.execute_activity(
        "assemble_conflict_resolution_context",
        resolution_input,
        start_to_close_timeout=_CONTEXT_TIMEOUT,
        result_type=ConflictResolutionCallInput,
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

    @workflow.signal
    async def batch_result_received(self, result: BatchResult) -> None:
        self._batch_results.append(result)

    @workflow.run
    async def run(self, input: ForgeTaskInput) -> TaskResult:
        self._sync_mode = input.sync_mode
        if input.plan:
            return await self._run_planned(input)
        return await self._run_single_step(input)

    # ------------------------------------------------------------------
    # Batch dispatch infrastructure
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
            result_type=BatchSubmitResult,
        )
        # Wait for the poller (14c) to deliver the result via signal
        await workflow.wait_condition(lambda: len(self._batch_results) > 0)
        result = self._batch_results.pop(0)
        if result.error:
            raise ApplicationError(f"Batch error: {result.error}")
        assert result.raw_response_json is not None
        return await workflow.execute_activity(
            "parse_llm_response",
            ParseResponseInput(
                raw_response_json=result.raw_response_json,
                output_type_name=output_type_name,
                task_id=context.task_id,
            ),
            start_to_close_timeout=_PARSE_TIMEOUT,
            result_type=ParsedLLMResponse,
        )

    # ------------------------------------------------------------------
    # Per-call-type dispatch methods
    # ------------------------------------------------------------------

    async def _call_generation(self, context: AssembledContext) -> LLMCallResult:
        """Dispatch generation LLM call via sync or batch path."""
        if self._sync_mode:
            return await workflow.execute_activity(
                "call_llm",
                context,
                start_to_close_timeout=_LLM_TIMEOUT,
                result_type=LLMCallResult,
            )
        parsed = await self._call_llm_batch(context, "LLMResponse")
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

    async def _call_planner_llm(self, planner_input: PlannerInput) -> PlanCallResult:
        """Dispatch planner LLM call via sync or batch path."""
        if self._sync_mode:
            return await workflow.execute_activity(
                "call_planner",
                planner_input,
                start_to_close_timeout=_LLM_TIMEOUT,
                result_type=PlanCallResult,
            )
        context = AssembledContext(
            task_id=planner_input.task_id,
            system_prompt=planner_input.system_prompt,
            user_prompt=planner_input.user_prompt,
            model_name=planner_input.model_name,
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
                result_type=ExplorationResponse,
            )
        context: AssembledContext = await workflow.execute_activity(
            "assemble_exploration_context",
            exploration_input,
            start_to_close_timeout=_CONTEXT_TIMEOUT,
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
                result_type=SanityCheckCallResult,
            )
        context = AssembledContext(
            task_id=sanity_input.task_id,
            system_prompt=sanity_input.system_prompt,
            user_prompt=sanity_input.user_prompt,
            model_name=sanity_input.model_name,
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
        """Dispatch conflict resolution LLM call via sync or batch path."""
        if self._sync_mode:
            return await workflow.execute_activity(
                "call_conflict_resolution",
                call_input,
                start_to_close_timeout=_CONFLICT_RESOLUTION_TIMEOUT,
                result_type=ConflictResolutionCallResult,
            )
        context = AssembledContext(
            task_id=call_input.task_id,
            system_prompt=call_input.system_prompt,
            user_prompt=call_input.user_prompt,
            model_name=call_input.model_name,
        )
        parsed = await self._call_llm_batch(
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
        capability_specs: list[ContextProviderSpec] | None = None,
        mcp_config: MCPConfig | None = None,
        mcp_tools: dict[str, MCPToolInfo] | None = None,
        skill_definitions: dict[str, SkillDefinition] | None = None,
    ) -> list[ContextResult]:
        """LLM-guided context exploration loop.

        The exploration LLM requests context from providers and actions
        (MCP tools / Skills) until it signals readiness (empty requests
        and empty actions) or the round limit is reached.

        Phase 15: When the LLM returns action requests, they are dispatched
        as ForgeActionWorkflow child workflows in parallel. Results flow
        back as ContextResult entries for the next exploration round.
        """
        specs = capability_specs or list(PROVIDER_SPECS)
        accumulated: list[ContextResult] = []

        for round_num in range(1, max_rounds + 1):
            exploration_input = ExplorationInput(
                task=task,
                available_providers=specs,
                accumulated_context=accumulated,
                round_number=round_num,
                max_rounds=max_rounds,
                repo_root=repo_root,
                model_name=model_name,
            )
            exploration_result = await self._call_exploration(exploration_input)

            has_requests = bool(exploration_result.requests)
            has_actions = bool(exploration_result.actions)

            if not has_requests and not has_actions:
                break  # LLM is ready to generate

            # --- Fulfill context requests (existing path) ---
            if has_requests:
                context_results = await workflow.execute_activity(
                    "fulfill_context_requests",
                    FulfillContextInput(
                        requests=exploration_result.requests,
                        repo_root=repo_root,
                        worktree_path=worktree_path,
                    ),
                    start_to_close_timeout=_EXPLORATION_FULFILL_TIMEOUT,
                    result_type=list[ContextResult],
                )
                accumulated.extend(context_results)

            # --- Dispatch action requests as child workflows (Phase 15) ---
            if has_actions:
                action_results = await self._dispatch_actions(
                    actions=exploration_result.actions,
                    mcp_config=mcp_config or MCPConfig(),
                    mcp_tools=mcp_tools or {},
                    skill_definitions=skill_definitions or {},
                    repo_root=repo_root,
                    worktree_path=worktree_path,
                    model_name=model_name,
                )
                accumulated.extend(action_results)

        return accumulated

    async def _dispatch_actions(
        self,
        actions: list[ActionRequest],
        mcp_config: MCPConfig,
        mcp_tools: dict[str, MCPToolInfo],
        skill_definitions: dict[str, SkillDefinition],
        repo_root: str,
        worktree_path: str,
        model_name: str = "",
    ) -> list[ContextResult]:
        """Dispatch action requests as ForgeActionWorkflow child workflows.

        All actions are launched in parallel and awaited together.
        Results are converted to ContextResult for inclusion in the
        accumulated exploration context.
        """
        handles = []
        for action in actions:
            action_input = ActionWorkflowInput(
                action=action,
                mcp_config=mcp_config,
                mcp_tools=mcp_tools,
                skill_definitions=skill_definitions,
                repo_root=repo_root,
                worktree_path=worktree_path,
                model_name=model_name,
                sync_mode=self._sync_mode,
            )
            action_id = f"forge-action-{workflow.info().workflow_id}-{action.capability}"
            handle = await workflow.start_child_workflow(
                ForgeActionWorkflow.run,
                action_input,
                id=action_id,
                task_queue=FORGE_TASK_QUEUE,
                execution_timeout=_ACTION_WORKFLOW_TIMEOUT,
            )
            handles.append((action, handle))

        results: list[ContextResult] = []
        for action, handle in handles:
            try:
                action_result: ActionWorkflowResult = await handle
                # Estimate tokens from result content
                estimated_tokens = len(action_result.result.content) // 4
                results.append(
                    ContextResult(
                        provider=action.capability,
                        content=action_result.result.content,
                        estimated_tokens=estimated_tokens,
                    )
                )
            except Exception as e:
                # Action failure is non-fatal: report error as context
                error_content = f"Action '{action.capability}' failed: {e}"
                results.append(
                    ContextResult(
                        provider=action.capability,
                        content=error_content,
                        estimated_tokens=len(error_content) // 4,
                    )
                )

        return results

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

    async def _discover_capabilities(
        self, input: ForgeTaskInput
    ) -> tuple[
        list[ContextProviderSpec],
        dict[str, MCPToolInfo],
        dict[str, SkillDefinition],
    ]:
        """Discover external capabilities (MCP tools + Skills) if configured.

        Returns (specs, mcp_tools, skill_definitions).
        If no MCP/Skills are configured, returns the default provider specs.
        """
        has_mcp = bool(input.mcp_config.mcp_servers)
        has_skills = bool(input.mcp_config.skills_dirs or input.skills_dirs)

        if not has_mcp and not has_skills:
            return list(PROVIDER_SPECS), {}, {}

        # Capability discovery must happen in an activity (D87)
        discovery_input = {
            "mcp_config": input.mcp_config.model_dump(),
            "skills_dirs": list(input.mcp_config.skills_dirs) + list(input.skills_dirs),
        }
        catalog_data: dict = await workflow.execute_activity(
            "discover_capabilities",
            discovery_input,
            start_to_close_timeout=_CAPABILITY_DISCOVERY_TIMEOUT,
            result_type=dict,
        )

        # Reconstruct typed objects from serialized activity result
        specs = [ContextProviderSpec(**s) for s in catalog_data.get("specs", [])]
        mcp_tools = {k: MCPToolInfo(**v) for k, v in catalog_data.get("mcp_tools", {}).items()}
        skill_defs = {
            k: SkillDefinition(**v) for k, v in catalog_data.get("skill_definitions", {}).items()
        }
        return specs, mcp_tools, skill_defs

    async def _run_single_step(self, input: ForgeTaskInput) -> TaskResult:
        task = input.task
        max_attempts = input.max_attempts
        prior_errors: list[ValidationResult] = []

        # Resolve models for this workflow
        generation_model = resolve_model(CapabilityTier.GENERATION, input.model_routing)
        exploration_model = resolve_model(CapabilityTier.CLASSIFICATION, input.model_routing)

        # --- Discover external capabilities (Phase 15) ---
        capability_specs, mcp_tools, skill_definitions = await self._discover_capabilities(input)

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
                    prior_errors=prior_errors,
                    attempt=attempt,
                    max_attempts=max_attempts,
                ),
                start_to_close_timeout=_CONTEXT_TIMEOUT,
                result_type=AssembledContext,
            )

            # --- Exploration loop (Phase 7 + Phase 15) ---
            if input.max_exploration_rounds > 0:
                exploration_results = await self._run_exploration_loop(
                    task=task,
                    repo_root=input.repo_root,
                    worktree_path=wt_output.worktree_path,
                    max_rounds=input.max_exploration_rounds,
                    model_name=exploration_model,
                    capability_specs=capability_specs,
                    mcp_config=input.mcp_config,
                    mcp_tools=mcp_tools,
                    skill_definitions=skill_definitions,
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

            # --- Set model_name on context ---
            context = context.model_copy(update={"model_name": generation_model})

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
                llm_stats=build_llm_stats(llm_result),
                context_stats=context.context_stats,
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

        # Resolve models for this workflow
        planner_model = resolve_model(CapabilityTier.REASONING, input.model_routing)
        exploration_model = resolve_model(CapabilityTier.CLASSIFICATION, input.model_routing)

        # --- Discover external capabilities (Phase 15) ---
        capability_specs, mcp_tools, skill_definitions = await self._discover_capabilities(input)

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

        # --- Exploration loop for planner (Phase 7 + Phase 15) ---
        if input.max_exploration_rounds > 0:
            exploration_results = await self._run_exploration_loop(
                task=task,
                repo_root=input.repo_root,
                worktree_path=wt_output.worktree_path,
                max_rounds=input.max_exploration_rounds,
                model_name=exploration_model,
                capability_specs=capability_specs,
                mcp_config=input.mcp_config,
                mcp_tools=mcp_tools,
                skill_definitions=skill_definitions,
            )
            exploration_section = self._format_exploration_context(exploration_results)
            if exploration_section:
                planner_input = PlannerInput(
                    task_id=planner_input.task_id,
                    system_prompt=planner_input.system_prompt + exploration_section,
                    user_prompt=planner_input.user_prompt,
                )

        # --- Set model_name and thinking config on planner input ---
        planner_update: dict[str, object] = {"model_name": planner_model}
        if input.thinking.enabled:
            planner_update["thinking_budget_tokens"] = input.thinking.budget_tokens
            planner_update["thinking_effort"] = input.thinking.effort
        planner_input = planner_input.model_copy(update=planner_update)

        # --- Call planner ---
        planner_result = await self._call_planner_llm(planner_input)
        plan: Plan = planner_result.plan
        p_stats = build_planner_stats(planner_result)
        step_results: list[StepResult] = []
        all_output_files: dict[str, str] = {}
        sanity_check_count = 0

        # --- Execute steps sequentially (while loop enables plan mutation on revise) ---
        step_index = 0
        while step_index < len(plan.steps):
            step = plan.steps[step_index]

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

            step_succeeded = False
            prior_errors: list[ValidationResult] = []

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
                        prior_errors=prior_errors,
                        attempt=attempt,
                        max_attempts=max_step_attempts,
                    ),
                    start_to_close_timeout=_CONTEXT_TIMEOUT,
                    result_type=AssembledContext,
                )

                # --- Set model_name on context ---
                context = context.model_copy(update={"model_name": step_model})

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
                        result_type=CommitChangesOutput,
                    )
                    step_results.append(
                        StepResult(
                            step_id=step.step_id,
                            status=TransitionSignal.SUCCESS,
                            output_files=output_files,
                            validation_results=validation_results,
                            commit_sha=commit_output.commit_sha,
                            llm_stats=build_llm_stats(llm_result),
                        )
                    )
                    all_output_files.update(output_files)
                    step_succeeded = True
                    break

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
                        llm_stats=build_llm_stats(llm_result),
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
                    planner_stats=p_stats,
                    sanity_check_count=sanity_check_count,
                )

            if not step_succeeded:
                # Exhausted retries for this step — should have hit TERMINAL above
                msg = f"Step {step.step_id} exhausted all attempts without a terminal transition"
                raise RuntimeError(msg)

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
                    plan = Plan(
                        task_id=plan.task_id,
                        steps=plan.steps[: step_index + 1] + revised,
                        explanation=plan.explanation,
                    )

            step_index += 1

        # --- All steps succeeded ---
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
                task=input.task,
                plan=plan,
                completed_steps=step_results,
                remaining_steps=remaining_steps,
                repo_root=input.repo_root,
                worktree_path=wt_output.worktree_path,
            ),
            start_to_close_timeout=_CONTEXT_TIMEOUT,
            result_type=SanityCheckInput,
        )

        # Set model and thinking config
        update: dict[str, object] = {"model_name": reasoning_model}
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
        if failures:
            error_parts = [f"{r.sub_task_id}: {r.error}" for r in failures]
            return StepResult(
                step_id=step.step_id,
                status=TransitionSignal.FAILURE_TERMINAL,
                sub_task_results=sub_task_results,
                error="; ".join(error_parts),
            )

        # --- Detect and resolve file conflicts ---
        from forge.activities.conflict_resolution import detect_file_conflicts

        non_conflicting, conflicts = detect_file_conflicts(
            sub_task_results, wt_output.worktree_path
        )
        conflict_resolution_result: ConflictResolutionCallResult | None = None

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

    @workflow.signal
    async def batch_result_received(self, result: BatchResult) -> None:
        self._batch_results.append(result)

    @workflow.run
    async def run(self, input: SubTaskInput) -> SubTaskResult:
        self._sync_mode = input.sync_mode
        if input.sub_task.sub_tasks and input.depth < input.max_depth:
            return await self._run_nested_fan_out(input)
        return await self._run_single_step(input)

    # ------------------------------------------------------------------
    # Batch dispatch infrastructure
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
            result_type=BatchSubmitResult,
        )
        await workflow.wait_condition(lambda: len(self._batch_results) > 0)
        result = self._batch_results.pop(0)
        if result.error:
            raise ApplicationError(f"Batch error: {result.error}")
        assert result.raw_response_json is not None
        return await workflow.execute_activity(
            "parse_llm_response",
            ParseResponseInput(
                raw_response_json=result.raw_response_json,
                output_type_name=output_type_name,
                task_id=context.task_id,
            ),
            start_to_close_timeout=_PARSE_TIMEOUT,
            result_type=ParsedLLMResponse,
        )

    # ------------------------------------------------------------------
    # Per-call-type dispatch methods
    # ------------------------------------------------------------------

    async def _call_generation(self, context: AssembledContext) -> LLMCallResult:
        """Dispatch generation LLM call via sync or batch path."""
        if self._sync_mode:
            return await workflow.execute_activity(
                "call_llm",
                context,
                start_to_close_timeout=_LLM_TIMEOUT,
                result_type=LLMCallResult,
            )
        parsed = await self._call_llm_batch(context, "LLMResponse")
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

    async def _call_conflict_resolution(
        self, call_input: ConflictResolutionCallInput
    ) -> ConflictResolutionCallResult:
        """Dispatch conflict resolution LLM call via sync or batch path."""
        if self._sync_mode:
            return await workflow.execute_activity(
                "call_conflict_resolution",
                call_input,
                start_to_close_timeout=_CONFLICT_RESOLUTION_TIMEOUT,
                result_type=ConflictResolutionCallResult,
            )
        context = AssembledContext(
            task_id=call_input.task_id,
            system_prompt=call_input.system_prompt,
            user_prompt=call_input.user_prompt,
            model_name=call_input.model_name,
        )
        parsed = await self._call_llm_batch(
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
                result_type=AssembledContext,
            )

            # --- Set model_name from parent ---
            if input.model_name:
                context = context.model_copy(update={"model_name": input.model_name})

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
                result_type=str,
            )
            signal = TransitionSignal(signal_value)

            # --- Collect output files before cleanup ---
            output_files = write_result.output_files
            digest = llm_result.response.explanation

            # --- Remove worktree (always — sub-tasks don't commit) ---
            await workflow.execute_activity(
                "remove_worktree_activity",
                RemoveWorktreeInput(
                    repo_root=input.repo_root,
                    task_id=compound_id,
                    force=True,
                ),
                start_to_close_timeout=_GIT_TIMEOUT,
                result_type=type(None),
            )

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
            result_type=CreateWorktreeOutput,
        )

        # --- Validate unique sub-task IDs ---
        nested_ids = [st.sub_task_id for st in nested_sub_tasks]
        if len(nested_ids) != len(set(nested_ids)):
            await workflow.execute_activity(
                "remove_worktree_activity",
                RemoveWorktreeInput(
                    repo_root=input.repo_root,
                    task_id=compound_id,
                    force=True,
                ),
                start_to_close_timeout=_GIT_TIMEOUT,
                result_type=type(None),
            )
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
            await workflow.execute_activity(
                "remove_worktree_activity",
                RemoveWorktreeInput(
                    repo_root=input.repo_root,
                    task_id=compound_id,
                    force=True,
                ),
                start_to_close_timeout=_GIT_TIMEOUT,
                result_type=type(None),
            )
            error_parts = [f"{r.sub_task_id}: {r.error}" for r in failures]
            return SubTaskResult(
                sub_task_id=input.sub_task.sub_task_id,
                status=TransitionSignal.FAILURE_TERMINAL,
                sub_task_results=sub_task_results,
                error="; ".join(error_parts),
            )

        # --- Detect and resolve file conflicts ---
        from forge.activities.conflict_resolution import detect_file_conflicts

        non_conflicting, conflicts = detect_file_conflicts(
            sub_task_results, wt_output.worktree_path
        )
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
            )
            conflict_resolution_result = await self._call_conflict_resolution(cr_call_input)

            conflict_paths = {c.file_path for c in conflicts}
            resolved_paths = set(conflict_resolution_result.resolved_files.keys())
            missing = conflict_paths - resolved_paths
            if missing:
                await workflow.execute_activity(
                    "remove_worktree_activity",
                    RemoveWorktreeInput(
                        repo_root=input.repo_root,
                        task_id=compound_id,
                        force=True,
                    ),
                    start_to_close_timeout=_GIT_TIMEOUT,
                    result_type=type(None),
                )
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
                result_type=str,
            )
            signal = TransitionSignal(signal_value)

            if signal != TransitionSignal.SUCCESS:
                await workflow.execute_activity(
                    "remove_worktree_activity",
                    RemoveWorktreeInput(
                        repo_root=input.repo_root,
                        task_id=compound_id,
                        force=True,
                    ),
                    start_to_close_timeout=_GIT_TIMEOUT,
                    result_type=type(None),
                )
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
        await workflow.execute_activity(
            "remove_worktree_activity",
            RemoveWorktreeInput(
                repo_root=input.repo_root,
                task_id=compound_id,
                force=True,
            ),
            start_to_close_timeout=_GIT_TIMEOUT,
            result_type=type(None),
        )

        return SubTaskResult(
            sub_task_id=input.sub_task.sub_task_id,
            status=TransitionSignal.SUCCESS,
            output_files=merged_files,
            validation_results=validation_results,
            sub_task_results=sub_task_results,
            conflict_resolution=conflict_resolution_result,
        )


# ===========================================================================
# Phase 15: Action child workflow
# ===========================================================================


@workflow.defn
class ForgeActionWorkflow:
    """Execute a single action (MCP tool call or Skill execution).

    Dispatches to the appropriate activity based on capability_type:
    - MCP_TOOL: direct dispatch via execute_mcp_tool activity
    - SKILL: LLM-mediated dispatch via execute_skill activity

    Results are returned as ActionWorkflowResult, which the parent
    workflow converts to ContextResult for the exploration loop.
    """

    @workflow.run
    async def run(self, input: ActionWorkflowInput) -> ActionWorkflowResult:
        action = input.action

        if action.capability_type == CapabilityType.MCP_TOOL:
            result = await self._execute_mcp_tool(input)
        elif action.capability_type == CapabilityType.SKILL:
            result = await self._execute_skill(input)
        else:
            result = ActionResult(
                capability=action.capability,
                capability_type=action.capability_type,
                content=f"Unknown capability type: {action.capability_type}",
                success=False,
                estimated_tokens=0,
            )

        return ActionWorkflowResult(result=result)

    async def _execute_mcp_tool(self, input: ActionWorkflowInput) -> ActionResult:
        """Dispatch an MCP tool call (D86: direct dispatch)."""
        action = input.action
        tool_info = input.mcp_tools.get(action.capability)
        if tool_info is None:
            return ActionResult(
                capability=action.capability,
                capability_type=CapabilityType.MCP_TOOL,
                content=f"MCP tool '{action.capability}' not found in catalog",
                success=False,
                estimated_tokens=0,
            )

        # Look up server config
        server_config = input.mcp_config.mcp_servers.get(tool_info.server_name)
        if server_config is None:
            return ActionResult(
                capability=action.capability,
                capability_type=CapabilityType.MCP_TOOL,
                content=f"MCP server '{tool_info.server_name}' not found in config",
                success=False,
                estimated_tokens=0,
            )

        mcp_result: ExecuteMCPToolResult = await workflow.execute_activity(
            "execute_mcp_tool",
            ExecuteMCPToolInput(
                server_config=server_config,
                server_name=tool_info.server_name,
                tool_name=tool_info.tool_name,
                arguments=action.params,
            ),
            start_to_close_timeout=_MCP_TOOL_TIMEOUT,
            result_type=ExecuteMCPToolResult,
        )

        content = mcp_result.content
        estimated_tokens = len(content) // 4

        return ActionResult(
            capability=action.capability,
            capability_type=CapabilityType.MCP_TOOL,
            content=content,
            success=mcp_result.success,
            estimated_tokens=estimated_tokens,
        )

    async def _execute_skill(self, input: ActionWorkflowInput) -> ActionResult:
        """Dispatch a Skill execution (D86: LLM-mediated subagent)."""
        action = input.action
        skill_def = input.skill_definitions.get(action.capability)
        if skill_def is None:
            return ActionResult(
                capability=action.capability,
                capability_type=CapabilityType.SKILL,
                content=f"Skill '{action.capability}' not found in catalog",
                success=False,
                estimated_tokens=0,
            )

        skill_result: ExecuteSkillResult = await workflow.execute_activity(
            "execute_skill",
            ExecuteSkillInput(
                skill=skill_def,
                action_params=action.params,
                reasoning=action.reasoning,
                repo_root=input.repo_root,
                worktree_path=input.worktree_path,
                model_name=input.model_name,
            ),
            start_to_close_timeout=_SKILL_EXECUTION_TIMEOUT,
            result_type=ExecuteSkillResult,
        )

        content = skill_result.content
        estimated_tokens = len(content) // 4

        return ActionResult(
            capability=action.capability,
            capability_type=CapabilityType.SKILL,
            content=content,
            success=skill_result.success,
            estimated_tokens=estimated_tokens,
        )
