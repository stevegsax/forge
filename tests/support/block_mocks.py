"""Scenario-shaped mock factories for the T5.2/T5.3 workflow tests.

These four factories predate the T5.5 harness and already use its pattern: each
builds its by-name mocks inside a call, closing over the caller's own recorder
lists, so they hold no module state. They live here rather than in one of the
test files because two files use ``step_block_activities`` and no test module
may import another test module (guide D7).

They are deliberately *not* folded into
:func:`tests.support.workflow_harness.build_activities`: each registers a
deliberately minimal activity set and writes its own call-log format, and the
tests that use them assert on exact call *sequences*. Merging them into the
canonical set would change what those assertions prove — which the T5.5
migration is not allowed to do (guide D5).
"""

from collections.abc import Callable

from temporalio import activity
from temporalio.exceptions import ApplicationError

from forge.activities.conflict_resolution import classify_file_conflicts
from forge.models import (
    AssembleContextInput,
    AssembledContext,
    AssembleSanityCheckContextInput,
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    BatchFetchResult,
    BatchStatusInput,
    BatchStatusResult,
    BatchSubmitInput,
    BatchSubmitResult,
    CommitChangesInput,
    CommitChangesOutput,
    ConflictResolutionCallInput,
    ConflictResolutionCallResult,
    ConflictResolutionInput,
    ContextRequest,
    ContextResult,
    CreateWorktreeInput,
    CreateWorktreeOutput,
    DetectFileConflictsInput,
    DetectFileConflictsOutput,
    ExplorationCallResult,
    ExplorationInput,
    ExplorationResponse,
    FetchBatchResultInput,
    FileOutput,
    FulfillContextInput,
    LLMCallResult,
    LLMResponse,
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
    SubTask,
    ValidateOutputInput,
    ValidationResult,
    WriteFilesInput,
    WriteOutputInput,
    WriteResult,
)
from forge.persist_models import PersistInteraction, PersistRequest, PersistResult
from tests.support.workflow_harness import DEFAULT_LLM_RESPONSE, FAIL_VALIDATION, PASS_VALIDATION

__all__ = [
    "exploration_batch_activities",
    "five_arm_activities",
    "gather_activities",
    "interactions",
    "step_block_activities",
]


def interactions(persisted: list[PersistRequest]) -> list[PersistInteraction]:
    """The interaction records among a run's survivable writes."""
    return [req for req in persisted if isinstance(req, PersistInteraction)]


def step_block_activities(
    calls: list[str],
    *,
    raise_in: str = "",
    plan: Plan | None = None,
    worktree_paths: list[str] | None = None,
    fail_first_validation: bool = False,
    persisted: list[PersistRequest] | None = None,
) -> list[Callable[..., object]]:
    """By-name mocks bound to *calls*.

    ``raise_in`` names an activity that fails with a non-retryable
    ApplicationError — the mid-step exception the cleanup wrap must survive.
    ``worktree_paths`` hands out a distinct worktree path per create call, so a
    test can tell which attempt's worktree an activity received.
    ``persisted`` collects every survivable write, for tests that assert on the
    interaction records a run produced.
    """
    created = 0

    def _fail_if_selected(name: str) -> None:
        if name == raise_in:
            raise ApplicationError(f"{name} blew up", non_retryable=True)

    @activity.defn(name="persist_to_store")
    async def persist_to_store(req: PersistRequest) -> PersistResult:
        if persisted is not None:
            persisted.append(req)
        return PersistResult(kind=req.kind, applied=True)

    @activity.defn(name="create_worktree_activity")
    async def create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
        nonlocal created
        path = (
            worktree_paths[created]
            if worktree_paths
            else f"/tmp/repo/.forge-worktrees/{input.task_id}"
        )
        created += 1
        calls.append(f"create_worktree:{path}")
        return CreateWorktreeOutput(worktree_path=path, branch_name=f"forge/{input.task_id}")

    @activity.defn(name="remove_worktree_activity")
    async def remove_worktree(input: RemoveWorktreeInput) -> None:
        # force=True is what makes the activity delete the forge/<task_id>
        # branch too (tests/test_git.py pins the real branch deletion).
        calls.append(f"remove_worktree:{input.task_id}:force={input.force}")

    @activity.defn(name="reset_worktree_activity")
    async def reset_worktree(input: ResetWorktreeInput) -> None:
        calls.append("reset_worktree")

    @activity.defn(name="commit_changes_activity")
    async def commit_changes(input: CommitChangesInput) -> CommitChangesOutput:
        calls.append(f"commit:{input.status}")
        return CommitChangesOutput(commit_sha="c" * 40)

    @activity.defn(name="assemble_context")
    async def assemble_context(input: AssembleContextInput) -> AssembledContext:
        calls.append(f"assemble_context:{input.worktree_path}")
        return AssembledContext(
            task_id=input.task_id, system_prompt="system prompt", user_prompt="user prompt"
        )

    @activity.defn(name="assemble_planner_context")
    async def assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
        calls.append("assemble_planner_context")
        return PlannerInput(
            task_id=input.task_id, system_prompt="planner system", user_prompt="planner user"
        )

    @activity.defn(name="call_planner")
    async def call_planner(input: PlannerInput) -> PlanCallResult:
        calls.append("call_planner")
        assert plan is not None
        return PlanCallResult(
            task_id=input.task_id,
            plan=plan,
            model_name="mock-planner",
            input_tokens=300,
            output_tokens=150,
            latency_ms=500.0,
        )

    @activity.defn(name="assemble_step_context")
    async def assemble_step_context(input: AssembleStepContextInput) -> AssembledContext:
        calls.append(f"assemble_step_context:{input.step.step_id}")
        return AssembledContext(
            task_id=input.task_id, system_prompt="step system", user_prompt="step user"
        )

    @activity.defn(name="assemble_sub_task_context")
    async def assemble_sub_task_context(input: AssembleSubTaskContextInput) -> AssembledContext:
        calls.append(f"assemble_sub_task_context:{input.worktree_path}")
        return AssembledContext(
            task_id=input.parent_task_id, system_prompt="sub system", user_prompt="sub user"
        )

    @activity.defn(name="call_exploration_llm")
    async def call_exploration_llm(input: ExplorationInput) -> ExplorationCallResult:
        calls.append(f"call_exploration_llm:{input.worktree_path}")
        # T5.3: the sync activity returns the full envelope — the prompts it
        # assembled internally plus the call's spend — so the workflow has an
        # interaction record to persist for this arm like every other one.
        return ExplorationCallResult(
            task_id=input.task_id,
            response=ExplorationResponse(
                requests=[ContextRequest(provider="file_content", reasoning="need a peek")]
            ),
            system_prompt="exploration system",
            user_prompt="exploration user",
            model_name="mock-explorer",
            input_tokens=41,
            output_tokens=17,
            latency_ms=90.0,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=5,
            stop_reason="end_turn",
        )

    @activity.defn(name="fulfill_context_requests")
    async def fulfill_context_requests(input: FulfillContextInput) -> list[ContextResult]:
        calls.append(f"fulfill_context_requests:{input.worktree_path}")
        return [
            ContextResult(provider="file_content", content="explored content", estimated_tokens=10)
        ]

    @activity.defn(name="call_llm")
    async def call_llm(context: AssembledContext) -> LLMCallResult:
        calls.append(f"call_llm:{context.worktree_path}")
        return LLMCallResult(
            task_id=context.task_id,
            response=LLMResponse(
                files=[FileOutput(file_path="hello.py", content="print('hello')\n")],
                explanation="Created hello module."
                + (" [explored]" if "Exploration Results" in context.system_prompt else ""),
            ),
            model_name="mock-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
        )

    @activity.defn(name="write_output")
    async def write_output(input: WriteOutputInput) -> WriteResult:
        calls.append("write_output")
        _fail_if_selected("write_output")
        files = input.llm_result.response.files
        return WriteResult(
            task_id=input.llm_result.task_id,
            files_written=[f.file_path for f in files],
            output_files={f.file_path: f.content for f in files},
        )

    @activity.defn(name="validate_output")
    async def validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
        calls.append("validate_output")
        # Attempt 1 fails, every later attempt passes: enough to exercise the
        # retry path without any scripted global state.
        failures = sum(1 for c in calls if c == "validate_output")
        return [PASS_VALIDATION] if failures > 1 or not fail_first_validation else [FAIL_VALIDATION]

    return [
        persist_to_store,
        create_worktree,
        remove_worktree,
        reset_worktree,
        commit_changes,
        assemble_context,
        assemble_planner_context,
        call_planner,
        assemble_step_context,
        assemble_sub_task_context,
        call_exploration_llm,
        fulfill_context_requests,
        call_llm,
        write_output,
        validate_output,
    ]


def exploration_batch_activities(
    calls: list[str],
    persisted: list[PersistRequest],
) -> list[Callable[..., object]]:
    """By-name batch-lane mocks for a single-step run with one exploration round."""

    @activity.defn(name="persist_to_store")
    async def persist_to_store(req: PersistRequest) -> PersistResult:
        persisted.append(req)
        return PersistResult(kind=req.kind, applied=True)

    @activity.defn(name="create_worktree_activity")
    async def create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
        return CreateWorktreeOutput(
            worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
            branch_name=f"forge/{input.task_id}",
        )

    @activity.defn(name="remove_worktree_activity")
    async def remove_worktree(input: RemoveWorktreeInput) -> None:
        return None

    @activity.defn(name="commit_changes_activity")
    async def commit_changes(input: CommitChangesInput) -> CommitChangesOutput:
        return CommitChangesOutput(commit_sha="f" * 40)

    @activity.defn(name="assemble_context")
    async def assemble_context(input: AssembleContextInput) -> AssembledContext:
        calls.append("assemble_context")
        return AssembledContext(
            task_id=input.task_id, system_prompt="system prompt", user_prompt="user prompt"
        )

    @activity.defn(name="assemble_exploration_context")
    async def assemble_exploration_context(input: ExplorationInput) -> AssembledContext:
        calls.append("assemble_exploration_context")
        return AssembledContext(
            task_id=input.task_id,
            system_prompt="explore system",
            user_prompt="explore user",
        )

    @activity.defn(name="submit_batch_request")
    async def submit_batch_request(input: BatchSubmitInput) -> BatchSubmitResult:
        calls.append(f"submit_batch_request:{input.output_type_name}")
        return BatchSubmitResult(
            request_id=input.request_id,
            batch_id=f"batch-{input.request_id}",
            provider="anthropic",
        )

    @activity.defn(name="batch_status")
    async def batch_status(input: BatchStatusInput) -> BatchStatusResult:
        return BatchStatusResult(batch_id=input.batch_id, state="ended")

    @activity.defn(name="fetch_batch_result")
    async def fetch_batch_result(input: FetchBatchResultInput) -> BatchFetchResult:
        return BatchFetchResult(raw_response_json='{"mock": true}')

    @activity.defn(name="parse_llm_response")
    async def parse_llm_response(input: ParseResponseInput) -> ParsedLLMResponse:
        calls.append(f"parse_llm_response:{input.output_type_name}")
        if input.output_type_name == "ExplorationResponse":
            return ParsedLLMResponse(
                parsed_json=ExplorationResponse(requests=[]).model_dump_json(),
                model_name="mock-explorer",
                input_tokens=123,
                output_tokens=45,
                cache_creation_input_tokens=7,
                cache_read_input_tokens=9,
                stop_reason="end_turn",
            )
        return ParsedLLMResponse(
            parsed_json=DEFAULT_LLM_RESPONSE.model_dump_json(),
            model_name="mock-batch-model",
            input_tokens=100,
            output_tokens=50,
        )

    @activity.defn(name="write_output")
    async def write_output(input: WriteOutputInput) -> WriteResult:
        files = input.llm_result.response.files
        return WriteResult(
            task_id=input.llm_result.task_id,
            files_written=[f.file_path for f in files],
            output_files={f.file_path: f.content for f in files},
        )

    @activity.defn(name="validate_output")
    async def validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
        return [PASS_VALIDATION]

    return [
        persist_to_store,
        create_worktree,
        remove_worktree,
        commit_changes,
        assemble_context,
        assemble_exploration_context,
        submit_batch_request,
        batch_status,
        fetch_batch_result,
        parse_llm_response,
        write_output,
        validate_output,
    ]


def gather_activities(
    calls: list[str],
    persisted: list[PersistRequest],
    *,
    crash_sub_task: str = "",
    raise_in: str = "",
    plan: Plan | None = None,
    conflicts: bool = False,
) -> list[Callable[..., object]]:
    """By-name mocks for a fan-out gather (parent or nested), sync mode.

    ``crash_sub_task`` names the sub-task whose context assembly fails
    non-retryably — that child's workflow fails, so the parent's await raises,
    which is the shape per-child isolation must absorb. ``raise_in`` fails one
    named gather activity outright (the mid-gather exception an owned worktree
    must be cleaned up after). ``conflicts`` makes both children write the same
    path so the conflict branch runs.
    """

    @activity.defn(name="persist_to_store")
    async def persist_to_store(req: PersistRequest) -> PersistResult:
        persisted.append(req)
        return PersistResult(kind=req.kind, applied=True)

    @activity.defn(name="create_worktree_activity")
    async def create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
        calls.append(f"create_worktree:{input.task_id}")
        return CreateWorktreeOutput(
            worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
            branch_name=f"forge/{input.task_id}",
        )

    @activity.defn(name="remove_worktree_activity")
    async def remove_worktree(input: RemoveWorktreeInput) -> None:
        calls.append(f"remove_worktree:{input.task_id}:force={input.force}")

    @activity.defn(name="reset_worktree_activity")
    async def reset_worktree(input: ResetWorktreeInput) -> None:
        calls.append("reset_worktree")

    @activity.defn(name="commit_changes_activity")
    async def commit_changes(input: CommitChangesInput) -> CommitChangesOutput:
        calls.append(f"commit:{input.message or input.status}")
        return CommitChangesOutput(commit_sha="g" * 40)

    @activity.defn(name="assemble_planner_context")
    async def assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
        return PlannerInput(
            task_id=input.task_id, system_prompt="planner system", user_prompt="planner user"
        )

    @activity.defn(name="call_planner")
    async def call_planner(input: PlannerInput) -> PlanCallResult:
        assert plan is not None
        return PlanCallResult(
            task_id=input.task_id,
            plan=plan,
            model_name="mock-planner",
            input_tokens=300,
            output_tokens=150,
            latency_ms=500.0,
        )

    @activity.defn(name="assemble_sub_task_context")
    async def assemble_sub_task_context(input: AssembleSubTaskContextInput) -> AssembledContext:
        sub_task_id = input.sub_task.sub_task_id
        calls.append(f"assemble_sub_task_context:{sub_task_id}")
        if sub_task_id == crash_sub_task:
            raise ApplicationError(f"{sub_task_id} context blew up", non_retryable=True)
        return AssembledContext(
            task_id=input.parent_task_id,
            system_prompt=f"sub system for {sub_task_id}",
            user_prompt=f"execute {sub_task_id}",
        )

    @activity.defn(name="call_llm")
    async def call_llm(context: AssembledContext) -> LLMCallResult:
        sub_task_id = context.system_prompt.rsplit(" ", 1)[-1]
        file_path = "shared.py" if conflicts else f"{sub_task_id}.py"
        calls.append(f"call_llm:{sub_task_id}")
        return LLMCallResult(
            task_id=context.task_id,
            response=LLMResponse(
                files=[FileOutput(file_path=file_path, content=f"# from {sub_task_id}\n")],
                explanation=f"{sub_task_id} output",
            ),
            model_name="mock-model",
            input_tokens=50,
            output_tokens=25,
            latency_ms=100.0,
        )

    @activity.defn(name="write_output")
    async def write_output(input: WriteOutputInput) -> WriteResult:
        files = input.llm_result.response.files
        return WriteResult(
            task_id=input.llm_result.task_id,
            files_written=[f.file_path for f in files],
            output_files={f.file_path: f.content for f in files},
        )

    @activity.defn(name="write_files")
    async def write_files(input: WriteFilesInput) -> WriteResult:
        calls.append(f"write_files:{sorted(input.files)}")
        return WriteResult(task_id=input.task_id, files_written=sorted(input.files))

    @activity.defn(name="validate_output")
    async def validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
        return [PASS_VALIDATION]

    @activity.defn(name="detect_file_conflicts_activity")
    async def detect_file_conflicts(
        input: DetectFileConflictsInput,
    ) -> DetectFileConflictsOutput:
        calls.append("detect_file_conflicts")
        if raise_in == "detect_file_conflicts_activity":
            raise ApplicationError("detect blew up", non_retryable=True)
        non_conflicting, found = classify_file_conflicts(input.sub_task_results)
        return DetectFileConflictsOutput(non_conflicting_files=non_conflicting, conflicts=found)

    return [
        persist_to_store,
        create_worktree,
        remove_worktree,
        reset_worktree,
        commit_changes,
        assemble_planner_context,
        call_planner,
        assemble_sub_task_context,
        call_llm,
        write_output,
        write_files,
        validate_output,
        detect_file_conflicts,
    ]


def five_arm_activities(
    calls: list[str],
    persisted: list[PersistRequest],
) -> list[Callable[..., object]]:
    """Sync-lane mocks for a planned run that touches every dispatch arm.

    The plan is a regular step followed by a fan-out step whose two children
    write the *same* file: the regular step triggers the between-steps sanity
    check (the driver skips it after a fan-out step and after the last one) and
    the shared file forces conflict resolution. One exploration round runs
    during planning.

    The children *declare* distinct targets and both *write* ``shared.py``. That
    is deliberate since T5.6: the preflight gate rejects a plan whose sub-tasks
    declare overlapping targets, so a declared overlap can no longer reach
    execution — an undeclared write is the shape a real conflict now takes, and
    conflict detection has always read what the children produced rather than
    what they promised.
    """
    plan = Plan(
        task_id="five-arm-task",
        steps=[
            PlanStep(step_id="step-1", description="Warm up.", target_files=["first.py"]),
            PlanStep(
                step_id="fan-step",
                description="Two children, one shared file.",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="st1", description="a", target_files=["st1.py"]),
                    SubTask(sub_task_id="st2", description="b", target_files=["st2.py"]),
                ],
            ),
        ],
        explanation="A regular step (so the sanity check fires between steps), then a fan-out.",
    )

    @activity.defn(name="persist_to_store")
    async def persist_to_store(req: PersistRequest) -> PersistResult:
        persisted.append(req)
        return PersistResult(kind=req.kind, applied=True)

    @activity.defn(name="create_worktree_activity")
    async def create_worktree(input: CreateWorktreeInput) -> CreateWorktreeOutput:
        return CreateWorktreeOutput(
            worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
            branch_name=f"forge/{input.task_id}",
        )

    @activity.defn(name="remove_worktree_activity")
    async def remove_worktree(input: RemoveWorktreeInput) -> None:
        return None

    @activity.defn(name="reset_worktree_activity")
    async def reset_worktree(input: ResetWorktreeInput) -> None:
        return None

    @activity.defn(name="commit_changes_activity")
    async def commit_changes(input: CommitChangesInput) -> CommitChangesOutput:
        return CommitChangesOutput(commit_sha="h" * 40)

    @activity.defn(name="assemble_planner_context")
    async def assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
        return PlannerInput(
            task_id=input.task_id, system_prompt="planner system", user_prompt="planner user"
        )

    @activity.defn(name="call_planner")
    async def call_planner(input: PlannerInput) -> PlanCallResult:
        calls.append("call_planner")
        return PlanCallResult(
            task_id=input.task_id,
            plan=plan,
            model_name="mock-planner",
            input_tokens=300,
            output_tokens=150,
            latency_ms=500.0,
        )

    @activity.defn(name="call_exploration_llm")
    async def call_exploration_llm(input: ExplorationInput) -> ExplorationCallResult:
        calls.append("call_exploration_llm")
        # Empty requests end the loop after one round.
        return ExplorationCallResult(
            task_id=input.task_id,
            response=ExplorationResponse(requests=[]),
            system_prompt="exploration system",
            user_prompt="exploration user",
            model_name="mock-explorer",
            input_tokens=41,
            output_tokens=17,
            latency_ms=90.0,
        )

    @activity.defn(name="assemble_sub_task_context")
    async def assemble_sub_task_context(input: AssembleSubTaskContextInput) -> AssembledContext:
        return AssembledContext(
            task_id=input.parent_task_id,
            system_prompt=f"sub system for {input.sub_task.sub_task_id}",
            user_prompt=f"execute {input.sub_task.sub_task_id}",
        )

    @activity.defn(name="assemble_step_context")
    async def assemble_step_context(input: AssembleStepContextInput) -> AssembledContext:
        return AssembledContext(
            task_id=input.task_id,
            system_prompt=f"step system for {input.step.step_id}",
            user_prompt=f"execute {input.step.step_id}",
        )

    @activity.defn(name="call_llm")
    async def call_llm(context: AssembledContext) -> LLMCallResult:
        who = context.system_prompt.rsplit(" ", 1)[-1]
        calls.append(f"call_llm:{who}")
        file_path = "shared.py" if who in ("st1", "st2") else f"{who}.py"
        return LLMCallResult(
            task_id=context.task_id,
            response=LLMResponse(
                files=[FileOutput(file_path=file_path, content=f"# from {who}\n")],
                explanation=f"{who} output",
            ),
            model_name="mock-model",
            input_tokens=50,
            output_tokens=25,
            latency_ms=100.0,
        )

    @activity.defn(name="write_output")
    async def write_output(input: WriteOutputInput) -> WriteResult:
        files = input.llm_result.response.files
        return WriteResult(
            task_id=input.llm_result.task_id,
            files_written=[f.file_path for f in files],
            output_files={f.file_path: f.content for f in files},
        )

    @activity.defn(name="write_files")
    async def write_files(input: WriteFilesInput) -> WriteResult:
        return WriteResult(task_id=input.task_id, files_written=sorted(input.files))

    @activity.defn(name="validate_output")
    async def validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
        return [PASS_VALIDATION]

    @activity.defn(name="detect_file_conflicts_activity")
    async def detect_file_conflicts(
        input: DetectFileConflictsInput,
    ) -> DetectFileConflictsOutput:
        non_conflicting, found = classify_file_conflicts(input.sub_task_results)
        return DetectFileConflictsOutput(non_conflicting_files=non_conflicting, conflicts=found)

    @activity.defn(name="assemble_conflict_resolution_context")
    async def assemble_cr_context(
        input: ConflictResolutionInput,
    ) -> ConflictResolutionCallInput:
        return ConflictResolutionCallInput(
            task_id=input.task_id,
            step_id=input.step_id,
            system_prompt="conflict system",
            user_prompt="conflict user",
            model_name=input.model_name,
            thinking=input.thinking,
        )

    @activity.defn(name="call_conflict_resolution")
    async def call_conflict_resolution(
        input: ConflictResolutionCallInput,
    ) -> ConflictResolutionCallResult:
        calls.append("call_conflict_resolution")
        return ConflictResolutionCallResult(
            task_id=input.task_id,
            resolved_files={"shared.py": "# merged\n"},
            explanation="Combined both.",
            model_name="mock-reasoning",
            input_tokens=200,
            output_tokens=100,
            latency_ms=300.0,
        )

    @activity.defn(name="assemble_sanity_check_context")
    async def assemble_sanity_check_context(
        input: AssembleSanityCheckContextInput,
    ) -> SanityCheckInput:
        return SanityCheckInput(
            task_id=input.task_id, system_prompt="sanity system", user_prompt="sanity user"
        )

    @activity.defn(name="call_sanity_check")
    async def call_sanity_check(input: SanityCheckInput) -> SanityCheckCallResult:
        calls.append("call_sanity_check")
        return SanityCheckCallResult(
            task_id=input.task_id,
            response=SanityCheckResponse(
                verdict=SanityCheckVerdict.CONTINUE, explanation="On track."
            ),
            model_name="mock-reasoning",
            input_tokens=180,
            output_tokens=60,
            latency_ms=250.0,
        )

    return [
        persist_to_store,
        create_worktree,
        remove_worktree,
        reset_worktree,
        commit_changes,
        assemble_planner_context,
        call_planner,
        call_exploration_llm,
        assemble_sub_task_context,
        assemble_step_context,
        call_llm,
        write_output,
        write_files,
        validate_output,
        detect_file_conflicts,
        assemble_cr_context,
        call_conflict_resolution,
        assemble_sanity_check_context,
        call_sanity_check,
    ]
