"""Regenerate committed Temporal workflow histories for the replay suite.

One command regenerates every history::

    uv run python -m tests.replay.regenerate      # or: make replay-histories

Each scenario runs a real forge workflow to completion on the **time-skipping
test server** (never a real Temporal server, so a 25h wait fast-forwards in
milliseconds and no production resource is touched) with by-name mock
activities, then writes its event history to
``tests/replay/histories/<scenario>.json`` via ``WorkflowHistory.to_json()``.
``tests/test_replay.py`` replays every committed history through a temporalio
``Replayer``; a workflow-logic change that would emit a different command
sequence — the failure mode that silently breaks an in-flight batch wait in
production — fails that replay in CI instead.

**When to regenerate:** only after a *deliberate* change to workflow logic (the
shape of the command/event sequence a workflow emits — a new activity call, a
new timer, a reordering). Regenerating to silence an unexplained replay failure
defeats the guard: that failure is exactly what the suite exists to catch.
Confirm the regenerated histories still replay (run ``test_replay.py``) before
committing them.

**Scenarios** (histories kept small and bounded — no deep fan-out):

- ``single_step_batch_success`` — ``ForgeTaskWorkflow``, ``plan=False``,
  batch mode: submit -> a couple of ``in_progress`` polls, then ``ended`` ->
  fetch -> parse -> success.
- ``single_step_batch_ceiling`` — batch never ends -> the 25h wait ceiling ->
  ``MISSING`` persist -> terminal-failure path (T1.6b symmetry).
- ``fan_out_batch`` — ``plan=True`` with one fan-out step of two sub-tasks in
  batch mode (child workflows + conflict-free merge). The two children's own
  histories are captured as ``fan_out_batch__child_st1``/``__st2`` so the
  ``ForgeSubTaskWorkflow`` event sequence is replayed too.
- ``single_step_sync`` — sync mode, the cheapest baseline (no batch transport).

T5.5 added six more, all on the sync lane, covering command shapes the six
above never emitted — the four it owed (a plan's per-step context assembly, a
depth-2 nested gather, the exploration loop, the between-steps sanity check)
plus two of its stretch set:

- ``planned_step_sync`` — ``plan=True``, two sequential steps: planner ->
  ``assemble_step_context`` -> generate -> commit, twice.
- ``nested_fan_out_sync`` — a fan-out step whose one sub-task has two nested
  sub-tasks (``max_fan_out_depth=2``). The depth-1 node's own history is
  captured as ``nested_fan_out_sync__child_st1`` — that is the nested-gather
  (owned-worktree) event sequence.
- ``exploration_sync`` — one exploration round before generation:
  ``call_exploration_llm`` -> ``fulfill_context_requests`` -> generate.
- ``sanity_check_sync`` — three steps at ``sanity_check_interval=1``, so the
  sanity check fires between steps and is skipped after the last one.
- ``conflict_resolution_sync`` — two fan-out children writing the same file,
  resolved by the conflict-resolution arm.
- ``worktree_reset_retry_sync`` — a planned step that fails validation once,
  so ``reset_worktree_activity`` appears before the retry.

**Selecting scenarios:** with no arguments every scenario is regenerated (what
``make replay-histories`` does). Pass names to regenerate only those::

    uv run python -m tests.replay.regenerate exploration_sync

The first four scenarios keep the self-contained mock activities defined below;
the T5.5 scenarios drive the shared harness in
``tests/support/workflow_harness.py`` instead of growing a second copy of it.
Mock activities run in the worker (the imperative shell, outside the workflow
sandbox), so module-level scenario state keyed by the unique per-submit
``batch_id`` is safe and cannot leak across scenarios.
"""

import asyncio
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from sax_platform.contracts.constants import FORGE_TASK_QUEUE
from temporalio import activity
from temporalio.client import WorkflowHistory
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from forge.activities.conflict_resolution import classify_file_conflicts
from forge.models import (
    AssembleContextInput,
    AssembledContext,
    AssembleSubTaskContextInput,
    BatchFetchResult,
    BatchStatusInput,
    BatchStatusResult,
    BatchSubmitInput,
    BatchSubmitResult,
    CommitChangesInput,
    CommitChangesOutput,
    ConflictResolutionCallResult,
    CreateWorktreeInput,
    CreateWorktreeOutput,
    DetectFileConflictsInput,
    DetectFileConflictsOutput,
    FetchBatchResultInput,
    FileOutput,
    ForgeTaskInput,
    LLMCallResult,
    LLMResponse,
    ParsedLLMResponse,
    ParseResponseInput,
    Plan,
    PlannerInput,
    PlanStep,
    RemoveWorktreeInput,
    ResetWorktreeInput,
    SubTask,
    TaskDefinition,
    TransitionSignal,
    ValidateOutputInput,
    ValidationResult,
    WriteFilesInput,
    WriteOutputInput,
    WriteResult,
)
from forge.persist_models import PersistRequest, PersistResult
from forge.workflows import ForgeSubTaskWorkflow, ForgeTaskWorkflow
from tests.support.workflow_harness import ScenarioState, build_activities

HISTORIES_DIR = Path(__file__).parent / "histories"

# ---------------------------------------------------------------------------
# Scenario knobs (set per scenario before its worker runs; read by the mock
# activities in the worker process — never inside the sandboxed workflow).
# ---------------------------------------------------------------------------

# batch_id -> number of batch_status polls seen so far. Keyed by the unique
# per-submit batch_id, so scenarios never contaminate one another.
_STATUS_POLLS: dict[str, int] = {}
# How many ``in_progress`` polls each batch returns before ``ended``. A value
# far larger than 25h / poll_interval stalls a batch to its wait ceiling.
_IN_PROGRESS_BEFORE_END = {"count": 0}


def _reset_knobs(*, in_progress_before_end: int) -> None:
    _STATUS_POLLS.clear()
    _IN_PROGRESS_BEFORE_END["count"] = in_progress_before_end


def _parsed(
    model: LLMResponse | Plan,
    *,
    model_name: str = "mock-model",
    input_tokens: int = 100,
    output_tokens: int = 50,
    latency_ms: float = 200.0,
) -> ParsedLLMResponse:
    return ParsedLLMResponse(
        parsed_json=model.model_dump_json(),
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )


def _sub_id_from_worktree(worktree_path: str) -> str:
    """Recover a fan-out sub-task id from its compound-id worktree path.

    A sub-task worktree is ``.../<parent>.sub.<sub_task_id>``; the root task's
    worktree has no ``.sub.`` marker. Lets the shared parse mock hand each child
    a distinct output file so the merge stays conflict-free.
    """
    if ".sub." in worktree_path:
        return worktree_path.rsplit(".sub.", 1)[-1]
    return ""


# ---------------------------------------------------------------------------
# By-name mock activities (compact versions of the batch pattern in
# the tests/workflows/ suites). Registered once and shared across all scenarios;
# unused activities in a given scenario are harmless.
# ---------------------------------------------------------------------------


@activity.defn(name="persist_to_store")
async def persist_to_store(req: PersistRequest) -> PersistResult:
    """No-op survivable-write mock — never touches a database."""
    return PersistResult(kind=req.kind, applied=True)


@activity.defn(name="create_worktree_activity")
async def create_worktree_activity(input: CreateWorktreeInput) -> CreateWorktreeOutput:
    return CreateWorktreeOutput(
        worktree_path=f"/tmp/repo/.forge-worktrees/{input.task_id}",
        branch_name=f"forge/{input.task_id}",
    )


@activity.defn(name="remove_worktree_activity")
async def remove_worktree_activity(input: RemoveWorktreeInput) -> None:
    return None


@activity.defn(name="reset_worktree_activity")
async def reset_worktree_activity(input: ResetWorktreeInput) -> None:
    return None


@activity.defn(name="commit_changes_activity")
async def commit_changes_activity(input: CommitChangesInput) -> CommitChangesOutput:
    return CommitChangesOutput(commit_sha="a" * 40)


@activity.defn(name="assemble_context")
async def assemble_context(input: AssembleContextInput) -> AssembledContext:
    return AssembledContext(
        task_id=input.task_id,
        system_prompt="system prompt",
        user_prompt="user prompt",
        worktree_path=input.worktree_path,
    )


@activity.defn(name="assemble_planner_context")
async def assemble_planner_context(input: AssembleContextInput) -> PlannerInput:
    return PlannerInput(
        task_id=input.task_id,
        system_prompt="planner system",
        user_prompt="planner user",
    )


@activity.defn(name="assemble_sub_task_context")
async def assemble_sub_task_context(input: AssembleSubTaskContextInput) -> AssembledContext:
    return AssembledContext(
        task_id=input.parent_task_id,
        system_prompt=f"sub-task prompt for {input.sub_task.sub_task_id}",
        user_prompt=f"execute {input.sub_task.sub_task_id}",
        worktree_path=input.worktree_path,
    )


@activity.defn(name="call_llm")
async def call_llm(context: AssembledContext) -> LLMCallResult:
    """Sync-path generation (scenario ``single_step_sync``)."""
    return LLMCallResult(
        task_id=context.task_id,
        response=LLMResponse(
            files=[FileOutput(file_path="hello.py", content="print('hello')\n")],
            explanation="Created hello module.",
        ),
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
    )


@activity.defn(name="submit_batch_request")
async def submit_batch_request(input: BatchSubmitInput) -> BatchSubmitResult:
    """Echo the workflow-minted request_id and encode it into the batch_id.

    The workflow always mints ``request_id`` (T4.1); a unique per-submit
    batch_id keeps each waiter's poll counter isolated.
    """
    return BatchSubmitResult(
        request_id=input.request_id,
        batch_id=f"batch-{input.request_id}",
        provider="anthropic",
    )


@activity.defn(name="batch_status")
async def batch_status(input: BatchStatusInput) -> BatchStatusResult:
    """Report ``in_progress`` for the first N polls of a batch, then ``ended``.

    N == ``_IN_PROGRESS_BEFORE_END``; a huge N stalls the batch to its 25h wait
    ceiling (the ceiling scenario).
    """
    seen = _STATUS_POLLS.get(input.batch_id, 0)
    _STATUS_POLLS[input.batch_id] = seen + 1
    if seen < _IN_PROGRESS_BEFORE_END["count"]:
        return BatchStatusResult(batch_id=input.batch_id, state="in_progress")
    return BatchStatusResult(batch_id=input.batch_id, state="ended")


@activity.defn(name="fetch_batch_result")
async def fetch_batch_result(input: FetchBatchResultInput) -> BatchFetchResult:
    """Return this waiter's inline canned body; parse dispatches on output type."""
    return BatchFetchResult(raw_response_json='{"mock": true}')


@activity.defn(name="parse_llm_response")
async def parse_llm_response(input: ParseResponseInput) -> ParsedLLMResponse:
    """Classify a fetched body: a two-child fan-out Plan, or an LLMResponse.

    The LLMResponse writes to a per-child file recovered from the worktree path
    so two fan-out children stay conflict-free.
    """
    if input.output_type_name == "Plan":
        plan = Plan(
            task_id=input.task_id,
            steps=[
                PlanStep(
                    step_id="fan-step",
                    description="Two-child fan-out step.",
                    target_files=[],
                    sub_tasks=[
                        SubTask(
                            sub_task_id="st1",
                            description="Produce st1.py.",
                            target_files=["st1.py"],
                        ),
                        SubTask(
                            sub_task_id="st2",
                            description="Produce st2.py.",
                            target_files=["st2.py"],
                        ),
                    ],
                ),
            ],
            explanation="One fan-out step, two sub-tasks.",
        )
        return _parsed(plan, model_name="mock-planner", input_tokens=300, output_tokens=150)
    sub_id = _sub_id_from_worktree(input.worktree_path)
    file_path = f"{sub_id}.py" if sub_id else "hello.py"
    response = LLMResponse(
        files=[FileOutput(file_path=file_path, content=f"# {file_path}\n")],
        explanation=f"Created {file_path}.",
    )
    return _parsed(response)


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
    return WriteResult(
        task_id=input.task_id,
        files_written=list(input.files.keys()),
        output_files=dict(input.files),
    )


@activity.defn(name="validate_output")
async def validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    return [ValidationResult(check_name="ruff_lint", passed=True, summary="passed")]


@activity.defn(name="detect_file_conflicts_activity")
async def detect_file_conflicts_activity(
    input: DetectFileConflictsInput,
) -> DetectFileConflictsOutput:
    non_conflicting, conflicts = classify_file_conflicts(input.sub_task_results)
    return DetectFileConflictsOutput(non_conflicting_files=non_conflicting, conflicts=conflicts)


_ACTIVITIES = [
    persist_to_store,
    create_worktree_activity,
    remove_worktree_activity,
    reset_worktree_activity,
    commit_changes_activity,
    assemble_context,
    assemble_planner_context,
    assemble_sub_task_context,
    call_llm,
    submit_batch_request,
    batch_status,
    fetch_batch_result,
    parse_llm_response,
    write_output,
    write_files,
    validate_output,
    detect_file_conflicts_activity,
]


# ---------------------------------------------------------------------------
# Scenario drivers
# ---------------------------------------------------------------------------


async def _run_and_fetch(
    env: WorkflowEnvironment,
    *,
    workflow_id: str,
    task_input: ForgeTaskInput,
) -> None:
    """Run one root ForgeTaskWorkflow to completion under mocked activities."""
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow],
        activities=_ACTIVITIES,
    ):
        await env.client.execute_workflow(
            ForgeTaskWorkflow.run,
            task_input,
            id=workflow_id,
            task_queue=FORGE_TASK_QUEUE,
        )


def _write_history(name: str, history: WorkflowHistory) -> int:
    path = HISTORIES_DIR / f"{name}.json"
    path.write_text(history.to_json())
    return path.stat().st_size


async def _fetch_history(env: WorkflowEnvironment, workflow_id: str) -> WorkflowHistory:
    handle = env.client.get_workflow_handle(workflow_id)
    return await handle.fetch_history()


async def _scenario_single_step_batch_success(env: WorkflowEnvironment) -> list[tuple[str, int]]:
    _reset_knobs(in_progress_before_end=2)
    workflow_id = "single_step_batch_success"
    task_input = ForgeTaskInput(
        task=TaskDefinition(
            task_id="single_step_batch_success",
            description="Write a hello module.",
            target_files=["hello.py"],
        ),
        repo_root="/tmp/repo",
        max_attempts=2,
        max_exploration_rounds=0,
        sync_mode=False,
    )
    await _run_and_fetch(env, workflow_id=workflow_id, task_input=task_input)
    history = await _fetch_history(env, workflow_id)
    return [(workflow_id, _write_history(workflow_id, history))]


async def _scenario_single_step_batch_ceiling(env: WorkflowEnvironment) -> list[tuple[str, int]]:
    # The batch never ends: stall it past the 25h ceiling. A larger poll interval
    # keeps the committed history bounded (fewer identical poll cycles) while
    # exercising the identical timer -> status -> ... -> MISSING command shape.
    _reset_knobs(in_progress_before_end=10_000_000)
    workflow_id = "single_step_batch_ceiling"
    task_input = ForgeTaskInput(
        task=TaskDefinition(
            task_id="single_step_batch_ceiling",
            description="Timeout scenario.",
            target_files=["x.py"],
        ),
        repo_root="/tmp/repo",
        max_attempts=1,
        max_exploration_rounds=0,
        sync_mode=False,
        batch_poll_interval_seconds=9000,
    )
    await _run_and_fetch(env, workflow_id=workflow_id, task_input=task_input)
    history = await _fetch_history(env, workflow_id)
    return [(workflow_id, _write_history(workflow_id, history))]


async def _scenario_fan_out_batch(env: WorkflowEnvironment) -> list[tuple[str, int]]:
    _reset_knobs(in_progress_before_end=0)
    workflow_id = "fan_out_batch"
    task_input = ForgeTaskInput(
        task=TaskDefinition(task_id="fan_out_batch", description="Build a thing."),
        repo_root="/tmp/repo",
        plan=True,
        max_exploration_rounds=0,
        sync_mode=False,
    )
    await _run_and_fetch(env, workflow_id=workflow_id, task_input=task_input)
    written = [(workflow_id, _write_history(workflow_id, await _fetch_history(env, workflow_id)))]
    # Capture each child ForgeSubTaskWorkflow history too, so the sub-task event
    # sequence is replayed (the parent history only records child completions).
    for sub_id in ("st1", "st2"):
        child_id = f"forge-subtask-fan_out_batch.sub.{sub_id}"
        name = f"fan_out_batch__child_{sub_id}"
        written.append((name, _write_history(name, await _fetch_history(env, child_id))))
    return written


async def _scenario_single_step_sync(env: WorkflowEnvironment) -> list[tuple[str, int]]:
    _reset_knobs(in_progress_before_end=0)
    workflow_id = "single_step_sync"
    task_input = ForgeTaskInput(
        task=TaskDefinition(
            task_id="single_step_sync",
            description="Write a hello module.",
            target_files=["hello.py"],
        ),
        repo_root="/tmp/repo",
        max_attempts=2,
        max_exploration_rounds=0,
        sync_mode=True,
    )
    await _run_and_fetch(env, workflow_id=workflow_id, task_input=task_input)
    history = await _fetch_history(env, workflow_id)
    return [(workflow_id, _write_history(workflow_id, history))]


# ---------------------------------------------------------------------------
# T5.5 scenarios — sync lane, driven through the shared per-test harness
# ---------------------------------------------------------------------------


async def _run_harness_scenario(
    env: WorkflowEnvironment,
    *,
    workflow_id: str,
    task_input: ForgeTaskInput,
    state: ScenarioState,
) -> None:
    """Run one root workflow under the shared harness's canonical mock set."""
    async with Worker(
        env.client,
        task_queue=FORGE_TASK_QUEUE,
        workflows=[ForgeTaskWorkflow, ForgeSubTaskWorkflow],
        activities=build_activities(state),
    ):
        await env.client.execute_workflow(
            ForgeTaskWorkflow.run,
            task_input,
            id=workflow_id,
            task_queue=FORGE_TASK_QUEUE,
        )


def _sync_task(task_id: str, **kwargs: object) -> ForgeTaskInput:
    return ForgeTaskInput(
        task=TaskDefinition(task_id=task_id, description="Build a thing."),
        repo_root="/tmp/repo",
        max_exploration_rounds=0,
        sync_mode=True,
        **kwargs,  # type: ignore[arg-type]
    )


async def _scenario_planned_step_sync(env: WorkflowEnvironment) -> list[tuple[str, int]]:
    """A two-step plan on the sync lane — pins ``assemble_step_context``."""
    workflow_id = "planned_step_sync"
    state = ScenarioState(
        plan=Plan(
            task_id=workflow_id,
            steps=[
                PlanStep(
                    step_id="step-1", description="Create models.", target_files=["models.py"]
                ),
                PlanStep(step_id="step-2", description="Create API.", target_files=["api.py"]),
            ],
            explanation="Two sequential steps.",
        ),
        llm_responses={
            "step-1": LLMResponse(
                files=[FileOutput(file_path="models.py", content="# models\n")],
                explanation="Created models.py.",
            ),
            "step-2": LLMResponse(
                files=[FileOutput(file_path="api.py", content="# api\n")],
                explanation="Created api.py.",
            ),
        },
    )
    await _run_harness_scenario(
        env, workflow_id=workflow_id, task_input=_sync_task(workflow_id, plan=True), state=state
    )
    return [(workflow_id, _write_history(workflow_id, await _fetch_history(env, workflow_id)))]


async def _scenario_nested_fan_out_sync(env: WorkflowEnvironment) -> list[tuple[str, int]]:
    """Depth-2 fan-out: the depth-1 node runs the owned-worktree nested gather."""
    workflow_id = "nested_fan_out_sync"
    state = ScenarioState(
        plan=Plan(
            task_id=workflow_id,
            steps=[
                PlanStep(
                    step_id="fan-step",
                    description="One nesting sub-task.",
                    target_files=[],
                    sub_tasks=[
                        SubTask(
                            sub_task_id="st1",
                            description="Nested node.",
                            target_files=[],
                            sub_tasks=[
                                SubTask(
                                    sub_task_id="gc1",
                                    description="Produce gc1.py.",
                                    target_files=["gc1.py"],
                                ),
                                SubTask(
                                    sub_task_id="gc2",
                                    description="Produce gc2.py.",
                                    target_files=["gc2.py"],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            explanation="One fan-out step, one nesting sub-task.",
        ),
        llm_responses={
            f"{workflow_id}.sub.st1.sub.gc1": LLMResponse(
                files=[FileOutput(file_path="gc1.py", content="# gc1\n")],
                explanation="Created gc1.py.",
            ),
            f"{workflow_id}.sub.st1.sub.gc2": LLMResponse(
                files=[FileOutput(file_path="gc2.py", content="# gc2\n")],
                explanation="Created gc2.py.",
            ),
        },
    )
    await _run_harness_scenario(
        env,
        workflow_id=workflow_id,
        task_input=_sync_task(workflow_id, plan=True, max_fan_out_depth=2),
        state=state,
    )
    written = [(workflow_id, _write_history(workflow_id, await _fetch_history(env, workflow_id)))]
    # The depth-1 node's own history is the nested-gather sequence.
    child_id = f"forge-subtask-{workflow_id}.sub.st1"
    name = f"{workflow_id}__child_st1"
    written.append((name, _write_history(name, await _fetch_history(env, child_id))))
    return written


async def _scenario_exploration_sync(env: WorkflowEnvironment) -> list[tuple[str, int]]:
    """One exploration round before a single-step generation."""
    workflow_id = "exploration_sync"
    state = ScenarioState()
    task_input = ForgeTaskInput(
        task=TaskDefinition(
            task_id=workflow_id,
            description="Write a hello module.",
            target_files=["hello.py"],
        ),
        repo_root="/tmp/repo",
        max_attempts=1,
        max_exploration_rounds=1,
        sync_mode=True,
    )
    await _run_harness_scenario(env, workflow_id=workflow_id, task_input=task_input, state=state)
    return [(workflow_id, _write_history(workflow_id, await _fetch_history(env, workflow_id)))]


async def _scenario_sanity_check_sync(env: WorkflowEnvironment) -> list[tuple[str, int]]:
    """Three steps at interval 1: the sanity check fires between steps, not after the last."""
    workflow_id = "sanity_check_sync"
    state = ScenarioState(
        plan=Plan(
            task_id=workflow_id,
            steps=[
                PlanStep(step_id="s1", description="Step 1.", target_files=["a.py"]),
                PlanStep(step_id="s2", description="Step 2.", target_files=["b.py"]),
                PlanStep(step_id="s3", description="Step 3.", target_files=["c.py"]),
            ],
            explanation="Three steps.",
        ),
    )
    await _run_harness_scenario(
        env,
        workflow_id=workflow_id,
        task_input=_sync_task(workflow_id, plan=True, sanity_check_interval=1),
        state=state,
    )
    return [(workflow_id, _write_history(workflow_id, await _fetch_history(env, workflow_id)))]


async def _scenario_conflict_resolution_sync(env: WorkflowEnvironment) -> list[tuple[str, int]]:
    """Two fan-out children write the same file; the resolution arm merges it."""
    workflow_id = "conflict_resolution_sync"
    shared = LLMResponse(
        files=[FileOutput(file_path="shared.py", content="# from a child\n")],
        explanation="Wrote shared.py.",
    )
    state = ScenarioState(
        plan=Plan(
            task_id=workflow_id,
            steps=[
                PlanStep(
                    step_id="fan-step",
                    description="Two children, one shared file.",
                    target_files=[],
                    sub_tasks=[
                        SubTask(sub_task_id="st1", description="a", target_files=["shared.py"]),
                        SubTask(sub_task_id="st2", description="b", target_files=["shared.py"]),
                    ],
                ),
            ],
            explanation="One conflicting fan-out step.",
        ),
        llm_responses={"": shared},
        conflict_responses={
            workflow_id: ConflictResolutionCallResult(
                task_id=workflow_id,
                resolved_files={"shared.py": "# merged\n"},
                explanation="Combined both.",
                model_name="mock-reasoning",
                input_tokens=200,
                output_tokens=100,
                latency_ms=300.0,
            )
        },
    )
    await _run_harness_scenario(
        env,
        workflow_id=workflow_id,
        task_input=_sync_task(workflow_id, plan=True, resolve_conflicts=True),
        state=state,
    )
    return [(workflow_id, _write_history(workflow_id, await _fetch_history(env, workflow_id)))]


async def _scenario_worktree_reset_retry_sync(env: WorkflowEnvironment) -> list[tuple[str, int]]:
    """A planned step fails validation once, so the borrowed worktree is reset."""
    workflow_id = "worktree_reset_retry_sync"
    state = ScenarioState(
        plan=Plan(
            task_id=workflow_id,
            steps=[PlanStep(step_id="step-1", description="Create.", target_files=["a.py"])],
            explanation="One step, retried once.",
        ),
        transitions={
            workflow_id: [
                TransitionSignal.FAILURE_RETRYABLE.value,
                TransitionSignal.SUCCESS.value,
            ]
        },
    )
    await _run_harness_scenario(
        env,
        workflow_id=workflow_id,
        task_input=_sync_task(workflow_id, plan=True, max_step_attempts=2),
        state=state,
    )
    return [(workflow_id, _write_history(workflow_id, await _fetch_history(env, workflow_id)))]


def _sandbox_env() -> None:
    """Point every keyed reader at throwaway resources before anything imports.

    The ambient shell env points at production (FORGE_DB_URL -> Supabase, AWS ->
    real S3). Nothing in this run should construct a real client — every activity
    is mocked — but override the vars defensively so an accidental engine/client
    construction cannot reach production.
    """
    tmp = tempfile.mkdtemp(prefix="forge-replay-")
    os.environ["FORGE_DB_URL"] = f"sqlite:///{Path(tmp) / 'replay.db'}"
    os.environ["FORGE_OCR_S3_BUCKET"] = "forge-replay-none"
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


SCENARIOS = {
    "single_step_batch_success": _scenario_single_step_batch_success,
    "single_step_batch_ceiling": _scenario_single_step_batch_ceiling,
    "fan_out_batch": _scenario_fan_out_batch,
    "single_step_sync": _scenario_single_step_sync,
    "planned_step_sync": _scenario_planned_step_sync,
    "nested_fan_out_sync": _scenario_nested_fan_out_sync,
    "exploration_sync": _scenario_exploration_sync,
    "sanity_check_sync": _scenario_sanity_check_sync,
    "conflict_resolution_sync": _scenario_conflict_resolution_sync,
    "worktree_reset_retry_sync": _scenario_worktree_reset_retry_sync,
}


async def main(names: Sequence[str] = ()) -> None:
    _sandbox_env()
    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        msg = f"unknown scenario(s): {', '.join(unknown)}; known: {', '.join(SCENARIOS)}"
        raise SystemExit(msg)
    scenarios = tuple(SCENARIOS[name] for name in names) or tuple(SCENARIOS.values())
    # Same data converter the production client and the test ``temporal_env``
    # fixture use, so recorded payloads are encoded exactly as in production and
    # ``tests/test_replay.py`` (which replays with the same converter) is faithful.
    from temporalio.contrib.pydantic import pydantic_data_converter

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        for scenario in scenarios:
            for name, size in await scenario(env):
                print(f"  wrote {name}.json ({size:,} bytes)")
    print(f"Regenerated {HISTORIES_DIR.relative_to(Path.cwd())}/*.json")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
