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

This module is self-contained: it defines its own compact by-name mock
activities modeled on the batch pattern in ``tests/test_workflows.py`` rather
than importing that suite. Mock activities run in the worker (the imperative
shell, outside the workflow sandbox), so module-level scenario state keyed by
the unique per-submit ``batch_id`` is safe and cannot leak across scenarios.
"""

import asyncio
import os
import tempfile
from pathlib import Path

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
    TransitionInput,
    TransitionSignal,
    ValidateOutputInput,
    ValidationResult,
    WriteFilesInput,
    WriteOutputInput,
    WriteResult,
)
from forge.persist_models import PersistRequest, PersistResult
from forge.workflows import FORGE_TASK_QUEUE, ForgeSubTaskWorkflow, ForgeTaskWorkflow

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
# tests/test_workflows.py). Registered once and shared across all scenarios;
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


@activity.defn(name="evaluate_transition")
async def evaluate_transition(input: TransitionInput) -> str:
    return TransitionSignal.SUCCESS.value


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
    evaluate_transition,
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


async def main() -> None:
    _sandbox_env()
    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = (
        _scenario_single_step_batch_success,
        _scenario_single_step_batch_ceiling,
        _scenario_fan_out_batch,
        _scenario_single_step_sync,
    )
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
    asyncio.run(main())
