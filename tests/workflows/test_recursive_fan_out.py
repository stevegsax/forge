"""Recursive (depth ≥ 2) fan-out scenarios driven through ``ForgeSubTaskWorkflow``.

Migrated from ``tests/test_workflows.py`` in T5.5 — and this is the file the
flake fix lives in. ``TestRecursiveFanOutNestedFailure::test_failure_propagates``
failed roughly one run in eight because the old section scripted *one* shared
transition list (``["success", "failure_terminal"]``) drained in call-arrival
order: the terminal token was meant for ``gc2``, but two grandchildren run in
parallel and sometimes ``gc1`` drew it first, so the failure the test asserts
about ``gc2`` landed on the wrong child. Scripting is now keyed by each
grandchild's own compound id (D3), which no scheduling order can permute. The
assertions are unchanged.

The same defect was latent in the passing tests here: the old fallback handler
assigned output files by ``count % 3``, so which child produced which file was
also arrival-dependent — green only because those outputs happened to be
conflict-free. Every response is now keyed too.

Batch lane, as before the migration (``sync_mode`` defaults to ``False``).
"""

from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

from forge.models import (
    ConflictResolutionCallResult,
    FileOutput,
    LLMResponse,
    ModelConfig,
    SubTask,
    SubTaskInput,
    ThinkingPolicy,
    TransitionSignal,
)
from forge.presets import THINKING_MAX_TOKENS
from tests.support.workflow_harness import ScenarioState, run_sub_task

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment

TERMINAL = TransitionSignal.FAILURE_TERMINAL.value

NODE = "parent-task.sub.st1"
GC1 = f"{NODE}.sub.gc1"
GC2 = f"{NODE}.sub.gc2"
MID = f"{NODE}.sub.mid"
MID_GC1 = f"{MID}.sub.gc1"
MID_GC2 = f"{MID}.sub.gc2"

GRANDCHILD_RESPONSES = MappingProxyType(
    {
        GC1: LLMResponse(
            files=[FileOutput(file_path="gc1.py", content="# gc1\n")],
            explanation="Grandchild 1 output.",
        ),
        GC2: LLMResponse(
            files=[FileOutput(file_path="gc2.py", content="# gc2\n")],
            explanation="Grandchild 2 output.",
        ),
    }
)


def _conflicting(*keys: str) -> dict[str, LLMResponse]:
    """Every named child writes ``conflict.py`` — the input to the conflict branch."""
    return {
        key: LLMResponse(
            files=[
                FileOutput(file_path="conflict.py", content=f"# from {key.rsplit('.', 1)[-1]}\n")
            ],
            explanation=f"{key.rsplit('.', 1)[-1]} output",
        )
        for key in keys
    }


class TestRecursiveFanOut:
    """2-level fan-out success. Sub-task has nested sub-tasks, all succeed."""

    @pytest.fixture
    def recursive_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Create schema components.",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="gc1", description="Create models.", target_files=["gc1.py"]
                    ),
                    SubTask(
                        sub_task_id="gc2", description="Create validators.", target_files=["gc2.py"]
                    ),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=2,
            depth=0,
            max_depth=2,
        )

    async def test_recursive_success(
        self, env: "WorkflowEnvironment", recursive_input: SubTaskInput
    ) -> None:
        state = ScenarioState(llm_responses=GRANDCHILD_RESPONSES)
        result = await run_sub_task(env, recursive_input, state)
        assert result.status == TransitionSignal.SUCCESS
        assert result.sub_task_id == "st1"

    async def test_merged_output_files_propagate(
        self, env: "WorkflowEnvironment", recursive_input: SubTaskInput
    ) -> None:
        state = ScenarioState(llm_responses=GRANDCHILD_RESPONSES)
        result = await run_sub_task(env, recursive_input, state)
        assert "gc1.py" in result.output_files
        assert "gc2.py" in result.output_files

    async def test_nested_sub_task_results_populated(
        self, env: "WorkflowEnvironment", recursive_input: SubTaskInput
    ) -> None:
        state = ScenarioState(llm_responses=GRANDCHILD_RESPONSES)
        result = await run_sub_task(env, recursive_input, state)
        assert len(result.sub_task_results) == 2
        ids = {r.sub_task_id for r in result.sub_task_results}
        assert ids == {"gc1", "gc2"}

    async def test_worktrees_created_and_removed(
        self, env: "WorkflowEnvironment", recursive_input: SubTaskInput
    ) -> None:
        state = ScenarioState(llm_responses=GRANDCHILD_RESPONSES)
        await run_sub_task(env, recursive_input, state)
        # Parent sub-task worktree + 2 grandchild worktrees
        assert state.count("create_worktree") == 3
        assert state.count("remove_worktree") == 3


class TestRecursiveFanOutDepthLimit:
    """max_depth=1, depth=1 with nested sub-tasks.

    The sub-task has nested sub_tasks but depth >= max_depth, so it
    runs single-step (ignores its sub_tasks).
    """

    @pytest.fixture
    def depth_limited_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Create schema.",
                target_files=["leaf.py"],
                sub_tasks=[
                    SubTask(
                        sub_task_id="gc1",
                        description="Nested child (should be ignored).",
                        target_files=["gc1.py"],
                    ),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=2,
            depth=1,
            max_depth=1,
        )

    async def test_runs_single_step(
        self, env: "WorkflowEnvironment", depth_limited_input: SubTaskInput
    ) -> None:
        state = ScenarioState(
            llm_responses={
                NODE: LLMResponse(
                    files=[FileOutput(file_path="leaf.py", content="# leaf\n")],
                    explanation="Leaf output.",
                )
            }
        )
        result = await run_sub_task(env, depth_limited_input, state)
        assert result.status == TransitionSignal.SUCCESS
        # Should have run single-step: the generation arm ran, not a nested fan-out.
        # (On the batch lane the generation call is the LLMResponse parse; the old
        # section's parse handler logged this entry as "call_llm".)
        assert "parse_llm_response:LLMResponse" in state.call_log
        # Only one worktree created (leaf, not grandchild)
        assert state.count("create_worktree") == 1

    async def test_no_nested_sub_task_results(
        self, env: "WorkflowEnvironment", depth_limited_input: SubTaskInput
    ) -> None:
        state = ScenarioState()
        result = await run_sub_task(env, depth_limited_input, state)
        assert result.sub_task_results == []


class TestRecursiveFanOutNestedFailure:
    """Grandchild fails terminal. Verify failure propagates up through all levels."""

    @pytest.fixture
    def nested_failure_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Create schema components.",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="gc1", description="Create models.", target_files=["gc1.py"]
                    ),
                    SubTask(
                        sub_task_id="gc2", description="Create validators.", target_files=["gc2.py"]
                    ),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=1,
            depth=0,
            max_depth=2,
        )

    async def test_failure_propagates(
        self, env: "WorkflowEnvironment", nested_failure_input: SubTaskInput
    ) -> None:
        # Keyed to gc2: the child the assertion is about. gc1 has no scripted
        # transition and passes. This is the flake fix — the old shared list made
        # which grandchild failed a function of scheduling order.
        state = ScenarioState(
            llm_responses=GRANDCHILD_RESPONSES,
            transitions={GC2: [TERMINAL]},
        )
        result = await run_sub_task(env, nested_failure_input, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "gc2" in result.error

    async def test_worktrees_cleaned_up(
        self, env: "WorkflowEnvironment", nested_failure_input: SubTaskInput
    ) -> None:
        state = ScenarioState(
            llm_responses=GRANDCHILD_RESPONSES,
            transitions={GC2: [TERMINAL]},
        )
        await run_sub_task(env, nested_failure_input, state)
        # All worktrees should be removed even on failure
        assert state.count("remove_worktree") >= 3  # parent + 2 grandchildren


class TestRecursiveFanOutNestedFileConflict:
    """Two grandchildren produce the same file → conflict resolution attempted."""

    @pytest.fixture
    def conflict_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Create components.",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="gc1",
                        description="Create module.",
                        target_files=["conflict.py"],
                    ),
                    SubTask(
                        sub_task_id="gc2",
                        description="Create module.",
                        target_files=["conflict.py"],
                    ),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=1,
            depth=0,
            max_depth=2,
        )

    async def test_conflict_resolution_attempted(
        self, env: "WorkflowEnvironment", conflict_input: SubTaskInput
    ) -> None:
        """Nested conflict triggers LLM resolution; incomplete resolution fails."""
        # No conflict response scripted → the default resolves nothing → incomplete.
        state = ScenarioState(llm_responses=_conflicting(GC1, GC2))
        result = await run_sub_task(env, conflict_input, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "Conflict resolution incomplete" in result.error
        assert "conflict.py" in result.error
        assert state.called("assemble_conflict_resolution_context")
        assert "parse_llm_response:ConflictResolutionResponse" in state.call_log

    async def test_nested_conflict_resolution_succeeds(
        self, env: "WorkflowEnvironment", conflict_input: SubTaskInput
    ) -> None:
        """Nested conflict resolved successfully → sub-task succeeds."""
        state = ScenarioState(
            llm_responses=_conflicting(GC1, GC2),
            conflict_responses={
                NODE: ConflictResolutionCallResult(
                    task_id=NODE,
                    resolved_files={"conflict.py": "# merged gc1+gc2\n"},
                    explanation="Combined both.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            },
        )
        result = await run_sub_task(env, conflict_input, state)
        assert result.status == TransitionSignal.SUCCESS
        assert result.output_files["conflict.py"] == "# merged gc1+gc2\n"
        assert result.conflict_resolution is not None


class TestRecursiveFanOutNoResolveConflicts:
    """Regression (T1.5): a nested fan-out with resolve_conflicts=False and a
    real conflict must terminate via the D27 terminal fallback, never resolve.

    Before T1.5 the nested gather ignored resolve_conflicts and always ran LLM
    resolution — this asserts the flag is now honored at depth >= 1.
    """

    @pytest.fixture
    def no_resolve_conflict_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Create components.",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="gc1",
                        description="Create module.",
                        target_files=["conflict.py"],
                    ),
                    SubTask(
                        sub_task_id="gc2",
                        description="Create module.",
                        target_files=["conflict.py"],
                    ),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=1,
            depth=0,
            max_depth=2,
            resolve_conflicts=False,
        )

    async def test_falls_back_to_d27_terminal(
        self, env: "WorkflowEnvironment", no_resolve_conflict_input: SubTaskInput
    ) -> None:
        state = ScenarioState(llm_responses=_conflicting(GC1, GC2))
        result = await run_sub_task(env, no_resolve_conflict_input, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "File conflict" in result.error
        assert "conflict.py" in result.error
        # D27 fallback: LLM resolution must NOT be invoked.
        assert not state.called("assemble_conflict_resolution_context")
        assert "parse_llm_response:ConflictResolutionResponse" not in state.call_log

    async def test_worktrees_cleaned_up_on_terminal(
        self, env: "WorkflowEnvironment", no_resolve_conflict_input: SubTaskInput
    ) -> None:
        state = ScenarioState(llm_responses=_conflicting(GC1, GC2))
        await run_sub_task(env, no_resolve_conflict_input, state)
        # Nested node + 2 grandchildren all created and removed (D16).
        assert state.count("create_worktree") == 3
        assert state.count("remove_worktree") == 3


class TestRecursiveFanOutPropagatesThinkingAndRouting:
    """thinking + model_routing propagate to a depth-1 nested node and reach its
    conflict-resolution LLM-call inputs (not the pre-T1.5 hardcoded defaults).

    Uses a 3-level tree (depth 0 -> 1 -> 2) so the resolving node sits at
    depth 1: st1 (depth 0) nests mid (depth 1), which nests gc1/gc2 (depth 2)
    that both target the same file.
    """

    @pytest.fixture
    def deep_conflict_input(self) -> SubTaskInput:
        return SubTaskInput(
            parent_task_id="parent-task",
            parent_description="Build an API.",
            sub_task=SubTask(
                sub_task_id="st1",
                description="Top nested node.",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="mid",
                        description="Mid nested node.",
                        target_files=[],
                        sub_tasks=[
                            SubTask(
                                sub_task_id="gc1",
                                description="Create module.",
                                target_files=["conflict.py"],
                            ),
                            SubTask(
                                sub_task_id="gc2",
                                description="Create module.",
                                target_files=["conflict.py"],
                            ),
                        ],
                    ),
                ],
            ),
            repo_root="/tmp/repo",
            parent_branch="forge/parent-task",
            max_attempts=1,
            depth=0,
            max_depth=2,
            # model_name is the leaf-generation model; conflict resolution must
            # route via model_routing.reasoning instead (distinct value).
            model_name="anthropic:sub-generation-model",
            model_routing=ModelConfig(reasoning="anthropic:custom-reasoning-model"),
            thinking=ThinkingPolicy(enabled=True, effort="max"),
            # Runs at the default 600s poll interval: the mode-aware batch
            # _child_timeout (T4.1 ST3c) sizes the depth-1 ``mid`` node from its
            # permitted batch-wait budget ((max_attempts + remaining) * 25h), so the
            # two sequential batch phases (await grandchildren, then conflict
            # resolution) fit comfortably.
        )

    async def test_thinking_and_routing_reach_depth_one_resolution(
        self, env: "WorkflowEnvironment", deep_conflict_input: SubTaskInput
    ) -> None:
        state = ScenarioState(
            llm_responses=_conflicting(MID_GC1, MID_GC2),
            conflict_responses={
                MID: ConflictResolutionCallResult(
                    task_id=MID,
                    resolved_files={"conflict.py": "# merged\n"},
                    explanation="Combined both.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            },
        )
        result = await run_sub_task(env, deep_conflict_input, state)

        assert result.status == TransitionSignal.SUCCESS
        # Exactly one resolution, at the depth-1 node ("...st1.sub.mid"). The
        # canonical assemble mock threads model_name and thinking through, as the
        # real activity does — a mock that dropped them would mask propagation
        # bugs downstream of this activity.
        assert len(state.conflict_inputs) == 1
        cr_input = state.conflict_inputs[0]
        assert cr_input.task_id == MID
        # thinking propagated (parent's, not the pre-T1.5 hardcoded default).
        assert cr_input.thinking == ThinkingPolicy(enabled=True, effort="max")
        # model_routing propagated: REASONING tier resolved from the parent's
        # ModelConfig, not the pre-T1.5 ModelConfig()/model_name override.
        assert cr_input.model_name == "anthropic:custom-reasoning-model"
        # Conflict resolution is thinking-enabled here, so its batch submit
        # must carry the explicit adaptive-thinking cap, not the generic
        # batch_submit_and_wait default (4096).
        submitted = state.submits_by_type["ConflictResolutionResponse"]
        assert submitted.thinking == ThinkingPolicy(enabled=True, effort="max")
        assert submitted.max_tokens == THINKING_MAX_TOKENS
