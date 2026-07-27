"""Fan-out step scenarios: a plan step with sub_tasks gathers child workflows.

Migrated from ``tests/test_workflows.py`` in T5.5. The scripting change that
matters here is D3: the old section drove both children from one shared
``subtask_transitions`` list drained by whichever child's ``validate_output``
arrived first. Each child now owns its own key (its compound sub-task id), so
the outcome is bound to the child it is about and no scheduling order can
hand it to the sibling. The assertions are unchanged.

Batch lane, as before the migration (``sync_mode`` defaults to ``False``).
"""

from types import MappingProxyType
from typing import TYPE_CHECKING

from forge.models import (
    ConflictResolutionCallResult,
    FileOutput,
    ForgeTaskInput,
    LLMResponse,
    Plan,
    PlanStep,
    SubTask,
    SubTaskInput,
    TaskDefinition,
    TransitionSignal,
)
from tests.support.workflow_harness import ScenarioState, run_task

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment

SUCCESS = TransitionSignal.SUCCESS.value
TERMINAL = TransitionSignal.FAILURE_TERMINAL.value

FAN_OUT_PLAN = Plan(
    task_id="fanout-task",
    steps=[
        PlanStep(
            step_id="fan-step",
            description="Fan-out step.",
            target_files=[],
            sub_tasks=[
                SubTask(
                    sub_task_id="st1", description="Create schema.", target_files=["schema.py"]
                ),
                SubTask(
                    sub_task_id="st2", description="Create routes.", target_files=["routes.py"]
                ),
            ],
        ),
    ],
    explanation="Single fan-out step.",
)

FANOUT_TASK = TaskDefinition(
    task_id="fanout-task",
    description="Build schema and routes in parallel.",
)

FANOUT_INPUT = ForgeTaskInput(
    task=FANOUT_TASK,
    repo_root="/tmp/repo",
    plan=True,
    max_sub_task_attempts=2,
    max_exploration_rounds=0,
)

NO_RESOLVE_INPUT = ForgeTaskInput(
    task=FANOUT_TASK,
    repo_root="/tmp/repo",
    plan=True,
    max_sub_task_attempts=2,
    max_exploration_rounds=0,
    resolve_conflicts=False,
)

# Compound ids: the key space for per-child scripting (validate identity and
# call identity coincide for a leaf child).
ST1 = "fanout-task.sub.st1"
ST2 = "fanout-task.sub.st2"

DEFAULT_CHILD_RESPONSES = MappingProxyType(
    {
        ST1: LLMResponse(
            files=[FileOutput(file_path="schema.py", content="# schema\n")],
            explanation="Created schema.",
        ),
        ST2: LLMResponse(
            files=[FileOutput(file_path="routes.py", content="# routes\n")],
            explanation="Created routes.",
        ),
    }
)


def _scenario(**kwargs: object) -> ScenarioState:
    """A two-child fan-out scenario with the section's default child outputs."""
    kwargs.setdefault("llm_responses", DEFAULT_CHILD_RESPONSES)
    return ScenarioState(plan=FAN_OUT_PLAN, **kwargs)  # type: ignore[arg-type]


def _conflicting(st1_content: str, st2_content: str, path: str = "shared.py") -> dict[str, object]:
    """Both children write the same path — the input to the conflict branch."""
    return {
        ST1: LLMResponse(
            files=[FileOutput(file_path=path, content=st1_content)],
            explanation="st1 output",
        ),
        ST2: LLMResponse(
            files=[FileOutput(file_path=path, content=st2_content)],
            explanation="st2 output",
        ),
    }


class TestFanOutStep:
    """Fan-out step with two sub-tasks, both succeed."""

    async def test_all_children_succeed(self, env: "WorkflowEnvironment") -> None:
        state = _scenario()
        result = await run_task(env, FANOUT_INPUT, state)
        assert result.status == TransitionSignal.SUCCESS

    async def test_step_results_populated(self, env: "WorkflowEnvironment") -> None:
        state = _scenario()
        result = await run_task(env, FANOUT_INPUT, state)
        assert len(result.step_results) == 1
        sr = result.step_results[0]
        assert sr.step_id == "fan-step"
        assert sr.status == TransitionSignal.SUCCESS

    async def test_sub_task_results_populated(self, env: "WorkflowEnvironment") -> None:
        state = _scenario()
        result = await run_task(env, FANOUT_INPUT, state)
        sr = result.step_results[0]
        assert len(sr.sub_task_results) == 2
        ids = {r.sub_task_id for r in sr.sub_task_results}
        assert ids == {"st1", "st2"}

    async def test_merged_output_files(self, env: "WorkflowEnvironment") -> None:
        state = _scenario()
        result = await run_task(env, FANOUT_INPUT, state)
        assert "schema.py" in result.output_files
        assert "routes.py" in result.output_files

    async def test_write_files_called(self, env: "WorkflowEnvironment") -> None:
        state = _scenario()
        await run_task(env, FANOUT_INPUT, state)
        assert state.called("write_files")

    async def test_commit_with_fan_out_message(self, env: "WorkflowEnvironment") -> None:
        state = _scenario()
        await run_task(env, FANOUT_INPUT, state)
        commits = state.entries("commit")
        assert any("fan-out gather" in c for c in commits)


class TestFanOutChildFailure:
    """One child fails terminally → fan-out step fails."""

    async def test_one_child_fails(self, env: "WorkflowEnvironment") -> None:
        state = _scenario(transitions={ST2: [TERMINAL]})
        result = await run_task(env, FANOUT_INPUT, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL

    async def test_error_references_sub_task(self, env: "WorkflowEnvironment") -> None:
        state = _scenario(transitions={ST2: [TERMINAL]})
        result = await run_task(env, FANOUT_INPUT, state)
        assert result.error is not None
        assert "fan-out failed" in result.error


class TestFanOutFileConflict:
    """Two sub-tasks produce the same file with resolve_conflicts=False → D27 terminal error."""

    async def test_file_conflict_detected(self, env: "WorkflowEnvironment") -> None:
        # Both sub-tasks return the same file path
        state = _scenario(
            llm_responses=_conflicting("# from st1\n", "# from st2\n", path="conflict.py")
        )
        result = await run_task(env, NO_RESOLVE_INPUT, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "File conflict" in result.error


class TestFanOutConflictResolution:
    """Two sub-tasks produce same file, LLM resolves the conflict."""

    async def test_resolution_succeeds(self, env: "WorkflowEnvironment") -> None:
        """Conflict is resolved, merged output passes validation, step succeeds."""
        state = _scenario(
            llm_responses=_conflicting(
                "# from st1\ndef foo(): pass\n", "# from st2\ndef bar(): pass\n"
            ),
            conflict_responses={
                "fanout-task": ConflictResolutionCallResult(
                    task_id="fanout-task",
                    resolved_files={"shared.py": "# merged\ndef foo(): pass\ndef bar(): pass\n"},
                    explanation="Combined both functions.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            },
        )
        result = await run_task(env, FANOUT_INPUT, state)
        assert result.status == TransitionSignal.SUCCESS
        assert state.called("assemble_conflict_resolution_context")
        # Batch lane: the conflict-resolution arm is the ConflictResolutionResponse
        # parse (the old section's parse handler logged this as
        # "call_conflict_resolution", which is what it stood in for).
        assert "parse_llm_response:ConflictResolutionResponse" in state.call_log
        sr = result.step_results[0]
        assert sr.conflict_resolution is not None
        # File contents live once, in the top-level TaskResult.output_files (T5.1);
        # the embedded step carries only paths + digests.
        assert "shared.py" in result.output_files
        assert "merged" in result.output_files["shared.py"]
        assert "shared.py" in sr.output_digests

    async def test_resolution_missing_path_fails(self, env: "WorkflowEnvironment") -> None:
        """Resolution LLM omits a conflict path → step fails terminal."""
        state = _scenario(
            llm_responses=_conflicting("# from st1\n", "# from st2\n"),
            conflict_responses={
                "fanout-task": ConflictResolutionCallResult(
                    task_id="fanout-task",
                    resolved_files={},  # Missing shared.py!
                    explanation="Oops, forgot.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            },
        )
        result = await run_task(env, FANOUT_INPUT, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "Conflict resolution incomplete" in result.error
        assert "shared.py" in result.error

    async def test_mixed_conflicting_and_non_conflicting(self, env: "WorkflowEnvironment") -> None:
        """Sub-tasks produce one conflicting and two non-conflicting files."""
        state = _scenario(
            llm_responses={
                ST1: LLMResponse(
                    files=[
                        FileOutput(file_path="shared.py", content="# from st1\n"),
                        FileOutput(file_path="unique_a.py", content="# unique a\n"),
                    ],
                    explanation="st1 output",
                ),
                ST2: LLMResponse(
                    files=[
                        FileOutput(file_path="shared.py", content="# from st2\n"),
                        FileOutput(file_path="unique_b.py", content="# unique b\n"),
                    ],
                    explanation="st2 output",
                ),
            },
            conflict_responses={
                "fanout-task": ConflictResolutionCallResult(
                    task_id="fanout-task",
                    resolved_files={"shared.py": "# merged shared\n"},
                    explanation="Merged shared.py.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            },
        )
        result = await run_task(env, FANOUT_INPUT, state)
        assert result.status == TransitionSignal.SUCCESS
        # File contents live once, in the top-level TaskResult.output_files (T5.1).
        assert result.output_files["shared.py"] == "# merged shared\n"
        assert result.output_files["unique_a.py"] == "# unique a\n"
        assert result.output_files["unique_b.py"] == "# unique b\n"

    async def test_resolution_disabled_falls_back_to_terminal(
        self, env: "WorkflowEnvironment"
    ) -> None:
        """resolve_conflicts=False, falls back to D27 terminal error."""
        state = _scenario(
            llm_responses=_conflicting("# from st1\n", "# from st2\n", path="conflict.py")
        )
        result = await run_task(env, NO_RESOLVE_INPUT, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert "File conflict" in (result.error or "")
        # Conflict resolution activities should NOT be called
        assert not state.called("assemble_conflict_resolution_context")
        assert not state.called("call_conflict_resolution")
        assert "parse_llm_response:ConflictResolutionResponse" not in state.call_log

    async def test_validation_failure_after_resolution(self, env: "WorkflowEnvironment") -> None:
        """Resolution succeeds but merged output fails validation → terminal error."""
        state = _scenario(
            llm_responses=_conflicting("# from st1\n", "# from st2\n"),
            conflict_responses={
                "fanout-task": ConflictResolutionCallResult(
                    task_id="fanout-task",
                    resolved_files={"shared.py": "# bad merge\n"},
                    explanation="Merged.",
                    model_name="mock-reasoning",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=300.0,
                ),
            },
            # Both children succeed (they have no scripted transitions), then the
            # parent's merged-output validation fails — one key each, so the
            # parent's terminal token can never be drawn by a child.
            transitions={"fanout-task": [TERMINAL]},
        )
        result = await run_task(env, FANOUT_INPUT, state)
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.error is not None
        assert "Merged output validation failed" in result.error


class TestRecursiveBackwardCompat:
    """Existing flat fan-out works unchanged with default max_fan_out_depth=1."""

    async def test_flat_fanout_still_works(self, env: "WorkflowEnvironment") -> None:
        state = _scenario()
        result = await run_task(env, FANOUT_INPUT, state)
        assert result.status == TransitionSignal.SUCCESS
        assert len(result.step_results) == 1
        sr = result.step_results[0]
        assert len(sr.sub_task_results) == 2

    async def test_default_max_fan_out_depth(self) -> None:
        """ForgeTaskInput defaults to max_fan_out_depth=1."""
        task_input = ForgeTaskInput(
            task=TaskDefinition(task_id="t", description="d"),
            repo_root="/tmp/repo",
        )
        assert task_input.max_fan_out_depth == 1

    async def test_subtask_input_default_depth(self) -> None:
        """SubTaskInput defaults to depth=0, max_depth=1."""
        st_input = SubTaskInput(
            parent_task_id="p",
            parent_description="d",
            sub_task=SubTask(sub_task_id="s", description="d", target_files=["f.py"]),
            repo_root="/tmp/repo",
            parent_branch="main",
        )
        assert st_input.depth == 0
        assert st_input.max_depth == 1
