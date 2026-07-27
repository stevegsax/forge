"""Tests for forge.blocks.gather — the policy table and the block's pure surface.

No Temporal here: the two-row table, the spec's ownership invariant, the child
input the gather builds (the T1.5 propagation fix, now in one place), and the
duplicate-id / commit-message helpers are plain values and functions. The
activity and child sequencing they drive is covered by the tests/workflows/ suites.
"""

from __future__ import annotations

import pytest

from forge.blocks.gather import (
    GATHER_POLICIES,
    SUB_TASK_WORKFLOW,
    GatherPolicy,
    GatherSpec,
    build_child_input,
    duplicate_sub_task_ids,
    gather_commit_message,
)
from forge.models import (
    CreateWorktreeOutput,
    ModelConfig,
    SubTask,
    TaskDomain,
    ThinkingPolicy,
    ValidationConfig,
)
from forge.workflows import ForgeSubTaskWorkflow

_WT = CreateWorktreeOutput(
    worktree_path="/tmp/repo/.forge-worktrees/t", branch_name="forge/parent-branch"
)

_SUB_TASKS = (
    SubTask(sub_task_id="st1", description="Create schema.", target_files=["schema.py"]),
    SubTask(sub_task_id="st2", description="Create routes.", target_files=["routes.py"]),
)


def _spec(**overrides: object) -> GatherSpec:
    """A nested (owned-worktree) spec, overridable field by field."""
    fields: dict[str, object] = {
        "mode": "nested_fan_out",
        "task_id": "parent-task.sub.st1",
        "step_id": "st1",
        "repo_root": "/tmp/repo",
        "base_branch": "forge/parent-task",
        "sub_tasks": _SUB_TASKS,
        "task_description": "Build an API.",
        "step_description": "Nested node.",
        "validation": ValidationConfig(),
        "domain": TaskDomain.CODE_GENERATION,
        "child_depth": 1,
        "max_depth": 2,
        "child_max_attempts": 3,
        "child_model_name": "anthropic:sub-generation-model",
        "resolve_conflicts": True,
        "model_routing": ModelConfig(reasoning="anthropic:custom-reasoning-model"),
        "thinking": ThinkingPolicy(enabled=True, effort="max"),
        "sync_mode": True,
        "log_messages": True,
        "batch_poll_interval_seconds": 900,
    }
    fields.update(overrides)
    return GatherSpec(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The policy table
# ---------------------------------------------------------------------------


class TestGatherPolicies:
    def test_exactly_two_legal_modes(self) -> None:
        assert set(GATHER_POLICIES) == {"fan_out_step", "nested_fan_out"}

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("fan_out_step", GatherPolicy(worktree="borrowed", commit=True)),
            ("nested_fan_out", GatherPolicy(worktree="owned", commit=False)),
        ],
    )
    def test_row(self, mode: str, expected: GatherPolicy) -> None:
        assert GATHER_POLICIES[mode] == expected  # type: ignore[index]

    def test_only_the_borrowed_row_commits(self) -> None:
        """D16: a sub-task never commits — its output travels home in the result."""
        assert GATHER_POLICIES["nested_fan_out"].commit is False

    def test_child_workflow_name_matches_the_class(self) -> None:
        """The block starts children by name to avoid importing forge.workflows
        (which imports the block); this pins the string against the class."""
        assert ForgeSubTaskWorkflow.__name__ == SUB_TASK_WORKFLOW


# ---------------------------------------------------------------------------
# Spec invariants
# ---------------------------------------------------------------------------


class TestGatherSpecOwnership:
    def test_borrowed_mode_requires_a_worktree(self) -> None:
        with pytest.raises(ValueError, match="borrows its worktree"):
            _spec(mode="fan_out_step", borrowed_worktree=None)

    def test_owned_mode_rejects_a_borrowed_worktree(self) -> None:
        """An owned gather removes its worktree at the end — handing it the
        caller's would delete the plan's worktree out from under the driver."""
        with pytest.raises(ValueError, match="creates its own worktree"):
            _spec(mode="nested_fan_out", borrowed_worktree=_WT)

    def test_valid_specs_construct(self) -> None:
        assert _spec().borrowed_worktree is None
        assert _spec(mode="fan_out_step", borrowed_worktree=_WT).borrowed_worktree is _WT


# ---------------------------------------------------------------------------
# Child input construction — the T1.5 propagation fix, in one place
# ---------------------------------------------------------------------------


class TestBuildChildInput:
    def test_propagates_every_inherited_field(self) -> None:
        """resolve_conflicts / thinking / model_routing are the three the nested
        copy dropped before T1.5; the rest ride along the same construction."""
        child = build_child_input(_spec(), _SUB_TASKS[0], "forge/parent-task.sub.st1")

        assert child.resolve_conflicts is True
        assert child.thinking == ThinkingPolicy(enabled=True, effort="max")
        assert child.model_routing == ModelConfig(reasoning="anthropic:custom-reasoning-model")
        assert child.domain == TaskDomain.CODE_GENERATION
        assert child.sync_mode is True
        assert child.log_messages is True
        assert child.batch_poll_interval_seconds == 900
        assert child.validation == ValidationConfig()

    def test_identity_and_depth_come_from_the_spec(self) -> None:
        child = build_child_input(_spec(), _SUB_TASKS[1], "forge/parent-task.sub.st1")

        assert child.parent_task_id == "parent-task.sub.st1"
        assert child.parent_description == "Build an API."
        assert child.parent_branch == "forge/parent-task.sub.st1"
        assert child.sub_task.sub_task_id == "st2"
        assert child.depth == 1
        assert child.max_depth == 2
        assert child.max_attempts == 3
        assert child.model_name == "anthropic:sub-generation-model"

    def test_top_level_children_start_at_depth_zero(self) -> None:
        spec = _spec(
            mode="fan_out_step",
            borrowed_worktree=_WT,
            base_branch="",
            task_id="fanout-task",
            step_id="fan-step",
            child_depth=0,
            max_depth=1,
        )
        child = build_child_input(spec, _SUB_TASKS[0], _WT.branch_name)

        assert (child.depth, child.max_depth) == (0, 1)
        assert child.parent_task_id == "fanout-task"
        assert child.parent_branch == "forge/parent-branch"


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


class TestGatherHelpers:
    def test_duplicate_ids_detected(self) -> None:
        dup = (
            SubTask(sub_task_id="st1", description="a", target_files=["a.py"]),
            SubTask(sub_task_id="st1", description="b", target_files=["b.py"]),
        )
        assert duplicate_sub_task_ids(dup) is True
        assert duplicate_sub_task_ids(_SUB_TASKS) is False

    def test_commit_message_names_the_task_and_step(self) -> None:
        spec = _spec(mode="fan_out_step", borrowed_worktree=_WT, base_branch="")
        assert gather_commit_message(spec) == "forge(parent-task.sub.st1): step st1 fan-out gather"
