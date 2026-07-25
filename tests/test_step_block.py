"""Tests for forge.blocks.step — the mode table and the block's pure surface.

No Temporal here: the table, the spec's ownership invariant, and the context
stamp are plain values and functions, so they get microsecond unit tests. The
activity sequencing they drive is covered by tests/test_workflows.py.
"""

from __future__ import annotations

import pytest

from forge.blocks.step import (
    MODE_POLICIES,
    ModePolicy,
    StepSpec,
    stamp_context,
)
from forge.models import (
    AssembleContextInput,
    AssembledContext,
    AssembleStepContextInput,
    ContextConfig,
    CreateWorktreeOutput,
    PlanStep,
    ValidationConfig,
)

_WT = CreateWorktreeOutput(worktree_path="/tmp/repo/.forge-worktrees/t", branch_name="forge/t")


def _assemble_input() -> AssembleContextInput:
    return AssembleContextInput(
        task_id="t",
        description="do the thing",
        target_files=["a.py"],
        context_files=[],
        context_config=ContextConfig(),
        repo_root="/tmp/repo",
        worktree_path="",
    )


def _spec(**overrides: object) -> StepSpec:
    """A single-step spec, overridable field by field."""
    fields: dict[str, object] = {
        "mode": "single_step",
        "task_id": "t",
        "repo_root": "/tmp/repo",
        "base_branch": "main",
        "assemble_input": _assemble_input(),
        "max_attempts": 2,
        "validation": ValidationConfig(),
    }
    fields.update(overrides)
    return StepSpec(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The mode table
# ---------------------------------------------------------------------------


class TestModePolicies:
    def test_exactly_three_legal_modes(self) -> None:
        assert set(MODE_POLICIES) == {"single_step", "planned_step", "sub_task"}

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            (
                "single_step",
                ModePolicy(
                    assemble_activity="assemble_context",
                    worktree="fresh-keep",
                    commit="task",
                ),
            ),
            (
                "planned_step",
                ModePolicy(
                    assemble_activity="assemble_step_context",
                    worktree="borrowed",
                    commit="step",
                ),
            ),
            (
                "sub_task",
                ModePolicy(
                    assemble_activity="assemble_sub_task_context",
                    worktree="fresh-dispose",
                    commit="never",
                ),
            ),
        ],
    )
    def test_row(self, mode: str, expected: ModePolicy) -> None:
        assert MODE_POLICIES[mode] == expected

    def test_table_is_read_only(self) -> None:
        """The table is a constant — no workflow may rewrite a policy at runtime."""
        with pytest.raises(TypeError):
            MODE_POLICIES["single_step"] = MODE_POLICIES["sub_task"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# StepSpec's worktree-ownership invariant
# ---------------------------------------------------------------------------


class TestStepSpecOwnership:
    def test_borrowed_mode_requires_a_worktree(self) -> None:
        """A borrowed mode without a handle would make the block create (and reset) one."""
        with pytest.raises(ValueError, match="borrows its worktree"):
            _spec(
                mode="planned_step",
                assemble_input=AssembleStepContextInput(
                    task_id="t",
                    task_description="d",
                    context_config=ContextConfig(),
                    step_index=0,
                    total_steps=1,
                    repo_root="/tmp/repo",
                    worktree_path="/tmp/repo/.forge-worktrees/t",
                    step=PlanStep(step_id="step-1", description="d", target_files=["a.py"]),
                ),
            )

    def test_fresh_mode_rejects_a_borrowed_worktree(self) -> None:
        with pytest.raises(ValueError, match="creates its own worktree"):
            _spec(borrowed_worktree=_WT)

    def test_fresh_mode_defaults_to_no_borrowed_worktree(self) -> None:
        assert _spec().borrowed_worktree is None


# ---------------------------------------------------------------------------
# stamp_context
# ---------------------------------------------------------------------------


class TestStampContext:
    def test_stamps_dispatch_fields(self) -> None:
        context = AssembledContext(task_id="t", system_prompt="s", user_prompt="u")
        stamped = stamp_context(_spec(model_name="opus", log_messages=True), context, "/wt")
        assert stamped.model_name == "opus"
        assert stamped.log_messages is True
        assert stamped.worktree_path == "/wt"

    def test_empty_model_name_leaves_the_context_model(self) -> None:
        """A sub-task inherits the parent's model only when the parent chose one."""
        context = AssembledContext(
            task_id="t", system_prompt="s", user_prompt="u", model_name="from-assemble"
        )
        stamped = stamp_context(_spec(model_name=""), context, "/wt")
        assert stamped.model_name == "from-assemble"

    def test_leaves_the_prompts_alone(self) -> None:
        context = AssembledContext(task_id="t", system_prompt="s", user_prompt="u")
        stamped = stamp_context(_spec(), context, "/wt")
        assert (stamped.system_prompt, stamped.user_prompt) == ("s", "u")
