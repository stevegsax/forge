"""Unit tests for the pure plan-structure checks (T5.6).

No Temporal, no fixtures beyond ``parametrize`` — every function under test is
pure, which is the point of keeping the gate's decision surface out of the
workflow (the ``tests/test_step_logic.py`` pattern). The workflow-level proof
that these findings actually stop a run lives in
``tests/workflows/test_preflight.py``.
"""

from __future__ import annotations

import pytest

from forge.models import (
    MAX_PLAN_REVISIONS,
    MAX_PLAN_STEPS,
    MAX_PLANNER_ATTEMPTS,
    Plan,
    PlanStep,
    SubTask,
    ThinkingPolicy,
)
from forge.plan_checks import (
    PREFLIGHT_CHECKS,
    PlanViolation,
    RevisedPlan,
    RevisionRejected,
    duplicate_step_ids,
    duplicate_sub_task_ids,
    escalate_thinking,
    forward_references,
    implausible_context_files,
    nodes_without_targets,
    overlapping_sub_task_targets,
    preflight_plan,
    retry_prompt_section,
    splice_revision,
    uncovered_task_targets,
    undersized_fan_outs,
    unsafe_target_paths,
    violation_summary,
)


def _plan(*steps: PlanStep, task_id: str = "t1") -> Plan:
    return Plan(task_id=task_id, steps=list(steps), explanation="e")


def _step(step_id: str, *, targets: list[str] | None = None, **kwargs: object) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        description="d",
        target_files=targets if targets is not None else [f"{step_id}.py"],
        **kwargs,  # type: ignore[arg-type]
    )


def _sub(sub_task_id: str, *, targets: list[str], children: list[SubTask] | None = None) -> SubTask:
    return SubTask(
        sub_task_id=sub_task_id,
        description="d",
        target_files=targets,
        sub_tasks=children,
    )


CLEAN_PLAN = _plan(
    _step("s1", targets=["a.py"]),
    PlanStep(
        step_id="s2",
        description="d",
        target_files=[],
        sub_tasks=[_sub("st1", targets=["b.py"]), _sub("st2", targets=["c.py"])],
    ),
)


# ---------------------------------------------------------------------------
# Individual finders
# ---------------------------------------------------------------------------


class TestUnsafeTargetPaths:
    @pytest.mark.parametrize("bad", ["/etc/passwd", "../outside.py", "a/../../b.py"])
    def test_rejects_absolute_and_traversal(self, bad: str) -> None:
        plan = _plan(_step("s1", targets=[bad]))
        assert unsafe_target_paths(plan) == (f"s1: {bad}",)

    def test_accepts_ordinary_relative_paths(self) -> None:
        assert unsafe_target_paths(CLEAN_PLAN) == ()

    def test_finds_nested_violations(self) -> None:
        """The pre-T5.6 checks stopped at depth 1, so this escaped even eval."""
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="d",
                target_files=[],
                sub_tasks=[
                    _sub("st1", targets=[], children=[_sub("gc1", targets=["/tmp/evil.py"])]),
                    _sub("st2", targets=["ok.py"]),
                ],
            )
        )
        assert unsafe_target_paths(plan) == ("s1/st1/gc1: /tmp/evil.py",)


class TestDuplicateStepIds:
    def test_reports_each_repeated_id_once(self) -> None:
        plan = _plan(_step("s1"), _step("s1"), _step("s1"), _step("s2"))
        assert duplicate_step_ids(plan) == ("s1",)

    def test_clean(self) -> None:
        assert duplicate_step_ids(CLEAN_PLAN) == ()


class TestDuplicateSubTaskIds:
    def test_siblings_collide(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="d",
                target_files=[],
                sub_tasks=[_sub("st1", targets=["a.py"]), _sub("st1", targets=["b.py"])],
            )
        )
        assert duplicate_sub_task_ids(plan) == ("s1/st1",)

    def test_same_id_under_different_parents_is_fine(self) -> None:
        """Identity is the compound ``<parent>.sub.<child>`` path, so this is unambiguous."""
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="d",
                target_files=[],
                sub_tasks=[
                    _sub("a", targets=[], children=[_sub("gc1", targets=["1.py"])]),
                    _sub("b", targets=[], children=[_sub("gc1", targets=["2.py"])]),
                ],
            )
        )
        assert duplicate_sub_task_ids(plan) == ()

    def test_nested_siblings_collide(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="d",
                target_files=[],
                sub_tasks=[
                    _sub(
                        "st1",
                        targets=[],
                        children=[_sub("gc1", targets=["1.py"]), _sub("gc1", targets=["2.py"])],
                    ),
                    _sub("st2", targets=["ok.py"]),
                ],
            )
        )
        assert duplicate_sub_task_ids(plan) == ("s1/st1/gc1",)


class TestOverlappingSubTaskTargets:
    def test_siblings_claiming_one_file(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="d",
                target_files=[],
                sub_tasks=[_sub("st1", targets=["shared.py"]), _sub("st2", targets=["shared.py"])],
            )
        )
        assert overlapping_sub_task_targets(plan) == ("s1: shared.py claimed by st1 and st2",)

    def test_grandchild_collides_with_uncle(self) -> None:
        """Effective targets, not declared ones: a grandchild's file surfaces in
        its parent's merged output, so this is a real collision at ``s1``."""
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="d",
                target_files=[],
                sub_tasks=[
                    _sub("st1", targets=[], children=[_sub("gc1", targets=["shared.py"])]),
                    _sub("st2", targets=["shared.py"]),
                ],
            )
        )
        assert overlapping_sub_task_targets(plan) == ("s1: shared.py claimed by st1 and st2",)

    def test_two_steps_may_touch_the_same_file(self) -> None:
        """Steps run sequentially — only *parallel* siblings can collide."""
        plan = _plan(_step("s1", targets=["a.py"]), _step("s2", targets=["a.py"]))
        assert overlapping_sub_task_targets(plan) == ()


class TestNodesWithoutTargets:
    def test_leaf_step_with_no_targets(self) -> None:
        assert nodes_without_targets(_plan(_step("s1", targets=[]))) == ("s1",)

    def test_fan_out_parents_are_exempt(self) -> None:
        assert nodes_without_targets(CLEAN_PLAN) == ()

    def test_nested_leaf(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="d",
                target_files=[],
                sub_tasks=[
                    _sub("st1", targets=[], children=[_sub("gc1", targets=[])]),
                    _sub("st2", targets=["ok.py"]),
                ],
            )
        )
        assert nodes_without_targets(plan) == ("s1/st1/gc1",)


class TestEvalOnlyFinders:
    """The four checks the live gate deliberately does not veto on."""

    def test_undersized_fan_out(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="d",
                target_files=[],
                sub_tasks=[_sub("st1", targets=["a.py"])],
            )
        )
        assert undersized_fan_outs(plan) == ("s1: 1 sub-task(s)",)
        # ...and the gate lets it through: a one-child fan-out executes correctly.
        assert preflight_plan(plan) == ()

    def test_uncovered_task_targets(self) -> None:
        assert uncovered_task_targets(CLEAN_PLAN, ["a.py", "missing.py"]) == ("missing.py",)

    def test_implausible_context_files(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1", description="d", target_files=["a.py"], context_files=["ghost.py"]
            )
        )
        assert implausible_context_files(plan, {"real.py"}) == ("s1: ghost.py",)

    def test_context_file_produced_by_an_earlier_step_is_plausible(self) -> None:
        plan = _plan(
            _step("s1", targets=["a.py"]),
            PlanStep(step_id="s2", description="d", target_files=["b.py"], context_files=["a.py"]),
        )
        assert implausible_context_files(plan, set()) == ()

    def test_forward_reference(self) -> None:
        plan = _plan(
            PlanStep(step_id="s1", description="d", target_files=["a.py"], context_files=["b.py"]),
            _step("s2", targets=["b.py"]),
        )
        assert forward_references(plan, set()) == ("s1: b.py",)

    def test_existing_repo_file_rewritten_later_is_not_a_forward_reference(self) -> None:
        """Why this check cannot run without the repo file set."""
        plan = _plan(
            PlanStep(
                step_id="s1", description="d", target_files=["a.py"], context_files=["config.py"]
            ),
            _step("s2", targets=["config.py"]),
        )
        assert forward_references(plan, {"config.py"}) == ()


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class TestPreflightPlan:
    def test_clean_plan_passes(self) -> None:
        assert preflight_plan(CLEAN_PLAN) == ()

    def test_reports_the_rule_and_the_offender(self) -> None:
        plan = _plan(_step("s1"), _step("s1"))
        assert preflight_plan(plan) == (PlanViolation(check="duplicate_step_ids", detail="s1"),)

    def test_collects_every_violation(self) -> None:
        plan = _plan(
            _step("s1", targets=["/abs.py"]),
            PlanStep(
                step_id="s1",
                description="d",
                target_files=[],
                sub_tasks=[_sub("st1", targets=["x.py"]), _sub("st1", targets=["x.py"])],
            ),
        )
        checks = {v.check for v in preflight_plan(plan)}
        assert checks == {
            "duplicate_step_ids",
            "duplicate_sub_task_ids",
            "overlapping_sub_task_targets",
            "unsafe_target_paths",
        }

    def test_gate_subset_is_the_documented_five(self) -> None:
        assert [name for name, _find in PREFLIGHT_CHECKS] == [
            "duplicate_step_ids",
            "duplicate_sub_task_ids",
            "overlapping_sub_task_targets",
            "unsafe_target_paths",
            "nodes_without_targets",
        ]

    def test_violation_summary(self) -> None:
        violations = (
            PlanViolation(check="duplicate_step_ids", detail="s1"),
            PlanViolation(check="nodes_without_targets", detail="s2"),
        )
        assert violation_summary(violations) == "duplicate_step_ids: s1; nodes_without_targets: s2"


class TestRetryContext:
    def test_section_names_every_violation(self) -> None:
        section = retry_prompt_section(
            (PlanViolation(check="duplicate_step_ids", detail="s1"),),
            attempt=2,
            max_attempts=MAX_PLANNER_ATTEMPTS,
        )
        assert "attempt 2 of 3" in section
        assert "- duplicate_step_ids: s1" in section

    def test_thinking_escalates_to_max_effort(self) -> None:
        escalated = escalate_thinking(ThinkingPolicy(enabled=False, effort="low"))
        assert escalated.enabled is True
        assert escalated.effort == "max"


# ---------------------------------------------------------------------------
# Revision splicing
# ---------------------------------------------------------------------------


class TestSpliceRevision:
    def test_replaces_the_remaining_steps(self) -> None:
        plan = _plan(_step("s1"), _step("s2"), _step("s3"))
        spliced = splice_revision(
            plan,
            completed_through=0,
            revised_steps=[_step("r1")],
            revision_count=0,
        )
        assert isinstance(spliced, RevisedPlan)
        assert [s.step_id for s in spliced.plan.steps] == ["s1", "r1"]

    def test_empty_revision_ends_the_plan(self) -> None:
        plan = _plan(_step("s1"), _step("s2"))
        spliced = splice_revision(plan, completed_through=0, revised_steps=[], revision_count=0)
        assert isinstance(spliced, RevisedPlan)
        assert [s.step_id for s in spliced.plan.steps] == ["s1"]

    def test_over_cap_splice_is_caught_not_raised(self) -> None:
        """The hazard: ``Plan(steps=...)`` over MAX_PLAN_STEPS raises a pydantic
        ValidationError inside workflow code — a workflow task Temporal retries
        forever. The cap must catch it and return a reason."""
        plan = _plan(_step("s1"), _step("s2"))
        revised = [_step(f"r{i}") for i in range(MAX_PLAN_STEPS)]
        spliced = splice_revision(
            plan, completed_through=0, revised_steps=revised, revision_count=0
        )
        assert isinstance(spliced, RevisionRejected)
        assert "exceed the step cap" in spliced.reason
        assert f"{MAX_PLAN_STEPS + 1} steps" in spliced.reason

    def test_revision_count_cap(self) -> None:
        plan = _plan(_step("s1"), _step("s2"))
        spliced = splice_revision(
            plan,
            completed_through=0,
            revised_steps=[_step("r1")],
            revision_count=MAX_PLAN_REVISIONS,
        )
        assert isinstance(spliced, RevisionRejected)
        assert "revision cap exceeded" in spliced.reason

    def test_structurally_invalid_revision_is_rejected(self) -> None:
        """A revised step colliding with a completed one would re-run its id."""
        plan = _plan(_step("s1"), _step("s2"))
        spliced = splice_revision(
            plan, completed_through=0, revised_steps=[_step("s1")], revision_count=0
        )
        assert isinstance(spliced, RevisionRejected)
        assert "duplicate_step_ids: s1" in spliced.reason

    def test_exactly_at_the_cap_is_allowed(self) -> None:
        plan = _plan(_step("s1"), _step("s2"))
        revised = [_step(f"r{i}") for i in range(MAX_PLAN_STEPS - 1)]
        spliced = splice_revision(
            plan, completed_through=0, revised_steps=revised, revision_count=0
        )
        assert isinstance(spliced, RevisedPlan)
        assert len(spliced.plan.steps) == MAX_PLAN_STEPS
