"""Tests for forge.eval.deterministic — structural plan checks."""

from __future__ import annotations

from forge.eval.deterministic import (
    check_all_task_targets_covered,
    check_context_files_plausible,
    check_fanout_steps_have_min_subtasks,
    check_no_forward_references,
    check_non_fanout_steps_have_targets,
    check_step_ids_unique,
    check_sub_task_ids_unique,
    check_sub_task_targets_non_overlapping,
    check_target_files_are_relative_paths,
    run_deterministic_checks,
)
from forge.eval.models import CheckStatus
from forge.models import Plan, PlanStep, SubTask, TaskDefinition

_TASK = TaskDefinition(task_id="t1", description="Test task.")


def _plan(*steps: PlanStep) -> Plan:
    return Plan(task_id="t1", steps=list(steps), explanation="Test plan.")


# ---------------------------------------------------------------------------
# check_target_files_are_relative_paths
# ---------------------------------------------------------------------------


class TestCheckTargetFilesAreRelativePaths:
    def test_pass_relative(self) -> None:
        plan = _plan(PlanStep(step_id="s1", description="x", target_files=["src/a.py"]))
        r = check_target_files_are_relative_paths(plan, _TASK)
        assert r.status == CheckStatus.PASS

    def test_fail_absolute(self) -> None:
        plan = _plan(PlanStep(step_id="s1", description="x", target_files=["/etc/passwd"]))
        r = check_target_files_are_relative_paths(plan, _TASK)
        assert r.status == CheckStatus.FAIL
        assert "/etc/passwd" in r.details[0]

    def test_fail_traversal(self) -> None:
        plan = _plan(PlanStep(step_id="s1", description="x", target_files=["../secret.py"]))
        r = check_target_files_are_relative_paths(plan, _TASK)
        assert r.status == CheckStatus.FAIL

    def test_fail_in_subtask(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="st1", description="x", target_files=["/bad"]),
                    SubTask(sub_task_id="st2", description="x", target_files=["ok.py"]),
                ],
            )
        )
        r = check_target_files_are_relative_paths(plan, _TASK)
        assert r.status == CheckStatus.FAIL
        assert len(r.details) == 1


# ---------------------------------------------------------------------------
# check_step_ids_unique
# ---------------------------------------------------------------------------


class TestCheckStepIdsUnique:
    def test_pass_unique(self) -> None:
        plan = _plan(
            PlanStep(step_id="s1", description="x", target_files=["a.py"]),
            PlanStep(step_id="s2", description="y", target_files=["b.py"]),
        )
        r = check_step_ids_unique(plan, _TASK)
        assert r.status == CheckStatus.PASS

    def test_fail_duplicate(self) -> None:
        plan = _plan(
            PlanStep(step_id="s1", description="x", target_files=["a.py"]),
            PlanStep(step_id="s1", description="y", target_files=["b.py"]),
        )
        r = check_step_ids_unique(plan, _TASK)
        assert r.status == CheckStatus.FAIL
        assert "s1" in r.details


# ---------------------------------------------------------------------------
# check_sub_task_ids_unique
# ---------------------------------------------------------------------------


class TestCheckSubTaskIdsUnique:
    def test_pass_no_subtasks(self) -> None:
        plan = _plan(PlanStep(step_id="s1", description="x", target_files=["a.py"]))
        r = check_sub_task_ids_unique(plan, _TASK)
        assert r.status == CheckStatus.PASS

    def test_pass_unique_subtasks(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="st1", description="a", target_files=["a.py"]),
                    SubTask(sub_task_id="st2", description="b", target_files=["b.py"]),
                ],
            )
        )
        r = check_sub_task_ids_unique(plan, _TASK)
        assert r.status == CheckStatus.PASS

    def test_fail_duplicate_within_step(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="st1", description="a", target_files=["a.py"]),
                    SubTask(sub_task_id="st1", description="b", target_files=["b.py"]),
                ],
            )
        )
        r = check_sub_task_ids_unique(plan, _TASK)
        assert r.status == CheckStatus.FAIL


# ---------------------------------------------------------------------------
# check_sub_task_targets_non_overlapping
# ---------------------------------------------------------------------------


class TestCheckSubTaskTargetsNonOverlapping:
    def test_pass_no_overlap(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="st1", description="a", target_files=["a.py"]),
                    SubTask(sub_task_id="st2", description="b", target_files=["b.py"]),
                ],
            )
        )
        r = check_sub_task_targets_non_overlapping(plan, _TASK)
        assert r.status == CheckStatus.PASS

    def test_fail_overlap(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="st1", description="a", target_files=["shared.py"]),
                    SubTask(sub_task_id="st2", description="b", target_files=["shared.py"]),
                ],
            )
        )
        r = check_sub_task_targets_non_overlapping(plan, _TASK)
        assert r.status == CheckStatus.FAIL
        assert "shared.py" in r.details[0]


# ---------------------------------------------------------------------------
# check_context_files_plausible
# ---------------------------------------------------------------------------


class TestCheckContextFilesPlausible:
    def test_skip_when_no_known_files(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=["a.py"],
                context_files=["b.py"],
            )
        )
        r = check_context_files_plausible(plan, _TASK)
        assert r.status == CheckStatus.SKIP

    def test_pass_file_in_repo(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=["a.py"],
                context_files=["existing.py"],
            )
        )
        r = check_context_files_plausible(plan, _TASK, known_repo_files={"existing.py"})
        assert r.status == CheckStatus.PASS

    def test_pass_file_from_earlier_step(self) -> None:
        plan = _plan(
            PlanStep(step_id="s1", description="x", target_files=["new.py"]),
            PlanStep(
                step_id="s2",
                description="y",
                target_files=["other.py"],
                context_files=["new.py"],
            ),
        )
        r = check_context_files_plausible(plan, _TASK, known_repo_files=set())
        assert r.status == CheckStatus.PASS

    def test_fail_unknown_file(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=["a.py"],
                context_files=["ghost.py"],
            )
        )
        r = check_context_files_plausible(plan, _TASK, known_repo_files=set())
        assert r.status == CheckStatus.FAIL
        assert "ghost.py" in r.details[0]

    def test_fail_subtask_context_unknown(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=[],
                sub_tasks=[
                    SubTask(
                        sub_task_id="st1",
                        description="a",
                        target_files=["a.py"],
                        context_files=["nope.py"],
                    ),
                    SubTask(sub_task_id="st2", description="b", target_files=["b.py"]),
                ],
            )
        )
        r = check_context_files_plausible(plan, _TASK, known_repo_files=set())
        assert r.status == CheckStatus.FAIL


# ---------------------------------------------------------------------------
# check_no_forward_references
# ---------------------------------------------------------------------------


class TestCheckNoForwardReferences:
    def test_skip_when_no_known_files(self) -> None:
        plan = _plan(PlanStep(step_id="s1", description="x", target_files=["a.py"]))
        r = check_no_forward_references(plan, _TASK)
        assert r.status == CheckStatus.SKIP

    def test_pass_no_references(self) -> None:
        plan = _plan(
            PlanStep(step_id="s1", description="x", target_files=["a.py"]),
            PlanStep(
                step_id="s2",
                description="y",
                target_files=["b.py"],
                context_files=["a.py"],
            ),
        )
        r = check_no_forward_references(plan, _TASK, known_repo_files=set())
        assert r.status == CheckStatus.PASS

    def test_fail_forward_reference(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=["a.py"],
                context_files=["b.py"],
            ),
            PlanStep(step_id="s2", description="y", target_files=["b.py"]),
        )
        r = check_no_forward_references(plan, _TASK, known_repo_files=set())
        assert r.status == CheckStatus.FAIL
        assert "s1: b.py" in r.details


# ---------------------------------------------------------------------------
# check_all_task_targets_covered
# ---------------------------------------------------------------------------


class TestCheckAllTaskTargetsCovered:
    def test_skip_no_task_targets(self) -> None:
        plan = _plan(PlanStep(step_id="s1", description="x", target_files=["a.py"]))
        r = check_all_task_targets_covered(plan, _TASK)
        assert r.status == CheckStatus.SKIP

    def test_pass_all_covered(self) -> None:
        task = TaskDefinition(task_id="t1", description="Test.", target_files=["a.py", "b.py"])
        plan = _plan(
            PlanStep(step_id="s1", description="x", target_files=["a.py"]),
            PlanStep(step_id="s2", description="y", target_files=["b.py"]),
        )
        r = check_all_task_targets_covered(plan, task)
        assert r.status == CheckStatus.PASS

    def test_fail_missing(self) -> None:
        task = TaskDefinition(
            task_id="t1", description="Test.", target_files=["a.py", "missing.py"]
        )
        plan = _plan(PlanStep(step_id="s1", description="x", target_files=["a.py"]))
        r = check_all_task_targets_covered(plan, task)
        assert r.status == CheckStatus.FAIL
        assert "missing.py" in r.details

    def test_covered_by_subtask(self) -> None:
        task = TaskDefinition(task_id="t1", description="Test.", target_files=["a.py"])
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="st1", description="a", target_files=["a.py"]),
                    SubTask(sub_task_id="st2", description="b", target_files=["b.py"]),
                ],
            )
        )
        r = check_all_task_targets_covered(plan, task)
        assert r.status == CheckStatus.PASS


# ---------------------------------------------------------------------------
# check_non_fanout_steps_have_targets
# ---------------------------------------------------------------------------


class TestCheckNonFanoutStepsHaveTargets:
    def test_pass_has_targets(self) -> None:
        plan = _plan(PlanStep(step_id="s1", description="x", target_files=["a.py"]))
        r = check_non_fanout_steps_have_targets(plan, _TASK)
        assert r.status == CheckStatus.PASS

    def test_fail_empty_targets(self) -> None:
        plan = _plan(PlanStep(step_id="s1", description="x", target_files=[]))
        r = check_non_fanout_steps_have_targets(plan, _TASK)
        assert r.status == CheckStatus.FAIL
        assert "s1" in r.details

    def test_pass_fanout_with_empty_targets(self) -> None:
        """Fan-out steps may have empty target_files since sub-tasks have them."""
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="st1", description="a", target_files=["a.py"]),
                    SubTask(sub_task_id="st2", description="b", target_files=["b.py"]),
                ],
            )
        )
        r = check_non_fanout_steps_have_targets(plan, _TASK)
        assert r.status == CheckStatus.PASS


# ---------------------------------------------------------------------------
# check_fanout_steps_have_min_subtasks
# ---------------------------------------------------------------------------


class TestCheckFanoutStepsHaveMinSubtasks:
    def test_pass_no_fanout(self) -> None:
        plan = _plan(PlanStep(step_id="s1", description="x", target_files=["a.py"]))
        r = check_fanout_steps_have_min_subtasks(plan, _TASK)
        assert r.status == CheckStatus.PASS

    def test_pass_two_subtasks(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="st1", description="a", target_files=["a.py"]),
                    SubTask(sub_task_id="st2", description="b", target_files=["b.py"]),
                ],
            )
        )
        r = check_fanout_steps_have_min_subtasks(plan, _TASK)
        assert r.status == CheckStatus.PASS

    def test_fail_one_subtask(self) -> None:
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=[],
                sub_tasks=[
                    SubTask(sub_task_id="st1", description="a", target_files=["a.py"]),
                ],
            )
        )
        r = check_fanout_steps_have_min_subtasks(plan, _TASK)
        assert r.status == CheckStatus.FAIL


# ---------------------------------------------------------------------------
# run_deterministic_checks
# ---------------------------------------------------------------------------


class TestRunDeterministicChecks:
    def test_all_pass(self) -> None:
        plan = _plan(
            PlanStep(step_id="s1", description="x", target_files=["a.py"]),
            PlanStep(step_id="s2", description="y", target_files=["b.py"]),
        )
        result = run_deterministic_checks(plan, _TASK)
        assert result.all_passed is True
        assert len(result.checks) == 9

    def test_some_fail(self) -> None:
        plan = _plan(
            PlanStep(step_id="s1", description="x", target_files=["/bad.py"]),
        )
        result = run_deterministic_checks(plan, _TASK)
        assert result.all_passed is False

    def test_with_known_repo_files(self) -> None:
        task = TaskDefinition(task_id="t1", description="Test.", target_files=["a.py"])
        plan = _plan(
            PlanStep(
                step_id="s1",
                description="x",
                target_files=["a.py"],
                context_files=["existing.py"],
            )
        )
        result = run_deterministic_checks(plan, task, known_repo_files={"existing.py"})
        assert result.all_passed is True
