"""Layer 1: Deterministic structural checks for plan quality.

All functions are pure — they take a Plan + TaskDefinition (+ optional repo
file set) and return a DeterministicCheckResult. No I/O.

Since T5.6 the *logic* of every check lives in :mod:`forge.plan_checks`, which
the live preflight gate (``blocks.dispatch.dispatch_planner``) runs at plan
acceptance. This module is the eval harness's presentation of the same finders:
one algorithm, two consumers, cross-referenced rather than copied. That is also
where the checks became recursive — a violation nested inside a sub-task's own
``sub_tasks`` used to be invisible to this harness.

Four checks here are eval-only, in the sense that the live gate does not veto on
them; :data:`forge.plan_checks.PREFLIGHT_CHECKS` documents why for each.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.eval.models import CheckStatus, DeterministicCheckResult, DeterministicResult
from forge.plan_checks import (
    duplicate_step_ids,
    duplicate_sub_task_ids,
    forward_references,
    implausible_context_files,
    nodes_without_targets,
    overlapping_sub_task_targets,
    uncovered_task_targets,
    undersized_fan_outs,
    unsafe_target_paths,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from forge.models import Plan, TaskDefinition


def _result(
    check_name: str,
    findings: Sequence[str],
    *,
    fail_message: str,
    pass_message: str,
) -> DeterministicCheckResult:
    """Wrap a finder's output in the harness's result model."""
    if findings:
        return DeterministicCheckResult(
            check_name=check_name,
            status=CheckStatus.FAIL,
            message=fail_message,
            details=list(findings),
        )
    return DeterministicCheckResult(
        check_name=check_name,
        status=CheckStatus.PASS,
        message=pass_message,
    )


def _skipped(check_name: str, message: str) -> DeterministicCheckResult:
    return DeterministicCheckResult(
        check_name=check_name,
        status=CheckStatus.SKIP,
        message=message,
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_target_files_are_relative_paths(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify no step/sub-task target files use absolute paths or '..' traversal."""
    bad = unsafe_target_paths(plan)
    return _result(
        "check_target_files_are_relative_paths",
        bad,
        fail_message=f"Found {len(bad)} absolute or traversal path(s).",
        pass_message="All target files are relative paths.",
    )


def check_step_ids_unique(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify all step IDs in the plan are unique."""
    dupes = duplicate_step_ids(plan)
    return _result(
        "check_step_ids_unique",
        dupes,
        fail_message=f"Duplicate step IDs: {', '.join(dupes)}.",
        pass_message="All step IDs are unique.",
    )


def check_sub_task_ids_unique(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify sub-task IDs are unique among the children of each parent."""
    dupes = duplicate_sub_task_ids(plan)
    return _result(
        "check_sub_task_ids_unique",
        dupes,
        fail_message=f"Duplicate sub-task IDs: {', '.join(dupes)}.",
        pass_message="All sub-task IDs are unique within their steps.",
    )


def check_sub_task_targets_non_overlapping(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify sibling sub-tasks don't share target files (D27)."""
    overlaps = overlapping_sub_task_targets(plan)
    return _result(
        "check_sub_task_targets_non_overlapping",
        overlaps,
        fail_message=f"Found {len(overlaps)} overlapping target(s) in fan-out steps.",
        pass_message="No overlapping sub-task targets.",
    )


def check_context_files_plausible(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify context files either exist in repo or are produced by earlier steps."""
    if known_repo_files is None:
        return _skipped("check_context_files_plausible", "No known_repo_files provided; skipping.")
    implausible = implausible_context_files(plan, known_repo_files)
    return _result(
        "check_context_files_plausible",
        implausible,
        fail_message=f"Found {len(implausible)} implausible context file(s).",
        pass_message="All context files are plausible.",
    )


def check_no_forward_references(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify no step references files produced only by a later step."""
    if known_repo_files is None:
        return _skipped("check_no_forward_references", "No known_repo_files provided; skipping.")
    refs = forward_references(plan, known_repo_files)
    return _result(
        "check_no_forward_references",
        refs,
        fail_message=f"Found {len(refs)} forward reference(s).",
        pass_message="No forward references found.",
    )


def check_all_task_targets_covered(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify every task target_file appears in at least one step."""
    if not task.target_files:
        return _skipped("check_all_task_targets_covered", "Task has no target files to check.")
    missing = uncovered_task_targets(plan, task.target_files)
    return _result(
        "check_all_task_targets_covered",
        missing,
        fail_message=f"{len(missing)} task target(s) not covered by plan.",
        pass_message="All task targets are covered by the plan.",
    )


def check_non_fanout_steps_have_targets(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify leaf steps and sub-tasks have non-empty target_files."""
    empty = nodes_without_targets(plan)
    return _result(
        "check_non_fanout_steps_have_targets",
        empty,
        fail_message=f"{len(empty)} non-fan-out step(s) have no target files.",
        pass_message="All non-fan-out steps have target files.",
    )


def check_fanout_steps_have_min_subtasks(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify fan-out steps have >= 2 sub-tasks."""
    bad = undersized_fan_outs(plan)
    return _result(
        "check_fanout_steps_have_min_subtasks",
        bad,
        fail_message=f"{len(bad)} fan-out step(s) have fewer than 2 sub-tasks.",
        pass_message="All fan-out steps have >= 2 sub-tasks.",
    )


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_target_files_are_relative_paths,
    check_step_ids_unique,
    check_sub_task_ids_unique,
    check_sub_task_targets_non_overlapping,
    check_context_files_plausible,
    check_no_forward_references,
    check_all_task_targets_covered,
    check_non_fanout_steps_have_targets,
    check_fanout_steps_have_min_subtasks,
]


def run_deterministic_checks(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicResult:
    """Run all deterministic checks and return aggregated results."""
    checks = [check(plan, task, known_repo_files) for check in ALL_CHECKS]
    all_passed = all(c.status != CheckStatus.FAIL for c in checks)
    return DeterministicResult(checks=checks, all_passed=all_passed)
