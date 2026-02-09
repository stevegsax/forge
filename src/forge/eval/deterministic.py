"""Layer 1: Deterministic structural checks for plan quality.

All functions are pure — they take a Plan + TaskDefinition (+ optional repo
file set) and return a DeterministicCheckResult. No I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.eval.models import CheckStatus, DeterministicCheckResult, DeterministicResult

if TYPE_CHECKING:
    from forge.models import Plan, TaskDefinition


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_target_files_are_relative_paths(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify no step/sub-task target files use absolute paths or '..' traversal."""
    bad: list[str] = []
    for step in plan.steps:
        for f in step.target_files:
            if f.startswith("/") or ".." in f.split("/"):
                bad.append(f"{step.step_id}: {f}")
        if step.sub_tasks:
            for st in step.sub_tasks:
                for f in st.target_files:
                    if f.startswith("/") or ".." in f.split("/"):
                        bad.append(f"{step.step_id}/{st.sub_task_id}: {f}")
    if bad:
        return DeterministicCheckResult(
            check_name="check_target_files_are_relative_paths",
            status=CheckStatus.FAIL,
            message=f"Found {len(bad)} absolute or traversal path(s).",
            details=bad,
        )
    return DeterministicCheckResult(
        check_name="check_target_files_are_relative_paths",
        status=CheckStatus.PASS,
        message="All target files are relative paths.",
    )


def check_step_ids_unique(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify all step IDs in the plan are unique."""
    seen: dict[str, int] = {}
    for step in plan.steps:
        seen[step.step_id] = seen.get(step.step_id, 0) + 1
    dupes = [sid for sid, count in seen.items() if count > 1]
    if dupes:
        return DeterministicCheckResult(
            check_name="check_step_ids_unique",
            status=CheckStatus.FAIL,
            message=f"Duplicate step IDs: {', '.join(dupes)}.",
            details=dupes,
        )
    return DeterministicCheckResult(
        check_name="check_step_ids_unique",
        status=CheckStatus.PASS,
        message="All step IDs are unique.",
    )


def check_sub_task_ids_unique(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify sub-task IDs are unique within each step."""
    dupes: list[str] = []
    for step in plan.steps:
        if not step.sub_tasks:
            continue
        seen: dict[str, int] = {}
        for st in step.sub_tasks:
            seen[st.sub_task_id] = seen.get(st.sub_task_id, 0) + 1
        for stid, count in seen.items():
            if count > 1:
                dupes.append(f"{step.step_id}/{stid}")
    if dupes:
        return DeterministicCheckResult(
            check_name="check_sub_task_ids_unique",
            status=CheckStatus.FAIL,
            message=f"Duplicate sub-task IDs: {', '.join(dupes)}.",
            details=dupes,
        )
    return DeterministicCheckResult(
        check_name="check_sub_task_ids_unique",
        status=CheckStatus.PASS,
        message="All sub-task IDs are unique within their steps.",
    )


def check_sub_task_targets_non_overlapping(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify sub-tasks within a step don't share target files (D27)."""
    overlaps: list[str] = []
    for step in plan.steps:
        if not step.sub_tasks:
            continue
        seen_files: dict[str, str] = {}
        for st in step.sub_tasks:
            for f in st.target_files:
                if f in seen_files:
                    overlaps.append(
                        f"{step.step_id}: {f} claimed by {seen_files[f]} and {st.sub_task_id}"
                    )
                else:
                    seen_files[f] = st.sub_task_id
    if overlaps:
        return DeterministicCheckResult(
            check_name="check_sub_task_targets_non_overlapping",
            status=CheckStatus.FAIL,
            message=f"Found {len(overlaps)} overlapping target(s) in fan-out steps.",
            details=overlaps,
        )
    return DeterministicCheckResult(
        check_name="check_sub_task_targets_non_overlapping",
        status=CheckStatus.PASS,
        message="No overlapping sub-task targets.",
    )


def check_context_files_plausible(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify context files either exist in repo or are produced by earlier steps."""
    if known_repo_files is None:
        return DeterministicCheckResult(
            check_name="check_context_files_plausible",
            status=CheckStatus.SKIP,
            message="No known_repo_files provided; skipping.",
        )

    # Build cumulative set of files produced by steps so far
    produced: set[str] = set()
    implausible: list[str] = []

    for step in plan.steps:
        # Check step-level context files
        for f in step.context_files:
            if f not in known_repo_files and f not in produced:
                implausible.append(f"{step.step_id}: {f}")

        # Check sub-task context files
        if step.sub_tasks:
            for st in step.sub_tasks:
                for f in st.context_files:
                    if f not in known_repo_files and f not in produced:
                        implausible.append(f"{step.step_id}/{st.sub_task_id}: {f}")

        # Add files this step produces
        for f in step.target_files:
            produced.add(f)
        if step.sub_tasks:
            for st in step.sub_tasks:
                for f in st.target_files:
                    produced.add(f)

    if implausible:
        return DeterministicCheckResult(
            check_name="check_context_files_plausible",
            status=CheckStatus.FAIL,
            message=f"Found {len(implausible)} implausible context file(s).",
            details=implausible,
        )
    return DeterministicCheckResult(
        check_name="check_context_files_plausible",
        status=CheckStatus.PASS,
        message="All context files are plausible.",
    )


def check_no_forward_references(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify no step references files produced only by a later step."""
    if known_repo_files is None:
        return DeterministicCheckResult(
            check_name="check_no_forward_references",
            status=CheckStatus.SKIP,
            message="No known_repo_files provided; skipping.",
        )

    # Collect all files produced by each step index
    step_outputs: list[set[str]] = []
    for step in plan.steps:
        outputs: set[str] = set(step.target_files)
        if step.sub_tasks:
            for st in step.sub_tasks:
                outputs.update(st.target_files)
        step_outputs.append(outputs)

    forward_refs: list[str] = []

    for i, step in enumerate(plan.steps):
        # Files available before this step: repo files + outputs of steps 0..i-1
        available = set(known_repo_files)
        for j in range(i):
            available.update(step_outputs[j])

        # Check context files
        for f in step.context_files:
            if f not in available:
                # It could be produced by this step or a later step
                for j in range(i, len(plan.steps)):
                    if f in step_outputs[j]:
                        forward_refs.append(f"{step.step_id}: {f}")
                        break

        if step.sub_tasks:
            for st in step.sub_tasks:
                for f in st.context_files:
                    if f not in available:
                        for j in range(i, len(plan.steps)):
                            if f in step_outputs[j]:
                                forward_refs.append(f"{step.step_id}/{st.sub_task_id}: {f}")
                                break

    if forward_refs:
        return DeterministicCheckResult(
            check_name="check_no_forward_references",
            status=CheckStatus.FAIL,
            message=f"Found {len(forward_refs)} forward reference(s).",
            details=forward_refs,
        )
    return DeterministicCheckResult(
        check_name="check_no_forward_references",
        status=CheckStatus.PASS,
        message="No forward references found.",
    )


def check_all_task_targets_covered(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify every task target_file appears in at least one step."""
    if not task.target_files:
        return DeterministicCheckResult(
            check_name="check_all_task_targets_covered",
            status=CheckStatus.SKIP,
            message="Task has no target files to check.",
        )

    plan_targets: set[str] = set()
    for step in plan.steps:
        plan_targets.update(step.target_files)
        if step.sub_tasks:
            for st in step.sub_tasks:
                plan_targets.update(st.target_files)

    missing = [f for f in task.target_files if f not in plan_targets]
    if missing:
        return DeterministicCheckResult(
            check_name="check_all_task_targets_covered",
            status=CheckStatus.FAIL,
            message=f"{len(missing)} task target(s) not covered by plan.",
            details=missing,
        )
    return DeterministicCheckResult(
        check_name="check_all_task_targets_covered",
        status=CheckStatus.PASS,
        message="All task targets are covered by the plan.",
    )


def check_non_fanout_steps_have_targets(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify non-fan-out steps have non-empty target_files."""
    empty: list[str] = []
    for step in plan.steps:
        if not step.sub_tasks and not step.target_files:
            empty.append(step.step_id)
    if empty:
        return DeterministicCheckResult(
            check_name="check_non_fanout_steps_have_targets",
            status=CheckStatus.FAIL,
            message=f"{len(empty)} non-fan-out step(s) have no target files.",
            details=empty,
        )
    return DeterministicCheckResult(
        check_name="check_non_fanout_steps_have_targets",
        status=CheckStatus.PASS,
        message="All non-fan-out steps have target files.",
    )


def check_fanout_steps_have_min_subtasks(
    plan: Plan,
    task: TaskDefinition,
    known_repo_files: set[str] | None = None,
) -> DeterministicCheckResult:
    """Verify fan-out steps have >= 2 sub-tasks."""
    bad: list[str] = []
    for step in plan.steps:
        if step.sub_tasks is not None and len(step.sub_tasks) < 2:
            bad.append(f"{step.step_id}: {len(step.sub_tasks)} sub-task(s)")
    if bad:
        return DeterministicCheckResult(
            check_name="check_fanout_steps_have_min_subtasks",
            status=CheckStatus.FAIL,
            message=f"{len(bad)} fan-out step(s) have fewer than 2 sub-tasks.",
            details=bad,
        )
    return DeterministicCheckResult(
        check_name="check_fanout_steps_have_min_subtasks",
        status=CheckStatus.PASS,
        message="All fan-out steps have >= 2 sub-tasks.",
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
