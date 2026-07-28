"""Tests for forge.step_logic — the pure step-logic core (T5.1, D95).

These are functional-core unit tests: no Temporal, no I/O, microsecond-fast.
They cover the full transition decision matrix (formerly the evaluate_transition
activity), the failure-summary/merge/slim/totals/id/timeout helpers, the result
builders' failure_kind wiring, and the payload-cap acceptance test that proves a
large fan-out's serialized result stays well under Temporal's ~2MB limit.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from forge.models import (
    BATCH_WAIT_CEILING,
    ConflictResolutionCallResult,
    ContextStats,
    FailureKind,
    FileConflict,
    FileConflictVersion,
    LLMStats,
    Plan,
    PlanStep,
    StepResult,
    SubTaskResult,
    TaskResult,
    TransitionSignal,
    ValidationResult,
)
from forge.step_logic import (
    MergedFiles,
    MissingResolutions,
    child_timeout,
    compound_sub_task_id,
    determine_transition,
    failure_summary,
    fan_out_step_failure,
    fan_out_success,
    file_digests,
    llm_totals,
    merge_resolution,
    nested_fan_out_failure,
    plan_preflight_failure,
    planned_failure,
    single_step_terminal,
    slim_result,
    sub_task_batch_wait_failure,
    sub_task_terminal,
    sub_task_workflow_id,
    subtask_failure_summary,
    sum_llm_stats,
)


def _passed(name: str = "ruff_lint") -> ValidationResult:
    return ValidationResult(check_name=name, passed=True, summary=f"{name} passed")


def _failed(name: str = "ruff_lint", summary: str = "errors") -> ValidationResult:
    return ValidationResult(check_name=name, passed=False, summary=summary)


def _stats(
    *, input_tokens: int = 100, output_tokens: int = 50, latency_ms: float = 200.0
) -> LLMStats:
    return LLMStats(
        model_name="mock-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=7,
    )


# ---------------------------------------------------------------------------
# determine_transition — full decision matrix
# ---------------------------------------------------------------------------


class TestDetermineTransition:
    @pytest.mark.parametrize(
        ("results", "attempt", "max_attempts", "expected"),
        [
            # all-passed / empty → SUCCESS regardless of attempt
            ([_passed()], 1, 2, TransitionSignal.SUCCESS),
            ([], 1, 2, TransitionSignal.SUCCESS),
            ([_passed("lint"), _passed("format")], 2, 2, TransitionSignal.SUCCESS),
            # any failure, attempt < max → RETRYABLE
            ([_failed()], 1, 2, TransitionSignal.FAILURE_RETRYABLE),
            ([_passed(), _failed("format", "bad")], 1, 2, TransitionSignal.FAILURE_RETRYABLE),
            ([_failed()], 2, 3, TransitionSignal.FAILURE_RETRYABLE),
            # any failure, attempt >= max → TERMINAL
            ([_failed()], 2, 2, TransitionSignal.FAILURE_TERMINAL),
            ([_failed()], 3, 3, TransitionSignal.FAILURE_TERMINAL),
            ([_passed(), _failed()], 2, 2, TransitionSignal.FAILURE_TERMINAL),
        ],
    )
    def test_matrix(
        self,
        results: list[ValidationResult],
        attempt: int,
        max_attempts: int,
        expected: TransitionSignal,
    ) -> None:
        assert determine_transition(results, attempt, max_attempts) == expected

    def test_default_max_attempts_is_two(self) -> None:
        assert determine_transition([_failed()], 1) == TransitionSignal.FAILURE_RETRYABLE
        assert determine_transition([_failed()], 2) == TransitionSignal.FAILURE_TERMINAL


# ---------------------------------------------------------------------------
# Failure-summary helpers
# ---------------------------------------------------------------------------


class TestFailureSummary:
    def test_joins_only_failing_summaries(self) -> None:
        results = [_passed("lint"), _failed("format", "bad format"), _failed("tests", "1 failed")]
        assert failure_summary(results) == "bad format; 1 failed"

    def test_empty_when_all_pass(self) -> None:
        assert failure_summary([_passed(), _passed("format")]) == ""

    def test_empty_on_empty(self) -> None:
        assert failure_summary([]) == ""


class TestSubtaskFailureSummary:
    def test_joins_id_and_error(self) -> None:
        term = TransitionSignal.FAILURE_TERMINAL
        failed = [
            SubTaskResult(sub_task_id="st1", status=term, error="boom"),
            SubTaskResult(sub_task_id="st2", status=term, error="nope"),
        ]
        assert subtask_failure_summary(failed) == "st1: boom; st2: nope"

    def test_empty_on_empty(self) -> None:
        assert subtask_failure_summary([]) == ""


# ---------------------------------------------------------------------------
# merge_resolution
# ---------------------------------------------------------------------------


def _conflict(path: str) -> FileConflict:
    return FileConflict(
        file_path=path,
        versions=[
            FileConflictVersion(source_id="st1", content="a"),
            FileConflictVersion(source_id="st2", content="b"),
        ],
    )


class TestMergeResolution:
    def test_all_resolved_returns_merged(self) -> None:
        result = merge_resolution(
            conflicts=[_conflict("shared.py")],
            resolved={"shared.py": "merged"},
            non_conflicting={"other.py": "kept"},
        )
        assert result == MergedFiles(files={"other.py": "kept", "shared.py": "merged"})

    def test_resolved_overrides_non_conflicting(self) -> None:
        result = merge_resolution(
            conflicts=[_conflict("shared.py")],
            resolved={"shared.py": "resolved-wins"},
            non_conflicting={"shared.py": "should-be-overwritten", "x.py": "x"},
        )
        assert isinstance(result, MergedFiles)
        assert result.files["shared.py"] == "resolved-wins"
        assert result.files["x.py"] == "x"

    def test_missing_returns_sorted_missing(self) -> None:
        result = merge_resolution(
            conflicts=[_conflict("b.py"), _conflict("a.py")],
            resolved={"a.py": "done"},
            non_conflicting={},
        )
        assert result == MissingResolutions(missing=("b.py",))

    def test_missing_is_sorted(self) -> None:
        result = merge_resolution(
            conflicts=[_conflict("z.py"), _conflict("a.py"), _conflict("m.py")],
            resolved={},
            non_conflicting={},
        )
        assert result == MissingResolutions(missing=("a.py", "m.py", "z.py"))

    def test_no_conflicts_returns_non_conflicting(self) -> None:
        result = merge_resolution(conflicts=[], resolved={}, non_conflicting={"a.py": "1"})
        assert result == MergedFiles(files={"a.py": "1"})


# ---------------------------------------------------------------------------
# file_digests / slimming
# ---------------------------------------------------------------------------


class TestFileDigests:
    def test_deterministic_sha256(self) -> None:
        # Known vector — pins the algorithm and utf-8 encoding without
        # re-deriving the expected value from the implementation.
        expected = "caf026f25d7140209f98072605307a438914b9ce6f3c14b23d15d9667241de52"
        assert file_digests({"a.py": "print('hi')\n"}) == {"a.py": expected}

    def test_empty(self) -> None:
        assert file_digests({}) == {}


class TestSlim:
    def test_slim_sub_task_moves_content_to_digests(self) -> None:
        original = SubTaskResult(
            sub_task_id="st1",
            status=TransitionSignal.SUCCESS,
            output_files={"a.py": "content-a"},
            llm_stats=_stats(),
        )
        slim = slim_result(original)
        assert slim.output_files == {}
        assert slim.output_digests == file_digests({"a.py": "content-a"})
        assert slim.llm_stats == original.llm_stats  # stats preserved

    def test_slim_is_recursive(self) -> None:
        child = SubTaskResult(
            sub_task_id="child",
            status=TransitionSignal.SUCCESS,
            output_files={"c.py": "cc"},
        )
        parent = SubTaskResult(
            sub_task_id="parent",
            status=TransitionSignal.SUCCESS,
            output_files={"p.py": "pp"},
            sub_task_results=[child],
        )
        slim = slim_result(parent)
        assert slim.output_files == {}
        assert slim.sub_task_results[0].output_files == {}
        assert slim.sub_task_results[0].output_digests == file_digests({"c.py": "cc"})

    def test_already_slim_returned_unchanged(self) -> None:
        original = SubTaskResult(
            sub_task_id="st1",
            status=TransitionSignal.SUCCESS,
            output_files={"a.py": "content-a"},
        )
        once = slim_result(original)
        twice = slim_result(once)
        # Idempotent AND cheap: an already-slim tree is returned by reference,
        # not re-copied node by node.
        assert twice is once

    def test_slim_step_drops_conflict_resolution_content_keeps_stats(self) -> None:
        cr = ConflictResolutionCallResult(
            task_id="t",
            resolved_files={"shared.py": "merged"},
            explanation="resolved",
            model_name="opus",
            input_tokens=10,
            output_tokens=20,
            latency_ms=5.0,
        )
        step = StepResult(
            step_id="s1",
            status=TransitionSignal.SUCCESS,
            output_files={"a.py": "aa"},
            conflict_resolution=cr,
        )
        slim = slim_result(step)
        assert slim.output_files == {}
        assert slim.output_digests == file_digests({"a.py": "aa"})
        assert slim.conflict_resolution is not None
        assert slim.conflict_resolution.resolved_files == {}
        assert slim.conflict_resolution.input_tokens == 10  # stats kept


class TestContentsTravelOnce:
    def test_both_populated_rejected(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            StepResult(
                step_id="s1",
                status=TransitionSignal.SUCCESS,
                output_files={"a.py": "content"},
                output_digests={"a.py": "deadbeef"},
            )
        with pytest.raises(ValueError, match="mutually exclusive"):
            SubTaskResult(
                sub_task_id="st1",
                status=TransitionSignal.SUCCESS,
                output_files={"a.py": "content"},
                output_digests={"a.py": "deadbeef"},
            )

    def test_file_count_in_either_state(self) -> None:
        step = StepResult(
            step_id="s1",
            status=TransitionSignal.SUCCESS,
            output_files={"a.py": "content", "b.py": "more"},
        )
        assert step.file_count == 2
        assert slim_result(step).file_count == 2


# ---------------------------------------------------------------------------
# llm_totals
# ---------------------------------------------------------------------------


class TestLLMTotals:
    def test_sums_across_tree(self) -> None:
        cr = ConflictResolutionCallResult(
            task_id="t",
            resolved_files={},
            explanation="x",
            model_name="opus",
            input_tokens=5,
            output_tokens=5,
            latency_ms=10.0,
            cache_creation_input_tokens=1,
            cache_read_input_tokens=2,
        )
        leaf = SubTaskResult(
            sub_task_id="st1",
            status=TransitionSignal.SUCCESS,
            llm_stats=_stats(input_tokens=10, output_tokens=20, latency_ms=100.0),
        )
        step = StepResult(
            step_id="s1",
            status=TransitionSignal.SUCCESS,
            sub_task_results=[leaf],
            conflict_resolution=cr,
        )
        result = TaskResult(
            task_id="t",
            status=TransitionSignal.SUCCESS,
            planner_stats=_stats(input_tokens=300, output_tokens=150, latency_ms=500.0),
            step_results=[step],
        )
        totals = llm_totals(result)
        # planner + leaf.llm_stats + step.conflict_resolution = 3 calls
        assert totals.call_count == 3
        assert totals.input_tokens == 300 + 10 + 5
        assert totals.output_tokens == 150 + 20 + 5
        assert totals.llm_time_ms == 500.0 + 100.0 + 10.0
        assert totals.cache_creation_input_tokens == 3 + 3 + 1  # planner+leaf from _stats, cr=1
        assert totals.cache_read_input_tokens == 7 + 7 + 2

    def test_single_step_counts_llm_stats(self) -> None:
        result = TaskResult(
            task_id="t",
            status=TransitionSignal.SUCCESS,
            llm_stats=_stats(input_tokens=42, output_tokens=8),
        )
        totals = llm_totals(result)
        assert totals.call_count == 1
        assert totals.input_tokens == 42

    def test_empty_result_is_zeroed(self) -> None:
        result = TaskResult(task_id="t", status=TransitionSignal.FAILURE_TERMINAL)
        totals = llm_totals(result)
        assert totals.call_count == 0
        assert totals.input_tokens == 0
        assert totals.llm_time_ms == 0.0


class TestSumLLMStats:
    """Several calls folded into one row (the preflight halt's planner attempts)."""

    def test_none_when_there_were_no_calls(self) -> None:
        assert sum_llm_stats([]) is None

    def test_counts_add_and_identity_comes_from_the_last_call(self) -> None:
        first = LLMStats(
            model_name="opus-a",
            input_tokens=300,
            output_tokens=150,
            latency_ms=500.0,
            cache_creation_input_tokens=1,
            cache_read_input_tokens=2,
            stop_reason="end_turn",
        )
        last = LLMStats(
            model_name="opus-b",
            input_tokens=400,
            output_tokens=200,
            latency_ms=700.0,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=4,
            stop_reason="max_tokens",
        )
        folded = sum_llm_stats([first, last])
        assert folded == LLMStats(
            model_name="opus-b",
            input_tokens=700,
            output_tokens=350,
            latency_ms=1200.0,
            cache_creation_input_tokens=4,
            cache_read_input_tokens=6,
            stop_reason="max_tokens",
        )


# ---------------------------------------------------------------------------
# id / timeout helpers
# ---------------------------------------------------------------------------


class TestIdHelpers:
    def test_compound_sub_task_id(self) -> None:
        assert compound_sub_task_id("parent-task", "st1") == "parent-task.sub.st1"

    def test_nested_compound(self) -> None:
        compound = compound_sub_task_id("parent-task", "st1")
        assert compound_sub_task_id(compound, "st2") == "parent-task.sub.st1.sub.st2"

    def test_sub_task_workflow_id(self) -> None:
        assert sub_task_workflow_id("parent-task.sub.st1") == "forge-subtask-parent-task.sub.st1"


class TestChildTimeout:
    def test_sync_mode_is_orchestration_only(self) -> None:
        # 15 + 5 * (max_depth - depth) minutes; no batch waits.
        assert child_timeout(0, 1, sync_mode=True, max_attempts=2) == timedelta(minutes=20)
        assert child_timeout(1, 3, sync_mode=True, max_attempts=2) == timedelta(minutes=25)

    def test_batch_mode_adds_wait_budget(self) -> None:
        # remaining=1, waits = max_attempts + remaining = 3; orchestration = 20 min.
        expected = 3 * BATCH_WAIT_CEILING + timedelta(minutes=20)
        assert child_timeout(0, 1, sync_mode=False, max_attempts=2) == expected


# ---------------------------------------------------------------------------
# Result builders — failure_kind wiring
# ---------------------------------------------------------------------------


class TestBuilderFailureKinds:
    def test_single_step_terminal_sets_validation_kind_and_error(self) -> None:
        result = single_step_terminal(
            task_id="t",
            output_files={"a.py": "x"},
            validation_results=[_failed("ruff_lint", "lint broke")],
            worktree_path="/wt",
            worktree_branch="forge/t",
            llm_stats=_stats(),
            context_stats=ContextStats(),
        )
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.failure_kind == "validation"
        assert result.error == "lint broke"

    def test_batch_wait_failure_owns_wording(self) -> None:
        result = sub_task_batch_wait_failure(sub_task_id="st1", exc=TimeoutError("ceiling hit"))
        assert result.failure_kind == "batch_wait"
        assert result.error == "Batch wait failed: TimeoutError: ceiling hit"

    def test_planned_failure_kind(self) -> None:
        result = planned_failure(
            task_id="t",
            failure_kind="step_failed",
            error="Step s1 failed: boom",
            output_files={},
            worktree_path="/wt",
            worktree_branch="forge/t",
            step_results=[],
            plan=Plan(
                task_id="t",
                steps=[PlanStep(step_id="s1", description="d", target_files=["a.py"])],
                explanation="e",
            ),
            planner_stats=_stats(),
            sanity_check_count=0,
        )
        assert result.failure_kind == "step_failed"

    def test_plan_preflight_failure_kind(self) -> None:
        """T5.6: the halt carries every attempt's spend and no plan (there is none)."""
        result = plan_preflight_failure(
            task_id="t",
            error="Plan rejected by preflight after 3 planner attempts: duplicate_step_ids: s1",
            worktree_path="/wt",
            worktree_branch="forge/t",
            planner_attempts=[_stats(), _stats(), _stats()],
        )
        assert result.status == TransitionSignal.FAILURE_TERMINAL
        assert result.failure_kind == "plan_preflight"
        assert result.plan is None
        # The whole halt, not just the last attempt: three calls at 100/50 each.
        assert result.planner_stats is not None
        assert result.planner_stats.input_tokens == 300
        assert result.planner_stats.output_tokens == 150
        assert result.planner_stats.latency_ms == 600.0
        assert result.llm_totals is not None
        assert result.llm_totals.call_count == 3
        # ...and run()'s one aggregation respects what this builder computed
        # rather than re-deriving call_count=1 from the single summed row.
        assert llm_totals(result) == result.llm_totals

    def test_missing_resolutions_owns_wording(self) -> None:
        missing = MissingResolutions(missing=("a.py", "b.py"))
        assert missing.message == (
            "Conflict resolution incomplete: missing resolved files: a.py, b.py"
        )

    @pytest.mark.parametrize(
        "kind",
        ["duplicate_sub_task_ids", "sub_task_failed", "conflict_incomplete", "merged_validation"],
    )
    def test_fan_out_step_failure_kinds(self, kind: FailureKind) -> None:
        result = fan_out_step_failure(step_id="s1", failure_kind=kind, error="e")
        assert result.failure_kind == kind
        assert result.status == TransitionSignal.FAILURE_TERMINAL

    def test_sub_task_terminal_summarizes_error(self) -> None:
        result = sub_task_terminal(
            sub_task_id="st1",
            validation_results=[_failed("tests", "2 failed")],
            llm_stats=_stats(),
        )
        assert result.failure_kind == "validation"
        assert result.error == "2 failed"

    def test_fan_out_success_slims_children(self) -> None:
        leaf = SubTaskResult(
            sub_task_id="st1",
            status=TransitionSignal.SUCCESS,
            output_files={"a.py": "big-content"},
            llm_stats=_stats(),
        )
        step = fan_out_success(
            step_id="s1",
            output_files={"a.py": "big-content"},
            validation_results=[_passed()],
            commit_sha="c" * 40,
            sub_task_results=[leaf],
            conflict_resolution=None,
        )
        # The step keeps its own merged output (the parent still needs it), but the
        # embedded child is slimmed to digests.
        assert step.output_files == {"a.py": "big-content"}
        assert step.sub_task_results[0].output_files == {}
        assert step.sub_task_results[0].output_digests == file_digests({"a.py": "big-content"})

    def test_nested_fan_out_failure_slims_children(self) -> None:
        child = SubTaskResult(
            sub_task_id="child",
            status=TransitionSignal.SUCCESS,
            output_files={"c.py": "cc"},
        )
        result = nested_fan_out_failure(
            sub_task_id="st1",
            failure_kind="merged_validation",
            error="Merged output validation failed: bad",
            sub_task_results=[child],
            output_files={"m.py": "merged"},
            validation_results=[_failed()],
        )
        assert result.failure_kind == "merged_validation"
        assert result.sub_task_results[0].output_files == {}


# ---------------------------------------------------------------------------
# Payload-cap acceptance test (T5.1 buried gem)
# ---------------------------------------------------------------------------


class TestPayloadCap:
    def test_fan_out_result_stays_well_under_2mb(self) -> None:
        """A multi-hundred-KB fan-out result, assembled through the builders exactly
        as the workflow does, must serialize well under Temporal's ~2MB cap — with
        contents living in exactly one place (the top-level TaskResult.output_files),
        not multiplied across per-step / per-sub-task / resolved-conflict copies.
        """
        n_sub_tasks = 24
        content = "x" * 25_000  # 25 KB per file → ~600 KB of unique content
        raw_content_bytes = n_sub_tasks * len(content)

        leaves = [
            SubTaskResult(
                sub_task_id=f"st{i}",
                status=TransitionSignal.SUCCESS,
                output_files={f"file_{i}.py": content},
                llm_stats=_stats(),
            )
            for i in range(n_sub_tasks)
        ]
        merged = {f"file_{i}.py": content for i in range(n_sub_tasks)}
        # A resolved conflict would (pre-T5.1) carry another full copy of a file.
        cr = ConflictResolutionCallResult(
            task_id="t",
            resolved_files={"file_0.py": content},
            explanation="resolved",
            model_name="opus",
            input_tokens=10,
            output_tokens=20,
            latency_ms=5.0,
        )
        fan_out_step = fan_out_success(
            step_id="s1",
            output_files=merged,
            validation_results=[_passed()],
            commit_sha="c" * 40,
            sub_task_results=leaves,
            conflict_resolution=cr,
        )
        # The planned driver folds the step's content into the single top-level map,
        # then embeds the slimmed step.
        all_output_files = dict(merged)
        step_results = [slim_result(fan_out_step)]
        result = TaskResult(
            task_id="t",
            status=TransitionSignal.SUCCESS,
            output_files=all_output_files,
            step_results=step_results,
            plan=Plan(
                task_id="t",
                steps=[PlanStep(step_id="s1", description="d", target_files=["a.py"])],
                explanation="e",
            ),
            planner_stats=_stats(),
        )

        serialized = result.model_dump_json()
        size = len(serialized.encode("utf-8"))

        # Content lives once: serialized size is close to a single copy of the
        # content, and far below both 2MB and the ~3-4x pre-slim multiplication.
        assert size < 2_000_000, f"serialized {size} bytes exceeds 2MB cap"
        assert size < raw_content_bytes * 2, "content appears to be multiplied, not slimmed"
        # The literal content string must appear exactly once per file — only in the
        # top-level output_files, never in the embedded step / sub-task / conflict copies.
        assert serialized.count(content) == n_sub_tasks
        # Every embedded child carries digests, not content.
        for child in step_results[0].sub_task_results:
            assert child.output_files == {}
            assert child.output_digests
        assert step_results[0].output_files == {}
        assert step_results[0].conflict_resolution is not None
        assert step_results[0].conflict_resolution.resolved_files == {}
