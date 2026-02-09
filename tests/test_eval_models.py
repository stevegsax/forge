"""Tests for forge.eval.models — evaluation data models."""

from __future__ import annotations

import datetime

import pytest
from pydantic import ValidationError

from forge.eval.models import (
    CheckStatus,
    DeterministicCheckResult,
    DeterministicResult,
    EvalCase,
    EvalComparison,
    EvalRunRecord,
    JudgeCriterion,
    JudgeScore,
    JudgeVerdict,
    PlanEvalResult,
)
from forge.models import Plan, PlanStep, TaskDefinition

# ---------------------------------------------------------------------------
# CheckStatus
# ---------------------------------------------------------------------------


class TestCheckStatus:
    def test_values(self) -> None:
        assert CheckStatus.PASS == "pass"
        assert CheckStatus.FAIL == "fail"
        assert CheckStatus.SKIP == "skip"


# ---------------------------------------------------------------------------
# DeterministicCheckResult
# ---------------------------------------------------------------------------


class TestDeterministicCheckResult:
    def test_pass_result(self) -> None:
        r = DeterministicCheckResult(
            check_name="check_step_ids_unique",
            status=CheckStatus.PASS,
            message="All step IDs are unique.",
        )
        assert r.check_name == "check_step_ids_unique"
        assert r.status == CheckStatus.PASS
        assert r.details == []

    def test_fail_result_with_details(self) -> None:
        r = DeterministicCheckResult(
            check_name="check_target_files_are_relative_paths",
            status=CheckStatus.FAIL,
            message="Found absolute paths.",
            details=["/etc/passwd", "../secret"],
        )
        assert r.status == CheckStatus.FAIL
        assert len(r.details) == 2


# ---------------------------------------------------------------------------
# DeterministicResult
# ---------------------------------------------------------------------------


class TestDeterministicResult:
    def test_default_all_passed(self) -> None:
        r = DeterministicResult()
        assert r.all_passed is True
        assert r.checks == []

    def test_with_failing_check(self) -> None:
        check = DeterministicCheckResult(
            check_name="x", status=CheckStatus.FAIL, message="bad"
        )
        r = DeterministicResult(checks=[check], all_passed=False)
        assert r.all_passed is False


# ---------------------------------------------------------------------------
# JudgeCriterion
# ---------------------------------------------------------------------------


class TestJudgeCriterion:
    def test_all_criteria(self) -> None:
        expected = {
            "completeness",
            "granularity",
            "ordering",
            "context_quality",
            "fan_out_appropriateness",
            "explanation_quality",
        }
        assert {c.value for c in JudgeCriterion} == expected


# ---------------------------------------------------------------------------
# JudgeScore
# ---------------------------------------------------------------------------


class TestJudgeScore:
    def test_valid_score(self) -> None:
        s = JudgeScore(
            criterion=JudgeCriterion.COMPLETENESS,
            score=4,
            rationale="Covers all targets.",
        )
        assert s.score == 4

    def test_score_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            JudgeScore(
                criterion=JudgeCriterion.COMPLETENESS,
                score=6,
                rationale="Too high.",
            )

    def test_score_below_minimum(self) -> None:
        with pytest.raises(ValidationError):
            JudgeScore(
                criterion=JudgeCriterion.COMPLETENESS,
                score=0,
                rationale="Too low.",
            )


# ---------------------------------------------------------------------------
# JudgeVerdict
# ---------------------------------------------------------------------------


class TestJudgeVerdict:
    def test_verdict(self) -> None:
        scores = [
            JudgeScore(criterion=JudgeCriterion.COMPLETENESS, score=5, rationale="Good."),
            JudgeScore(criterion=JudgeCriterion.GRANULARITY, score=3, rationale="OK."),
        ]
        v = JudgeVerdict(scores=scores, overall_assessment="Solid plan.")
        assert len(v.scores) == 2
        assert v.overall_assessment == "Solid plan."


# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------


class TestEvalCase:
    def test_minimal(self) -> None:
        task = TaskDefinition(task_id="t1", description="Do something.")
        case = EvalCase(case_id="case-1", task=task, repo_root="/tmp/repo")
        assert case.reference_plan is None
        assert case.tags == []

    def test_with_reference_plan(self) -> None:
        task = TaskDefinition(task_id="t1", description="Do something.")
        plan = Plan(
            task_id="t1",
            steps=[PlanStep(step_id="s1", description="Step 1", target_files=["a.py"])],
            explanation="Simple.",
        )
        case = EvalCase(
            case_id="case-1",
            task=task,
            repo_root="/tmp/repo",
            reference_plan=plan,
            tags=["simple"],
        )
        assert case.reference_plan is not None
        assert case.tags == ["simple"]


# ---------------------------------------------------------------------------
# PlanEvalResult
# ---------------------------------------------------------------------------


class TestPlanEvalResult:
    def test_deterministic_only(self) -> None:
        plan = Plan(
            task_id="t1",
            steps=[PlanStep(step_id="s1", description="Step 1", target_files=["a.py"])],
            explanation="Simple.",
        )
        det = DeterministicResult(checks=[], all_passed=True)
        result = PlanEvalResult(case_id="case-1", plan=plan, deterministic=det)
        assert result.judge is None
        assert isinstance(result.timestamp, datetime.datetime)

    def test_with_judge(self) -> None:
        plan = Plan(
            task_id="t1",
            steps=[PlanStep(step_id="s1", description="Step 1", target_files=["a.py"])],
            explanation="Simple.",
        )
        det = DeterministicResult(checks=[], all_passed=True)
        verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion=JudgeCriterion.COMPLETENESS, score=5, rationale="Good.")
            ],
            overall_assessment="Fine.",
        )
        result = PlanEvalResult(
            case_id="case-1", plan=plan, deterministic=det, judge=verdict
        )
        assert result.judge is not None


# ---------------------------------------------------------------------------
# EvalRunRecord
# ---------------------------------------------------------------------------


class TestEvalRunRecord:
    def test_minimal(self) -> None:
        record = EvalRunRecord(run_id="run-1", model_name="test-model")
        assert record.judge_model is None
        assert record.results == []
        assert isinstance(record.timestamp, datetime.datetime)


# ---------------------------------------------------------------------------
# EvalComparison
# ---------------------------------------------------------------------------


class TestEvalComparison:
    def test_comparison(self) -> None:
        comp = EvalComparison(
            baseline_run_id="run-1",
            candidate_run_id="run-2",
            regressions=["case-1"],
            improvements=["case-2"],
            summary="Mixed results.",
        )
        assert len(comp.regressions) == 1
        assert len(comp.improvements) == 1
