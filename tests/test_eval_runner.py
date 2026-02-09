"""Tests for forge.eval.runner — orchestration, save/load, comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.eval.models import (
    CheckStatus,
    DeterministicCheckResult,
    DeterministicResult,
    EvalCase,
    EvalRunRecord,
    JudgeCriterion,
    JudgeScore,
    JudgeVerdict,
    PlanEvalResult,
)
from forge.eval.runner import (
    build_eval_result,
    compare_runs,
    evaluate_plan,
    load_run,
    save_run,
)
from forge.models import Plan, PlanStep, TaskDefinition

_TASK = TaskDefinition(task_id="t1", description="Test task.", target_files=["a.py"])
_CASE = EvalCase(case_id="case-1", task=_TASK, repo_root="/tmp/repo")
_PLAN = Plan(
    task_id="t1",
    steps=[PlanStep(step_id="s1", description="Do it.", target_files=["a.py"])],
    explanation="Simple.",
)


# ---------------------------------------------------------------------------
# build_eval_result
# ---------------------------------------------------------------------------


class TestBuildEvalResult:
    def test_without_judge(self) -> None:
        det = DeterministicResult(checks=[], all_passed=True)
        result = build_eval_result("case-1", _PLAN, det)
        assert result.case_id == "case-1"
        assert result.judge is None

    def test_with_judge(self) -> None:
        det = DeterministicResult(checks=[], all_passed=True)
        verdict = JudgeVerdict(scores=[], overall_assessment="Fine.")
        result = build_eval_result("case-1", _PLAN, det, verdict)
        assert result.judge is not None


# ---------------------------------------------------------------------------
# evaluate_plan
# ---------------------------------------------------------------------------


class TestEvaluatePlan:
    @pytest.mark.asyncio
    async def test_deterministic_only(self) -> None:
        result = await evaluate_plan(_CASE, _PLAN, known_repo_files=set())
        assert result.case_id == "case-1"
        assert result.deterministic.all_passed is True
        assert result.judge is None

    @pytest.mark.asyncio
    async def test_with_known_repo_files(self) -> None:
        known = {"a.py", "existing.py"}
        plan = Plan(
            task_id="t1",
            steps=[
                PlanStep(
                    step_id="s1",
                    description="Do it.",
                    target_files=["a.py"],
                    context_files=["existing.py"],
                )
            ],
            explanation="Simple.",
        )
        result = await evaluate_plan(_CASE, plan, known_repo_files=known)
        assert result.deterministic.all_passed is True


# ---------------------------------------------------------------------------
# save_run / load_run
# ---------------------------------------------------------------------------


class TestSaveLoadRun:
    def test_round_trip(self, tmp_path: Path) -> None:
        det = DeterministicResult(
            checks=[
                DeterministicCheckResult(
                    check_name="test_check",
                    status=CheckStatus.PASS,
                    message="OK.",
                )
            ],
            all_passed=True,
        )
        eval_result = PlanEvalResult(
            case_id="case-1",
            plan=_PLAN,
            deterministic=det,
        )
        record = EvalRunRecord(
            run_id="run-1",
            model_name="test-model",
            results=[eval_result],
        )

        path = save_run(record, output_dir=tmp_path)
        assert path.exists()

        loaded = load_run(path)
        assert loaded.run_id == "run-1"
        assert len(loaded.results) == 1
        assert loaded.results[0].case_id == "case-1"

    def test_load_nonexistent(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_run(Path("/nonexistent/run.json"))

    def test_load_invalid(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        with pytest.raises(ValueError, match="Invalid eval run"):
            load_run(bad)

    def test_default_dir_uses_xdg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        record = EvalRunRecord(run_id="xdg-test", model_name="test")
        path = save_run(record)
        assert tmp_path in path.parents or path.parent == tmp_path / "forge" / "eval"
        assert path.exists()


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def _make_record(
        self,
        run_id: str,
        case_results: list[tuple[str, bool, float | None]],
    ) -> EvalRunRecord:
        """Helper: create a run record from (case_id, all_passed, avg_score) tuples."""
        results = []
        for case_id, all_passed, avg_score in case_results:
            det = DeterministicResult(checks=[], all_passed=all_passed)
            judge = None
            if avg_score is not None:
                judge = JudgeVerdict(
                    scores=[
                        JudgeScore(
                            criterion=JudgeCriterion.COMPLETENESS,
                            score=int(avg_score),
                            rationale="Test.",
                        )
                    ],
                    overall_assessment="Test.",
                )
            results.append(PlanEvalResult(
                case_id=case_id,
                plan=_PLAN,
                deterministic=det,
                judge=judge,
            ))
        return EvalRunRecord(
            run_id=run_id, model_name="test", results=results
        )

    def test_no_changes(self) -> None:
        baseline = self._make_record("r1", [("c1", True, None)])
        candidate = self._make_record("r2", [("c1", True, None)])
        comp = compare_runs(baseline, candidate)
        assert comp.regressions == []
        assert comp.improvements == []

    def test_deterministic_regression(self) -> None:
        baseline = self._make_record("r1", [("c1", True, None)])
        candidate = self._make_record("r2", [("c1", False, None)])
        comp = compare_runs(baseline, candidate)
        assert "c1" in comp.regressions

    def test_deterministic_improvement(self) -> None:
        baseline = self._make_record("r1", [("c1", False, None)])
        candidate = self._make_record("r2", [("c1", True, None)])
        comp = compare_runs(baseline, candidate)
        assert "c1" in comp.improvements

    def test_judge_regression(self) -> None:
        baseline = self._make_record("r1", [("c1", True, 5.0)])
        candidate = self._make_record("r2", [("c1", True, 2.0)])
        comp = compare_runs(baseline, candidate)
        assert "c1" in comp.regressions

    def test_judge_improvement(self) -> None:
        baseline = self._make_record("r1", [("c1", True, 2.0)])
        candidate = self._make_record("r2", [("c1", True, 5.0)])
        comp = compare_runs(baseline, candidate)
        assert "c1" in comp.improvements

    def test_summary_mentions_missing_cases(self) -> None:
        baseline = self._make_record("r1", [("c1", True, None), ("only-base", True, None)])
        candidate = self._make_record("r2", [("c1", True, None), ("only-cand", True, None)])
        comp = compare_runs(baseline, candidate)
        assert "only-base" in comp.summary
        assert "only-cand" in comp.summary
