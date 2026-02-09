"""Orchestration: run eval cases through deterministic + judge layers.

Provides functions to evaluate plans, save/load run records, and compare runs.
Storage follows XDG: ``$XDG_DATA_HOME/forge/eval/`` (defaults to
``~/.local/share/forge/eval/``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from forge.eval.deterministic import run_deterministic_checks
from forge.eval.judge import judge_plan
from forge.eval.models import (
    EvalComparison,
    EvalRunRecord,
    PlanEvalResult,
)

if TYPE_CHECKING:
    from forge.eval.models import DeterministicResult, EvalCase, JudgeVerdict
    from forge.models import Plan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _default_data_dir() -> Path:
    """Return the XDG data directory for eval results."""
    xdg = os.environ.get("XDG_DATA_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "forge" / "eval"


def build_eval_result(
    case_id: str,
    plan: Plan,
    deterministic: DeterministicResult,
    judge: JudgeVerdict | None = None,
) -> PlanEvalResult:
    """Build a PlanEvalResult from its components."""
    return PlanEvalResult(
        case_id=case_id,
        plan=plan,
        deterministic=deterministic,
        judge=judge,
    )


def compare_runs(baseline: EvalRunRecord, candidate: EvalRunRecord) -> EvalComparison:
    """Compare two evaluation runs and identify regressions/improvements.

    Uses deterministic pass/fail as the primary comparison signal.
    When judge scores are available, compares average scores.
    """
    baseline_map = {r.case_id: r for r in baseline.results}
    candidate_map = {r.case_id: r for r in candidate.results}

    common_ids = set(baseline_map) & set(candidate_map)
    regressions: list[str] = []
    improvements: list[str] = []

    for case_id in sorted(common_ids):
        b = baseline_map[case_id]
        c = candidate_map[case_id]

        # Compare deterministic results first
        if b.deterministic.all_passed and not c.deterministic.all_passed:
            regressions.append(case_id)
            continue
        if not b.deterministic.all_passed and c.deterministic.all_passed:
            improvements.append(case_id)
            continue

        # If both have judge verdicts, compare average scores
        if b.judge and c.judge:
            b_avg = sum(s.score for s in b.judge.scores) / max(len(b.judge.scores), 1)
            c_avg = sum(s.score for s in c.judge.scores) / max(len(c.judge.scores), 1)
            if c_avg < b_avg - 0.5:
                regressions.append(case_id)
            elif c_avg > b_avg + 0.5:
                improvements.append(case_id)

    summary_parts = [
        f"Compared {len(common_ids)} common case(s).",
        f"Regressions: {len(regressions)}.",
        f"Improvements: {len(improvements)}.",
    ]
    only_baseline = set(baseline_map) - set(candidate_map)
    only_candidate = set(candidate_map) - set(baseline_map)
    if only_baseline:
        summary_parts.append(f"Only in baseline: {', '.join(sorted(only_baseline))}.")
    if only_candidate:
        summary_parts.append(f"Only in candidate: {', '.join(sorted(only_candidate))}.")

    return EvalComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        regressions=regressions,
        improvements=improvements,
        summary=" ".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


async def evaluate_plan(
    case: EvalCase,
    plan: Plan,
    known_repo_files: set[str] | None = None,
    *,
    run_judge: bool = False,
    judge_model: str | None = None,
) -> PlanEvalResult:
    """Evaluate a single plan against a case.

    Runs deterministic checks always. Optionally runs the LLM judge.
    """
    deterministic = run_deterministic_checks(plan, case.task, known_repo_files)

    verdict: JudgeVerdict | None = None
    if run_judge:
        verdict = await judge_plan(case, plan, model_name=judge_model)

    return build_eval_result(case.case_id, plan, deterministic, verdict)


async def evaluate_corpus(
    cases: list[EvalCase],
    plans: dict[str, Plan],
    *,
    known_repo_files: dict[str, set[str]] | None = None,
    run_judge: bool = False,
    judge_model: str | None = None,
) -> list[PlanEvalResult]:
    """Evaluate a corpus of cases against their plans.

    Args:
        cases: List of eval cases.
        plans: Mapping of case_id -> Plan.
        known_repo_files: Optional mapping of case_id -> set of known files.
        run_judge: Whether to run the LLM judge.
        judge_model: Override judge model.

    Returns:
        List of PlanEvalResult for cases that have matching plans.
    """
    results: list[PlanEvalResult] = []
    for case in cases:
        plan = plans.get(case.case_id)
        if plan is None:
            logger.warning("No plan for case %s, skipping.", case.case_id)
            continue
        repo_files = (known_repo_files or {}).get(case.case_id)
        result = await evaluate_plan(
            case, plan, repo_files, run_judge=run_judge, judge_model=judge_model
        )
        results.append(result)
    return results


def save_run(record: EvalRunRecord, output_dir: Path | None = None) -> Path:
    """Save an EvalRunRecord to a JSON file.

    Returns the path to the saved file.
    """
    if output_dir is None:
        output_dir = _default_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{record.run_id}.json"
    path = output_dir / filename
    path.write_text(record.model_dump_json(indent=2))
    logger.info("Saved eval run to %s", path)
    return path


def load_run(path: Path) -> EvalRunRecord:
    """Load an EvalRunRecord from a JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is invalid.
    """
    content = path.read_text()
    try:
        return EvalRunRecord.model_validate_json(content)
    except Exception as e:
        msg = f"Invalid eval run record in {path}: {e}"
        raise ValueError(msg) from e
