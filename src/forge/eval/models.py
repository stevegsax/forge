"""Data models for the planner evaluation framework."""

from __future__ import annotations

import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from forge.models import Plan, TaskDefinition  # noqa: TC001


class CheckStatus(StrEnum):
    """Outcome of a single deterministic check."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class DeterministicCheckResult(BaseModel):
    """Result of a single deterministic check."""

    check_name: str
    status: CheckStatus
    message: str = Field(description="Human-readable summary.")
    details: list[str] = Field(
        default_factory=list,
        description="Specific items that caused failure.",
    )


class DeterministicResult(BaseModel):
    """Aggregated result of all deterministic checks on a plan."""

    checks: list[DeterministicCheckResult] = Field(default_factory=list)
    all_passed: bool = True


class JudgeCriterion(StrEnum):
    """Criteria the LLM judge scores on."""

    COMPLETENESS = "completeness"
    GRANULARITY = "granularity"
    ORDERING = "ordering"
    CONTEXT_QUALITY = "context_quality"
    FAN_OUT_APPROPRIATENESS = "fan_out_appropriateness"
    EXPLANATION_QUALITY = "explanation_quality"


class JudgeScore(BaseModel):
    """Score for a single judge criterion."""

    criterion: JudgeCriterion
    score: int = Field(ge=1, le=5, description="1-5 scale.")
    rationale: str = Field(description="Why this score was given.")


class JudgeVerdict(BaseModel):
    """Full verdict from the LLM judge."""

    scores: list[JudgeScore] = Field(description="One score per criterion.")
    overall_assessment: str = Field(description="Summary assessment of plan quality.")


class EvalCase(BaseModel):
    """A single evaluation case: a task + optional reference plan."""

    case_id: str
    task: TaskDefinition
    repo_root: str = Field(description="Path to the repo for this case.")
    reference_plan: Plan | None = Field(
        default=None,
        description="Optional known-good plan for comparison.",
    )
    tags: list[str] = Field(default_factory=list, description="Tags for filtering.")


class PlanEvalResult(BaseModel):
    """Evaluation result for a single plan against a case."""

    case_id: str
    plan: Plan
    deterministic: DeterministicResult
    judge: JudgeVerdict | None = None
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
    )


class EvalRunRecord(BaseModel):
    """Record of a full evaluation run across multiple cases."""

    run_id: str
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
    )
    model_name: str = Field(description="Model that produced the plans.")
    judge_model: str | None = Field(
        default=None,
        description="Model used as judge, if any.",
    )
    results: list[PlanEvalResult] = Field(default_factory=list)


class EvalComparison(BaseModel):
    """Comparison between two evaluation runs."""

    baseline_run_id: str
    candidate_run_id: str
    regressions: list[str] = Field(
        default_factory=list,
        description="Case IDs where candidate scored worse.",
    )
    improvements: list[str] = Field(
        default_factory=list,
        description="Case IDs where candidate scored better.",
    )
    summary: str = Field(description="Human-readable comparison summary.")
