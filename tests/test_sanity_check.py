"""Tests for forge.activities.sanity_check — pure functions and testable function."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.activities.sanity_check import (
    build_sanity_check_system_prompt,
    build_sanity_check_user_prompt,
    build_step_digest,
    execute_sanity_check_call,
)
from forge.models import (
    Plan,
    PlanStep,
    SanityCheckCallResult,
    SanityCheckInput,
    SanityCheckResponse,
    SanityCheckVerdict,
    StepResult,
    TaskDefinition,
    TransitionSignal,
)
from tests.conftest import build_mock_message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TASK = TaskDefinition(
    task_id="test-task",
    description="Build a REST API.",
    target_files=["api.py"],
)

_PLAN = Plan(
    task_id="test-task",
    steps=[
        PlanStep(step_id="step-1", description="Create models.", target_files=["models.py"]),
        PlanStep(step_id="step-2", description="Create API.", target_files=["api.py"]),
        PlanStep(step_id="step-3", description="Add tests.", target_files=["test_api.py"]),
    ],
    explanation="Three-step plan.",
)


def _make_step_result(
    step_id: str,
    status: TransitionSignal = TransitionSignal.SUCCESS,
    output_files: dict[str, str] | None = None,
    error: str | None = None,
) -> StepResult:
    return StepResult(
        step_id=step_id,
        status=status,
        output_files=output_files or {},
        error=error,
    )


# ---------------------------------------------------------------------------
# TestBuildStepDigest
# ---------------------------------------------------------------------------


class TestBuildStepDigest:
    def test_success_digest(self) -> None:
        sr = _make_step_result("step-1", output_files={"a.py": "x", "b.py": "y"})
        digest = build_step_digest(sr)
        assert "step-1" in digest
        assert "success" in digest
        assert "2 files" in digest

    def test_failure_digest_includes_error(self) -> None:
        sr = _make_step_result(
            "step-2",
            status=TransitionSignal.FAILURE_TERMINAL,
            error="lint failed",
        )
        digest = build_step_digest(sr)
        assert "step-2" in digest
        assert "failure_terminal" in digest
        assert "lint failed" in digest

    def test_zero_files(self) -> None:
        sr = _make_step_result("step-1")
        digest = build_step_digest(sr)
        assert "0 files" in digest


# ---------------------------------------------------------------------------
# TestBuildSanityCheckSystemPrompt
# ---------------------------------------------------------------------------


class TestBuildSanityCheckSystemPrompt:
    def test_contains_task_description(self) -> None:
        prompt = build_sanity_check_system_prompt(
            _TASK.task_id, _TASK.description, _PLAN, [], _PLAN.steps, project_instructions=""
        )
        assert "Build a REST API" in prompt

    def test_contains_completed_digests(self) -> None:
        completed = [
            _make_step_result("step-1", output_files={"models.py": "class M: pass"}),
        ]
        prompt = build_sanity_check_system_prompt(
            _TASK.task_id, _TASK.description, _PLAN, completed, _PLAN.steps[1:],
            project_instructions="",
        )
        assert "step-1" in prompt
        assert "1 files" in prompt

    def test_contains_remaining_steps(self) -> None:
        prompt = build_sanity_check_system_prompt(
            _TASK.task_id, _TASK.description, _PLAN, [], _PLAN.steps, project_instructions=""
        )
        assert "step-2" in prompt
        assert "step-3" in prompt
        assert "Create API" in prompt
        assert "Add tests" in prompt

    def test_contains_project_instructions(self) -> None:
        prompt = build_sanity_check_system_prompt(
            _TASK.task_id, _TASK.description, _PLAN, [], _PLAN.steps,
            project_instructions="## Project\nUse ruff.",
        )
        assert "Use ruff" in prompt

    def test_contains_verdict_instructions(self) -> None:
        prompt = build_sanity_check_system_prompt(
            _TASK.task_id, _TASK.description, _PLAN, [], _PLAN.steps, project_instructions=""
        )
        assert "continue" in prompt
        assert "revise" in prompt
        assert "abort" in prompt


# ---------------------------------------------------------------------------
# TestBuildSanityCheckUserPrompt
# ---------------------------------------------------------------------------


class TestBuildSanityCheckUserPrompt:
    def test_counts_in_prompt(self) -> None:
        prompt = build_sanity_check_user_prompt(2, 5)
        assert "2 of 5" in prompt

    def test_zero_completed(self) -> None:
        prompt = build_sanity_check_user_prompt(0, 3)
        assert "0 of 3" in prompt


# ---------------------------------------------------------------------------
# TestExecuteSanityCheckCall
# ---------------------------------------------------------------------------


class TestExecuteSanityCheckCall:
    @pytest.mark.asyncio
    async def test_returns_result_with_correct_fields(self) -> None:
        mock_response = SanityCheckResponse(
            verdict=SanityCheckVerdict.CONTINUE,
            explanation="Plan looks good.",
        )
        mock_message = build_mock_message(
            tool_name="sanity_check_response",
            tool_input=mock_response.model_dump(),
            input_tokens=100,
            output_tokens=50,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        input_data = SanityCheckInput(
            task_id="test-task",
            system_prompt="system",
            user_prompt="user",
        )

        result = await execute_sanity_check_call(input_data, mock_client)

        assert isinstance(result, SanityCheckCallResult)
        assert result.task_id == "test-task"
        assert result.response.verdict == SanityCheckVerdict.CONTINUE
        assert result.response.explanation == "Plan looks good."
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_revise_verdict_with_steps(self) -> None:
        revised = [
            PlanStep(step_id="new-1", description="Revised step.", target_files=["x.py"]),
        ]
        mock_response = SanityCheckResponse(
            verdict=SanityCheckVerdict.REVISE,
            explanation="Need to adjust.",
            revised_steps=revised,
        )
        mock_message = build_mock_message(
            tool_name="sanity_check_response",
            tool_input=mock_response.model_dump(),
            input_tokens=200,
            output_tokens=100,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        input_data = SanityCheckInput(
            task_id="test-task",
            system_prompt="system",
            user_prompt="user",
        )

        result = await execute_sanity_check_call(input_data, mock_client)

        assert result.response.verdict == SanityCheckVerdict.REVISE
        assert result.response.revised_steps is not None
        assert len(result.response.revised_steps) == 1
        assert result.response.revised_steps[0].step_id == "new-1"
