"""Tests for forge.activities.sanity_check — pure functions and testable function."""

from __future__ import annotations

import pytest
from sax_platform.llm import LLMRefused, LLMTruncated, Telemetry

from forge.activities.sanity_check import (
    DEFAULT_SANITY_CHECK_MAX_TOKENS,
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
from tests.conftest import build_mock_llm

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
            _TASK.task_id,
            _TASK.description,
            _PLAN,
            completed,
            _PLAN.steps[1:],
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
            _TASK.task_id,
            _TASK.description,
            _PLAN,
            [],
            _PLAN.steps,
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


def _telemetry(stop_reason: str) -> Telemetry:
    """Minimal Telemetry for constructing typed LLM failures in tests."""
    return Telemetry(
        model="test-model",
        stop_reason=stop_reason,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        request_id=None,
    )


class TestExecuteSanityCheckCall:
    @pytest.mark.asyncio
    async def test_returns_result_with_correct_fields(self) -> None:
        mock_response = SanityCheckResponse(
            verdict=SanityCheckVerdict.CONTINUE,
            explanation="Plan looks good.",
        )
        llm = build_mock_llm(
            output=mock_response,
            model="claude-opus-4-8",
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=7,
            cache_read_input_tokens=3,
        )

        input_data = SanityCheckInput(
            task_id="test-task",
            system_prompt="system",
            user_prompt="user",
        )

        result = await execute_sanity_check_call(input_data, llm)

        assert isinstance(result, SanityCheckCallResult)
        assert result.task_id == "test-task"
        assert result.response.verdict == SanityCheckVerdict.CONTINUE
        assert result.response.explanation == "Plan looks good."
        assert result.model_name == "claude-opus-4-8"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cache_creation_input_tokens == 7
        assert result.cache_read_input_tokens == 3
        assert result.stop_reason == "end_turn"
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_complete_called_with_expected_kwargs(self) -> None:
        mock_response = SanityCheckResponse(
            verdict=SanityCheckVerdict.CONTINUE,
            explanation="ok",
        )
        llm = build_mock_llm(output=mock_response)
        input_data = SanityCheckInput(
            task_id="test-task",
            system_prompt="system",
            user_prompt="user",
            model_name="anthropic:claude-sonnet-5",
        )

        await execute_sanity_check_call(input_data, llm)

        llm.complete.assert_awaited_once()
        call = llm.complete.await_args
        assert call.args[0] == [{"role": "user", "content": "user"}]
        assert call.kwargs["output_type"] is SanityCheckResponse
        # split_provider strips the provider prefix before the model reaches complete.
        assert call.kwargs["model"] == "claude-sonnet-5"
        assert call.kwargs["max_tokens"] == DEFAULT_SANITY_CHECK_MAX_TOKENS
        assert call.kwargs["system"] == "system"
        assert call.kwargs["thinking"] == input_data.thinking

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
        llm = build_mock_llm(output=mock_response, input_tokens=200, output_tokens=100)

        input_data = SanityCheckInput(
            task_id="test-task",
            system_prompt="system",
            user_prompt="user",
        )

        result = await execute_sanity_check_call(input_data, llm)

        assert result.response.verdict == SanityCheckVerdict.REVISE
        assert result.response.revised_steps is not None
        assert len(result.response.revised_steps) == 1
        assert result.response.revised_steps[0].step_id == "new-1"

    @pytest.mark.asyncio
    async def test_uses_thinking_enabled_max_tokens(self) -> None:
        """Sanity-check is thinking-enabled (D94): adaptive thinking now competes
        for tokens inside max_tokens, so the cap must be the explicit
        owner-adjudicated 16384, not the old 4096 default."""
        mock_response = SanityCheckResponse(
            verdict=SanityCheckVerdict.CONTINUE,
            explanation="Plan looks good.",
        )
        llm = build_mock_llm(output=mock_response)
        input_data = SanityCheckInput(
            task_id="test-task",
            system_prompt="system",
            user_prompt="user",
        )

        await execute_sanity_check_call(input_data, llm)

        assert DEFAULT_SANITY_CHECK_MAX_TOKENS == 16384
        assert llm.complete.await_args.kwargs["max_tokens"] == DEFAULT_SANITY_CHECK_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_refusal_propagates(self) -> None:
        llm = build_mock_llm(error=LLMRefused(category="policy", telemetry=_telemetry("refusal")))
        input_data = SanityCheckInput(task_id="t", system_prompt="s", user_prompt="u")

        with pytest.raises(LLMRefused):
            await execute_sanity_check_call(input_data, llm)

    @pytest.mark.asyncio
    async def test_truncation_propagates(self) -> None:
        llm = build_mock_llm(
            error=LLMTruncated(
                partial_text="partial", max_tokens=16384, telemetry=_telemetry("max_tokens")
            )
        )
        input_data = SanityCheckInput(task_id="t", system_prompt="s", user_prompt="u")

        with pytest.raises(LLMTruncated):
            await execute_sanity_check_call(input_data, llm)


# ---------------------------------------------------------------------------
# call_sanity_check shell
# ---------------------------------------------------------------------------


class TestCallSanityCheck:
    @pytest.mark.asyncio
    async def test_delegates_to_client_and_returns_result(self) -> None:
        from unittest.mock import MagicMock, patch

        from forge.activities.roots import LlmActivities

        mock_response = SanityCheckResponse(verdict=SanityCheckVerdict.CONTINUE, explanation="ok")
        llm = build_mock_llm(output=mock_response, model="claude-opus-4-8")

        with patch("forge.activities.roots.get_tracer") as mock_get_tracer:
            mock_span = MagicMock()
            mock_span.__enter__ = MagicMock(return_value=mock_span)
            mock_span.__exit__ = MagicMock(return_value=False)
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer

            sanity_input = SanityCheckInput(task_id="t-ok", system_prompt="s", user_prompt="u")
            result = await LlmActivities(llm).call_sanity_check(sanity_input)

        assert result.task_id == "t-ok"
        assert result.response.verdict == SanityCheckVerdict.CONTINUE
        llm.complete.assert_awaited_once()
