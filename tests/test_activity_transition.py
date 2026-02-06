"""Tests for forge.activities.transition — transition evaluation."""

from __future__ import annotations

import pytest

from forge.activities.transition import determine_transition, evaluate_transition
from forge.models import TransitionInput, TransitionSignal, ValidationResult

# ---------------------------------------------------------------------------
# determine_transition (pure function)
# ---------------------------------------------------------------------------


class TestDetermineTransition:
    def test_all_passed_returns_success(self) -> None:
        results = [
            ValidationResult(check_name="lint", passed=True, summary="ok"),
            ValidationResult(check_name="format", passed=True, summary="ok"),
        ]
        assert determine_transition(results, attempt=1) == TransitionSignal.SUCCESS

    def test_empty_results_returns_success(self) -> None:
        assert determine_transition([], attempt=1) == TransitionSignal.SUCCESS

    def test_first_attempt_failure_is_retryable(self) -> None:
        results = [
            ValidationResult(check_name="lint", passed=False, summary="errors"),
        ]
        assert determine_transition(results, attempt=1) == TransitionSignal.FAILURE_RETRYABLE

    def test_max_attempt_failure_is_terminal(self) -> None:
        results = [
            ValidationResult(check_name="lint", passed=False, summary="errors"),
        ]
        assert determine_transition(results, attempt=2) == TransitionSignal.FAILURE_TERMINAL

    def test_custom_max_attempts(self) -> None:
        results = [
            ValidationResult(check_name="lint", passed=False, summary="errors"),
        ]
        # attempt=2, max_attempts=3 → still retryable
        assert (
            determine_transition(results, attempt=2, max_attempts=3)
            == TransitionSignal.FAILURE_RETRYABLE
        )
        # attempt=3, max_attempts=3 → terminal
        assert (
            determine_transition(results, attempt=3, max_attempts=3)
            == TransitionSignal.FAILURE_TERMINAL
        )

    def test_mixed_results_with_one_failure(self) -> None:
        results = [
            ValidationResult(check_name="lint", passed=True, summary="ok"),
            ValidationResult(check_name="format", passed=False, summary="bad"),
        ]
        assert determine_transition(results, attempt=1) == TransitionSignal.FAILURE_RETRYABLE


# ---------------------------------------------------------------------------
# evaluate_transition (activity wrapper)
# ---------------------------------------------------------------------------


class TestEvaluateTransition:
    @pytest.mark.asyncio
    async def test_delegates_to_pure_function(self) -> None:
        input_data = TransitionInput(
            validation_results=[
                ValidationResult(check_name="lint", passed=True, summary="ok"),
            ],
            attempt=1,
        )
        result = await evaluate_transition(input_data)
        assert result == TransitionSignal.SUCCESS.value

    @pytest.mark.asyncio
    async def test_returns_string_value(self) -> None:
        input_data = TransitionInput(
            validation_results=[
                ValidationResult(check_name="lint", passed=False, summary="errors"),
            ],
            attempt=2,
        )
        result = await evaluate_transition(input_data)
        assert result == "failure_terminal"
        assert isinstance(result, str)
