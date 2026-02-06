"""Transition evaluation activity for Forge.

Determines the next workflow state based on validation results.
Fully deterministic — no I/O.
"""

from __future__ import annotations

from temporalio import activity

from forge.models import TransitionInput, TransitionSignal, ValidationResult

# ---------------------------------------------------------------------------
# Pure function
# ---------------------------------------------------------------------------


def determine_transition(
    results: list[ValidationResult],
    attempt: int,
    max_attempts: int = 2,
) -> TransitionSignal:
    """Decide the workflow transition based on validation outcomes.

    - All passed (or empty) → SUCCESS
    - Any failed + attempt < max_attempts → FAILURE_RETRYABLE
    - Any failed + attempt >= max_attempts → FAILURE_TERMINAL
    """
    all_passed = all(r.passed for r in results)
    if all_passed:
        return TransitionSignal.SUCCESS
    if attempt < max_attempts:
        return TransitionSignal.FAILURE_RETRYABLE
    return TransitionSignal.FAILURE_TERMINAL


# ---------------------------------------------------------------------------
# Temporal activity
# ---------------------------------------------------------------------------


@activity.defn
async def evaluate_transition(input: TransitionInput) -> str:
    """Activity wrapper — delegates to the pure function and returns signal value."""
    signal = determine_transition(
        input.validation_results,
        input.attempt,
        input.max_attempts,
    )
    return signal.value
