"""Shared batch poll loop for the D88 timer-loop transport (T4.2, ST1).

The waiting workflow is the recipient of its own batch result: it submits, then
polls the provider's normalized status on a timer until the batch ends or a 25h
ceiling passes. That loop skeleton — deadline, sleep, status-dispatch — is
identical for every consumer (forge's Anthropic waiters, ocr's Mistral waiters),
so it lives here once. What differs per consumer is (a) *how* the status is
fetched (an injected ``status_fn`` that wraps the consumer's own status
activity) and (b) *how long* to wait between polls (an injected :class:`PollSchedule`).

The loop owns no persistence and raises no exceptions for outcomes: it returns a
plain state string and the caller decides what ledger row to write and whether to
raise. That keeps the outcome-to-side-effect mapping (which is consumer-specific)
out of the shared skeleton.

Sandbox / determinism
----------------------
This module imports only ``temporalio.workflow`` at framework level (plus
stdlib), so it is safe to import inside the Temporal workflow sandbox via
``workflow.unsafe.imports_passed_through()``. Everything it touches inside the
loop is replay-safe: ``workflow.now`` (deterministic clock), ``workflow.sleep``
(a timer command), and ``workflow.random`` (deterministic PRNG seeded from
history — no command emitted). The schedules are pure frozen dataclasses whose
jitter is drawn from that caller-supplied ``random.Random``; with
``jitter_fraction == 0`` the sleep sequence is exactly deterministic.
"""

import math
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from temporalio import workflow

__all__ = [
    "BATCH_WAIT_CEILING",
    "BackoffSchedule",
    "FixedInterval",
    "PollSchedule",
    "wait_batch_ended",
]

# The per-wait batch ceiling: a single batch wait may legally block a workflow up
# to this long before the timer loop gives up. One source of truth for the 25h
# number — forge re-exports it (``forge.models`` for the CLI/timeout math,
# ``forge.workflow_blocks`` for the loop) rather than duplicating the literal.
BATCH_WAIT_CEILING: Final = timedelta(hours=25)


class PollSchedule(Protocol):
    """A per-poll sleep policy: how long to wait before the next status poll.

    Pure and replay-safe. ``next_sleep`` is a total function of the poll index,
    the remaining wait budget, and a caller-supplied deterministic PRNG (used
    only for jitter); it returns the sleep duration, already clamped so it never
    overshoots ``remaining``.
    """

    def next_sleep(self, *, attempt: int, remaining: timedelta, rng: random.Random) -> timedelta:
        """Sleep before poll ``attempt`` (0-based), clamped to ``remaining``."""
        ...


@dataclass(frozen=True, slots=True)
class FixedInterval:
    """Constant cadence: sleep ``interval`` before every poll (clamped to remaining).

    Forge's schedule under the timer-loop transport. Reproduces the exact sleep
    sequence forge's ``batch_submit_and_wait`` emitted before the loop was
    extracted — ``min(interval, remaining)`` every iteration — so the committed
    replay histories still replay without regeneration.
    """

    interval: timedelta

    def next_sleep(self, *, attempt: int, remaining: timedelta, rng: random.Random) -> timedelta:
        """Return ``min(interval, remaining)`` — ``attempt`` and ``rng`` unused."""
        return min(self.interval, remaining)


@dataclass(frozen=True, slots=True, kw_only=True)
class BackoffSchedule:
    """Adaptive backoff with per-waiter jitter (ocr's schedule; used from ST2).

    Shaped to the OCR processing-time distribution (mode ≈ 30 min): poll at a
    steady ``initial`` cadence while a wait is young (its first ``fast_window``
    worth of polls clears most jobs), then double each subsequent poll up to
    ``cap`` so a long tail costs few history events. ``jitter_fraction`` spreads a
    burst of concurrent waiters so they don't hammer the provider status endpoint
    in lockstep.

    ``fast_window // initial`` fixes how many polls run at the fast cadence (e.g.
    1h // 300s = 12); poll ``attempt`` at or beyond that index sleeps
    ``initial * factor ** (index-past-window)``, capped at ``cap``. Jitter is
    symmetric (``± jitter_fraction`` of the base) and drawn from the injected
    PRNG; ``jitter_fraction == 0`` leaves the base untouched and never advances
    the PRNG, so the sequence is exactly deterministic.
    """

    initial: timedelta = timedelta(seconds=300)
    fast_window: timedelta = timedelta(hours=1)
    factor: float = 2.0
    cap: timedelta = timedelta(seconds=1800)
    jitter_fraction: float = 0.0

    def next_sleep(self, *, attempt: int, remaining: timedelta, rng: random.Random) -> timedelta:
        """Return the (jittered, capped) backoff sleep, clamped to ``remaining``."""
        fast_polls = self.fast_window // self.initial
        if attempt < fast_polls:
            base_seconds = self.initial.total_seconds()
        else:
            # Clamp the exponent to the doublings needed to reach the cap: once
            # ``initial * factor ** step >= cap`` the result is pinned at the cap,
            # so a far-out attempt need not raise ``factor`` to a huge power (which
            # would ``OverflowError`` on ``float ** int``).
            step = min(attempt - fast_polls + 1, self._steps_to_cap)
            base_seconds = min(
                self.initial.total_seconds() * self.factor**step,
                self.cap.total_seconds(),
            )
        sleep_seconds = self._apply_jitter(base_seconds, rng)
        return min(timedelta(seconds=sleep_seconds), remaining)

    @property
    def _steps_to_cap(self) -> int:
        """Doublings for ``initial`` to reach ``cap`` (0 when it can't grow)."""
        if self.factor <= 1.0 or self.cap <= self.initial:
            return 0
        return math.ceil(math.log(self.cap / self.initial, self.factor))

    def _apply_jitter(self, base_seconds: float, rng: random.Random) -> float:
        """Symmetric ``± jitter_fraction`` jitter; a no-op (PRNG untouched) at 0."""
        if self.jitter_fraction <= 0:
            return base_seconds
        spread = base_seconds * self.jitter_fraction
        return base_seconds + rng.uniform(-spread, spread)


async def wait_batch_ended(
    status_fn: Callable[[], Awaitable[str]],
    *,
    schedule: PollSchedule,
    ceiling: timedelta = BATCH_WAIT_CEILING,
) -> str:
    """Poll a provider batch on a timer until it ends or the ceiling expires.

    The shared timer-loop skeleton (D88, T4.2). Computes a deadline from
    ``workflow.now() + ceiling``, then repeatedly: check the remaining budget,
    sleep ``schedule.next_sleep(...)`` (already clamped to remaining), and await
    ``status_fn`` for the provider's normalized state string. Returns:

    - ``"ended"`` when the batch reached ``ended`` (the caller fetches its result);
    - the terminal state verbatim (``"failed"`` / ``"expired"`` / ``"canceled"``)
      when the provider reported one;
    - ``"gave_up"`` when the ceiling expired before the batch ended.

    Any other state loops. The loop writes no ledger rows and raises nothing for
    an outcome — the caller owns persistence and the non-retryable ``ApplicationError``
    (the outcome-to-side-effect mapping is consumer-specific). ``status_fn`` is the
    only I/O; it must wrap a status-poll activity (never inline a provider call in
    workflow code).
    """
    deadline = workflow.now() + ceiling
    rng = workflow.random()
    attempt = 0
    while True:
        remaining = deadline - workflow.now()
        if remaining <= timedelta(0):
            return "gave_up"
        await workflow.sleep(schedule.next_sleep(attempt=attempt, remaining=remaining, rng=rng))
        state = await status_fn()
        if state == "ended":
            return "ended"
        if state in ("failed", "expired", "canceled"):
            return state
        attempt += 1
