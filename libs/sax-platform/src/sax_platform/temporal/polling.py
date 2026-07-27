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
history — no command emitted). The schedule is a pure frozen dataclass, so the
sleep sequence is exactly deterministic.
"""

import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from temporalio import workflow

__all__ = [
    "BATCH_WAIT_CEILING",
    "FixedInterval",
    "PollSchedule",
    "wait_batch_ended",
]

# The per-wait batch ceiling: a single batch wait may legally block a workflow up
# to this long before the timer loop gives up. One source of truth for the 25h
# number — forge re-exports it (``forge.models`` for the CLI/timeout math,
# ``forge.presets`` for the timer loop's wait ceiling) rather than duplicating
# the literal.
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
