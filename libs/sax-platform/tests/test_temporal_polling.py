"""Tests for the shared batch poll loop (``sax_platform.temporal.polling``).

Two layers, matching the module's Functional-Core / Imperative-Shell split:

* **Schedule math** — ``FixedInterval`` is a pure frozen dataclass, exercised
  directly with a seeded ``random.Random`` (no workflow context): the constant
  cadence and the remaining-time clamp.
* **The loop** — ``wait_batch_ended`` calls ``workflow.now`` / ``workflow.sleep``
  / ``workflow.random``, so the tests patch those three with a fake clock (sleep
  advances it) and drive the loop with a scripted ``status_fn``. This pins the
  outcome strings and the sleep sequence without a real Temporal environment.
"""

import random
from datetime import UTC, datetime, timedelta

import pytest

from sax_platform.temporal import polling
from sax_platform.temporal.polling import (
    BATCH_WAIT_CEILING,
    FixedInterval,
    wait_batch_ended,
)

# ---------------------------------------------------------------------------
# Schedule math — pure, seeded rng
# ---------------------------------------------------------------------------


class TestFixedInterval:
    def test_returns_interval_when_below_remaining(self) -> None:
        sched = FixedInterval(timedelta(seconds=300))
        got = sched.next_sleep(attempt=0, remaining=timedelta(hours=25), rng=random.Random(0))
        assert got == timedelta(seconds=300)

    def test_ignores_attempt(self) -> None:
        sched = FixedInterval(timedelta(seconds=300))
        rng = random.Random(0)
        first = sched.next_sleep(attempt=0, remaining=timedelta(hours=25), rng=rng)
        later = sched.next_sleep(attempt=99, remaining=timedelta(hours=25), rng=rng)
        assert first == later == timedelta(seconds=300)

    def test_clamps_to_remaining(self) -> None:
        sched = FixedInterval(timedelta(seconds=300))
        got = sched.next_sleep(attempt=0, remaining=timedelta(seconds=120), rng=random.Random(0))
        assert got == timedelta(seconds=120)

    def test_does_not_advance_rng(self) -> None:
        sched = FixedInterval(timedelta(seconds=300))
        rng = random.Random(0)
        before = rng.getstate()
        sched.next_sleep(attempt=3, remaining=timedelta(hours=1), rng=rng)
        assert rng.getstate() == before


# ---------------------------------------------------------------------------
# The loop — fake clock + scripted status_fn
# ---------------------------------------------------------------------------


class _FakeClock:
    """A workflow clock whose ``sleep`` advances ``now`` (replaces the timer)."""

    def __init__(self, start: datetime) -> None:
        self.now = start
        self.sleeps: list[timedelta] = []

    def read(self) -> datetime:
        return self.now

    async def sleep(self, duration: timedelta) -> None:
        self.sleeps.append(duration)
        self.now += duration


def _scripted_status(states: list[str]) -> tuple[object, list[int]]:
    """A ``status_fn`` returning ``states`` in order; also reports its call count."""
    calls = [0]

    async def status_fn() -> str:
        state = states[calls[0]]
        calls[0] += 1
        return state

    return status_fn, calls


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Patch ``workflow.now/sleep/random`` with a deterministic fake clock."""
    fake = _FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(polling.workflow, "now", fake.read)
    monkeypatch.setattr(polling.workflow, "sleep", fake.sleep)
    monkeypatch.setattr(polling.workflow, "random", lambda: random.Random(0))
    return fake


class TestWaitBatchEnded:
    async def test_returns_ended_on_first_ended_status(self, clock: _FakeClock) -> None:
        status_fn, calls = _scripted_status(["ended"])
        outcome = await wait_batch_ended(status_fn, schedule=FixedInterval(timedelta(seconds=300)))
        assert outcome == "ended"
        assert calls[0] == 1
        assert clock.sleeps == [timedelta(seconds=300)]  # sleeps before the first poll

    async def test_loops_through_non_terminal_states(self, clock: _FakeClock) -> None:
        status_fn, calls = _scripted_status(["validating", "in_progress", "ended"])
        outcome = await wait_batch_ended(status_fn, schedule=FixedInterval(timedelta(seconds=300)))
        assert outcome == "ended"
        assert calls[0] == 3
        assert clock.sleeps == [timedelta(seconds=300)] * 3

    @pytest.mark.parametrize("terminal", ["failed", "expired", "canceled"])
    async def test_returns_terminal_state_verbatim(self, clock: _FakeClock, terminal: str) -> None:
        status_fn, _ = _scripted_status([terminal])
        outcome = await wait_batch_ended(status_fn, schedule=FixedInterval(timedelta(seconds=300)))
        assert outcome == terminal

    async def test_gives_up_when_ceiling_expires(self, clock: _FakeClock) -> None:
        # ceiling 600s, interval 300s -> two polls, then remaining hits 0.
        status_fn, calls = _scripted_status(["in_progress", "in_progress"])
        outcome = await wait_batch_ended(
            status_fn,
            schedule=FixedInterval(timedelta(seconds=300)),
            ceiling=timedelta(seconds=600),
        )
        assert outcome == "gave_up"
        assert calls[0] == 2
        assert clock.sleeps == [timedelta(seconds=300), timedelta(seconds=300)]

    async def test_final_sleep_clamped_to_remaining(self, clock: _FakeClock) -> None:
        # ceiling 400s < interval 300s twice: first sleep 300, second clamped to 100.
        status_fn, _ = _scripted_status(["in_progress", "in_progress"])
        outcome = await wait_batch_ended(
            status_fn,
            schedule=FixedInterval(timedelta(seconds=300)),
            ceiling=timedelta(seconds=400),
        )
        assert outcome == "gave_up"
        assert clock.sleeps == [timedelta(seconds=300), timedelta(seconds=100)]

    async def test_default_ceiling_is_the_shared_constant(self, clock: _FakeClock) -> None:
        assert timedelta(hours=25) == BATCH_WAIT_CEILING
        status_fn, _ = _scripted_status(["ended"])
        # Default ceiling path (no explicit ceiling kwarg) still resolves.
        outcome = await wait_batch_ended(status_fn, schedule=FixedInterval(timedelta(seconds=300)))
        assert outcome == "ended"
