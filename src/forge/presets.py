"""Activity timeout, retry, and token presets for forge's workflows (T5.3 ST8).

One home for the values every workflow module schedules activities with. They
lived in four copies — ``workflows.py``, the former ``workflow_blocks.py``,
``blocks/step.py``, and ``blocks/dispatch.py`` — where three of them carried
hand-synchronized duplicates of the same git/context/write/validate numbers, and
``blocks/gather.py`` had to reach into another block's privates to avoid minting
a fifth.

These values are not decoration: a timeout and a retry policy are recorded in
the ``ScheduleActivityTask`` command a workflow emits, so **changing one changes
the command sequence** and breaks the replay of an in-flight execution (and the
committed histories under ``tests/replay/``). Moving the definitions here is
command-neutral; editing a number is not. Treat every constant below as pinned
unless a task deliberately re-tunes it and regenerates the histories.

Pure module: no I/O, no workflow state, safe to import anywhere (including into
the Temporal workflow sandbox).
"""

from __future__ import annotations

from datetime import timedelta

from sax_platform.temporal.polling import BATCH_WAIT_CEILING
from temporalio.common import RetryPolicy

__all__ = [
    "BATCH_FETCH_TIMEOUT",
    "BATCH_POLL_FLOOR",
    "BATCH_POLL_INTERVAL",
    "BATCH_STATUS_TIMEOUT",
    "BATCH_WAIT_TIMEOUT",
    "CONFLICT_RESOLUTION_TIMEOUT",
    "CONTEXT_TIMEOUT",
    "DEFAULT_MAX_TOKENS",
    "EXPLORATION_FULFILL_TIMEOUT",
    "EXPLORATION_LLM_TIMEOUT",
    "GIT_RETRY",
    "GIT_TIMEOUT",
    "LLM_HEARTBEAT",
    "LLM_TIMEOUT",
    "PARSE_TIMEOUT",
    "SANITY_CHECK_TIMEOUT",
    "SUBMIT_TIMEOUT",
    "THINKING_MAX_TOKENS",
    "VALIDATE_HEARTBEAT",
    "VALIDATE_TIMEOUT",
    "WRITE_RETRY",
    "WRITE_TIMEOUT",
]


# ---------------------------------------------------------------------------
# Local work: git, context assembly, writing, validation
# ---------------------------------------------------------------------------

GIT_TIMEOUT = timedelta(seconds=30)
CONTEXT_TIMEOUT = timedelta(seconds=30)
WRITE_TIMEOUT = timedelta(seconds=30)
VALIDATE_TIMEOUT = timedelta(minutes=2)
EXPLORATION_FULFILL_TIMEOUT = timedelta(minutes=2)


# ---------------------------------------------------------------------------
# LLM calls (sync lane)
# ---------------------------------------------------------------------------

LLM_TIMEOUT = timedelta(minutes=5)
SANITY_CHECK_TIMEOUT = timedelta(minutes=5)
EXPLORATION_LLM_TIMEOUT = timedelta(minutes=5)
CONFLICT_RESOLUTION_TIMEOUT = timedelta(minutes=5)


# ---------------------------------------------------------------------------
# Batch transport (T4.1, D88)
# ---------------------------------------------------------------------------

SUBMIT_TIMEOUT = timedelta(seconds=60)
PARSE_TIMEOUT = timedelta(seconds=30)
BATCH_STATUS_TIMEOUT = timedelta(seconds=60)
BATCH_FETCH_TIMEOUT = timedelta(minutes=5)
# One source of truth for the 25h ceiling: sax_platform.temporal.polling owns it
# (T4.2 ST1); forge.models re-exports it for step_logic.child_timeout /
# derive_execution_timeout.
BATCH_WAIT_TIMEOUT = BATCH_WAIT_CEILING
# Timer-loop poll cadence (T4.1, D88). Default 600s; floored at 300s in the loop
# as defense in depth (the ``batch_poll_interval_seconds`` input fields also
# validate ge=300). A batch is never done instantly, so the loop sleeps before
# its first status poll.
BATCH_POLL_INTERVAL = timedelta(seconds=600)
BATCH_POLL_FLOOR = timedelta(seconds=300)


# ---------------------------------------------------------------------------
# Heartbeats — detect worker crashes during long-running activities
# ---------------------------------------------------------------------------

LLM_HEARTBEAT = timedelta(seconds=60)
# Validation subprocesses run via asyncio.to_thread (T1.4), so the heartbeat
# loop keeps firing (every 30s) during a check. 60s makes this a real crash
# detector rather than the 120s workaround for a blocked event loop.
VALIDATE_HEARTBEAT = timedelta(seconds=60)


# ---------------------------------------------------------------------------
# Retry policies
#
# LLM_RETRY and IO_RETRY are the shared presets from
# sax_platform.temporal.retries (T3.4); these two are forge-specific because
# their non-retryable error types are forge's own.
# ---------------------------------------------------------------------------

GIT_RETRY = RetryPolicy(
    maximum_attempts=2,
    non_retryable_error_types=["CommitError", "RepoDiscoveryError"],
)
WRITE_RETRY = RetryPolicy(
    maximum_attempts=2,
    non_retryable_error_types=["OutputWriteError", "EditApplicationError"],
)


# ---------------------------------------------------------------------------
# Output token caps
# ---------------------------------------------------------------------------

# The cap for a thinking-disabled batch call — generation and exploration.
DEFAULT_MAX_TOKENS = 4096

# Explicit cap for the three thinking-enabled batch call paths (planner,
# sanity-check, conflict-resolution): adaptive thinking now competes for
# tokens inside max_tokens instead of riding on top of it, so the old 4096
# default batch-lane cap left too little room for both the thinking budget
# and the structured output it must still emit. Sized for adaptive thinking +
# structured output on the batch lane; tokens-vs-cap telemetry decides future
# tuning (owner-adjudicated, 2026-07 Phase 3 code review). The generation
# path stays thinking-disabled and keeps its own (lower) cap untouched.
THINKING_MAX_TOKENS = 16384
