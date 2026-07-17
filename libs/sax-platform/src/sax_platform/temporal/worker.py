"""Temporal worker scaffold: sandbox runner, ``Worker`` kwargs, and a graceful
SIGINT/SIGTERM drain loop shared across the platform and its consumer apps.

Ported from forge's ``src/forge/worker.py`` (T3.4, ST2) with one behavior
fix: ``graceful_shutdown_timeout`` defaults to 5 minutes here, not 30 seconds.
Forge's sync activities include LLM calls with up to a 5-minute
``start_to_close`` timeout (``src/forge/workflows.py``); a 30-second drain
window cancels one of those mid-flight on every deploy. This module's default
is sized to be at least as long as the longest sync activity in the fleet —
apps with shorter activities may pass a smaller value.

This module imports ``temporalio.worker`` (real client/worker machinery), so
it must NEVER be imported from inside a Temporal workflow or activity
sandbox — that is what ``sax_platform.temporal.retries`` and
``sax_platform.temporal.heartbeat`` are for. ``sax_platform.temporal``'s
``__init__`` exports this module lazily for exactly that reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from temporalio.client import Client

__all__ = [
    "DEFAULT_GRACEFUL_SHUTDOWN",
    "DEFAULT_MAX_CONCURRENT_ACTIVITIES",
    "build_sandbox_runner",
    "run_worker",
    "worker_kwargs",
]

logger = logging.getLogger(__name__)

_DEFAULT_PASSTHROUGH_MODULES: tuple[str, ...] = ("pydantic", "pydantic_core")

# Activities in this fleet spawn subprocesses (git, validators, OCR/LLM SDK
# calls) — unbounded concurrency is wrong for a single host. 8 is a modest,
# explicit default sized for one worker process on one desktop; apps with a
# different activity shape (heavier or lighter) override it per call.
DEFAULT_MAX_CONCURRENT_ACTIVITIES: Final[int] = 8

# Must be >= the longest sync-activity start_to_close timeout in the fleet so
# a graceful drain never cancels an in-flight call — forge's LLM activities
# run up to 5 minutes (see module docstring). This is the AC fix: forge's
# prior hardcoded 30s value canceled in-flight LLM calls on every deploy.
DEFAULT_GRACEFUL_SHUTDOWN: Final[timedelta] = timedelta(minutes=5)


def build_sandbox_runner(
    passthrough_modules: Sequence[str] = _DEFAULT_PASSTHROUGH_MODULES,
) -> SandboxedWorkflowRunner:
    """Build the workflow sandbox runner, passing selected modules through unsandboxed.

    Passing pydantic/pydantic_core through the sandbox (the default) matters
    because the shared pydantic data converter serializes workflow args/results,
    so pydantic_core (the Rust extension) gets imported lazily the first time a
    model is (de)serialized inside a workflow run — after the sandbox has
    snapshotted its modules, which triggers a "imported after initial workflow
    load" UserWarning. Reusing the host's already-loaded modules is safe
    (they're deterministic) and silences it.
    """
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(*passthrough_modules)
    )


def worker_kwargs(
    *,
    task_queue: str,
    workflows: list[type],
    activities: list[Callable[..., Any]],
    workflow_runner: SandboxedWorkflowRunner,
    max_concurrent_activities: int,
    graceful_shutdown_timeout: timedelta,
) -> dict[str, Any]:
    """Pure: assemble the ``Worker(...)`` kwargs.

    Kept separate from ``run_worker`` so the shape (which fields are set, to
    what) is unit-testable without a live client, an event loop, or a real
    Temporal connection.
    """
    return {
        "task_queue": task_queue,
        "workflows": workflows,
        "activities": activities,
        "workflow_runner": workflow_runner,
        "max_concurrent_activities": max_concurrent_activities,
        "graceful_shutdown_timeout": graceful_shutdown_timeout,
    }


async def _run_until_shutdown(worker: Worker, stop: asyncio.Event) -> None:
    """Run ``worker`` until it exits on its own or ``stop`` requests a drain.

    Races ``worker.run()`` against ``stop.wait()``. If ``stop`` fires first (a
    signal handler set it), this requests a graceful shutdown and then waits
    for ``run()`` to return — ``Worker.shutdown()`` and ``Worker.run()`` both
    block until the same underlying drain completes, so the second await
    resolves right away. If ``run()`` finishes first — clean exit or a crash —
    the stop waiter is cancelled and ``run_task`` is awaited so any exception
    propagates to the caller unchanged.

    Ported verbatim (structurally) from ``forge.worker._run_until_shutdown``.
    """
    run_task: asyncio.Task[None] = asyncio.create_task(worker.run())
    stop_task: asyncio.Task[bool] = asyncio.create_task(stop.wait())
    done, _pending = await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    if run_task in done:
        stop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_task
        await run_task  # re-raise a crash, if any
        return
    await worker.shutdown()
    await run_task


def _request_shutdown(
    sig: signal.Signals, stop: asyncio.Event, graceful_shutdown_timeout: timedelta
) -> None:
    """Signal handler: log receipt and flag the run loop to begin a graceful drain."""
    logger.info(
        "received %s — draining (graceful_shutdown_timeout=%s)",
        sig.name,
        graceful_shutdown_timeout,
    )
    stop.set()


async def run_worker(
    client: Client,
    *,
    task_queue: str,
    workflows: list[type],
    activities: list[Callable[..., Any]],
    max_concurrent_activities: int = DEFAULT_MAX_CONCURRENT_ACTIVITIES,
    graceful_shutdown_timeout: timedelta = DEFAULT_GRACEFUL_SHUTDOWN,
    passthrough_modules: Sequence[str] = _DEFAULT_PASSTHROUGH_MODULES,
) -> None:
    """Build a ``Worker`` on ``client`` and run it until SIGINT/SIGTERM requests
    a graceful drain.

    Owns exactly the ``Worker`` construction plus the signal-handled drain
    loop (ported from ``forge.worker.run_worker``) — connecting to Temporal,
    running DB migrations, registering schedules, and registering output
    types stay app-specific and are the caller's job before invoking this.
    """
    runner = build_sandbox_runner(passthrough_modules)
    worker = Worker(
        client,
        **worker_kwargs(
            task_queue=task_queue,
            workflows=workflows,
            activities=activities,
            workflow_runner=runner,
            max_concurrent_activities=max_concurrent_activities,
            graceful_shutdown_timeout=graceful_shutdown_timeout,
        ),
    )

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    handled_signals = (signal.SIGTERM, signal.SIGINT)
    for sig in handled_signals:
        loop.add_signal_handler(sig, _request_shutdown, sig, stop, graceful_shutdown_timeout)

    try:
        await _run_until_shutdown(worker, stop)
        logger.info("worker exited cleanly")
    finally:
        for sig in handled_signals:
            loop.remove_signal_handler(sig)
