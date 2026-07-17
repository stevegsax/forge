"""Shared heartbeat helper for long-running Temporal activities."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from temporalio import activity

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_DEFAULT_INTERVAL_SECONDS = 30


@asynccontextmanager
async def heartbeat_during(
    interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
) -> AsyncIterator[None]:
    """Emit Temporal heartbeats at regular intervals while the body executes.

    Use this to wrap long-running awaits (e.g., LLM API calls) so that
    Temporal can detect worker crashes and deliver cancellation signals.
    """

    async def _loop() -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            activity.heartbeat()

    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
