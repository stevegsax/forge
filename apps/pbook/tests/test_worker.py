"""Tests for pbook.worker."""

from __future__ import annotations

import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pbook.worker as worker_mod


class TestRequestShutdown:
    def test_sets_stop_event(self) -> None:
        stop = asyncio.Event()

        worker_mod._request_shutdown(signal.SIGTERM, stop)

        assert stop.is_set()


class TestRunUntilShutdown:
    @pytest.mark.asyncio
    async def test_stop_triggers_shutdown_then_waits_for_run(self) -> None:
        """Setting ``stop`` drains gracefully: shutdown() is awaited, then run()."""
        stop = asyncio.Event()
        worker = MagicMock()
        run_gate = asyncio.Event()

        async def _run() -> None:
            await run_gate.wait()

        async def _shutdown() -> None:
            # Mirrors real temporalio behavior: shutdown() and run() both
            # unblock once the same underlying drain completes.
            run_gate.set()

        worker.run = AsyncMock(side_effect=_run)
        worker.shutdown = AsyncMock(side_effect=_shutdown)

        task = asyncio.create_task(worker_mod._run_until_shutdown(worker, stop))
        await asyncio.sleep(0)  # let run_task start awaiting run_gate
        stop.set()
        await asyncio.wait_for(task, timeout=1)

        worker.shutdown.assert_awaited_once_with()
        worker.run.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_run_completing_first_skips_shutdown(self) -> None:
        stop = asyncio.Event()
        worker = MagicMock()
        worker.run = AsyncMock()
        worker.shutdown = AsyncMock()

        await worker_mod._run_until_shutdown(worker, stop)

        worker.shutdown.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_crash_propagates_without_shutdown(self) -> None:
        stop = asyncio.Event()
        worker = MagicMock()
        worker.run = AsyncMock(side_effect=RuntimeError("worker boom"))
        worker.shutdown = AsyncMock()

        with pytest.raises(RuntimeError, match="worker boom"):
            await worker_mod._run_until_shutdown(worker, stop)

        worker.shutdown.assert_not_awaited()


class TestRunWorker:
    @pytest.mark.asyncio
    async def test_bootstraps_worker_and_runs_until_clean_exit(self) -> None:
        mock_client = MagicMock()
        mock_worker_instance = MagicMock()
        mock_worker_instance.run = AsyncMock()

        with (
            patch.object(worker_mod, "_register_llm_provider", MagicMock()),
            patch.object(worker_mod, "_register_output_types", MagicMock()),
            patch.object(worker_mod, "_migrate_if_configured", MagicMock()),
            patch("pbook.log_config.setup_logging", MagicMock()),
            patch.object(worker_mod.Client, "connect", AsyncMock(return_value=mock_client)),
            patch.object(worker_mod, "Worker", return_value=mock_worker_instance) as mock_worker,
        ):
            await worker_mod.run_worker(address="localhost:7233")

        mock_worker.assert_called_once()
        worker_kwargs = mock_worker.call_args.kwargs
        assert worker_kwargs["task_queue"] == worker_mod.PBOOK_TASK_QUEUE
        assert worker_kwargs["graceful_shutdown_timeout"].total_seconds() == 30
        mock_worker_instance.run.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_worker_crash_propagates(self) -> None:
        mock_client = MagicMock()
        mock_worker_instance = MagicMock()
        mock_worker_instance.run = AsyncMock(side_effect=RuntimeError("worker boom"))

        with (
            patch.object(worker_mod, "_register_llm_provider", MagicMock()),
            patch.object(worker_mod, "_register_output_types", MagicMock()),
            patch.object(worker_mod, "_migrate_if_configured", MagicMock()),
            patch("pbook.log_config.setup_logging", MagicMock()),
            patch.object(worker_mod.Client, "connect", AsyncMock(return_value=mock_client)),
            patch.object(worker_mod, "Worker", return_value=mock_worker_instance),
            pytest.raises(RuntimeError, match="worker boom"),
        ):
            await worker_mod.run_worker(address="localhost:7233")

    @pytest.mark.asyncio
    async def test_registers_and_removes_sigterm_sigint_handlers(self) -> None:
        mock_client = MagicMock()
        mock_worker_instance = MagicMock()
        mock_worker_instance.run = AsyncMock()

        loop = asyncio.get_running_loop()
        add_calls: list[tuple[int, tuple[object, ...]]] = []
        remove_calls: list[int] = []

        def _fake_add_signal_handler(sig: int, callback: object, *args: object) -> None:
            add_calls.append((sig, args))

        def _fake_remove_signal_handler(sig: int) -> bool:
            remove_calls.append(sig)
            return True

        with (
            patch.object(worker_mod, "_register_llm_provider", MagicMock()),
            patch.object(worker_mod, "_register_output_types", MagicMock()),
            patch.object(worker_mod, "_migrate_if_configured", MagicMock()),
            patch("pbook.log_config.setup_logging", MagicMock()),
            patch.object(worker_mod.Client, "connect", AsyncMock(return_value=mock_client)),
            patch.object(worker_mod, "Worker", return_value=mock_worker_instance),
            patch.object(loop, "add_signal_handler", _fake_add_signal_handler),
            patch.object(loop, "remove_signal_handler", _fake_remove_signal_handler),
        ):
            await worker_mod.run_worker(address="localhost:7233")

        registered = {sig for sig, _args in add_calls}
        assert registered == {signal.SIGTERM, signal.SIGINT}
        for sig, args in add_calls:
            assert args[0] == sig
        assert set(remove_calls) == {signal.SIGTERM, signal.SIGINT}
