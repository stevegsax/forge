"""Tests for forge.worker."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import forge.worker as worker_mod


class TestInitStore:
    def test_raises_when_store_url_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from forge.store import StoreConfigError

        mock_run_migrations = MagicMock()
        monkeypatch.delenv("FORGE_DB_URL", raising=False)
        monkeypatch.setattr("forge.store.run_migrations", mock_run_migrations)

        with pytest.raises(StoreConfigError, match="FORGE_DB_URL"):
            worker_mod._init_store()

        mock_run_migrations.assert_not_called()

    def test_runs_migrations_against_configured_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        url = f"sqlite:///{tmp_path / 'forge.db'}"
        mock_run_migrations = MagicMock()

        monkeypatch.setenv("FORGE_DB_URL", url)
        monkeypatch.setattr("forge.store.run_migrations", mock_run_migrations)

        worker_mod._init_store()

        mock_run_migrations.assert_called_once_with(url)


class TestEnsureSchedule:
    @pytest.mark.asyncio
    async def test_creates_schedule_with_expected_interval(self) -> None:
        client = MagicMock()
        client.create_schedule = AsyncMock()

        await worker_mod._ensure_schedule(
            client,
            schedule_id="forge-batch-poller",
            workflow_name="BatchPollerWorkflow",
            workflow_arg=worker_mod.BatchPollerInput(),
            interval=timedelta(seconds=300),
        )

        client.create_schedule.assert_awaited_once()
        schedule_id, schedule = client.create_schedule.await_args.args
        assert schedule_id == "forge-batch-poller"
        assert schedule.action.id == "forge-batch-poller-run"
        assert schedule.action.task_queue == worker_mod.FORGE_TASK_QUEUE
        assert schedule.spec.intervals[0].every == timedelta(seconds=300)
        assert schedule.state.note == "Forge schedule: forge-batch-poller"

    @pytest.mark.asyncio
    async def test_updates_existing_schedule_when_already_running(self) -> None:
        handle = MagicMock()
        handle.update = AsyncMock()
        client = MagicMock()
        client.create_schedule = AsyncMock(
            side_effect=worker_mod.ScheduleAlreadyRunningError()
        )
        client.get_schedule_handle.return_value = handle

        await worker_mod._ensure_schedule(
            client,
            schedule_id="forge-extraction-schedule",
            workflow_name="ForgeExtractionWorkflow",
            workflow_arg=worker_mod.ExtractionWorkflowInput(),
            interval=timedelta(hours=4),
        )

        client.get_schedule_handle.assert_called_once_with("forge-extraction-schedule")
        handle.update.assert_awaited_once()

        updater = handle.update.await_args.args[0]
        existing_schedule = worker_mod.Schedule(
            action=worker_mod.ScheduleActionStartWorkflow(
                "OldWorkflow",
                {},
                id="old-run",
                task_queue="old-queue",
            ),
            spec=worker_mod.ScheduleSpec(
                intervals=[worker_mod.ScheduleIntervalSpec(every=timedelta(minutes=5))]
            ),
            state=worker_mod.ScheduleState(note="old"),
        )
        update_input = SimpleNamespace(
            description=SimpleNamespace(schedule=existing_schedule)
        )

        update = await updater(update_input)
        assert update.schedule.spec.intervals[0].every == timedelta(hours=4)
        assert update.schedule.action.id == "old-run"


class TestRunWorker:
    @pytest.mark.asyncio
    async def test_bootstraps_worker_and_registers_schedules(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_client = MagicMock()
        mock_worker_instance = MagicMock()
        mock_worker_instance.run = AsyncMock()
        mock_ensure_schedule = AsyncMock()
        mock_set_temporal_client = MagicMock()
        mock_init_store = MagicMock()
        mock_init_tracing = MagicMock()
        mock_shutdown_tracing = MagicMock()
        mock_silence_noisy_loggers = MagicMock()

        monkeypatch.setenv("FORGE_TEMPORAL_ADDRESS", "temporal.example:7233")

        with (
            patch.object(worker_mod, "_init_store", mock_init_store),
            patch.object(worker_mod, "_ensure_schedule", mock_ensure_schedule),
            patch.object(worker_mod, "set_temporal_client", mock_set_temporal_client),
            patch.object(
                worker_mod.Client, "connect", AsyncMock(return_value=mock_client)
            ) as mock_connect,
            patch.object(worker_mod, "Worker", return_value=mock_worker_instance) as mock_worker,
            patch("forge.tracing.init_tracing", mock_init_tracing),
            patch("forge.tracing.shutdown_tracing", mock_shutdown_tracing),
            patch("forge.logging_config.silence_noisy_loggers", mock_silence_noisy_loggers),
        ):
            await worker_mod.run_worker(
                batch_poll_interval=300,
                extraction_interval=7200,
                identity="worker-123",
            )

        mock_init_store.assert_called_once_with()
        mock_init_tracing.assert_called_once_with()
        mock_silence_noisy_loggers.assert_called_once_with()
        mock_connect.assert_awaited_once_with(
            "temporal.example:7233",
            data_converter=worker_mod.pydantic_data_converter,
            identity="worker-123",
        )
        mock_set_temporal_client.assert_called_once_with(mock_client)
        assert mock_ensure_schedule.await_count == 2

        first_call = mock_ensure_schedule.await_args_list[0]
        assert first_call.args[0] is mock_client
        assert first_call.kwargs["schedule_id"] == "forge-batch-poller"
        assert first_call.kwargs["workflow_name"] == "BatchPollerWorkflow"
        assert first_call.kwargs["interval"] == timedelta(seconds=300)

        second_call = mock_ensure_schedule.await_args_list[1]
        assert second_call.kwargs["schedule_id"] == "forge-extraction-schedule"
        assert second_call.kwargs["workflow_name"] == "ForgeExtractionWorkflow"
        assert second_call.kwargs["interval"] == timedelta(seconds=7200)

        mock_worker.assert_called_once()
        worker_kwargs = mock_worker.call_args.kwargs
        assert worker_kwargs["task_queue"] == worker_mod.FORGE_TASK_QUEUE
        assert worker_mod.OcrSubmitWorkflow in worker_kwargs["workflows"]
        assert worker_mod.OcrGatherWorkflow in worker_kwargs["workflows"]
        assert worker_mod.poll_batch_results in worker_kwargs["activities"]
        assert worker_kwargs["graceful_shutdown_timeout"] == timedelta(seconds=30)

        # Ingestion workflows and activity should be registered when pbook
        # is installed (which it is in the test environment).
        if worker_mod._INGESTION_AVAILABLE:
            assert worker_mod.TranscriptIngestionWorkflow in worker_kwargs["workflows"]
            assert worker_mod.BatchIngestionWorkflow in worker_kwargs["workflows"]
            assert worker_mod.prepare_transcript in worker_kwargs["activities"]

        mock_worker_instance.run.assert_awaited_once_with()
        mock_shutdown_tracing.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_shutdown_tracing_runs_when_worker_fails(self) -> None:
        mock_client = MagicMock()
        mock_worker_instance = MagicMock()
        mock_worker_instance.run = AsyncMock(side_effect=RuntimeError("worker boom"))

        with (
            patch.object(worker_mod, "_init_store", MagicMock()),
            patch.object(worker_mod, "_ensure_schedule", AsyncMock()),
            patch.object(worker_mod, "set_temporal_client", MagicMock()),
            patch.object(worker_mod.Client, "connect", AsyncMock(return_value=mock_client)),
            patch.object(worker_mod, "Worker", return_value=mock_worker_instance),
            patch("forge.tracing.init_tracing", MagicMock()),
            patch("forge.tracing.shutdown_tracing", MagicMock()) as mock_shutdown_tracing,
            patch("forge.logging_config.silence_noisy_loggers", MagicMock()),
        ):
            with pytest.raises(RuntimeError, match="worker boom"):
                await worker_mod.run_worker(address="localhost:7233")

        mock_shutdown_tracing.assert_called_once_with()
