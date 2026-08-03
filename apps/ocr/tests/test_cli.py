"""Tests for the OCR CLI (``ocr worker``, ``submit``, ``list``, ``export``, ``mark``,
``unmark``).

Every command funnels through the single ``_start_and_wait`` shell helper, so most
command tests mock that one seam and assert on the workflow name, input payload,
workflow-id prefix, and address/timeout wiring passed to it. ``_start_and_wait``
itself is exercised directly (mocking ``connect_temporal``) so its Temporal-calling
body is covered too, not just the pass-through from each command.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
import sqlalchemy as sa
from click.testing import CliRunner
from sax_platform.contracts.constants import OCR_TASK_QUEUE

from ocr.cli import (
    EXIT_CONFIG_ERROR,
    EXIT_INFRASTRUCTURE_ERROR,
    EXIT_PROBE_ERROR,
    TrackerStatusReport,
    _auto_id,
    _echo,
    _start_and_wait,
    _start_submit,
    derive_tracker_status,
    format_migration_target,
    main,
    tracker_status_lines,
)
from ocr.models import (
    OcrExportResult,
    OcrListJobsResult,
    OcrMarkResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_result(value: object = None):
    """Return an async function that returns *value*.

    Use as ``mock.side_effect = _async_result(TaskResult(...))`` on a plain
    MagicMock to make ``asyncio.run(mock(...))`` return *value*.

    Unlike ``AsyncMock``, this avoids orphaned internal coroutines that trigger
    'coroutine was never awaited' warnings when the mock is called via
    ``asyncio.run()`` rather than ``await``.
    """

    async def _fn(*_args: object, **_kwargs: object) -> object:
        return value

    return _fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestAutoId:
    def test_prefix_and_length(self) -> None:
        result = _auto_id("ocr-submit")
        assert result.startswith("ocr-submit-")
        assert len(result) == len("ocr-submit-") + 8

    def test_unique_across_calls(self) -> None:
        assert _auto_id("ocr-submit") != _auto_id("ocr-submit")


class TestEcho:
    def test_pydantic_result_uses_model_dump_json(self, capsys: pytest.CaptureFixture) -> None:
        result = OcrMarkResult(document_id="doc-1", found=True)
        _echo(result)
        printed = capsys.readouterr().out
        parsed = json.loads(printed)
        assert parsed == {"document_id": "doc-1", "found": True}

    def test_non_pydantic_result_falls_back_to_json_dumps(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        _echo({"status": "done", "pages": 5})
        printed = capsys.readouterr().out
        assert json.loads(printed) == {"status": "done", "pages": 5}

    def test_non_serializable_value_uses_default_str(self, capsys: pytest.CaptureFixture) -> None:
        # No model_dump_json, and not directly JSON-serializable -> exercises the
        # json.dumps(..., default=str) fallback branch.
        _echo({"created_at": timedelta(hours=1)})
        printed = capsys.readouterr().out
        assert json.loads(printed) == {"created_at": "1:00:00"}


# ---------------------------------------------------------------------------
# _start_and_wait (the async shell helper every command funnels through)
# ---------------------------------------------------------------------------


class TestStartAndWait:
    @pytest.mark.asyncio
    async def test_connects_starts_and_awaits_result(self) -> None:
        mock_handle = AsyncMock()
        mock_handle.result.return_value = {"ok": True}
        mock_client = AsyncMock()
        mock_client.start_workflow.return_value = mock_handle

        with patch(
            "ocr.cli.connect_temporal", new=AsyncMock(return_value=mock_client)
        ) as mock_connect:
            result = await _start_and_wait(
                "OcrSubmitWorkflow",
                {"file_path": "/tmp/x.pdf"},
                workflow_id="wf-1",
                address="localhost:7233",
                timeout_hours=2.0,
            )

        assert result == {"ok": True}
        # _connect_checked threads the resolved namespace (forge-test in the suite,
        # from the autouse forge_env fixture) into the shared connect chokepoint.
        mock_connect.assert_awaited_once()
        assert mock_connect.await_args.args == ("localhost:7233",)
        assert mock_connect.await_args.kwargs["namespace"] == "forge-test"
        mock_client.start_workflow.assert_awaited_once_with(
            "OcrSubmitWorkflow",
            {"file_path": "/tmp/x.pdf"},
            id="wf-1",
            task_queue=OCR_TASK_QUEUE,
        )
        mock_handle.result.assert_awaited_once_with(rpc_timeout=timedelta(hours=2.0))


class TestStartSubmit:
    @pytest.mark.asyncio
    async def test_starts_without_waiting_with_derived_timeout(self) -> None:
        """Submit starts the workflow (with a ~26h execution timeout) and returns
        its id — it never awaits ``handle.result``."""
        from sax_platform.temporal.polling import BATCH_WAIT_CEILING

        mock_handle = AsyncMock()
        mock_handle.id = "ocr-submit-abc"
        mock_client = AsyncMock()
        mock_client.start_workflow.return_value = mock_handle

        with patch(
            "ocr.cli.connect_temporal", new=AsyncMock(return_value=mock_client)
        ) as mock_connect:
            started_id = await _start_submit(
                {"file_path": "/tmp/x.pdf"}, workflow_id="ocr-submit-abc", address="localhost:7233"
            )

        assert started_id == "ocr-submit-abc"
        mock_connect.assert_awaited_once()
        assert mock_connect.await_args.args == ("localhost:7233",)
        assert mock_connect.await_args.kwargs["namespace"] == "forge-test"
        mock_client.start_workflow.assert_awaited_once_with(
            "OcrSubmitWorkflow",
            {"file_path": "/tmp/x.pdf"},
            id="ocr-submit-abc",
            task_queue=OCR_TASK_QUEUE,
            execution_timeout=BATCH_WAIT_CEILING + timedelta(hours=1),
        )
        # It must NOT block on the workflow result (the run can take up to 25h).
        mock_handle.result.assert_not_awaited()


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


class TestWorkerCommand:
    def test_runs_worker_with_default_address(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FORGE_WORKER_IDENTITY", raising=False)
        with patch("ocr.worker.run_worker") as mock_run:
            mock_run.side_effect = _async_result(None)
            result = cli_runner.invoke(main, ["worker"])
            assert result.exit_code == 0
            # No address default: the conftest's FORGE_TEMPORAL_ADDRESS reaches
            # the option through its envvar.
            mock_run.assert_called_once_with("127.0.0.1:7233", identity=None)

    def test_runs_worker_with_custom_address(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FORGE_WORKER_IDENTITY", raising=False)
        with patch("ocr.worker.run_worker") as mock_run:
            mock_run.side_effect = _async_result(None)
            result = cli_runner.invoke(
                main, ["worker", "--temporal-address", "temporal.internal:7233"]
            )
            assert result.exit_code == 0
            mock_run.assert_called_once_with("temporal.internal:7233", identity=None)

    def test_worker_identity_option_reaches_run_worker(self, cli_runner: CliRunner) -> None:
        with patch("ocr.worker.run_worker") as mock_run:
            mock_run.side_effect = _async_result(None)
            result = cli_runner.invoke(main, ["worker", "--worker-identity", "prod-ocr-worker"])
            assert result.exit_code == 0
            mock_run.assert_called_once_with("127.0.0.1:7233", identity="prod-ocr-worker")

    def test_worker_identity_env_var_reaches_run_worker(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The launchd/tmux lanes set the base this way (FORGE_WORKER_IDENTITY),
        # never on the command line.
        monkeypatch.setenv("FORGE_WORKER_IDENTITY", "dev-ocr-worker")
        with patch("ocr.worker.run_worker") as mock_run:
            mock_run.side_effect = _async_result(None)
            result = cli_runner.invoke(main, ["worker"])
            assert result.exit_code == 0
            mock_run.assert_called_once_with("127.0.0.1:7233", identity="dev-ocr-worker")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmitCommand:
    def test_success_uses_defaults(self, cli_runner: CliRunner) -> None:
        with patch("ocr.cli._start_submit") as mock_start:
            mock_start.side_effect = _async_result("ocr-submit-xyz")
            result = cli_runner.invoke(main, ["submit", "/tmp/doc.pdf"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed == {"workflow_id": "ocr-submit-xyz", "status": "started"}

        call_args, call_kwargs = mock_start.call_args
        wf_input = call_args[0]
        assert wf_input.file_path == "/tmp/doc.pdf"
        assert wf_input.model_name == "mistral:mistral-ocr-latest"
        assert wf_input.skip_duplicate_detection is False
        assert call_kwargs["workflow_id"].startswith("ocr-submit-")
        # No address default: the value is the conftest's FORGE_TEMPORAL_ADDRESS.
        assert call_kwargs["address"] == "127.0.0.1:7233"

    def test_custom_model_and_skip_duplicate_flag(self, cli_runner: CliRunner) -> None:
        with patch("ocr.cli._start_submit") as mock_start:
            mock_start.side_effect = _async_result("ocr-submit-2")
            result = cli_runner.invoke(
                main,
                [
                    "submit",
                    "/tmp/doc.pdf",
                    "--model",
                    "mistral:custom",
                    "--skip-duplicate-detection",
                ],
            )

        assert result.exit_code == 0
        wf_input = mock_start.call_args[0][0]
        assert wf_input.model_name == "mistral:custom"
        assert wf_input.skip_duplicate_detection is True

    def test_error_exits_with_infrastructure_code(self, cli_runner: CliRunner) -> None:
        with patch("ocr.cli._start_submit") as mock_start:
            mock_start.side_effect = RuntimeError("Connection refused")
            result = cli_runner.invoke(main, ["submit", "/tmp/doc.pdf"])

        assert result.exit_code == EXIT_INFRASTRUCTURE_ERROR
        assert "Error: Connection refused" in result.output


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestListCommand:
    def test_success_uses_defaults(self, cli_runner: CliRunner) -> None:
        list_result = OcrListJobsResult(jobs=[], total=0)
        with patch("ocr.cli._start_and_wait") as mock_start:
            mock_start.side_effect = _async_result(list_result)
            result = cli_runner.invoke(main, ["list"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed == {"jobs": [], "total": 0}

        call_args, call_kwargs = mock_start.call_args
        assert call_args[0] == "OcrListJobsWorkflow"
        wf_input = call_args[1]
        assert wf_input.limit == 50
        assert wf_input.status_filter == ""
        assert call_kwargs["workflow_id"].startswith("ocr-list-")

    def test_custom_limit_and_status_filter(self, cli_runner: CliRunner) -> None:
        list_result = OcrListJobsResult(jobs=[], total=0)
        with patch("ocr.cli._start_and_wait") as mock_start:
            mock_start.side_effect = _async_result(list_result)
            result = cli_runner.invoke(main, ["list", "--limit", "10", "--status", "errored"])

        assert result.exit_code == 0
        wf_input = mock_start.call_args[0][1]
        assert wf_input.limit == 10
        assert wf_input.status_filter == "errored"

    def test_error_exits_with_infrastructure_code(self, cli_runner: CliRunner) -> None:
        with patch("ocr.cli._start_and_wait") as mock_start:
            mock_start.side_effect = RuntimeError("boom")
            result = cli_runner.invoke(main, ["list"])

        assert result.exit_code == EXIT_INFRASTRUCTURE_ERROR
        assert "Error: boom" in result.output


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


class TestExportCommand:
    def test_success_uses_default_output_dir(self, cli_runner: CliRunner) -> None:
        export_result = OcrExportResult(
            document_id="doc-1",
            export_dir="/data/export/doc-1",
            markdown_path="/data/export/doc-1/doc.md",
            image_count=2,
        )
        with patch("ocr.cli._start_and_wait") as mock_start:
            mock_start.side_effect = _async_result(export_result)
            result = cli_runner.invoke(main, ["export", "doc-1"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["document_id"] == "doc-1"
        assert parsed["image_count"] == 2

        call_args, call_kwargs = mock_start.call_args
        assert call_args[0] == "OcrExportWorkflow"
        wf_input = call_args[1]
        assert wf_input.document_id == "doc-1"
        assert wf_input.output_dir == ""
        assert call_kwargs["workflow_id"].startswith("ocr-export-")

    def test_custom_output_dir(self, cli_runner: CliRunner) -> None:
        export_result = OcrExportResult(
            document_id="doc-1",
            export_dir="/custom/out",
            markdown_path="/custom/out/doc.md",
            image_count=0,
        )
        with patch("ocr.cli._start_and_wait") as mock_start:
            mock_start.side_effect = _async_result(export_result)
            result = cli_runner.invoke(main, ["export", "doc-1", "--output-dir", "/custom/out"])

        assert result.exit_code == 0
        wf_input = mock_start.call_args[0][1]
        assert wf_input.output_dir == "/custom/out"

    def test_error_exits_with_infrastructure_code(self, cli_runner: CliRunner) -> None:
        with patch("ocr.cli._start_and_wait") as mock_start:
            mock_start.side_effect = RuntimeError("not found")
            result = cli_runner.invoke(main, ["export", "doc-1"])

        assert result.exit_code == EXIT_INFRASTRUCTURE_ERROR
        assert "Error: not found" in result.output


# ---------------------------------------------------------------------------
# mark / unmark
# ---------------------------------------------------------------------------


class TestMarkCommand:
    def test_success(self, cli_runner: CliRunner) -> None:
        mark_result = OcrMarkResult(document_id="doc-1", found=True)
        with patch("ocr.cli._start_and_wait") as mock_start:
            mock_start.side_effect = _async_result(mark_result)
            result = cli_runner.invoke(main, ["mark", "doc-1"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed == {"document_id": "doc-1", "found": True}

        call_args, call_kwargs = mock_start.call_args
        assert call_args[0] == "OcrMarkForRemovalWorkflow"
        assert call_args[1].document_id == "doc-1"
        assert call_kwargs["workflow_id"].startswith("ocr-mark-")

    def test_error_exits_with_infrastructure_code(self, cli_runner: CliRunner) -> None:
        with patch("ocr.cli._start_and_wait") as mock_start:
            mock_start.side_effect = RuntimeError("unavailable")
            result = cli_runner.invoke(main, ["mark", "doc-1"])

        assert result.exit_code == EXIT_INFRASTRUCTURE_ERROR
        assert "Error: unavailable" in result.output


class TestUnmarkCommand:
    def test_success(self, cli_runner: CliRunner) -> None:
        mark_result = OcrMarkResult(document_id="doc-1", found=False)
        with patch("ocr.cli._start_and_wait") as mock_start:
            mock_start.side_effect = _async_result(mark_result)
            result = cli_runner.invoke(main, ["unmark", "doc-1"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed == {"document_id": "doc-1", "found": False}

        call_args, call_kwargs = mock_start.call_args
        assert call_args[0] == "OcrClearRemovalMarkWorkflow"
        assert call_args[1].document_id == "doc-1"
        assert call_kwargs["workflow_id"].startswith("ocr-unmark-")

    def test_error_exits_with_infrastructure_code(self, cli_runner: CliRunner) -> None:
        with patch("ocr.cli._start_and_wait") as mock_start:
            mock_start.side_effect = RuntimeError("unavailable")
            result = cli_runner.invoke(main, ["unmark", "doc-1"])

        assert result.exit_code == EXIT_INFRASTRUCTURE_ERROR
        assert "Error: unavailable" in result.output


# ---------------------------------------------------------------------------
# tracker-status: pure verdict derivation (functional core)
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _heartbeat(
    last_run_at: datetime,
    *,
    live_jobs: int = 2,
    hints_sent: int = 1,
    cycles_total: int = 5,
) -> dict[str, object]:
    """A raw ``get_tracker_heartbeat`` row shaped like the store returns it."""
    return {
        "id": 1,
        "last_run_at": last_run_at,
        "live_jobs": live_jobs,
        "hints_sent": hints_sent,
        "cycles_total": cycles_total,
    }


class TestDeriveTrackerStatus:
    @pytest.mark.parametrize(
        ("heartbeat", "live_jobs_now", "expected_status", "expected_exit", "expected_age"),
        [
            # No heartbeat row: never-ran. Exit 2 only when work is waiting.
            (None, 0, "never-ran", 1, None),
            (None, 3, "never-ran", 2, None),
            # Fresh (age <= threshold): healthy regardless of live-job count.
            (_heartbeat(_NOW - timedelta(seconds=10)), 0, "fresh", 0, 10),
            (_heartbeat(_NOW - timedelta(seconds=10)), 5, "fresh", 0, 10),
            (_heartbeat(_NOW), 9, "fresh", 0, 0),
            # Exactly at the threshold is still fresh (<=).
            (_heartbeat(_NOW - timedelta(seconds=300)), 0, "fresh", 0, 300),
            # Stale (age > threshold): exit 1 with no live jobs, exit 2 with live jobs.
            (_heartbeat(_NOW - timedelta(seconds=400)), 0, "stale", 1, 400),
            (_heartbeat(_NOW - timedelta(seconds=400)), 2, "stale", 2, 400),
        ],
    )
    def test_verdict_table(
        self,
        heartbeat: dict[str, object] | None,
        live_jobs_now: int,
        expected_status: str,
        expected_exit: int,
        expected_age: int | None,
    ) -> None:
        report = derive_tracker_status(
            heartbeat, now=_NOW, stale_after_seconds=300, live_jobs_now=live_jobs_now
        )
        assert report.status == expected_status
        assert report.exit_code == expected_exit
        assert report.heartbeat_age_seconds == expected_age
        assert report.live_jobs_now == live_jobs_now

    def test_never_ran_nulls_all_heartbeat_fields(self) -> None:
        report = derive_tracker_status(None, now=_NOW, stale_after_seconds=300, live_jobs_now=0)
        assert report.last_run_at is None
        assert report.cycles_total is None
        assert report.live_jobs_last_cycle is None
        assert report.hints_sent_last_cycle is None

    def test_heartbeat_fields_pass_through(self) -> None:
        row = _heartbeat(_NOW - timedelta(seconds=30), live_jobs=7, hints_sent=4, cycles_total=99)
        report = derive_tracker_status(row, now=_NOW, stale_after_seconds=300, live_jobs_now=1)
        assert report.cycles_total == 99
        assert report.live_jobs_last_cycle == 7
        assert report.hints_sent_last_cycle == 4

    def test_naive_last_run_at_is_treated_as_utc(self) -> None:
        """A naive ``last_run_at`` (as SQLite reads it) is normalized to UTC, not crashed."""
        naive = datetime(2026, 1, 1, 11, 50, 0)  # deliberately naive (sqlite readback)
        report = derive_tracker_status(
            _heartbeat(naive), now=_NOW, stale_after_seconds=300, live_jobs_now=0
        )
        assert report.heartbeat_age_seconds == 600
        assert report.status == "stale"
        assert report.last_run_at == datetime(2026, 1, 1, 11, 50, 0, tzinfo=UTC)

    def test_aware_non_utc_last_run_at_is_normalized(self) -> None:
        """An aware, non-UTC ``last_run_at`` (Postgres-style) is converted to UTC."""
        aware = datetime(2026, 1, 1, 13, 55, 0, tzinfo=timezone(timedelta(hours=2)))  # 11:55 UTC
        report = derive_tracker_status(
            _heartbeat(aware), now=_NOW, stale_after_seconds=300, live_jobs_now=0
        )
        assert report.last_run_at == datetime(2026, 1, 1, 11, 55, 0, tzinfo=UTC)
        assert report.heartbeat_age_seconds == 300
        assert report.status == "fresh"


class TestTrackerStatusLines:
    def test_never_ran_renders_none_placeholders(self) -> None:
        report = TrackerStatusReport(
            last_run_at=None,
            heartbeat_age_seconds=None,
            cycles_total=None,
            live_jobs_last_cycle=None,
            hints_sent_last_cycle=None,
            live_jobs_now=0,
            status="never-ran",
            exit_code=1,
        )
        lines = tracker_status_lines(report)
        assert lines[0] == "last_run_at: none"
        assert "heartbeat_age_seconds: none" in lines
        assert "cycles_total: none" in lines
        assert "status: never-ran" in lines


# ---------------------------------------------------------------------------
# tracker-status: CLI shell (direct DB read; no Temporal)
# ---------------------------------------------------------------------------


def _create_batch_jobs(engine: sa.Engine) -> None:
    """Create the platform-owned ``batch_jobs`` table the live-job query LEFT-joins."""
    from sax_platform.contracts.batch_jobs import metadata as bj_metadata

    bj_metadata.create_all(engine)


def _seed_heartbeat(
    engine: sa.Engine,
    *,
    last_run_at: datetime,
    live_jobs: int = 0,
    hints_sent: int = 0,
) -> None:
    from ocr.store import record_tracker_heartbeat

    record_tracker_heartbeat(engine, now=last_run_at, live_jobs=live_jobs, hints_sent=hints_sent)


def _seed_live_job(engine: sa.Engine, *, request_id: str, batch_id: str, workflow_id: str) -> None:
    """A live ``ocr_job_status`` row plus its routable ``batch_jobs`` ledger row."""
    from sax_platform.contracts.batch_jobs import batch_jobs

    from ocr.store import upsert_ocr_job_status

    upsert_ocr_job_status(
        engine,
        request_id=request_id,
        document_id=f"d-{request_id}",
        file_path=f"/{request_id}.pdf",
        status="submitted",
    )
    with engine.begin() as conn:
        conn.execute(
            sa.insert(batch_jobs).values(
                id=request_id,
                batch_id=batch_id,
                workflow_id=workflow_id,
                status="submitted",
                provider="mistral",
            )
        )


def _parse_report(output: str) -> dict[str, str]:
    """Parse ``key: value`` report lines into a dict."""
    report: dict[str, str] = {}
    for line in output.splitlines():
        if ": " in line:
            key, _, value = line.partition(": ")
            report[key] = value
    return report


class TestTrackerStatusCommand:
    def test_fresh_heartbeat_exit_zero_all_fields(
        self, cli_runner: CliRunner, store_engine: sa.Engine
    ) -> None:
        _create_batch_jobs(store_engine)
        _seed_heartbeat(store_engine, last_run_at=datetime.now(UTC), live_jobs=2, hints_sent=1)

        result = cli_runner.invoke(main, ["tracker-status"])

        assert result.exit_code == 0
        report = _parse_report(result.output)
        assert report["status"] == "fresh"
        assert report["cycles_total"] == "1"
        assert report["live_jobs_last_cycle"] == "2"
        assert report["hints_sent_last_cycle"] == "1"
        assert report["live_jobs_now"] == "0"
        for field in (
            "checked_at_gmt",
            "last_run_at",
            "heartbeat_age_seconds",
            "cycles_total",
            "live_jobs_last_cycle",
            "hints_sent_last_cycle",
            "live_jobs_now",
            "status",
        ):
            assert field in report

    def test_stale_heartbeat_no_live_jobs_exit_one(
        self, cli_runner: CliRunner, store_engine: sa.Engine
    ) -> None:
        _create_batch_jobs(store_engine)
        _seed_heartbeat(store_engine, last_run_at=datetime.now(UTC) - timedelta(seconds=1000))

        result = cli_runner.invoke(main, ["tracker-status"])

        assert result.exit_code == 1
        report = _parse_report(result.output)
        assert report["status"] == "stale"
        assert report["live_jobs_now"] == "0"

    def test_stale_heartbeat_with_live_job_exit_two(
        self, cli_runner: CliRunner, store_engine: sa.Engine
    ) -> None:
        _create_batch_jobs(store_engine)
        _seed_heartbeat(store_engine, last_run_at=datetime.now(UTC) - timedelta(seconds=1000))
        _seed_live_job(store_engine, request_id="r1", batch_id="batch-1", workflow_id="wf-1")

        result = cli_runner.invoke(main, ["tracker-status"])

        assert result.exit_code == 2
        report = _parse_report(result.output)
        assert report["status"] == "stale"
        assert report["live_jobs_now"] == "1"

    def test_no_heartbeat_row_never_ran_exit_one(
        self, cli_runner: CliRunner, store_engine: sa.Engine
    ) -> None:
        _create_batch_jobs(store_engine)  # no heartbeat seeded

        result = cli_runner.invoke(main, ["tracker-status"])

        assert result.exit_code == 1
        report = _parse_report(result.output)
        assert report["status"] == "never-ran"
        assert report["last_run_at"] == "none"
        assert report["heartbeat_age_seconds"] == "none"
        assert report["cycles_total"] == "none"

    def test_checked_at_gmt_always_first_and_aware_utc(
        self, cli_runner: CliRunner, store_engine: sa.Engine
    ) -> None:
        _create_batch_jobs(store_engine)
        _seed_heartbeat(store_engine, last_run_at=datetime.now(UTC))

        result = cli_runner.invoke(main, ["tracker-status"])

        lines = result.output.splitlines()
        assert lines[0].startswith("checked_at_gmt: ")
        parsed = datetime.fromisoformat(lines[0].split(": ", 1)[1])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)

    def test_unreachable_store_fails_probe_exit_three(self, cli_runner: CliRunner) -> None:
        """An unreachable store fails the probe: exit 3, ``status: error`` on stdout,
        the exception message on stderr, and ``checked_at_gmt`` still printed first."""
        with patch("sax_platform.db.get_store_engine", side_effect=RuntimeError("db down")):
            result = cli_runner.invoke(main, ["tracker-status"])

        assert result.exit_code == EXIT_PROBE_ERROR
        out_lines = result.stdout.splitlines()
        assert out_lines[0].startswith("checked_at_gmt: ")
        assert "status: error" in out_lines
        # The reason goes to stderr as the bare exception message (no "Error:" prefix).
        assert "db down" in result.stderr
        assert "status: error" not in result.stderr

    def test_missing_forge_db_url_fails_fast_exit_three(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No FORGE_DB_URL: config fail-fast — exit 3, ``status: error`` on stdout,
        an actionable FORGE_DB_URL one-liner on stderr, ``checked_at_gmt`` still first.

        The autouse ``forge_db_url`` fixture exported a tmp sqlite URL; delete it to
        simulate a shell that was never pointed at the shared forge database.
        """
        monkeypatch.delenv("FORGE_DB_URL", raising=False)

        result = cli_runner.invoke(main, ["tracker-status"])

        assert result.exit_code == EXIT_PROBE_ERROR
        out_lines = result.stdout.splitlines()
        assert out_lines[0].startswith("checked_at_gmt: ")
        assert "status: error" in out_lines
        assert "FORGE_DB_URL is not set" in result.stderr
        assert "forge.env" in result.stderr
        assert "5434" in result.stderr

    def test_rejects_non_positive_stale_after(
        self, cli_runner: CliRunner, store_engine: sa.Engine
    ) -> None:
        result = cli_runner.invoke(main, ["tracker-status", "--stale-after", "0"])
        assert result.exit_code != 0
        assert "stale-after" in result.output.lower()


# ---------------------------------------------------------------------------
# ST-G2 environment guard (T0.9)
# ---------------------------------------------------------------------------


class TestEnvGuard:
    """Every ocr command refuses to run without an explicitly declared FORGE_ENV.

    Since ``--env`` became position-independent the guard runs at the per-command
    seam (``_EnvCommand.invoke``), immediately before the command body — not in
    the group callback. So ``--help`` and other parse-only paths work env-less,
    while any actual command execution without a declared FORGE_ENV exits
    ``EXIT_CONFIG_ERROR`` (78) — outside the ``tracker-status`` 0/1/2/3 contract —
    with the guard's actionable message on stderr. For ``tracker-status`` a guard
    failure means the probe never ran, so no ``checked_at_gmt`` line is printed.
    ``FORGE_ENV=test`` comes from the autouse ``forge_env`` fixture; the failure
    cases override it.
    """

    def test_help_succeeds_without_env(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Re-pinned contract: help is a parse-only path, so it works env-less at
        # both the group and command levels (the guard no longer fires for it).
        monkeypatch.delenv("FORGE_ENV", raising=False)
        top = cli_runner.invoke(main, ["--help"])
        assert top.exit_code == 0
        sub = cli_runner.invoke(main, ["tracker-status", "--help"])
        assert sub.exit_code == 0
        assert "--env" in sub.output

    def test_missing_forge_env_exits_78_probe_never_runs(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FORGE_ENV", raising=False)
        result = cli_runner.invoke(main, ["tracker-status"])
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "no default environment" in result.stderr
        # A guard failure short-circuits before the probe body: no report line.
        assert "checked_at_gmt" not in result.output

    def test_invalid_forge_env_exits_78(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FORGE_ENV", "staging")
        # Real execution (not --help): the guard fires before the command body,
        # ahead of any Temporal connect.
        result = cli_runner.invoke(main, ["list"])
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "not a valid environment" in result.stderr

    def test_prod_without_ack_refused(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FORGE_ENV", "prod")
        monkeypatch.delenv("FORGE_ENV_TAG", raising=False)
        monkeypatch.delenv("FORGE_PROD_ACK", raising=False)
        result = cli_runner.invoke(main, ["list"])
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "explicit act" in result.stderr

    def test_test_env_proceeds_probe_runs(
        self, cli_runner: CliRunner, store_engine: sa.Engine
    ) -> None:
        # FORGE_ENV=test (autouse) passes the guard, so tracker-status runs to a
        # real verdict and prints checked_at_gmt first — never exit 78.
        _create_batch_jobs(store_engine)
        result = cli_runner.invoke(main, ["tracker-status"])
        assert result.exit_code != EXIT_CONFIG_ERROR
        assert result.output.splitlines()[0].startswith("checked_at_gmt: ")


# ---------------------------------------------------------------------------
# --env profile flag (T0.9 follow-up)
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_environ() -> Iterator[None]:
    """Snapshot ``os.environ`` and restore it on teardown.

    ``--env`` mutates the real process environment (that is the whole feature),
    and those writes are not tracked by ``monkeypatch``, so they would leak
    between tests. This fixture restores a full snapshot afterward.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


class TestEnvProfileFlag:
    """``ocr --env NAME|PATH`` loads a profile before the guard runs.

    It applies the parsed KEY=VALUE pairs to the process environment (overwriting
    ambient values), declares FORGE_ENV, then the guard runs unchanged. It never
    sets FORGE_PROD_ACK — ``--env prod`` still fails without a separately-exported
    ack. All cases use ``restore_environ`` so the direct ``os.environ`` writes
    don't leak.
    """

    def test_path_profile_applies_vars_and_proceeds(
        self, cli_runner: CliRunner, tmp_path: Path, restore_environ: None
    ) -> None:
        profile = tmp_path / "dev.env"
        profile.write_text('export FORGE_ENV_TAG="dev"\nFORGE_DB_URL=sqlite:///from-profile.db\n')

        result = cli_runner.invoke(main, ["--env", str(profile), "list", "--help"])

        assert result.exit_code == 0
        assert os.environ["FORGE_DB_URL"] == "sqlite:///from-profile.db"
        assert os.environ["FORGE_ENV"] == "dev"

    def test_name_resolves_under_xdg_config_home(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        restore_environ: None,
    ) -> None:
        envs = tmp_path / "config" / "forge" / "envs"
        envs.mkdir(parents=True)
        (envs / "dev.env").write_text("FORGE_ENV_TAG=dev\nFORGE_DB_URL=sqlite:///named.db\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = cli_runner.invoke(main, ["--env", "dev", "list", "--help"])

        assert result.exit_code == 0
        assert os.environ["FORGE_DB_URL"] == "sqlite:///named.db"
        assert os.environ["FORGE_ENV"] == "dev"

    def test_profile_overrides_ambient_var(
        self, cli_runner: CliRunner, tmp_path: Path, restore_environ: None
    ) -> None:
        os.environ["FORGE_DB_URL"] = "sqlite:///ambient.db"
        profile = tmp_path / "dev.env"
        profile.write_text("FORGE_ENV_TAG=dev\nFORGE_DB_URL=sqlite:///override.db\n")

        result = cli_runner.invoke(main, ["--env", str(profile), "list", "--help"])

        assert result.exit_code == 0
        assert os.environ["FORGE_DB_URL"] == "sqlite:///override.db"

    def test_tag_mismatch_exits_78(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        restore_environ: None,
    ) -> None:
        envs = tmp_path / "config" / "forge" / "envs"
        envs.mkdir(parents=True)
        (envs / "dev.env").write_text("FORGE_ENV_TAG=prod\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        # Real execution (not --help), so the per-command guard actually runs.
        result = cli_runner.invoke(main, ["--env", "dev", "list"])

        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "does not match" in result.stderr

    def test_env_prod_still_requires_ack_both_positions(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        restore_environ: None,
    ) -> None:
        # The non-bypass proof, in BOTH positions: --env prod loads a prod-tagged
        # profile but never supplies FORGE_PROD_ACK, so the guard still refuses
        # whether --env sits before or after the subcommand.
        envs = tmp_path / "config" / "forge" / "envs"
        envs.mkdir(parents=True)
        (envs / "prod.env").write_text("FORGE_ENV_TAG=prod\nFORGE_DB_URL=sqlite:///prod.db\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.delenv("FORGE_PROD_ACK", raising=False)

        for argv in (["--env", "prod", "list"], ["list", "--env", "prod"]):
            result = cli_runner.invoke(main, argv)
            assert result.exit_code == EXIT_CONFIG_ERROR, argv
            assert "explicit act" in result.stderr, argv

    def test_applies_in_subcommand_position(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        restore_environ: None,
    ) -> None:
        # --env after the subcommand is parsed by the command (not the group), so
        # its profile is applied and guarded there. A tag mismatch proves it took
        # effect: without the subcommand-position apply the autouse FORGE_ENV=test
        # would pass the guard and the probe would print checked_at_gmt.
        envs = tmp_path / "config" / "forge" / "envs"
        envs.mkdir(parents=True)
        (envs / "dev.env").write_text("FORGE_ENV_TAG=prod\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = cli_runner.invoke(main, ["tracker-status", "--env", "dev"])

        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "does not match" in result.stderr
        assert "checked_at_gmt" not in result.output

    def test_subcommand_env_wins_over_group_env(
        self, cli_runner: CliRunner, tmp_path: Path, restore_environ: None
    ) -> None:
        # --env at both levels: the command-level profile applies last, so its
        # value wins. Both are dev-tagged so the guard passes; the applied
        # FORGE_DB_URL reflects the winner regardless of the probe's own verdict.
        group_profile = tmp_path / "group.env"
        group_profile.write_text(f"FORGE_ENV_TAG=dev\nFORGE_DB_URL=sqlite:///{tmp_path}/group.db\n")
        cmd_profile = tmp_path / "cmd.env"
        cmd_profile.write_text(f"FORGE_ENV_TAG=dev\nFORGE_DB_URL=sqlite:///{tmp_path}/cmd.db\n")

        result = cli_runner.invoke(
            main, ["--env", str(group_profile), "tracker-status", "--env", str(cmd_profile)]
        )

        assert result.exit_code != EXIT_CONFIG_ERROR
        assert os.environ["FORGE_DB_URL"] == f"sqlite:///{tmp_path}/cmd.db"

    def test_missing_file_exits_78(
        self, cli_runner: CliRunner, tmp_path: Path, restore_environ: None
    ) -> None:
        missing = tmp_path / "nope.env"
        # apply-profile fails inside the group callback, before any subcommand.
        result = cli_runner.invoke(main, ["--env", str(missing), "list"])
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert str(missing) in result.stderr

    def test_path_profile_without_tag_exits_78(
        self, cli_runner: CliRunner, tmp_path: Path, restore_environ: None
    ) -> None:
        profile = tmp_path / "notag.env"
        profile.write_text("FORGE_DB_URL=sqlite:///x.db\n")
        result = cli_runner.invoke(main, ["--env", str(profile), "list"])
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "FORGE_ENV_TAG" in result.stderr


# ---------------------------------------------------------------------------
# Staging-lane isolation: --env dev threads its namespace into the connect,
# and a dev env without a declared namespace is refused before connecting.
# ---------------------------------------------------------------------------


class TestNamespaceCoherence:
    """A ``--env dev`` profile carries its Temporal namespace into every connect."""

    def test_dev_profile_derives_the_dev_namespace(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        restore_environ: None,
    ) -> None:
        # The profile declares only the environment: a connecting command (list)
        # must derive forge-dev and the dev server from it.
        monkeypatch.delenv("FORGE_TEMPORAL_ADDRESS", raising=False)
        profile = tmp_path / "dev.env"
        profile.write_text("FORGE_ENV_TAG=dev\nFORGE_DB_URL=sqlite:///x.db\n")

        mock_handle = AsyncMock()
        mock_handle.result.return_value = {"jobs": []}
        mock_client = AsyncMock()
        mock_client.start_workflow.return_value = mock_handle

        with patch(
            "ocr.cli.connect_temporal", new=AsyncMock(return_value=mock_client)
        ) as mock_connect:
            result = cli_runner.invoke(main, ["--env", str(profile), "list"])

        assert result.exit_code == 0
        mock_connect.assert_awaited_once()
        assert mock_connect.await_args.kwargs["namespace"] == "forge-dev"
        assert mock_connect.await_args.args[0] == "127.0.0.1:7236"

    def test_dev_pointed_at_the_prod_server_refuses_to_connect(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, restore_environ: None
    ) -> None:
        # The environment is coherent; the *server* is production's. The old
        # namespace-name check never read the address and could not see this.
        monkeypatch.setenv("FORGE_ENV", "dev")
        monkeypatch.setenv("FORGE_TEMPORAL_ADDRESS", "127.0.0.1:7243")

        with patch("ocr.cli.connect_temporal", new=AsyncMock()) as mock_connect:
            result = cli_runner.invoke(main, ["list"])

        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "FORGE_TEMPORAL_ADDRESS" in result.stderr
        mock_connect.assert_not_awaited()

    def test_direct_db_commands_unaffected_by_namespace(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, restore_environ: None
    ) -> None:
        # tracker-status and migrate never connect to Temporal, so a dev env with
        # no namespace (which would fail a Temporal command) leaves them alone.
        monkeypatch.setenv("FORGE_ENV", "dev")
        monkeypatch.delenv("FORGE_TEMPORAL_NAMESPACE", raising=False)

        status_result = cli_runner.invoke(main, ["tracker-status"])
        migrate_result = cli_runner.invoke(main, ["migrate"])

        assert status_result.exit_code != EXIT_CONFIG_ERROR
        assert migrate_result.exit_code != EXIT_CONFIG_ERROR


# ---------------------------------------------------------------------------
# migrate — pure target-line formatter (functional core)
# ---------------------------------------------------------------------------


class TestFormatMigrationTarget:
    def test_postgres_url_hides_credentials(self) -> None:
        line = format_migration_target("postgresql+psycopg2://u:secretpw@db.host:5432/forge")
        assert "secretpw" not in line
        assert "u:" not in line
        assert "db.host:5432/forge" in line
        assert line.startswith("alembic_version_ocr -> ")

    def test_sqlite_url_shows_file_path(self) -> None:
        line = format_migration_target("sqlite:////var/data/ocr.db")
        assert line == "alembic_version_ocr -> /var/data/ocr.db"

    def test_postgres_without_port(self) -> None:
        line = format_migration_target("postgresql://u:p@host/forge")
        assert line == "alembic_version_ocr -> host/forge"


# ---------------------------------------------------------------------------
# migrate — CLI shell
# ---------------------------------------------------------------------------


class TestMigrateCommand:
    def test_creates_tables_on_sqlite(self, cli_runner: CliRunner, forge_db_url: str) -> None:
        result = cli_runner.invoke(main, ["migrate"])

        assert result.exit_code == 0
        engine = sa.create_engine(forge_db_url)
        try:
            table_names = sa.inspect(engine).get_table_names()
        finally:
            engine.dispose()
        assert "ocr_tracker_heartbeat" in table_names

    def test_prints_credential_free_target_line(
        self, cli_runner: CliRunner, forge_db_url: str
    ) -> None:
        result = cli_runner.invoke(main, ["migrate"])

        assert result.exit_code == 0
        last_line = result.output.strip().splitlines()[-1]
        assert last_line.startswith("alembic_version_ocr -> ")
        # The autouse forge_db_url is a sqlite file URL — no credentials to leak,
        # but assert the chain name and that no user:pass fragment appears.
        assert "://" not in last_line

    def test_missing_forge_db_url_exits_78(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FORGE_DB_URL", raising=False)
        result = cli_runner.invoke(main, ["migrate"])
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "FORGE_DB_URL is not set" in result.stderr


# ---------------------------------------------------------------------------
# db-change — the production apply path's front door
# ---------------------------------------------------------------------------


class TestDbChangeCommand:
    """``ocr db-change`` generates a sax-datastores request for the ocr chain.

    The generator is tested in ``libs/sax-platform``; these cover the wiring
    that is specific to ocr — the chain it reads, the product it files under
    (``forge``: ocr is not a registered product and its tables live in forge's
    database), and that it is outside the ``FORGE_ENV`` guard.
    """

    def test_generates_a_request_for_the_ocr_chain(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        result = cli_runner.invoke(
            main,
            [
                "db-change",
                "--from",
                "001",
                "--to",
                "002",
                "--title",
                "tracker-heartbeat",
                "--output-root",
                str(tmp_path),
                "--no-lint",
            ],
        )

        assert result.exit_code == 0, result.output
        directory = tmp_path / "0001-tracker-heartbeat"
        assert "CREATE TABLE ocr_tracker_heartbeat" in (directory / "change-1.sql").read_text()
        request = (directory / "request.md").read_text()
        # ocr files under forge's product and database; only the version table
        # distinguishes the chain.
        assert "| Product | `forge` |" in request
        assert "| Database | `forge_prod` (prod) |" in request
        assert "| Version table | `alembic_version_ocr` |" in request

    def test_an_unusable_range_exits_1_and_writes_nothing(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        result = cli_runner.invoke(
            main,
            [
                "db-change",
                "--from",
                "nope",
                "--title",
                "doomed",
                "--output-root",
                str(tmp_path),
                "--no-lint",
            ],
        )

        assert result.exit_code == 1
        assert "Cannot walk nope" in result.stderr
        assert list(tmp_path.iterdir()) == []

    def test_runs_without_a_declared_environment(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It opens no database, so the FORGE_ENV guard does not apply to it."""
        monkeypatch.delenv("FORGE_ENV", raising=False)

        result = cli_runner.invoke(
            main,
            [
                "db-change",
                "--from",
                "001",
                "--title",
                "no-env-needed",
                "--output-root",
                str(tmp_path),
                "--no-lint",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "0001-no-env-needed" / "change-1.sql").exists()
