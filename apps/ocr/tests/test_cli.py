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
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner
from sax_platform.contracts.constants import OCR_TASK_QUEUE

from ocr.cli import EXIT_INFRASTRUCTURE_ERROR, _auto_id, _echo, _start_and_wait, _start_submit, main
from ocr.models import (
    OcrExportResult,
    OcrListJobsResult,
    OcrMarkResult,
)

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
        mock_connect.assert_awaited_once_with("localhost:7233")
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
        mock_connect.assert_awaited_once_with("localhost:7233")
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
    def test_runs_worker_with_default_address(self, cli_runner: CliRunner) -> None:
        with patch("ocr.worker.run_worker") as mock_run:
            mock_run.side_effect = _async_result(None)
            result = cli_runner.invoke(main, ["worker"])
            assert result.exit_code == 0
            mock_run.assert_called_once_with("localhost:7233")

    def test_runs_worker_with_custom_address(self, cli_runner: CliRunner) -> None:
        with patch("ocr.worker.run_worker") as mock_run:
            mock_run.side_effect = _async_result(None)
            result = cli_runner.invoke(
                main, ["worker", "--temporal-address", "temporal.internal:7233"]
            )
            assert result.exit_code == 0
            mock_run.assert_called_once_with("temporal.internal:7233")


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
        assert call_kwargs["address"] == "localhost:7233"

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
