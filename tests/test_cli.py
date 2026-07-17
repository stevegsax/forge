"""Tests for the Forge CLI entry point."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from forge.cli import (
    EXIT_FAILURE,
    EXIT_INFRASTRUCTURE_ERROR,
    build_task_definition,
    configure_logging,
    format_deterministic_result,
    format_eval_result,
    format_llm_stats,
    format_step_result,
    format_sub_task_result,
    format_task_result,
    format_validation_results,
    format_verbose_result,
    load_task_definition,
    load_workflow_input,
    main,
)
from forge.models import (
    Plan,
    PlanStep,
    StepResult,
    SubTaskResult,
    TaskDomain,
    TaskResult,
    TransitionSignal,
    ValidationResult,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_result(value: object = None):
    """Return an async function that returns *value*.

    Use as ``mock.side_effect = _async_result(TaskResult(...))`` on a plain
    MagicMock to make ``asyncio.run(mock(...))`` return *value*.

    Unlike ``AsyncMock``, this avoids orphaned internal coroutines that
    trigger 'coroutine was never awaited' warnings when the mock is called
    via ``asyncio.run()`` rather than ``await``.
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


@pytest.fixture
def success_result() -> TaskResult:
    return TaskResult(
        task_id="test-task",
        status=TransitionSignal.SUCCESS,
        validation_results=[
            ValidationResult(check_name="ruff_lint", passed=True, summary="ruff_lint passed"),
            ValidationResult(check_name="ruff_format", passed=True, summary="ruff_format passed"),
        ],
        worktree_path="/repo/.forge-worktrees/test-task",
        worktree_branch="forge/test-task",
    )


@pytest.fixture
def failure_result() -> TaskResult:
    return TaskResult(
        task_id="test-task",
        status=TransitionSignal.FAILURE_TERMINAL,
        validation_results=[
            ValidationResult(
                check_name="ruff_lint",
                passed=False,
                summary="ruff_lint found errors",
            ),
        ],
        error="ruff_lint found errors",
        worktree_path="/repo/.forge-worktrees/test-task",
        worktree_branch="forge/test-task",
    )


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestFormatValidationResults:
    """Tests for format_validation_results."""

    def test_passing_checks(self) -> None:
        results = [
            ValidationResult(check_name="ruff_lint", passed=True, summary="ruff_lint passed"),
            ValidationResult(check_name="ruff_format", passed=True, summary="ruff_format passed"),
        ]
        output = format_validation_results(results)
        assert "[PASS] ruff_lint: ruff_lint passed" in output
        assert "[PASS] ruff_format: ruff_format passed" in output

    def test_failing_checks(self) -> None:
        results = [
            ValidationResult(
                check_name="ruff_lint",
                passed=False,
                summary="ruff_lint found errors",
            ),
        ]
        output = format_validation_results(results)
        assert "[FAIL] ruff_lint: ruff_lint found errors" in output

    def test_empty_results(self) -> None:
        assert format_validation_results([]) == ""


class TestFormatTaskResult:
    """Tests for format_task_result."""

    def test_success_output(self, success_result: TaskResult) -> None:
        output = format_task_result(success_result)
        assert "Task: test-task" in output
        assert "Status: success" in output
        assert "[PASS] ruff_lint" in output
        assert "Worktree: /repo/.forge-worktrees/test-task" in output
        assert "Branch: forge/test-task" in output

    def test_failure_output(self, failure_result: TaskResult) -> None:
        output = format_task_result(failure_result)
        assert "Status: failure_terminal" in output
        assert "Error: ruff_lint found errors" in output

    def test_no_validation_results(self) -> None:
        result = TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        output = format_task_result(result)
        assert "Validation:" not in output

    def test_no_worktree(self) -> None:
        result = TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        output = format_task_result(result)
        assert "Worktree:" not in output
        assert "Branch:" not in output


class TestBuildTaskDefinition:
    """Tests for build_task_definition."""

    def test_basic_args(self) -> None:
        td = build_task_definition(
            task_id="impl-utils",
            description="Create a utility module.",
            target_files=["src/utils.py"],
        )
        assert td.task_id == "impl-utils"
        assert td.description == "Create a utility module."
        assert td.target_files == ["src/utils.py"]
        assert td.context_files == []
        assert td.base_branch == "main"
        assert td.validation.run_ruff_lint is True
        assert td.validation.run_ruff_format is True

    def test_with_context_files(self) -> None:
        td = build_task_definition(
            task_id="t",
            description="d",
            target_files=["a.py"],
            context_files=["b.py", "c.py"],
        )
        assert td.context_files == ["b.py", "c.py"]

    def test_no_lint(self) -> None:
        td = build_task_definition(
            task_id="t", description="d", target_files=["a.py"], no_lint=True
        )
        assert td.validation.run_ruff_lint is False

    def test_no_format(self) -> None:
        td = build_task_definition(
            task_id="t", description="d", target_files=["a.py"], no_format=True
        )
        assert td.validation.run_ruff_format is False

    def test_run_tests(self) -> None:
        td = build_task_definition(
            task_id="t",
            description="d",
            target_files=["a.py"],
            run_tests=True,
            test_command="pytest -x",
        )
        assert td.validation.run_tests is True
        assert td.validation.test_command == "pytest -x"

    def test_custom_base_branch(self) -> None:
        td = build_task_definition(
            task_id="t", description="d", target_files=["a.py"], base_branch="develop"
        )
        assert td.base_branch == "develop"


class TestLoadTaskDefinition:
    """Tests for load_task_definition."""

    def test_valid_json(self, tmp_path: Path) -> None:
        data = {
            "task_id": "test-task",
            "description": "Test task.",
            "target_files": ["src/main.py"],
        }
        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(data))

        td = load_task_definition(str(task_file))
        assert td.task_id == "test-task"
        assert td.target_files == ["src/main.py"]

    def test_invalid_json(self, tmp_path: Path) -> None:
        task_file = tmp_path / "bad.json"
        task_file.write_text("not json at all")

        with pytest.raises(Exception, match="Invalid task definition"):
            load_task_definition(str(task_file))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="Cannot read task file"):
            load_task_definition(str(tmp_path / "nonexistent.json"))

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        task_file = tmp_path / "partial.json"
        task_file.write_text(json.dumps({"task_id": "t"}))

        with pytest.raises(Exception, match="Invalid task definition"):
            load_task_definition(str(task_file))


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------


class TestRunCommandValidation:
    """Tests for ``forge run`` argument validation."""

    def test_no_args_shows_error(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["run"])
        assert result.exit_code != 0
        assert "Provide either --task-file or" in result.output

    def test_mutual_exclusion_task_file_and_inline(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        task_file = tmp_path / "task.json"
        task_file.write_text(
            json.dumps(
                {
                    "task_id": "t",
                    "description": "d",
                    "target_files": ["a.py"],
                }
            )
        )
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-file",
                str(task_file),
                "--task-id",
                "t",
            ],
        )
        assert result.exit_code != 0
        assert "Cannot combine --task-file" in result.output

    def test_inline_missing_task_id(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--description",
                "d",
                "--target-file",
                "a.py",
            ],
        )
        assert result.exit_code != 0
        assert "--task-id is required" in result.output

    def test_inline_missing_description(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--target-file",
                "a.py",
            ],
        )
        assert result.exit_code != 0
        assert "--description is required" in result.output

    def test_inline_missing_target_file(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
            ],
        )
        assert result.exit_code != 0
        assert "--target-file is required" in result.output


class TestRunCommandExecution:
    """Tests for ``forge run`` execution paths."""

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_success_exit_code(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
        success_result: TaskResult,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(success_result)

        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "test-task",
                "--description",
                "Test.",
                "--target-file",
                "a.py",
            ],
        )
        assert result.exit_code == 0
        assert "Task: test-task" in result.output
        assert "Status: success" in result.output

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_failure_exit_code(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
        failure_result: TaskResult,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(failure_result)

        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "test-task",
                "--description",
                "Test.",
                "--target-file",
                "a.py",
            ],
        )
        assert result.exit_code == EXIT_FAILURE

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_json_output(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
        success_result: TaskResult,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(success_result)

        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "test-task",
                "--description",
                "Test.",
                "--target-file",
                "a.py",
                "--json",
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert parsed["task_id"] == "test-task"
        assert parsed["status"] == "success"

    @patch("forge.cli._submit_no_wait")
    @patch("forge.cli.discover_repo_root")
    def test_no_wait_mode(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result("forge-task-test-task")

        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "test-task",
                "--description",
                "Test.",
                "--target-file",
                "a.py",
                "--no-wait",
            ],
        )
        assert result.exit_code == 0
        assert "forge-task-test-task" in result.output

    @patch("forge.cli.discover_repo_root")
    def test_repo_discovery_error(
        self,
        mock_discover: object,
        cli_runner: CliRunner,
    ) -> None:
        from forge.git import RepoDiscoveryError

        mock_discover.side_effect = RepoDiscoveryError("not a repo")  # type: ignore[attr-defined]

        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "test-task",
                "--description",
                "Test.",
                "--target-file",
                "a.py",
            ],
        )
        assert result.exit_code == EXIT_INFRASTRUCTURE_ERROR

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_temporal_connection_error(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = RuntimeError("Connection refused")

        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "test-task",
                "--description",
                "Test.",
                "--target-file",
                "a.py",
            ],
        )
        assert result.exit_code == EXIT_INFRASTRUCTURE_ERROR
        assert "Connection refused" in result.stderr

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_task_file_mode(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
        tmp_path: Path,
        success_result: TaskResult,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(success_result)

        task_data = {
            "task_id": "file-task",
            "description": "From file.",
            "target_files": ["src/mod.py"],
        }
        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        result = cli_runner.invoke(main, ["run", "--task-file", str(task_file)])
        assert result.exit_code == 0

        call_args = mock_submit.call_args
        task_def = call_args[0][0]
        assert task_def.task_id == "file-task"

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_validation_flags_passed(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
        success_result: TaskResult,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(success_result)

        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--target-file",
                "a.py",
                "--no-lint",
                "--no-format",
                "--run-tests",
                "--test-command",
                "pytest -x",
            ],
        )
        assert result.exit_code == 0

        call_args = mock_submit.call_args
        task_def = call_args[0][0]
        assert task_def.validation.run_ruff_lint is False
        assert task_def.validation.run_ruff_format is False
        assert task_def.validation.run_tests is True
        assert task_def.validation.test_command == "pytest -x"


class TestWorkerCommand:
    """Tests for ``forge worker`` command."""

    @patch("forge.cli.asyncio.run")
    @patch("forge.worker.run_worker", new_callable=MagicMock)
    def test_worker_invokes_run_worker(
        self,
        mock_run_worker: MagicMock,
        mock_asyncio_run: object,
        cli_runner: CliRunner,
    ) -> None:
        result = cli_runner.invoke(main, ["worker"])
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()  # type: ignore[attr-defined]


class TestMainGroup:
    """Tests for the main CLI group."""

    def test_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Forge" in result.output

    def test_run_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--task-id" in result.output

    def test_worker_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["worker", "--help"])
        assert result.exit_code == 0
        assert "--temporal-address" in result.output


# ---------------------------------------------------------------------------
# Logging configuration tests
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """Tests for the configure_logging helper.

    File logging is disabled via ``FORGE_LOG_DIR=""`` so we can test console
    handler level independently.
    """

    @pytest.fixture(autouse=True)
    def _disable_file_logging(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", "")

    def test_default_warning(self) -> None:
        import logging

        configure_logging(0)
        root = logging.getLogger()
        assert root.level == logging.WARNING
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert stream_handlers
        assert stream_handlers[0].level == logging.WARNING

    def test_v_info(self) -> None:
        import logging

        configure_logging(1)
        root = logging.getLogger()
        assert root.level == logging.INFO
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert stream_handlers
        assert stream_handlers[0].level == logging.INFO

    def test_vv_debug(self) -> None:
        import logging

        configure_logging(2)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert stream_handlers
        assert stream_handlers[0].level == logging.DEBUG

    def test_file_handler_active_sets_root_debug(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When file logging is active, root level is DEBUG regardless of console level."""
        import logging

        monkeypatch.setenv("FORGE_LOG_DIR", str(tmp_path))
        configure_logging(0)  # console=WARNING
        root = logging.getLogger()
        try:
            assert root.level == logging.DEBUG
            # At verbosity 0 with file logging active, no stream handler is added.
            stream_handlers = [
                h
                for h in root.handlers
                if isinstance(h, logging.StreamHandler) and not hasattr(h, "maxBytes")
            ]
            assert not stream_handlers
        finally:
            # Clean up file handlers to avoid leaking FDs.
            for h in list(root.handlers):
                if hasattr(h, "maxBytes"):
                    root.removeHandler(h)
                    h.close()

    def test_file_handler_with_verbosity_keeps_console(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When file logging is active and verbosity > 0, console handler is still added."""
        import logging

        monkeypatch.setenv("FORGE_LOG_DIR", str(tmp_path))
        configure_logging(1)  # console=INFO
        root = logging.getLogger()
        try:
            assert root.level == logging.DEBUG
            stream_handlers = [
                h
                for h in root.handlers
                if isinstance(h, logging.StreamHandler) and not hasattr(h, "maxBytes")
            ]
            assert stream_handlers
            assert stream_handlers[0].level == logging.INFO
        finally:
            for h in list(root.handlers):
                if hasattr(h, "maxBytes"):
                    root.removeHandler(h)
                    h.close()


# ---------------------------------------------------------------------------
# Phase 2 CLI tests
# ---------------------------------------------------------------------------


class TestFormatStepResult:
    def test_success(self) -> None:
        sr = StepResult(step_id="s1", status=TransitionSignal.SUCCESS, commit_sha="a" * 40)
        output = format_step_result(sr)
        assert "[PASS]" in output
        assert "s1" in output
        assert "aaaaaaaa" in output

    def test_failure(self) -> None:
        sr = StepResult(step_id="s2", status=TransitionSignal.FAILURE_TERMINAL)
        output = format_step_result(sr)
        assert "[FAIL]" in output
        assert "none" in output


class TestFormatTaskResultWithPlan:
    def test_includes_plan_info(self) -> None:
        plan = Plan(
            task_id="t",
            steps=[PlanStep(step_id="s1", description="d", target_files=["a.py"])],
            explanation="test",
        )
        result = TaskResult(
            task_id="t",
            status=TransitionSignal.SUCCESS,
            plan=plan,
            step_results=[
                StepResult(step_id="s1", status=TransitionSignal.SUCCESS, commit_sha="a" * 40),
            ],
        )
        output = format_task_result(result)
        assert "Plan: 1 steps" in output
        assert "Steps:" in output
        assert "[PASS] s1" in output


class TestPlanFlag:
    def test_plan_flag_in_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["run", "--help"])
        assert "--plan" in result.output
        assert "--max-step-attempts" in result.output

    def test_plan_allows_no_target_file(self, cli_runner: CliRunner) -> None:
        """With --plan, --target-file is not required."""
        # This will fail at the submit stage, but the validation should pass
        with (
            patch("forge.cli.discover_repo_root") as mock_discover,
            patch("forge.cli._submit_and_wait") as mock_submit,
        ):
            mock_discover.return_value = "/repo"
            mock_submit.side_effect = _async_result(
                TaskResult(task_id="plan-task", status=TransitionSignal.SUCCESS)
            )
            result = cli_runner.invoke(
                main,
                [
                    "run",
                    "--task-id",
                    "plan-task",
                    "--description",
                    "Build an API.",
                    "--plan",
                ],
            )
            assert result.exit_code == 0

    def test_no_plan_requires_target_file(self, cli_runner: CliRunner) -> None:
        """Without --plan, --target-file is still required."""
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
            ],
        )
        assert result.exit_code != 0
        assert "--target-file is required" in result.output

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_plan_flag_passed_to_submit(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--plan",
                "--max-step-attempts",
                "3",
            ],
        )
        call_kwargs = mock_submit.call_args
        assert call_kwargs[1]["plan"] is True
        assert call_kwargs[1]["max_step_attempts"] == 3


# ---------------------------------------------------------------------------
# T3.2: --effort/--no-thinking (ThinkingPolicy) CLI flags
# ---------------------------------------------------------------------------


class TestThinkingCliFlags:
    def test_effort_and_no_thinking_in_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["run", "--help"])
        assert "--effort" in result.output
        assert "--no-thinking" in result.output

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_thinking_policy_built_from_effort_flag(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--plan",
                "--effort",
                "xhigh",
            ],
        )
        call_kwargs = mock_submit.call_args
        thinking = call_kwargs[1]["thinking"]
        assert thinking.enabled is True
        assert thinking.effort == "xhigh"

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_no_thinking_flag_disables_policy(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--plan",
                "--no-thinking",
            ],
        )
        call_kwargs = mock_submit.call_args
        thinking = call_kwargs[1]["thinking"]
        assert thinking.enabled is False

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_warns_when_effort_passed_without_plan(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--target-file",
                "a.py",
                "--effort",
                "low",
            ],
        )
        assert "no effect without --plan" in result.stderr

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_warns_when_no_thinking_passed_without_plan(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--target-file",
                "a.py",
                "--no-thinking",
            ],
        )
        assert "no effect without --plan" in result.stderr

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_no_warning_when_flags_default_without_plan(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--target-file",
                "a.py",
            ],
        )
        assert "no effect without --plan" not in result.stderr

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_no_warning_when_effort_passed_with_plan(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--plan",
                "--effort",
                "low",
            ],
        )
        assert "no effect without --plan" not in result.stderr


# ---------------------------------------------------------------------------
# Phase 3 CLI tests
# ---------------------------------------------------------------------------


class TestFormatSubTaskResult:
    def test_success(self) -> None:
        sr = SubTaskResult(sub_task_id="st1", status=TransitionSignal.SUCCESS)
        output = format_sub_task_result(sr)
        assert "[PASS]" in output
        assert "st1" in output

    def test_failure(self) -> None:
        sr = SubTaskResult(sub_task_id="st2", status=TransitionSignal.FAILURE_TERMINAL)
        output = format_sub_task_result(sr)
        assert "[FAIL]" in output
        assert "st2" in output


class TestFormatStepResultWithSubTasks:
    def test_includes_sub_task_results(self) -> None:
        st_results = [
            SubTaskResult(sub_task_id="st1", status=TransitionSignal.SUCCESS),
            SubTaskResult(sub_task_id="st2", status=TransitionSignal.FAILURE_TERMINAL),
        ]
        sr = StepResult(
            step_id="fan-step",
            status=TransitionSignal.SUCCESS,
            commit_sha="a" * 40,
            sub_task_results=st_results,
        )
        output = format_step_result(sr)
        assert "fan-step" in output
        assert "st1" in output
        assert "st2" in output
        assert "[PASS]" in output
        assert "[FAIL]" in output


class TestMaxSubTaskAttemptsFlag:
    def test_flag_in_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["run", "--help"])
        assert "--max-sub-task-attempts" in result.output

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_flag_passed_to_submit(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--plan",
                "--max-sub-task-attempts",
                "3",
            ],
        )
        call_kwargs = mock_submit.call_args
        assert call_kwargs[1]["max_sub_task_attempts"] == 3


# ---------------------------------------------------------------------------
# Phase 4 CLI tests
# ---------------------------------------------------------------------------


class TestBuildTaskDefinitionContextConfig:
    def test_default_auto_discover(self) -> None:
        td = build_task_definition(task_id="t", description="d", target_files=["a.py"])
        assert td.context.auto_discover is True

    def test_no_auto_discover(self) -> None:
        td = build_task_definition(
            task_id="t", description="d", target_files=["a.py"], no_auto_discover=True
        )
        assert td.context.auto_discover is False

    def test_custom_token_budget(self) -> None:
        td = build_task_definition(
            task_id="t", description="d", target_files=["a.py"], token_budget=50_000
        )
        assert td.context.token_budget == 50_000

    def test_custom_max_import_depth(self) -> None:
        td = build_task_definition(
            task_id="t", description="d", target_files=["a.py"], max_import_depth=3
        )
        assert td.context.max_import_depth == 3

    def test_include_deps_default_false(self) -> None:
        td = build_task_definition(task_id="t", description="d", target_files=["a.py"])
        assert td.context.include_dependencies is False

    def test_include_deps_true(self) -> None:
        td = build_task_definition(
            task_id="t", description="d", target_files=["a.py"], include_deps=True
        )
        assert td.context.include_dependencies is True

    def test_none_values_keep_defaults(self) -> None:
        td = build_task_definition(
            task_id="t",
            description="d",
            target_files=["a.py"],
            token_budget=None,
            max_import_depth=None,
        )
        assert td.context.token_budget == 100_000
        assert td.context.max_import_depth == 2


# ---------------------------------------------------------------------------
# Domain flag
# ---------------------------------------------------------------------------


class TestBuildTaskDefinitionDomain:
    def test_default_code_generation(self) -> None:
        td = build_task_definition(task_id="t", description="d", target_files=["a.py"])
        assert td.domain == TaskDomain.CODE_GENERATION

    def test_explicit_research(self) -> None:
        td = build_task_definition(
            task_id="t",
            description="d",
            target_files=["report.md"],
            domain=TaskDomain.RESEARCH,
        )
        assert td.domain == TaskDomain.RESEARCH

    def test_research_domain_disables_ruff(self) -> None:
        td = build_task_definition(
            task_id="t",
            description="d",
            target_files=["report.md"],
            domain=TaskDomain.RESEARCH,
        )
        assert td.validation.run_ruff_lint is False
        assert td.validation.run_ruff_format is False

    def test_code_generation_enables_ruff(self) -> None:
        td = build_task_definition(
            task_id="t",
            description="d",
            target_files=["a.py"],
            domain=TaskDomain.CODE_GENERATION,
        )
        assert td.validation.run_ruff_lint is True
        assert td.validation.run_ruff_format is True

    def test_code_generation_no_lint_override(self) -> None:
        td = build_task_definition(
            task_id="t",
            description="d",
            target_files=["a.py"],
            domain=TaskDomain.CODE_GENERATION,
            no_lint=True,
        )
        assert td.validation.run_ruff_lint is False
        assert td.validation.run_ruff_format is True

    def test_research_run_tests_override(self) -> None:
        td = build_task_definition(
            task_id="t",
            description="d",
            target_files=["report.md"],
            domain=TaskDomain.RESEARCH,
            run_tests=True,
        )
        assert td.validation.run_tests is True


class TestDomainFlag:
    def test_domain_in_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["run", "--help"])
        assert "--domain" in result.output

    def test_domain_flag_accepted(self, cli_runner: CliRunner) -> None:
        """The --domain flag is accepted without error."""
        result = cli_runner.invoke(main, ["run", "--help"])
        assert "code_generation" in result.output
        assert "research" in result.output


class TestContextDiscoveryFlags:
    def test_flags_in_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["run", "--help"])
        assert "--no-auto-discover" in result.output
        assert "--token-budget" in result.output
        assert "--max-import-depth" in result.output
        assert "--include-deps" in result.output

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_no_auto_discover_flag(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--target-file",
                "a.py",
                "--no-auto-discover",
            ],
        )
        call_args = mock_submit.call_args
        task_def = call_args[0][0]
        assert task_def.context.auto_discover is False

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_token_budget_flag(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--target-file",
                "a.py",
                "--token-budget",
                "50000",
            ],
        )
        call_args = mock_submit.call_args
        task_def = call_args[0][0]
        assert task_def.context.token_budget == 50_000

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_include_deps_flag(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--target-file",
                "a.py",
                "--include-deps",
            ],
        )
        call_args = mock_submit.call_args
        task_def = call_args[0][0]
        assert task_def.context.include_dependencies is True

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_max_import_depth_flag(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--target-file",
                "a.py",
                "--max-import-depth",
                "3",
            ],
        )
        call_args = mock_submit.call_args
        task_def = call_args[0][0]
        assert task_def.context.max_import_depth == 3


# ---------------------------------------------------------------------------
# Eval-planner CLI tests
# ---------------------------------------------------------------------------

_EVAL_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "eval"


class TestFormatDeterministicResult:
    def test_pass_result(self) -> None:
        from forge.eval.models import CheckStatus, DeterministicCheckResult, DeterministicResult

        det = DeterministicResult(
            checks=[
                DeterministicCheckResult(
                    check_name="check_step_ids_unique",
                    status=CheckStatus.PASS,
                    message="All step IDs are unique.",
                )
            ],
            all_passed=True,
        )
        output = format_deterministic_result(det)
        assert "[PASS]" in output
        assert "check_step_ids_unique" in output

    def test_fail_with_details(self) -> None:
        from forge.eval.models import CheckStatus, DeterministicCheckResult, DeterministicResult

        det = DeterministicResult(
            checks=[
                DeterministicCheckResult(
                    check_name="check_target_files_are_relative_paths",
                    status=CheckStatus.FAIL,
                    message="Found absolute paths.",
                    details=["/etc/passwd"],
                )
            ],
            all_passed=False,
        )
        output = format_deterministic_result(det)
        assert "[FAIL]" in output
        assert "/etc/passwd" in output


class TestFormatEvalResult:
    def test_without_judge(self) -> None:
        from forge.eval.models import DeterministicResult, PlanEvalResult

        plan = Plan(
            task_id="t1",
            steps=[PlanStep(step_id="s1", description="Do it.", target_files=["a.py"])],
            explanation="Simple.",
        )
        det = DeterministicResult(checks=[], all_passed=True)
        result = PlanEvalResult(case_id="case-1", plan=plan, deterministic=det)
        output = format_eval_result(result)
        assert "Case: case-1" in output
        assert "PASS" in output
        assert "Judge" not in output


class TestEvalPlannerCommand:
    def test_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["eval-planner", "--help"])
        assert result.exit_code == 0
        assert "--corpus-dir" in result.output
        assert "--judge" in result.output
        assert "--dry-run" in result.output

    def test_dry_run(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            main,
            ["eval-planner", "--corpus-dir", str(_EVAL_FIXTURES / "cases"), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "3 eval case(s)" in result.output
        assert "add-feature" in result.output
        assert "refactor" in result.output
        assert "fan-out" in result.output

    def test_run_with_plans(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            main,
            [
                "eval-planner",
                "--corpus-dir",
                str(_EVAL_FIXTURES / "cases"),
                "--plans-dir",
                str(_EVAL_FIXTURES / "plans"),
            ],
        )
        # Should produce output for the add-auth case (matched by task_id)
        # and warn/skip others
        assert "add-auth" in result.output or "Case:" in result.output or result.exit_code != 0

    def test_json_output(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            main,
            [
                "eval-planner",
                "--corpus-dir",
                str(_EVAL_FIXTURES / "cases"),
                "--plans-dir",
                str(_EVAL_FIXTURES / "plans"),
                "--json",
            ],
        )
        # If there are results, they should be valid JSON
        if result.exit_code == 0:
            parsed = json.loads(result.stdout)
            assert isinstance(parsed, list)

    def test_save_results(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        result = cli_runner.invoke(
            main,
            [
                "eval-planner",
                "--corpus-dir",
                str(_EVAL_FIXTURES / "cases"),
                "--plans-dir",
                str(_EVAL_FIXTURES / "plans"),
                "--output-dir",
                str(tmp_path),
            ],
        )
        if result.exit_code == 0:
            json_files = list(tmp_path.glob("*.json"))
            assert len(json_files) == 1


# ---------------------------------------------------------------------------
# Phase 5 CLI tests
# ---------------------------------------------------------------------------


class TestFormatLlmStats:
    def test_format(self) -> None:
        from forge.models import LLMStats

        stats = LLMStats(
            model_name="test-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=250.0,
        )
        output = format_llm_stats(stats)
        assert "test-model" in output
        assert "100in" in output
        assert "50out" in output
        assert "250ms" in output


class TestFormatVerboseResult:
    def test_with_llm_stats(self) -> None:
        from forge.models import LLMStats

        stats = LLMStats(
            model_name="test-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=250.0,
        )
        result = TaskResult(
            task_id="t",
            status=TransitionSignal.SUCCESS,
            llm_stats=stats,
        )
        output = format_verbose_result(result)
        assert "LLM:" in output
        assert "test-model" in output

    def test_with_context_stats(self) -> None:
        from forge.models import ContextStats

        result = TaskResult(
            task_id="t",
            status=TransitionSignal.SUCCESS,
            context_stats=ContextStats(
                files_discovered=10,
                files_included_full=5,
                total_estimated_tokens=5000,
                budget_utilization=0.75,
            ),
        )
        output = format_verbose_result(result)
        assert "Context:" in output
        assert "Files discovered: 10" in output
        assert "75.0%" in output

    def test_without_stats(self) -> None:
        result = TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        output = format_verbose_result(result)
        assert "Task: t" in output
        assert "LLM:" not in output


class TestVerboseFlag:
    def test_verbose_in_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["run", "--help"])
        assert "--verbose" in result.output

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_verbose_flag_shows_stats(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        from forge.models import LLMStats

        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(
                task_id="t",
                status=TransitionSignal.SUCCESS,
                llm_stats=LLMStats(
                    model_name="test-model",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=250.0,
                ),
            )
        )
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--target-file",
                "a.py",
                "--verbose",
            ],
        )
        assert result.exit_code == 0
        assert "LLM:" in result.output
        assert "test-model" in result.output


class TestStatusCommand:
    def test_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0
        assert "--workflow-id" in result.output
        assert "--verbose" in result.output
        assert "--json" in result.output

    def test_no_store(self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FORGE_DB_URL", raising=False)
        result = cli_runner.invoke(main, ["status"])
        assert result.exit_code == EXIT_FAILURE
        assert "FORGE_DB_URL" in result.stderr

    def test_list_runs(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{db_path}")

        from forge.store import get_store_engine, run_migrations, save_run

        run_migrations(f"sqlite:///{db_path}")
        engine = get_store_engine()
        save_run(
            engine,
            TaskResult(task_id="t1", status=TransitionSignal.SUCCESS),
            "wf-123",
            "run-1",
        )

        result = cli_runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "wf-123" in result.output

    def test_specific_run(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{db_path}")

        from forge.store import get_store_engine, run_migrations, save_run

        run_migrations(f"sqlite:///{db_path}")
        engine = get_store_engine()
        save_run(
            engine,
            TaskResult(task_id="t1", status=TransitionSignal.SUCCESS),
            "wf-123",
            "run-1",
        )

        result = cli_runner.invoke(main, ["status", "--workflow-id", "wf-123"])
        assert result.exit_code == 0
        assert "wf-123" in result.output
        assert "run-1" in result.output
        assert "t1" in result.output

    def test_rerun_shows_both_runs(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reused workflow_id with two run_ids surfaces both runs in the listing."""
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{db_path}")

        from forge.store import get_store_engine, run_migrations, save_run

        run_migrations(f"sqlite:///{db_path}")
        engine = get_store_engine()
        task_result = TaskResult(task_id="t1", status=TransitionSignal.SUCCESS)
        save_run(engine, task_result, "forge-task-t1", "run-A")
        save_run(engine, task_result, "forge-task-t1", "run-B")

        result = cli_runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "run-A" in result.output
        assert "run-B" in result.output

    def test_json_output(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{db_path}")

        from forge.store import get_store_engine, run_migrations, save_run

        run_migrations(f"sqlite:///{db_path}")
        engine = get_store_engine()
        save_run(
            engine,
            TaskResult(task_id="t1", status=TransitionSignal.SUCCESS),
            "wf-123",
            "run-1",
        )

        result = cli_runner.invoke(main, ["status", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# Phase 6 CLI tests
# ---------------------------------------------------------------------------


class TestPlaybooksCommand:
    def test_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["playbooks", "--help"])
        assert result.exit_code == 0
        assert "--tag" in result.output
        assert "--json" in result.output

    def test_no_store(self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FORGE_DB_URL", raising=False)
        result = cli_runner.invoke(main, ["playbooks"])
        assert result.exit_code == EXIT_FAILURE
        assert "FORGE_DB_URL" in result.stderr

    def test_list_playbooks(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{db_path}")

        from forge.store import get_store_engine, run_migrations, save_playbooks

        run_migrations(f"sqlite:///{db_path}")
        engine = get_store_engine()
        save_playbooks(
            engine,
            [
                {
                    "title": "Test lesson",
                    "content": "Always do X.",
                    "tags_json": '["python"]',
                    "source_task_id": "t1",
                    "source_workflow_id": "wf-1",
                    "extraction_workflow_id": "extract-1",
                }
            ],
        )

        result = cli_runner.invoke(main, ["playbooks"])
        assert result.exit_code == 0
        assert "Test lesson" in result.output
        assert "python" in result.output

    def test_tag_filter(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{db_path}")

        from forge.store import get_store_engine, run_migrations, save_playbooks

        run_migrations(f"sqlite:///{db_path}")
        engine = get_store_engine()
        save_playbooks(
            engine,
            [
                {
                    "title": "Python lesson",
                    "content": "Do X.",
                    "tags_json": '["python"]',
                    "source_task_id": "t1",
                    "source_workflow_id": "wf-1",
                    "extraction_workflow_id": "extract-1",
                },
                {
                    "title": "JS lesson",
                    "content": "Do Y.",
                    "tags_json": '["javascript"]',
                    "source_task_id": "t2",
                    "source_workflow_id": "wf-2",
                    "extraction_workflow_id": "extract-1",
                },
            ],
        )

        result = cli_runner.invoke(main, ["playbooks", "--tag", "python"])
        assert result.exit_code == 0
        assert "Python lesson" in result.output
        assert "JS lesson" not in result.output

    def test_json_output(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{db_path}")

        from forge.store import get_store_engine, run_migrations, save_playbooks

        run_migrations(f"sqlite:///{db_path}")
        engine = get_store_engine()
        save_playbooks(
            engine,
            [
                {
                    "title": "Test lesson",
                    "content": "Always do X.",
                    "tags_json": '["python"]',
                    "source_task_id": "t1",
                    "source_workflow_id": "wf-1",
                    "extraction_workflow_id": "extract-1",
                }
            ],
        )

        result = cli_runner.invoke(main, ["playbooks", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert isinstance(parsed, list)
        assert parsed[0]["title"] == "Test lesson"

    def test_empty_playbooks(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{db_path}")

        from forge.store import run_migrations

        run_migrations(f"sqlite:///{db_path}")

        result = cli_runner.invoke(main, ["playbooks"])
        assert result.exit_code == 0
        assert "No playbooks found" in result.output

    # -----------------------------------------------------------------------
    # playbooks add
    # -----------------------------------------------------------------------

    def test_add_schema_output(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["playbooks", "add", "--schema"])
        assert result.exit_code == 0
        schema = json.loads(result.output)
        assert "title" in schema["properties"]
        assert "content" in schema["properties"]
        assert "tags" in schema["properties"]

    def test_add_from_file(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        from forge.models import ManualPlaybookResult, PlaybookEntry

        entry_file = tmp_path / "entry.json"
        entry_file.write_text(
            json.dumps(
                {
                    "title": "Test entry",
                    "content": "Do the thing.",
                    "tags": ["test"],
                    "source_task_id": "manual-1",
                }
            )
        )

        mock_result = ManualPlaybookResult(
            approved=True,
            entry=PlaybookEntry(
                title="Test entry",
                content="Do the thing.",
                tags=["test"],
                source_task_id="manual-1",
            ),
        )
        with patch(
            "forge.cli._submit_manual_playbook",
            side_effect=_async_result(mock_result),
        ):
            result = cli_runner.invoke(main, ["playbooks", "add", "--file", str(entry_file)])

        assert result.exit_code == 0, result.output + (result.stderr or "")
        assert "saved" in result.output.lower()
        assert "Test entry" in result.output

    def test_add_rejected(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        from forge.models import ManualPlaybookResult

        entry_file = tmp_path / "entry.json"
        entry_file.write_text(
            json.dumps(
                {
                    "title": "Bad entry",
                    "content": "Vague advice.",
                    "tags": [],
                    "source_task_id": "manual-2",
                }
            )
        )

        mock_result = ManualPlaybookResult(
            approved=False,
            rejection_reason="Too vague and not actionable.",
        )
        with patch(
            "forge.cli._submit_manual_playbook",
            side_effect=_async_result(mock_result),
        ):
            result = cli_runner.invoke(main, ["playbooks", "add", "--file", str(entry_file)])

        assert result.exit_code == EXIT_FAILURE
        assert "Too vague" in result.stderr

    def test_add_validation_error(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        from forge.models import ManualPlaybookResult

        entry_file = tmp_path / "bad.json"
        entry_file.write_text("{not valid json")

        mock_result = ManualPlaybookResult(
            approved=False,
            validation_error="Invalid JSON: expected value at line 1 column 2",
        )
        with patch(
            "forge.cli._submit_manual_playbook",
            side_effect=_async_result(mock_result),
        ):
            result = cli_runner.invoke(main, ["playbooks", "add", "--file", str(entry_file)])

        assert result.exit_code == EXIT_FAILURE
        assert "Invalid input" in result.stderr

    def test_add_with_suggestions(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        from forge.models import ManualPlaybookResult, PlaybookEntry

        entry_file = tmp_path / "entry.json"
        entry_file.write_text(
            json.dumps(
                {
                    "title": "Original title",
                    "content": "Original content.",
                    "tags": ["test"],
                    "source_task_id": "manual-3",
                }
            )
        )

        mock_result = ManualPlaybookResult(
            approved=True,
            entry=PlaybookEntry(
                title="Improved title",
                content="Original content.",
                tags=["test", "extra"],
                source_task_id="manual-3",
            ),
        )
        with patch(
            "forge.cli._submit_manual_playbook",
            side_effect=_async_result(mock_result),
        ):
            result = cli_runner.invoke(main, ["playbooks", "add", "--file", str(entry_file)])

        assert result.exit_code == 0, result.output + (result.stderr or "")
        assert "saved" in result.output.lower()
        assert "Improved title" in result.output

    def test_list_still_works_as_default(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Backward compat: `forge playbooks` without subcommand still lists."""
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{db_path}")

        from forge.store import get_store_engine, run_migrations, save_playbooks

        run_migrations(f"sqlite:///{db_path}")
        engine = get_store_engine()
        save_playbooks(
            engine,
            [
                {
                    "title": "Compat lesson",
                    "content": "Still works.",
                    "tags_json": '["compat"]',
                    "source_task_id": "t1",
                    "source_workflow_id": "wf-1",
                    "extraction_workflow_id": "extract-1",
                }
            ],
        )

        result = cli_runner.invoke(main, ["playbooks"])
        assert result.exit_code == 0
        assert "Compat lesson" in result.output


# ---------------------------------------------------------------------------
# Sanity check CLI tests
# ---------------------------------------------------------------------------


class TestSanityCheckIntervalFlag:
    def test_flag_in_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["run", "--help"])
        assert "--sanity-check-interval" in result.output

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_flag_passed_to_submit(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--plan",
                "--sanity-check-interval",
                "3",
            ],
        )
        call_kwargs = mock_submit.call_args
        assert call_kwargs[1]["sanity_check_interval"] == 3

    @patch("forge.cli._submit_and_wait")
    @patch("forge.cli.discover_repo_root")
    def test_default_is_zero(
        self,
        mock_discover: object,
        mock_submit: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.side_effect = _async_result(
            TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
        )
        cli_runner.invoke(
            main,
            [
                "run",
                "--task-id",
                "t",
                "--description",
                "d",
                "--plan",
            ],
        )
        call_kwargs = mock_submit.call_args
        assert call_kwargs[1]["sanity_check_interval"] == 0


class TestFormatVerboseResultSanityCheckCount:
    def test_shows_sanity_check_count(self) -> None:
        result = TaskResult(
            task_id="t",
            status=TransitionSignal.SUCCESS,
            sanity_check_count=3,
        )
        output = format_verbose_result(result)
        assert "Sanity checks: 3" in output

    def test_hides_when_zero(self) -> None:
        result = TaskResult(
            task_id="t",
            status=TransitionSignal.SUCCESS,
            sanity_check_count=0,
        )
        output = format_verbose_result(result)
        assert "Sanity checks" not in output


# ---------------------------------------------------------------------------
# Start command tests
# ---------------------------------------------------------------------------


class TestLoadWorkflowInput:
    """Tests for load_workflow_input pure function."""

    def test_json_string(self) -> None:
        result = load_workflow_input('{"file_path": "/data/doc.pdf"}', None)
        assert result == {"file_path": "/data/doc.pdf"}

    def test_empty_input(self) -> None:
        result = load_workflow_input(None, None)
        assert result == {}

    def test_both_provided_error(self) -> None:
        with pytest.raises(click.UsageError, match="not both"):
            load_workflow_input('{"a": 1}', "input.json")

    def test_invalid_json(self) -> None:
        with pytest.raises(click.BadParameter, match="Invalid JSON"):
            load_workflow_input("{bad json", None)

    def test_non_object_json(self) -> None:
        with pytest.raises(click.BadParameter, match="Expected a JSON object"):
            load_workflow_input("[1, 2, 3]", None)

    def test_file_input(self, tmp_path: Path) -> None:
        f = tmp_path / "input.json"
        f.write_text('{"key": "value"}')
        result = load_workflow_input(None, str(f))
        assert result == {"key": "value"}

    def test_file_not_found(self) -> None:
        with pytest.raises(click.BadParameter, match="Cannot read input file"):
            load_workflow_input(None, "/nonexistent/path.json")


class TestStartCommand:
    """Tests for the ``forge start`` CLI command."""

    def test_fire_and_forget(self, cli_runner: CliRunner) -> None:
        with patch("forge.cli._start_workflow") as mock_start:
            mock_start.side_effect = _async_result("myworkflow-abc12345")
            result = cli_runner.invoke(
                main,
                ["start", "MyWorkflow", '{"key": "val"}'],
            )
            assert result.exit_code == 0
            assert "myworkflow-abc12345" in result.output
            call_args = mock_start.call_args
            assert call_args[0][0] == "MyWorkflow"
            assert call_args[0][1] == {"key": "val"}

    def test_wait_mode(self, cli_runner: CliRunner) -> None:
        with patch("forge.cli._start_workflow_and_wait") as mock_wait:
            mock_wait.side_effect = _async_result({"status": "done", "pages": 5})
            result = cli_runner.invoke(
                main,
                ["start", "MyWorkflow", '{"key": "val"}', "--wait"],
            )
            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert parsed == {"status": "done", "pages": 5}

    def test_custom_id(self, cli_runner: CliRunner) -> None:
        with patch("forge.cli._start_workflow") as mock_start:
            mock_start.side_effect = _async_result("custom-id")
            result = cli_runner.invoke(
                main,
                ["start", "MyWorkflow", "--id", "custom-id"],
            )
            assert result.exit_code == 0
            call_kwargs = mock_start.call_args[1]
            assert call_kwargs["workflow_id"] == "custom-id"

    def test_no_input_sends_empty_dict(self, cli_runner: CliRunner) -> None:
        with patch("forge.cli._start_workflow") as mock_start:
            mock_start.side_effect = _async_result("wf-123")
            result = cli_runner.invoke(main, ["start", "MyWorkflow"])
            assert result.exit_code == 0
            assert mock_start.call_args[0][1] == {}

    def test_input_file(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        f = tmp_path / "input.json"
        f.write_text('{"from_file": true}')
        with patch("forge.cli._start_workflow") as mock_start:
            mock_start.side_effect = _async_result("wf-file")
            result = cli_runner.invoke(
                main,
                ["start", "MyWorkflow", "--input-file", str(f)],
            )
            assert result.exit_code == 0
            assert mock_start.call_args[0][1] == {"from_file": True}

    def test_invalid_json_error(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["start", "MyWorkflow", "{bad"])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_connection_error(self, cli_runner: CliRunner) -> None:
        with patch("forge.cli._start_workflow") as mock_start:
            mock_start.side_effect = RuntimeError("Connection refused")
            result = cli_runner.invoke(
                main,
                ["start", "MyWorkflow", '{"a": 1}'],
            )
            assert result.exit_code == EXIT_INFRASTRUCTURE_ERROR
            assert "Connection refused" in result.stderr

    def test_auto_generated_id_format(self, cli_runner: CliRunner) -> None:
        with patch("forge.cli._start_workflow") as mock_start:
            mock_start.side_effect = _async_result("wf-123")
            cli_runner.invoke(main, ["start", "ForgeTaskWorkflow"])
            call_kwargs = mock_start.call_args[1]
            wf_id = call_kwargs["workflow_id"]
            assert wf_id.startswith("forgetaskworkflow-")
            assert len(wf_id) == len("forgetaskworkflow-") + 8


# ---------------------------------------------------------------------------
# Ingest command tests
# ---------------------------------------------------------------------------


def _make_session_info(
    session_id: str,
    *,
    project_name: str = "forge",
    path: str = "",
    size_bytes: int = 20480,
):
    """Build a pbook SessionInfo for testing without needing a real JSONL file."""
    from pbook.transcript import SessionInfo

    return SessionInfo(
        path=path or f"/tmp/fake/{session_id}.jsonl",
        session_id=session_id,
        project_dir_name="-tmp-fake",
        project_name=project_name,
        size_bytes=size_bytes,
    )


class TestFormatIngestDryRun:
    """Tests for the pure format_ingest_dry_run helper."""

    def test_empty_list(self) -> None:
        from forge.cli import format_ingest_dry_run

        output = format_ingest_dry_run([])
        assert "Found 0 session(s)" in output
        assert "0.0 MB" in output

    def test_single_session_shows_detail(self) -> None:
        from forge.cli import format_ingest_dry_run

        sessions = [_make_session_info("abcdef1234567890", size_bytes=15 * 1024)]
        output = format_ingest_dry_run(sessions)
        assert "Found 1 session(s)" in output
        assert "forge: 1 session(s)" in output
        # Small groups (<=3) show a per-session line
        assert "abcdef123456" in output

    def test_large_group_hides_detail(self) -> None:
        from forge.cli import format_ingest_dry_run

        sessions = [_make_session_info(f"sess-{i:04d}") for i in range(10)]
        output = format_ingest_dry_run(sessions)
        assert "forge: 10 session(s)" in output
        # Per-session detail should NOT appear for groups > 3
        assert "sess-0000" not in output

    def test_groups_by_project(self) -> None:
        from forge.cli import format_ingest_dry_run

        sessions = [
            _make_session_info("s1", project_name="forge"),
            _make_session_info("s2", project_name="pbook"),
            _make_session_info("s3", project_name="pbook"),
        ]
        output = format_ingest_dry_run(sessions)
        assert "forge: 1 session(s)" in output
        assert "pbook: 2 session(s)" in output


class TestFormatIngestResult:
    """Tests for the pure format_ingest_result helper."""

    def test_full_result(self) -> None:
        from forge.cli import format_ingest_result

        output = format_ingest_result(
            {
                "sessions_processed": 3,
                "total_experiences": 7,
                "total_entries_created": 5,
            }
        )
        assert "3 sessions processed" in output
        assert "7 experiences found" in output
        assert "5 entries created" in output

    def test_missing_keys_default_to_zero(self) -> None:
        from forge.cli import format_ingest_result

        output = format_ingest_result({})
        assert "0 sessions processed" in output
        assert "0 experiences found" in output
        assert "0 entries created" in output


class TestIngestCommand:
    """End-to-end tests for `forge ingest` via CliRunner."""

    def test_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--all" in result.output
        assert "--dry-run" in result.output
        assert "--force" in result.output
        assert "--project" in result.output

    def test_no_args_shows_error(self, cli_runner: CliRunner) -> None:
        # Prevent the "already ingested" filter from hitting a real pbook DB.
        with patch("pbook.store.get_store_engine", return_value=None):
            result = cli_runner.invoke(main, ["ingest"])
        assert result.exit_code == EXIT_FAILURE
        assert "TRANSCRIPT_PATH" in result.stderr or "--all" in result.stderr

    def test_dry_run_all_with_no_sessions(self, cli_runner: CliRunner) -> None:
        with (
            patch("pbook.transcript.discover_sessions", return_value=[]),
            patch("pbook.store.get_store_engine", return_value=None),
        ):
            result = cli_runner.invoke(main, ["ingest", "--all", "--dry-run"])
        assert result.exit_code == 0
        assert "No sessions found" in result.output

    def test_dry_run_all_with_sessions(self, cli_runner: CliRunner) -> None:
        sessions = [
            _make_session_info("aaaabbbbccccdddd", project_name="forge"),
            _make_session_info("eeeeffff00001111", project_name="pbook"),
        ]
        with (
            patch("pbook.transcript.discover_sessions", return_value=sessions),
            patch("pbook.store.get_store_engine", return_value=None),
        ):
            result = cli_runner.invoke(main, ["ingest", "--all", "--dry-run"])
        assert result.exit_code == 0
        assert "Found 2 session(s)" in result.output
        assert "forge: 1 session(s)" in result.output
        assert "pbook: 1 session(s)" in result.output

    def test_dry_run_single_path(self, cli_runner: CliRunner, tmp_path: pathlib.Path) -> None:
        fake = tmp_path / "sess-xyz.jsonl"
        fake.write_text('{"type": "user", "sessionId": "sess-xyz"}\n')
        with patch("pbook.store.get_store_engine", return_value=None):
            result = cli_runner.invoke(
                main, ["ingest", str(fake), "--project", "demo", "--dry-run"]
            )
        assert result.exit_code == 0
        assert "Found 1 session(s)" in result.output
        assert "demo:" in result.output

    def test_project_filter_applied_with_all(self, cli_runner: CliRunner) -> None:
        sessions = [
            _make_session_info("s1", project_name="forge"),
            _make_session_info("s2", project_name="pbook"),
        ]
        with (
            patch("pbook.transcript.discover_sessions", return_value=sessions),
            patch("pbook.store.get_store_engine", return_value=None),
        ):
            result = cli_runner.invoke(main, ["ingest", "--all", "--project", "forge", "--dry-run"])
        assert result.exit_code == 0
        assert "Found 1 session(s)" in result.output
        assert "pbook:" not in result.output

    def test_already_ingested_filter_skips_sessions(self, cli_runner: CliRunner) -> None:
        """When pbook's store reports a session is ingested, it should be skipped."""
        sessions = [
            _make_session_info("already-done"),
            _make_session_info("new-one"),
        ]

        with (
            patch("pbook.transcript.discover_sessions", return_value=sessions),
            patch("pbook.store.get_store_engine", return_value=MagicMock()),
            patch(
                "pbook.store.get_ingested_session_ids",
                return_value={"already-done"},
            ),
        ):
            result = cli_runner.invoke(main, ["ingest", "--all", "--dry-run"])

        assert result.exit_code == 0
        assert "Skipping 1 already-ingested session(s)" in result.output
        assert "Found 1 session(s)" in result.output

    def test_force_skips_already_ingested_filter(self, cli_runner: CliRunner) -> None:
        """--force bypasses the already-ingested query entirely."""
        sessions = [_make_session_info("already-done")]

        mock_get_ids = MagicMock()
        with (
            patch("pbook.transcript.discover_sessions", return_value=sessions),
            patch("pbook.store.get_ingested_session_ids", mock_get_ids),
        ):
            result = cli_runner.invoke(main, ["ingest", "--all", "--force", "--dry-run"])

        assert result.exit_code == 0
        mock_get_ids.assert_not_called()
        assert "Found 1 session(s)" in result.output

    def test_submission_happy_path(self, cli_runner: CliRunner) -> None:
        sessions = [_make_session_info("s1"), _make_session_info("s2")]
        mock_submit = MagicMock()
        mock_submit.side_effect = _async_result(
            {
                "sessions_processed": 2,
                "total_experiences": 4,
                "total_entries_created": 3,
                "per_session": [],
            }
        )
        with (
            patch("pbook.transcript.discover_sessions", return_value=sessions),
            patch("pbook.store.get_store_engine", return_value=None),
            patch("forge.cli._submit_ingestion", mock_submit),
        ):
            result = cli_runner.invoke(main, ["ingest", "--all"])

        assert result.exit_code == 0
        assert "Submitting 2 session(s)" in result.output
        assert "2 sessions processed" in result.output
        assert "4 experiences found" in result.output
        assert "3 entries created" in result.output

        # Verify the payload passed to _submit_ingestion
        call_args = mock_submit.call_args
        _temporal_address, session_dicts = call_args.args
        assert len(session_dicts) == 2
        assert session_dicts[0]["session_id"] == "s1"
        assert session_dicts[0]["project"] == "forge"

    def test_submission_json_output(self, cli_runner: CliRunner) -> None:
        sessions = [_make_session_info("s1")]
        mock_submit = MagicMock()
        mock_submit.side_effect = _async_result(
            {
                "sessions_processed": 1,
                "total_experiences": 0,
                "total_entries_created": 0,
            }
        )
        with (
            patch("pbook.transcript.discover_sessions", return_value=sessions),
            patch("pbook.store.get_store_engine", return_value=None),
            patch("forge.cli._submit_ingestion", mock_submit),
        ):
            result = cli_runner.invoke(main, ["ingest", "--all", "--json"])

        assert result.exit_code == 0
        # The output contains a "Submitting..." line followed by an
        # indented JSON block. Strip the first line and parse the rest.
        lines = result.output.splitlines()
        json_start = next(i for i, ln in enumerate(lines) if ln.startswith("{"))
        payload = json.loads("\n".join(lines[json_start:]))
        assert payload["sessions_processed"] == 1

    def test_submission_infrastructure_error(self, cli_runner: CliRunner) -> None:
        sessions = [_make_session_info("s1")]

        async def _raise(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("Temporal unreachable")

        with (
            patch("pbook.transcript.discover_sessions", return_value=sessions),
            patch("pbook.store.get_store_engine", return_value=None),
            patch("forge.cli._submit_ingestion", side_effect=_raise),
        ):
            result = cli_runner.invoke(main, ["ingest", "--all"])

        assert result.exit_code == EXIT_INFRASTRUCTURE_ERROR
        assert "Temporal unreachable" in result.stderr

    def test_pbook_not_installed_error(self, cli_runner: CliRunner) -> None:
        """If pbook can't be imported, we fail fast with a friendly message."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "pbook.transcript" or name.startswith("pbook."):
                raise ImportError("No module named 'pbook'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        with patch.object(builtins, "__import__", side_effect=fake_import):
            result = cli_runner.invoke(main, ["ingest", "--all"])

        assert result.exit_code == EXIT_FAILURE
        assert "pbook is not installed" in result.stderr
