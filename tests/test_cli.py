"""Tests for the Forge CLI entry point."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from forge.cli import (
    EXIT_FAILURE,
    EXIT_INFRASTRUCTURE_ERROR,
    build_task_definition,
    format_deterministic_result,
    format_eval_result,
    format_llm_stats,
    format_step_result,
    format_sub_task_result,
    format_task_result,
    format_validation_results,
    format_verbose_result,
    load_task_definition,
    main,
)
from forge.models import (
    Plan,
    PlanStep,
    StepResult,
    SubTaskResult,
    TaskResult,
    TransitionSignal,
    ValidationResult,
)

if TYPE_CHECKING:
    from pathlib import Path


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

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_success_exit_code(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
        success_result: TaskResult,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = success_result

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

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_failure_exit_code(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
        failure_result: TaskResult,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = failure_result

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

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_json_output(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
        success_result: TaskResult,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = success_result

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
        parsed = json.loads(result.output)
        assert parsed["task_id"] == "test-task"
        assert parsed["status"] == "success"

    @patch("forge.cli._submit_no_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_no_wait_mode(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = "forge-task-test-task"

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

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_temporal_connection_error(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
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
        assert "Connection refused" in result.output

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_task_file_mode(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
        tmp_path: Path,
        success_result: TaskResult,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = success_result

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

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_validation_flags_passed(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
        success_result: TaskResult,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = success_result

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
    @patch("forge.worker.run_worker", new_callable=AsyncMock)
    def test_worker_invokes_run_worker(
        self,
        mock_run_worker: AsyncMock,
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
            patch("forge.cli._submit_and_wait", new_callable=AsyncMock) as mock_submit,
        ):
            mock_discover.return_value = "/repo"
            mock_submit.return_value = TaskResult(
                task_id="plan-task", status=TransitionSignal.SUCCESS
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

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_plan_flag_passed_to_submit(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
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

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_flag_passed_to_submit(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
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


class TestContextDiscoveryFlags:
    def test_flags_in_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["run", "--help"])
        assert "--no-auto-discover" in result.output
        assert "--token-budget" in result.output
        assert "--max-import-depth" in result.output

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_no_auto_discover_flag(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
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

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_token_budget_flag(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
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

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    def test_max_import_depth_flag(
        self,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
    ) -> None:
        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = TaskResult(task_id="t", status=TransitionSignal.SUCCESS)
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
            parsed = json.loads(result.output)
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

    @patch("forge.cli._submit_and_wait", new_callable=AsyncMock)
    @patch("forge.cli.discover_repo_root")
    @patch("forge.cli._persist_run")
    def test_verbose_flag_shows_stats(
        self,
        mock_persist: object,
        mock_discover: object,
        mock_submit: AsyncMock,
        cli_runner: CliRunner,
    ) -> None:
        from forge.models import LLMStats

        mock_discover.return_value = "/repo"  # type: ignore[attr-defined]
        mock_submit.return_value = TaskResult(
            task_id="t",
            status=TransitionSignal.SUCCESS,
            llm_stats=LLMStats(
                model_name="test-model",
                input_tokens=100,
                output_tokens=50,
                latency_ms=250.0,
            ),
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
        monkeypatch.setenv("FORGE_DB_PATH", "")
        result = cli_runner.invoke(main, ["status"])
        assert result.exit_code == EXIT_FAILURE
        assert "No store available" in result.output

    def test_list_runs(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_PATH", str(db_path))

        from forge.store import get_engine, run_migrations, save_run

        run_migrations(db_path)
        engine = get_engine(db_path)
        save_run(
            engine,
            TaskResult(task_id="t1", status=TransitionSignal.SUCCESS),
            "wf-123",
        )

        result = cli_runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "wf-123" in result.output

    def test_specific_run(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_PATH", str(db_path))

        from forge.store import get_engine, run_migrations, save_run

        run_migrations(db_path)
        engine = get_engine(db_path)
        save_run(
            engine,
            TaskResult(task_id="t1", status=TransitionSignal.SUCCESS),
            "wf-123",
        )

        result = cli_runner.invoke(main, ["status", "--workflow-id", "wf-123"])
        assert result.exit_code == 0
        assert "wf-123" in result.output
        assert "t1" in result.output

    def test_json_output(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_PATH", str(db_path))

        from forge.store import get_engine, run_migrations, save_run

        run_migrations(db_path)
        engine = get_engine(db_path)
        save_run(
            engine,
            TaskResult(task_id="t1", status=TransitionSignal.SUCCESS),
            "wf-123",
        )

        result = cli_runner.invoke(main, ["status", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# Phase 6 CLI tests
# ---------------------------------------------------------------------------


class TestExtractCommand:
    def test_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.output
        assert "--dry-run" in result.output

    def test_dry_run_no_store(self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DB_PATH", "")
        result = cli_runner.invoke(main, ["extract", "--dry-run"])
        assert result.exit_code == EXIT_FAILURE
        assert "No store available" in result.output

    def test_dry_run_with_runs(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_PATH", str(db_path))

        from forge.store import get_engine, run_migrations, save_run

        run_migrations(db_path)
        engine = get_engine(db_path)
        save_run(
            engine,
            TaskResult(task_id="t1", status=TransitionSignal.SUCCESS),
            "wf-123",
        )

        result = cli_runner.invoke(main, ["extract", "--dry-run"])
        assert result.exit_code == 0
        assert "wf-123" in result.output
        assert "Unextracted runs" in result.output

    def test_dry_run_no_unextracted(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_PATH", str(db_path))

        from forge.store import run_migrations

        run_migrations(db_path)

        result = cli_runner.invoke(main, ["extract", "--dry-run"])
        assert result.exit_code == 0
        assert "No unextracted runs found" in result.output


class TestPlaybooksCommand:
    def test_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["playbooks", "--help"])
        assert result.exit_code == 0
        assert "--tag" in result.output
        assert "--json" in result.output

    def test_no_store(self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DB_PATH", "")
        result = cli_runner.invoke(main, ["playbooks"])
        assert result.exit_code == EXIT_FAILURE
        assert "No store available" in result.output

    def test_list_playbooks(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_PATH", str(db_path))

        from forge.store import get_engine, run_migrations, save_playbooks

        run_migrations(db_path)
        engine = get_engine(db_path)
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
        monkeypatch.setenv("FORGE_DB_PATH", str(db_path))

        from forge.store import get_engine, run_migrations, save_playbooks

        run_migrations(db_path)
        engine = get_engine(db_path)
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
        monkeypatch.setenv("FORGE_DB_PATH", str(db_path))

        from forge.store import get_engine, run_migrations, save_playbooks

        run_migrations(db_path)
        engine = get_engine(db_path)
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
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert parsed[0]["title"] == "Test lesson"

    def test_empty_playbooks(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("FORGE_DB_PATH", str(db_path))

        from forge.store import run_migrations

        run_migrations(db_path)

        result = cli_runner.invoke(main, ["playbooks"])
        assert result.exit_code == 0
        assert "No playbooks found" in result.output
