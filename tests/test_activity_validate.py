"""Tests for forge.activities.validate — validation checks."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest
from sax_platform.temporal.heartbeat import heartbeat_during
from temporalio.testing import ActivityEnvironment

from forge.activities import validate as validate_module
from forge.activities.validate import (
    _run_command,
    _run_ruff_format_check,
    _run_ruff_format_fix,
    _run_ruff_lint,
    _run_ruff_lint_fix,
    parse_check_result,
    validate_output,
)
from forge.models import ValidateOutputInput, ValidationConfig
from forge.subprocess_result import SubprocessResult

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# parse_check_result (pure function)
# ---------------------------------------------------------------------------


class TestParseCheckResult:
    def test_passing_result(self) -> None:
        result = SubprocessResult(returncode=0, stdout="", stderr="")
        vr = parse_check_result("lint", result)
        assert vr.passed is True
        assert vr.check_name == "lint"
        assert "passed" in vr.summary

    def test_failing_result(self) -> None:
        result = SubprocessResult(returncode=1, stdout="error on line 5", stderr="")
        vr = parse_check_result("lint", result)
        assert vr.passed is False
        assert "error on line 5" in vr.summary

    def test_long_output_truncated_in_summary(self) -> None:
        long_output = "x" * 300
        result = SubprocessResult(returncode=1, stdout=long_output, stderr="")
        vr = parse_check_result("lint", result)
        assert len(vr.summary) < len(long_output)
        assert vr.summary.endswith("...")
        assert vr.details == long_output

    def test_stderr_used_when_stdout_empty(self) -> None:
        result = SubprocessResult(returncode=1, stdout="", stderr="stderr msg")
        vr = parse_check_result("check", result)
        assert "stderr msg" in vr.summary


# ---------------------------------------------------------------------------
# ruff checks (imperative shell, real subprocess)
# ---------------------------------------------------------------------------


class TestRunRuffLint:
    def test_valid_python_passes(self, tmp_path: Path, ruff_config: Path) -> None:
        (tmp_path / "good.py").write_text("x = 1\n")
        result = _run_ruff_lint(tmp_path, ["good.py"])
        assert result.passed is True

    def test_invalid_python_fails(self, tmp_path: Path, ruff_config: Path) -> None:
        (tmp_path / "bad.py").write_text("import os\nimport sys\n")
        result = _run_ruff_lint(tmp_path, ["bad.py"])
        assert result.passed is False
        assert "ruff_lint" in result.check_name


class TestRunRuffFormatCheck:
    def test_formatted_python_passes(self, tmp_path: Path, ruff_config: Path) -> None:
        (tmp_path / "good.py").write_text("x = 1\n")
        result = _run_ruff_format_check(tmp_path, ["good.py"])
        assert result.passed is True

    def test_unformatted_python_fails(self, tmp_path: Path, ruff_config: Path) -> None:
        (tmp_path / "bad.py").write_text("x=1")
        result = _run_ruff_format_check(tmp_path, ["bad.py"])
        assert result.passed is False


# ---------------------------------------------------------------------------
# validate_output (activity)
# ---------------------------------------------------------------------------


class TestValidateOutput:
    @pytest.mark.asyncio
    async def test_valid_file_passes_all_checks(self, tmp_path: Path, ruff_config: Path) -> None:
        (tmp_path / "clean.py").write_text("x = 1\n")
        input_data = ValidateOutputInput(
            task_id="v1",
            worktree_path=str(tmp_path),
            files=["clean.py"],
            validation=ValidationConfig(run_ruff_lint=True, run_ruff_format=True),
        )
        results = await validate_output(input_data)
        assert len(results) == 2
        assert all(r.passed for r in results)

    @pytest.mark.asyncio
    async def test_invalid_file_fails(self, tmp_path: Path, ruff_config: Path) -> None:
        (tmp_path / "bad.py").write_text("import os\nimport sys\n")
        input_data = ValidateOutputInput(
            task_id="v2",
            worktree_path=str(tmp_path),
            files=["bad.py"],
            validation=ValidationConfig(auto_fix=False, run_ruff_lint=True, run_ruff_format=False),
        )
        results = await validate_output(input_data)
        assert len(results) == 1
        assert results[0].passed is False

    @pytest.mark.asyncio
    async def test_respects_disabled_checks(self, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("x = 1\n")
        input_data = ValidateOutputInput(
            task_id="v3",
            worktree_path=str(tmp_path),
            files=["f.py"],
            validation=ValidationConfig(
                run_ruff_lint=False,
                run_ruff_format=False,
                run_tests=False,
            ),
        )
        results = await validate_output(input_data)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_runs_test_command(self, tmp_path: Path) -> None:
        input_data = ValidateOutputInput(
            task_id="v4",
            worktree_path=str(tmp_path),
            files=[],
            validation=ValidationConfig(
                run_ruff_lint=False,
                run_ruff_format=False,
                run_tests=True,
                test_command="echo ok",
            ),
        )
        results = await validate_output(input_data)
        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].check_name == "tests"

    @pytest.mark.asyncio
    async def test_auto_fix_makes_fixable_code_pass(
        self, tmp_path: Path, ruff_config: Path
    ) -> None:
        """Code with typing.List passes validation when auto_fix=True."""
        code = "from typing import List\n\ndef f() -> List[str]:\n    return []\n"
        (tmp_path / "fixable.py").write_text(code)
        input_data = ValidateOutputInput(
            task_id="v5",
            worktree_path=str(tmp_path),
            files=["fixable.py"],
            validation=ValidationConfig(auto_fix=True, run_ruff_lint=True, run_ruff_format=True),
        )
        results = await validate_output(input_data)
        assert all(r.passed for r in results)

    @pytest.mark.asyncio
    async def test_auto_fix_disabled_leaves_fixable_issues(
        self, tmp_path: Path, ruff_config: Path
    ) -> None:
        """Same code fails validation when auto_fix=False."""
        code = "from typing import List\n\ndef f() -> List[str]:\n    return []\n"
        (tmp_path / "fixable.py").write_text(code)
        input_data = ValidateOutputInput(
            task_id="v6",
            worktree_path=str(tmp_path),
            files=["fixable.py"],
            validation=ValidationConfig(auto_fix=False, run_ruff_lint=True, run_ruff_format=True),
        )
        results = await validate_output(input_data)
        assert any(not r.passed for r in results)


# ---------------------------------------------------------------------------
# ruff lint fix (imperative shell, real subprocess)
# ---------------------------------------------------------------------------


class TestRunRuffLintFix:
    def test_fixes_auto_fixable_issues(self, tmp_path: Path, ruff_config: Path) -> None:
        """typing.List → list after lint fix."""
        p = tmp_path / "fixable.py"
        p.write_text("from typing import List\n\ndef f() -> List[str]:\n    return []\n")
        _run_ruff_lint_fix(tmp_path, ["fixable.py"])
        content = p.read_text()
        assert "List" not in content
        assert "list[str]" in content

    def test_leaves_unfixable_issues(self, tmp_path: Path, ruff_config: Path) -> None:
        """F841 (unused var) requires --unsafe-fixes, not fixed by default."""
        p = tmp_path / "unfixable.py"
        p.write_text("def f() -> None:\n    x = 1\n    return\n")
        _run_ruff_lint_fix(tmp_path, ["unfixable.py"])
        content = p.read_text()
        assert "x = 1" in content


# ---------------------------------------------------------------------------
# ruff format fix (imperative shell, real subprocess)
# ---------------------------------------------------------------------------


class TestRunRuffFormatFix:
    def test_formats_unformatted_code(self, tmp_path: Path, ruff_config: Path) -> None:
        """x=1 → x = 1 after format fix."""
        p = tmp_path / "ugly.py"
        p.write_text("x=1\n")
        _run_ruff_format_fix(tmp_path, ["ugly.py"])
        content = p.read_text()
        assert "x = 1" in content


# ---------------------------------------------------------------------------
# T1.4 — event loop is not blocked by validation subprocesses
# ---------------------------------------------------------------------------


class TestValidateOutputEventLoop:
    @pytest.mark.asyncio
    async def test_heartbeats_fire_during_slow_validation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A sleeping test command runs in a thread, so heartbeats keep firing.

        If the subprocess blocked the event loop (the pre-T1.4 behavior), the
        heartbeat loop could not run and ``beats`` would be empty.
        """
        # Shorten the interval so the assertion resolves in well under a second.
        # validate_output calls heartbeat_during() with no args, so patch the
        # name it looks up.
        fast_heartbeat = functools.partial(heartbeat_during, interval_seconds=0.02)
        monkeypatch.setattr(validate_module, "heartbeat_during", fast_heartbeat)

        input_data = ValidateOutputInput(
            task_id="hb1",
            worktree_path=str(tmp_path),
            files=[],
            validation=ValidationConfig(
                auto_fix=False,
                run_ruff_lint=False,
                run_ruff_format=False,
                run_tests=True,
                test_command="sleep 0.3",
            ),
        )

        env = ActivityEnvironment()
        beats: list[tuple[object, ...]] = []
        env.on_heartbeat = lambda *args: beats.append(args)

        results = await env.run(validate_output, input_data)

        assert len(beats) >= 1
        assert results[0].passed is True

    @pytest.mark.asyncio
    async def test_test_command_timeout_yields_failed_result(self, tmp_path: Path) -> None:
        """A test command exceeding the cap fails the check, not the activity."""
        input_data = ValidateOutputInput(
            task_id="to1",
            worktree_path=str(tmp_path),
            files=[],
            validation=ValidationConfig(
                auto_fix=False,
                run_ruff_lint=False,
                run_ruff_format=False,
                run_tests=True,
                test_command="sleep 5",
                test_timeout_seconds=1,
            ),
        )

        results = await validate_output(input_data)

        assert len(results) == 1
        assert results[0].check_name == "tests"
        assert results[0].passed is False
        assert "timed out" in results[0].summary


# ---------------------------------------------------------------------------
# T1.7 — model-influenced commands run under an allowlisted environment
# ---------------------------------------------------------------------------


class TestValidateEnvScrub:
    def test_secret_env_absent_from_child(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The validate seam does not leak worker secrets into a test command.

        Sentinels are set in the parent env; the child (``env``, dumped to
        stdout) must not see them, while an allowlisted var (PATH) survives so
        the test runner can still resolve its executables.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SENTINEL")
        monkeypatch.setenv("FORGE_DB_URL", "postgres://SENTINEL")

        result = _run_command(["sh", "-c", "env"], tmp_path)

        assert "ANTHROPIC_API_KEY" not in result.stdout
        assert "FORGE_DB_URL" not in result.stdout
        assert "SENTINEL" not in result.stdout
        assert "PATH=" in result.stdout  # allowlisted var still reaches the child

    @pytest.mark.asyncio
    async def test_real_pytest_passes_under_scrubbed_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A genuine pytest command still passes with the scrubbed env.

        Confirms the allowlist (PATH + VIRTUAL_ENV) is sufficient for the
        project's pytest to run — the scrub must not brick real test commands.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SENTINEL")
        (tmp_path / "test_smoke.py").write_text("def test_ok() -> None:\n    assert True\n")

        input_data = ValidateOutputInput(
            task_id="scrub1",
            worktree_path=str(tmp_path),
            files=[],
            validation=ValidationConfig(
                auto_fix=False,
                run_ruff_lint=False,
                run_ruff_format=False,
                run_tests=True,
                test_command="python -m pytest -q -p no:cacheprovider test_smoke.py",
            ),
        )

        results = await validate_output(input_data)

        assert len(results) == 1
        assert results[0].check_name == "tests"
        assert results[0].passed is True
