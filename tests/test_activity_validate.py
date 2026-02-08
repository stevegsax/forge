"""Tests for forge.activities.validate — validation checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.activities.validate import (
    _run_ruff_format_check,
    _run_ruff_format_fix,
    _run_ruff_lint,
    _run_ruff_lint_fix,
    _SubprocessResult,
    parse_check_result,
    validate_output,
)
from forge.models import ValidateOutputInput, ValidationConfig

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# parse_check_result (pure function)
# ---------------------------------------------------------------------------


class TestParseCheckResult:
    def test_passing_result(self) -> None:
        result = _SubprocessResult(returncode=0, stdout="", stderr="")
        vr = parse_check_result("lint", result)
        assert vr.passed is True
        assert vr.check_name == "lint"
        assert "passed" in vr.summary

    def test_failing_result(self) -> None:
        result = _SubprocessResult(returncode=1, stdout="error on line 5", stderr="")
        vr = parse_check_result("lint", result)
        assert vr.passed is False
        assert "error on line 5" in vr.summary

    def test_long_output_truncated_in_summary(self) -> None:
        long_output = "x" * 300
        result = _SubprocessResult(returncode=1, stdout=long_output, stderr="")
        vr = parse_check_result("lint", result)
        assert len(vr.summary) < len(long_output)
        assert vr.summary.endswith("...")
        assert vr.details == long_output

    def test_stderr_used_when_stdout_empty(self) -> None:
        result = _SubprocessResult(returncode=1, stdout="", stderr="stderr msg")
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
