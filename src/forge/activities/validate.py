"""Validation activity for Forge.

Runs deterministic checks (ruff lint, ruff format, tests) against generated files.

Design follows Function Core / Imperative Shell:
- Internal transport: _SubprocessResult
- Pure function: parse_check_result
- Imperative shell (fix): _run_ruff_lint_fix, _run_ruff_format_fix
- Imperative shell (check): _run_ruff_lint, _run_ruff_format_check, _run_tests
- Temporal activity: validate_output
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

from temporalio import activity

from forge.models import ValidateOutputInput, ValidationResult

SUBPROCESS_TIMEOUT_SECONDS = 60
RUFF_CONFIG = "tool-config/ruff.toml"


# ---------------------------------------------------------------------------
# Internal transport
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _SubprocessResult:
    """Result of a subprocess invocation. Internal transport only."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


# ---------------------------------------------------------------------------
# Subprocess wrapper
# ---------------------------------------------------------------------------


def _run_command(
    args: list[str],
    cwd: Path,
    timeout: int = SUBPROCESS_TIMEOUT_SECONDS,
) -> _SubprocessResult:
    """Execute a command and return the result. Does not raise on non-zero exit."""
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return _SubprocessResult(
        returncode=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


# ---------------------------------------------------------------------------
# Pure function
# ---------------------------------------------------------------------------


def parse_check_result(check_name: str, result: _SubprocessResult) -> ValidationResult:
    """Convert a subprocess result to a ValidationResult."""
    if result.ok:
        return ValidationResult(
            check_name=check_name,
            passed=True,
            summary=f"{check_name} passed",
        )

    output = result.stdout or result.stderr
    # Truncate for the summary, keep full output in details.
    summary_limit = 200
    summary = output[:summary_limit] + "..." if len(output) > summary_limit else output

    return ValidationResult(
        check_name=check_name,
        passed=False,
        summary=f"{check_name} failed: {summary}",
        details=output if len(output) > summary_limit else None,
    )


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def _run_ruff_lint_fix(worktree_path: Path, file_paths: list[str]) -> None:
    """Run ruff lint auto-fix on the given files. Ignores exit code."""
    _run_command(
        ["ruff", "check", "--config", RUFF_CONFIG, "--fix", *file_paths],
        cwd=worktree_path,
    )


def _run_ruff_format_fix(worktree_path: Path, file_paths: list[str]) -> None:
    """Run ruff format on the given files. Ignores exit code."""
    _run_command(
        ["ruff", "format", "--config", RUFF_CONFIG, *file_paths],
        cwd=worktree_path,
    )


def _run_ruff_lint(worktree_path: Path, file_paths: list[str]) -> ValidationResult:
    """Run ruff lint check on the given files."""
    result = _run_command(
        ["ruff", "check", "--config", RUFF_CONFIG, "--no-fix", *file_paths],
        cwd=worktree_path,
    )
    return parse_check_result("ruff_lint", result)


def _run_ruff_format_check(worktree_path: Path, file_paths: list[str]) -> ValidationResult:
    """Run ruff format check on the given files."""
    result = _run_command(
        ["ruff", "format", "--config", RUFF_CONFIG, "--check", *file_paths],
        cwd=worktree_path,
    )
    return parse_check_result("ruff_format", result)


def _run_tests(worktree_path: Path, test_command: str) -> ValidationResult:
    """Run the test command via shell."""
    result = _run_command(
        ["sh", "-c", test_command],
        cwd=worktree_path,
    )
    return parse_check_result("tests", result)


@activity.defn
async def validate_output(input: ValidateOutputInput) -> list[ValidationResult]:
    """Run enabled validation checks against generated files."""
    wt = Path(input.worktree_path)

    # Fix phase: auto-correct cosmetic issues before checking.
    # Lint fix first (may change imports), then format fix.
    if input.validation.auto_fix and input.files:
        _run_ruff_lint_fix(wt, input.files)
        _run_ruff_format_fix(wt, input.files)

    # Check phase: validate the (possibly fixed) files.
    results: list[ValidationResult] = []

    if input.validation.run_ruff_lint:
        results.append(_run_ruff_lint(wt, input.files))

    if input.validation.run_ruff_format:
        results.append(_run_ruff_format_check(wt, input.files))

    if input.validation.run_tests and input.validation.test_command:
        results.append(_run_tests(wt, input.validation.test_command))

    return results
