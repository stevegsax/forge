"""CLI entry point for Forge.

Provides ``forge run``, ``forge worker``, and ``forge eval-planner`` subcommands.

Follows Function Core / Imperative Shell:
- Pure functions: format_task_result, format_validation_results,
  build_task_definition, load_task_definition, format_eval_result,
  format_deterministic_result
- Async shell: _submit_and_wait, _submit_no_wait, _run_eval
- Click commands: main, run, worker, eval_planner
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from forge.git import RepoDiscoveryError, discover_repo_root
from forge.models import (
    ContextConfig,
    ForgeTaskInput,
    TaskDefinition,
    TaskResult,
    TransitionSignal,
    ValidationConfig,
)

if TYPE_CHECKING:
    from forge.eval.models import DeterministicResult, PlanEvalResult
    from forge.models import StepResult, SubTaskResult, ValidationResult

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INFRASTRUCTURE_ERROR = 3


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def format_validation_results(results: list[ValidationResult]) -> str:
    """Format a list of validation results as human-readable lines."""
    lines: list[str] = []
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{tag}] {r.check_name}: {r.summary}")
    return "\n".join(lines)


def format_sub_task_result(sub_task: SubTaskResult) -> str:
    """Format a single SubTaskResult as a compact line."""
    tag = "PASS" if sub_task.status == TransitionSignal.SUCCESS else "FAIL"
    return f"    [{tag}] {sub_task.sub_task_id}: {sub_task.status.value}"


def format_step_result(step: StepResult) -> str:
    """Format a single StepResult as a compact line, with sub-task details if present."""
    tag = "PASS" if step.status == TransitionSignal.SUCCESS else "FAIL"
    sha_short = step.commit_sha[:8] if step.commit_sha else "none"
    lines = [f"  [{tag}] {step.step_id}: {step.status.value} (commit: {sha_short})"]
    for st_result in step.sub_task_results:
        lines.append(format_sub_task_result(st_result))
    return "\n".join(lines)


def format_task_result(result: TaskResult) -> str:
    """Format a TaskResult for human-readable terminal output."""
    lines: list[str] = [
        f"Task: {result.task_id}",
        f"Status: {result.status.value}",
    ]

    if result.plan:
        lines.append(f"Plan: {len(result.plan.steps)} steps")

    if result.step_results:
        lines.append("")
        lines.append("Steps:")
        for sr in result.step_results:
            lines.append(format_step_result(sr))

    if result.validation_results:
        lines.append("")
        lines.append("Validation:")
        lines.append(format_validation_results(result.validation_results))

    if result.error:
        lines.append("")
        lines.append(f"Error: {result.error}")

    if result.worktree_path:
        lines.append("")
        lines.append(f"Worktree: {result.worktree_path}")
    if result.worktree_branch:
        lines.append(f"Branch: {result.worktree_branch}")

    return "\n".join(lines)


def build_task_definition(
    task_id: str,
    description: str,
    target_files: list[str],
    context_files: list[str] | None = None,
    base_branch: str = "main",
    no_lint: bool = False,
    no_format: bool = False,
    run_tests: bool = False,
    test_command: str | None = None,
    no_auto_discover: bool = False,
    token_budget: int | None = None,
    max_import_depth: int | None = None,
) -> TaskDefinition:
    """Build a TaskDefinition from CLI arguments."""
    context_config = ContextConfig(auto_discover=not no_auto_discover)
    if token_budget is not None:
        context_config = context_config.model_copy(update={"token_budget": token_budget})
    if max_import_depth is not None:
        context_config = context_config.model_copy(update={"max_import_depth": max_import_depth})

    return TaskDefinition(
        task_id=task_id,
        description=description,
        target_files=target_files,
        context_files=context_files or [],
        base_branch=base_branch,
        validation=ValidationConfig(
            run_ruff_lint=not no_lint,
            run_ruff_format=not no_format,
            run_tests=run_tests,
            test_command=test_command,
        ),
        context=context_config,
    )


def load_task_definition(path: str) -> TaskDefinition:
    """Load and validate a TaskDefinition from a JSON file.

    Raises:
        click.BadParameter: If the file cannot be read or is invalid JSON.
    """
    try:
        content = Path(path).read_text()
    except OSError as e:
        msg = f"Cannot read task file: {e}"
        raise click.BadParameter(msg, param_hint="'--task-file'") from e

    try:
        return TaskDefinition.model_validate_json(content)
    except Exception as e:
        msg = f"Invalid task definition: {e}"
        raise click.BadParameter(msg, param_hint="'--task-file'") from e


# ---------------------------------------------------------------------------
# Async shell
# ---------------------------------------------------------------------------


async def _submit_and_wait(
    task_def: TaskDefinition,
    repo_root: str,
    temporal_address: str,
    max_attempts: int,
    *,
    plan: bool = False,
    max_step_attempts: int = 2,
    max_sub_task_attempts: int = 2,
) -> TaskResult:
    """Submit a task to Temporal and wait for completion."""
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter

    from forge.workflows import FORGE_TASK_QUEUE, ForgeTaskWorkflow

    client = await Client.connect(temporal_address, data_converter=pydantic_data_converter)

    result = await client.execute_workflow(
        ForgeTaskWorkflow.run,
        ForgeTaskInput(
            task=task_def,
            repo_root=repo_root,
            max_attempts=max_attempts,
            plan=plan,
            max_step_attempts=max_step_attempts,
            max_sub_task_attempts=max_sub_task_attempts,
        ),
        id=f"forge-task-{task_def.task_id}",
        task_queue=FORGE_TASK_QUEUE,
    )
    return result


async def _submit_no_wait(
    task_def: TaskDefinition,
    repo_root: str,
    temporal_address: str,
    max_attempts: int,
    *,
    plan: bool = False,
    max_step_attempts: int = 2,
    max_sub_task_attempts: int = 2,
) -> str:
    """Submit a task to Temporal and return the workflow ID without waiting."""
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter

    from forge.workflows import FORGE_TASK_QUEUE, ForgeTaskWorkflow

    client = await Client.connect(temporal_address, data_converter=pydantic_data_converter)

    handle = await client.start_workflow(
        ForgeTaskWorkflow.run,
        ForgeTaskInput(
            task=task_def,
            repo_root=repo_root,
            max_attempts=max_attempts,
            plan=plan,
            max_step_attempts=max_step_attempts,
            max_sub_task_attempts=max_sub_task_attempts,
        ),
        id=f"forge-task-{task_def.task_id}",
        task_queue=FORGE_TASK_QUEUE,
    )
    return handle.id


# ---------------------------------------------------------------------------
# Click commands
# ---------------------------------------------------------------------------

DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"


@click.group()
@click.version_option(package_name="forge")
def main() -> None:
    """Forge — LLM task orchestrator."""


@main.command()
@click.option("--task-id", help="Unique task identifier.")
@click.option("--description", help="What the task should produce.")
@click.option("--target-file", multiple=True, help="File to create or modify (repeatable).")
@click.option("--context-file", multiple=True, help="File to include as context (repeatable).")
@click.option("--task-file", type=click.Path(exists=True), help="JSON file with task definition.")
@click.option("--json", "output_json", is_flag=True, help="Output TaskResult as JSON.")
@click.option("--no-wait", is_flag=True, help="Submit and print workflow ID without waiting.")
@click.option("--no-lint", is_flag=True, help="Disable ruff lint check.")
@click.option("--no-format", is_flag=True, help="Disable ruff format check.")
@click.option("--run-tests", is_flag=True, help="Enable test validation.")
@click.option("--test-command", help="Custom test command.")
@click.option(
    "--base-branch",
    default="main",
    show_default=True,
    help="Branch to create worktree from.",
)
@click.option("--max-attempts", default=2, show_default=True, type=int, help="Retry limit.")
@click.option("--plan", "use_plan", is_flag=True, help="Enable planning mode.")
@click.option(
    "--max-step-attempts",
    default=2,
    show_default=True,
    type=int,
    help="Retry limit per step in planning mode.",
)
@click.option(
    "--max-sub-task-attempts",
    default=2,
    show_default=True,
    type=int,
    help="Retry limit per sub-task in fan-out steps.",
)
@click.option("--no-auto-discover", is_flag=True, help="Disable automatic context discovery.")
@click.option(
    "--token-budget",
    type=int,
    default=None,
    help="Token budget for context (default: 100000).",
)
@click.option(
    "--max-import-depth",
    type=int,
    default=None,
    help="How deep to trace imports (default: 2).",
)
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
    help="Temporal server address.",
)
def run(
    task_id: str | None,
    description: str | None,
    target_file: tuple[str, ...],
    context_file: tuple[str, ...],
    task_file: str | None,
    output_json: bool,
    no_wait: bool,
    no_lint: bool,
    no_format: bool,
    run_tests: bool,
    test_command: str | None,
    base_branch: str,
    max_attempts: int,
    use_plan: bool,
    max_step_attempts: int,
    max_sub_task_attempts: int,
    no_auto_discover: bool,
    token_budget: int | None,
    max_import_depth: int | None,
    temporal_address: str,
) -> None:
    """Submit a task and wait for the result."""
    # --- Mutual exclusion: task-file vs inline ---
    inline_provided = any([task_id, description, target_file])
    if task_file and inline_provided:
        raise click.UsageError(
            "Cannot combine --task-file with --task-id/--description/--target-file."
        )
    if not task_file and not inline_provided:
        raise click.UsageError(
            "Provide either --task-file or --task-id/--description/--target-file."
        )

    # --- Build TaskDefinition ---
    if task_file:
        task_def = load_task_definition(task_file)
    else:
        if not task_id:
            raise click.UsageError("--task-id is required for inline task definition.")
        if not description:
            raise click.UsageError("--description is required for inline task definition.")
        if not target_file and not use_plan:
            raise click.UsageError(
                "--target-file is required for inline task definition (unless --plan is set)."
            )

        task_def = build_task_definition(
            task_id=task_id,
            description=description,
            target_files=list(target_file),
            context_files=list(context_file),
            base_branch=base_branch,
            no_lint=no_lint,
            no_format=no_format,
            run_tests=run_tests,
            test_command=test_command,
            no_auto_discover=no_auto_discover,
            token_budget=token_budget,
            max_import_depth=max_import_depth,
        )

    # --- Discover repo root ---
    try:
        repo_root = str(discover_repo_root())
    except RepoDiscoveryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)

    # --- Submit ---
    try:
        if no_wait:
            workflow_id = asyncio.run(
                _submit_no_wait(
                    task_def,
                    repo_root,
                    temporal_address,
                    max_attempts,
                    plan=use_plan,
                    max_step_attempts=max_step_attempts,
                    max_sub_task_attempts=max_sub_task_attempts,
                )
            )
            click.echo(workflow_id)
        else:
            result = asyncio.run(
                _submit_and_wait(
                    task_def,
                    repo_root,
                    temporal_address,
                    max_attempts,
                    plan=use_plan,
                    max_step_attempts=max_step_attempts,
                    max_sub_task_attempts=max_sub_task_attempts,
                )
            )
            if output_json:
                click.echo(result.model_dump_json(indent=2))
            else:
                click.echo(format_task_result(result))

            if result.status == TransitionSignal.FAILURE_TERMINAL:
                sys.exit(EXIT_FAILURE)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


@main.command()
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
    help="Temporal server address.",
)
def worker(temporal_address: str) -> None:
    """Start the Temporal worker."""
    from forge.worker import run_worker

    asyncio.run(run_worker(address=temporal_address))


# ---------------------------------------------------------------------------
# Eval-planner pure functions
# ---------------------------------------------------------------------------


def format_deterministic_result(det: DeterministicResult) -> str:
    """Format a DeterministicResult as human-readable lines."""
    from forge.eval.models import CheckStatus

    lines: list[str] = []
    for check in det.checks:
        tag = {CheckStatus.PASS: "PASS", CheckStatus.FAIL: "FAIL", CheckStatus.SKIP: "SKIP"}[
            check.status
        ]
        lines.append(f"  [{tag}] {check.check_name}: {check.message}")
        for detail in check.details:
            lines.append(f"         {detail}")
    return "\n".join(lines)


def format_eval_result(result: PlanEvalResult) -> str:
    """Format a PlanEvalResult for human-readable terminal output."""
    lines: list[str] = [
        f"Case: {result.case_id}",
        f"Plan: {len(result.plan.steps)} step(s)",
        f"Deterministic: {'PASS' if result.deterministic.all_passed else 'FAIL'}",
    ]
    lines.append(format_deterministic_result(result.deterministic))

    if result.judge:
        lines.append("")
        lines.append("Judge scores:")
        for score in result.judge.scores:
            lines.append(f"  {score.criterion.value}: {score.score}/5 — {score.rationale}")
        lines.append(f"  Overall: {result.judge.overall_assessment}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Eval-planner async shell
# ---------------------------------------------------------------------------


async def _run_eval(
    corpus_dir: str,
    plans_dir: str | None,
    *,
    run_judge: bool,
    judge_model: str | None,
) -> list[PlanEvalResult]:
    """Load corpus, discover plans, and run evaluation."""
    from forge.eval.corpus import discover_eval_cases, list_repo_files
    from forge.eval.deterministic import run_deterministic_checks
    from forge.eval.runner import build_eval_result
    from forge.models import Plan

    cases = discover_eval_cases(Path(corpus_dir))
    if not cases:
        click.echo("No eval cases found.", err=True)
        return []

    # Load plans from plans_dir if provided, otherwise use reference plans from cases
    plans: dict[str, Plan] = {}
    if plans_dir:
        plans_path = Path(plans_dir)
        if plans_path.is_dir():
            for json_file in sorted(plans_path.glob("*.json")):
                try:
                    content = json_file.read_text()
                    plan = Plan.model_validate_json(content)
                    plans[plan.task_id] = plan
                except Exception:
                    click.echo(f"Warning: skipping invalid plan {json_file}", err=True)

    results: list[PlanEvalResult] = []
    for case in cases:
        # Try to find a plan: by task_id from plans_dir, or reference_plan from case
        plan = plans.get(case.task.task_id) or case.reference_plan
        if plan is None:
            click.echo(f"Warning: no plan for case {case.case_id}, skipping.", err=True)
            continue

        repo_root = Path(case.repo_root)
        known_files = list_repo_files(repo_root) if repo_root.is_dir() else None

        det = run_deterministic_checks(plan, case.task, known_files)

        verdict = None
        if run_judge:
            from forge.eval.judge import judge_plan

            verdict = await judge_plan(case, plan, model_name=judge_model)

        result = build_eval_result(case.case_id, plan, det, verdict)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# eval-planner command
# ---------------------------------------------------------------------------


@main.command("eval-planner")
@click.option(
    "--corpus-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing eval case JSON files.",
)
@click.option(
    "--plans-dir",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing plan JSON files.",
)
@click.option("--judge/--no-judge", default=False, help="Run LLM judge scoring.")
@click.option(
    "--judge-model",
    default=None,
    help="Model to use as judge (default: claude-sonnet-4-5-20250929).",
)
@click.option("--dry-run", is_flag=True, help="List cases without evaluating.")
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Directory to save run results JSON.",
)
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON.")
def eval_planner(
    corpus_dir: str,
    plans_dir: str | None,
    judge: bool,
    judge_model: str | None,
    dry_run: bool,
    output_dir: str | None,
    output_json: bool,
) -> None:
    """Evaluate planner output against an eval corpus."""
    from forge.eval.corpus import discover_eval_cases

    cases = discover_eval_cases(Path(corpus_dir))
    if not cases:
        click.echo("No eval cases found.", err=True)
        sys.exit(EXIT_FAILURE)

    if dry_run:
        click.echo(f"Found {len(cases)} eval case(s):")
        for case in cases:
            tags = f" [{', '.join(case.tags)}]" if case.tags else ""
            click.echo(f"  {case.case_id}: {case.task.description}{tags}")
        return

    results = asyncio.run(
        _run_eval(corpus_dir, plans_dir, run_judge=judge, judge_model=judge_model)
    )

    if not results:
        click.echo("No results produced.", err=True)
        sys.exit(EXIT_FAILURE)

    if output_json:
        import json

        data = [r.model_dump(mode="json") for r in results]
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        for result in results:
            click.echo(format_eval_result(result))
            click.echo("")

    # Save if output-dir specified
    if output_dir:
        import uuid

        from forge.eval.models import EvalRunRecord

        record = EvalRunRecord(
            run_id=str(uuid.uuid4())[:8],
            model_name="unknown",
            judge_model=judge_model if judge else None,
            results=results,
        )
        from forge.eval.runner import save_run

        path = save_run(record, output_dir=Path(output_dir))
        click.echo(f"Results saved to {path}")

    # Exit with failure if any deterministic check failed
    if any(not r.deterministic.all_passed for r in results):
        sys.exit(EXIT_FAILURE)
