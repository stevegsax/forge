"""CLI entry point for Forge.

Provides ``forge run``, ``forge worker``, ``forge status``,
``forge eval-planner``, ``forge extract``, and ``forge playbooks`` subcommands.

Follows Function Core / Imperative Shell:
- Pure functions: format_task_result, format_validation_results,
  build_task_definition, load_task_definition, format_eval_result,
  format_deterministic_result, format_extraction_result, format_playbook_entry
- Async shell: _submit_and_wait, _submit_no_wait, _run_eval, _submit_extraction
- Click commands: main, run, worker, status, eval_planner, extract, playbooks
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from forge.domains import get_domain_config
from forge.git import RepoDiscoveryError, discover_repo_root
from forge.models import (
    ContextConfig,
    ForgeTaskInput,
    ModelConfig,
    TaskDefinition,
    TaskDomain,
    TaskResult,
    ThinkingConfig,
    TransitionSignal,
    ValidationConfig,
)

if TYPE_CHECKING:
    from forge.eval.models import DeterministicResult, PlanEvalResult
    from forge.models import (
        ExtractionWorkflowResult,
        LLMStats,
        StepResult,
        SubTaskResult,
        ValidationResult,
    )

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


def format_sub_task_result(sub_task: SubTaskResult, indent: int = 4) -> str:
    """Format a single SubTaskResult as a compact line, with nested sub-tasks if present."""
    prefix = " " * indent
    tag = "PASS" if sub_task.status == TransitionSignal.SUCCESS else "FAIL"
    lines = [f"{prefix}[{tag}] {sub_task.sub_task_id}: {sub_task.status.value}"]
    for nested in sub_task.sub_task_results:
        lines.append(format_sub_task_result(nested, indent=indent + 2))
    return "\n".join(lines)


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


def format_llm_stats(stats: LLMStats) -> str:
    """Format LLMStats as a compact human-readable string."""
    parts = [
        f"model={stats.model_name}",
        f"tokens={stats.input_tokens}in/{stats.output_tokens}out",
        f"latency={stats.latency_ms:.0f}ms",
    ]
    if stats.cache_creation_input_tokens or stats.cache_read_input_tokens:
        parts.append(
            f"cache={stats.cache_creation_input_tokens}write/{stats.cache_read_input_tokens}read"
        )
    return " ".join(parts)


def format_verbose_result(result: TaskResult) -> str:
    """Format a TaskResult with full interaction details from the store."""
    lines = [format_task_result(result)]

    if result.llm_stats:
        lines.append("")
        lines.append(f"LLM: {format_llm_stats(result.llm_stats)}")

    if result.planner_stats:
        lines.append(f"Planner: {format_llm_stats(result.planner_stats)}")

    if result.sanity_check_count > 0:
        lines.append(f"Sanity checks: {result.sanity_check_count}")

    if result.context_stats:
        cs = result.context_stats
        lines.append("")
        lines.append("Context:")
        lines.append(f"  Files discovered: {cs.files_discovered}")
        lines.append(f"  Full content: {cs.files_included_full}")
        lines.append(f"  Signatures only: {cs.files_included_signatures}")
        lines.append(f"  Estimated tokens: {cs.total_estimated_tokens}")
        lines.append(f"  Budget utilization: {cs.budget_utilization:.1%}")

    def _append_sub_task_stats(st: SubTaskResult, indent: int = 4) -> None:
        prefix = " " * indent
        if st.llm_stats:
            lines.append(f"{prefix}Sub-task {st.sub_task_id}: {format_llm_stats(st.llm_stats)}")
        for nested in st.sub_task_results:
            _append_sub_task_stats(nested, indent=indent + 2)

    for sr in result.step_results:
        if sr.llm_stats:
            lines.append(f"  Step {sr.step_id}: {format_llm_stats(sr.llm_stats)}")
        for st in sr.sub_task_results:
            _append_sub_task_stats(st)

    # Query store for interaction details
    try:
        from forge.store import get_db_path, get_engine, get_interactions

        db_path = get_db_path()
        if db_path is not None and db_path.exists():
            engine = get_engine(db_path)
            interactions = get_interactions(engine, result.task_id)
            if interactions:
                lines.append("")
                lines.append(f"Interactions ({len(interactions)}):")
                for ix in interactions:
                    role = ix["role"]
                    model = ix["model_name"]
                    tokens = f"{ix['input_tokens']}in/{ix['output_tokens']}out"
                    latency = f"{ix['latency_ms']:.0f}ms"
                    step_info = ""
                    if ix.get("step_id"):
                        step_info = f" step={ix['step_id']}"
                    if ix.get("sub_task_id"):
                        step_info += f" sub_task={ix['sub_task_id']}"
                    cache_info = ""
                    cache_write = ix.get("cache_creation_input_tokens", 0)
                    cache_read = ix.get("cache_read_input_tokens", 0)
                    if cache_write or cache_read:
                        cache_info = f" cache={cache_write}write/{cache_read}read"
                    lines.append(f"  [{role}]{step_info} {model} {tokens} {latency}{cache_info}")
    except Exception:
        pass

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
    include_deps: bool = False,
    domain: TaskDomain = TaskDomain.CODE_GENERATION,
) -> TaskDefinition:
    """Build a TaskDefinition from CLI arguments.

    Domain provides validation defaults; CLI flags override them.
    """
    domain_config = get_domain_config(domain)
    vd = domain_config.validation_defaults

    context_config = ContextConfig(
        auto_discover=not no_auto_discover,
        include_dependencies=include_deps,
    )
    if token_budget is not None:
        context_config = context_config.model_copy(update={"token_budget": token_budget})
    if max_import_depth is not None:
        context_config = context_config.model_copy(update={"max_import_depth": max_import_depth})

    return TaskDefinition(
        task_id=task_id,
        description=description,
        domain=domain,
        target_files=target_files,
        context_files=context_files or [],
        base_branch=base_branch,
        validation=ValidationConfig(
            auto_fix=vd.auto_fix,
            run_ruff_lint=vd.run_ruff_lint and not no_lint,
            run_ruff_format=vd.run_ruff_format and not no_format,
            run_tests=run_tests or vd.run_tests,
            test_command=test_command or vd.test_command,
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
    max_fan_out_depth: int = 1,
    max_exploration_rounds: int = 10,
    sanity_check_interval: int = 0,
    resolve_conflicts: bool = True,
    model_routing: ModelConfig | None = None,
    thinking: ThinkingConfig | None = None,
    sync_mode: bool = True,
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
            max_fan_out_depth=max_fan_out_depth,
            max_exploration_rounds=max_exploration_rounds,
            sanity_check_interval=sanity_check_interval,
            resolve_conflicts=resolve_conflicts,
            model_routing=model_routing or ModelConfig(),
            thinking=thinking or ThinkingConfig(),
            sync_mode=sync_mode,
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
    max_fan_out_depth: int = 1,
    max_exploration_rounds: int = 10,
    sanity_check_interval: int = 0,
    resolve_conflicts: bool = True,
    model_routing: ModelConfig | None = None,
    thinking: ThinkingConfig | None = None,
    sync_mode: bool = True,
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
            max_fan_out_depth=max_fan_out_depth,
            max_exploration_rounds=max_exploration_rounds,
            sanity_check_interval=sanity_check_interval,
            resolve_conflicts=resolve_conflicts,
            model_routing=model_routing or ModelConfig(),
            thinking=thinking or ThinkingConfig(),
            sync_mode=sync_mode,
        ),
        id=f"forge-task-{task_def.task_id}",
        task_queue=FORGE_TASK_QUEUE,
    )
    return handle.id


def _persist_run(result: TaskResult, workflow_id: str) -> None:
    """Best-effort persistence of run result to the store."""
    try:
        from forge.store import get_db_path, get_engine
        from forge.store import save_run as store_save_run

        db_path = get_db_path()
        if db_path is None:
            return
        engine = get_engine(db_path)
        store_save_run(engine, result, workflow_id)
    except Exception:
        pass


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
@click.option(
    "--max-fan-out-depth",
    default=1,
    show_default=True,
    type=int,
    help="Maximum recursive fan-out depth. 1 = flat fan-out only.",
)
@click.option("--verbose", is_flag=True, help="Show detailed LLM stats and interactions.")
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
    "--max-exploration-rounds",
    type=int,
    default=10,
    show_default=True,
    help="Max rounds of LLM-guided context exploration (0 disables).",
)
@click.option(
    "--include-deps",
    is_flag=True,
    help="Include dependency file contents in upfront context (default: off).",
)
@click.option("--no-explore", is_flag=True, help="Disable LLM-guided context exploration.")
@click.option(
    "--reasoning-model",
    default=None,
    help="Override the model used for REASONING tier (planning).",
)
@click.option(
    "--generation-model",
    default=None,
    help="Override the model used for GENERATION tier (code gen).",
)
@click.option(
    "--summarization-model",
    default=None,
    help="Override the model used for SUMMARIZATION tier (extraction).",
)
@click.option(
    "--classification-model",
    default=None,
    help="Override the model used for CLASSIFICATION tier (exploration).",
)
@click.option(
    "--thinking-budget",
    type=int,
    default=10000,
    show_default=True,
    help="Token budget for extended thinking in planner (Sonnet). Opus uses adaptive.",
)
@click.option("--no-thinking", is_flag=True, help="Disable extended thinking for planner.")
@click.option(
    "--sanity-check-interval",
    type=int,
    default=0,
    show_default=True,
    help="Run sanity check every N steps in planning mode (0 = disabled).",
)
@click.option(
    "--no-resolve-conflicts",
    is_flag=True,
    help="Disable LLM-based conflict resolution for fan-out file conflicts.",
)
@click.option(
    "--sync/--no-sync",
    "sync_mode",
    default=False,
    show_default=True,
    help="Use synchronous Messages API. --no-sync enables batch mode (default).",
)
@click.option(
    "--domain",
    type=click.Choice(["code_generation", "research", "code_review", "documentation", "generic"]),
    default="code_generation",
    show_default=True,
    help="Task domain: code_generation, research, code_review, documentation, generic.",
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
    max_fan_out_depth: int,
    verbose: bool,
    no_auto_discover: bool,
    token_budget: int | None,
    max_import_depth: int | None,
    include_deps: bool,
    max_exploration_rounds: int,
    no_explore: bool,
    reasoning_model: str | None,
    generation_model: str | None,
    summarization_model: str | None,
    classification_model: str | None,
    thinking_budget: int,
    no_thinking: bool,
    sanity_check_interval: int,
    no_resolve_conflicts: bool,
    sync_mode: bool,
    domain: str,
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
            include_deps=include_deps,
            domain=TaskDomain(domain),
        )

    # --- Discover repo root ---
    try:
        repo_root = str(discover_repo_root())
    except RepoDiscoveryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)

    # --- Compute exploration rounds ---
    effective_exploration_rounds = 0 if no_explore else max_exploration_rounds

    # --- Build model routing config ---
    model_overrides: dict[str, str] = {}
    if reasoning_model:
        model_overrides["reasoning"] = reasoning_model
    if generation_model:
        model_overrides["generation"] = generation_model
    if summarization_model:
        model_overrides["summarization"] = summarization_model
    if classification_model:
        model_overrides["classification"] = classification_model
    model_routing = ModelConfig(**model_overrides) if model_overrides else None

    # --- Build thinking config ---
    thinking = ThinkingConfig(
        enabled=not no_thinking,
        budget_tokens=thinking_budget,
    )

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
                    max_fan_out_depth=max_fan_out_depth,
                    max_exploration_rounds=effective_exploration_rounds,
                    sanity_check_interval=sanity_check_interval,
                    resolve_conflicts=not no_resolve_conflicts,
                    model_routing=model_routing,
                    thinking=thinking,
                    sync_mode=sync_mode,
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
                    max_fan_out_depth=max_fan_out_depth,
                    max_exploration_rounds=effective_exploration_rounds,
                    sanity_check_interval=sanity_check_interval,
                    resolve_conflicts=not no_resolve_conflicts,
                    model_routing=model_routing,
                    thinking=thinking,
                    sync_mode=sync_mode,
                )
            )

            # Persist run to store (best-effort)
            workflow_id = f"forge-task-{task_def.task_id}"
            _persist_run(result, workflow_id)

            if output_json:
                click.echo(result.model_dump_json(indent=2))
            elif verbose:
                click.echo(format_verbose_result(result))
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
@click.option(
    "--batch-poll-interval",
    type=int,
    default=60,
    show_default=True,
    help="Seconds between batch polling runs.",
)
@click.option(
    "--extraction-interval",
    type=int,
    default=14400,
    show_default=True,
    help="Seconds between knowledge extraction schedule runs (default 4 hours).",
)
def worker(temporal_address: str, batch_poll_interval: int, extraction_interval: int) -> None:
    """Start the Temporal worker."""
    from forge.worker import run_worker

    asyncio.run(
        run_worker(
            address=temporal_address,
            batch_poll_interval=batch_poll_interval,
            extraction_interval=extraction_interval,
        )
    )


# ---------------------------------------------------------------------------
# Status command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--workflow-id", default=None, help="Show details for a specific workflow run.")
@click.option("--verbose", is_flag=True, help="Show full interaction details.")
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
@click.option(
    "--limit",
    type=int,
    default=20,
    show_default=True,
    help="Number of recent runs to show.",
)
def status(
    workflow_id: str | None,
    verbose: bool,
    output_json: bool,
    limit: int,
) -> None:
    """List recent runs or show details for a specific workflow."""
    import json as json_mod

    from forge.store import get_db_path, get_engine, get_interactions, get_run, list_recent_runs

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        click.echo("No store available. Set FORGE_DB_PATH or run a workflow first.", err=True)
        sys.exit(EXIT_FAILURE)

    engine = get_engine(db_path)

    if workflow_id:
        run_data = get_run(engine, workflow_id)
        if run_data is None:
            click.echo(f"No run found for workflow ID: {workflow_id}", err=True)
            sys.exit(EXIT_FAILURE)

        if output_json:
            click.echo(json_mod.dumps(run_data, indent=2, default=str))
        else:
            click.echo(f"Workflow: {run_data['workflow_id']}")
            click.echo(f"Task: {run_data['task_id']}")
            click.echo(f"Status: {run_data['status']}")
            click.echo(f"Created: {run_data['created_at']}")

            if verbose:
                task_id = run_data["task_id"]
                interactions = get_interactions(engine, task_id)
                if interactions:
                    click.echo("")
                    click.echo(f"Interactions ({len(interactions)}):")
                    for ix in interactions:
                        role = ix["role"]
                        model = ix["model_name"]
                        tokens = f"{ix['input_tokens']}in/{ix['output_tokens']}out"
                        latency = f"{ix['latency_ms']:.0f}ms"
                        step_info = ""
                        if ix.get("step_id"):
                            step_info = f" step={ix['step_id']}"
                        if ix.get("sub_task_id"):
                            step_info += f" sub_task={ix['sub_task_id']}"
                        click.echo(f"  [{role}]{step_info} {model} {tokens} {latency}")

                        click.echo(f"    System prompt: {ix['system_prompt'][:200]}...")
                        click.echo(f"    User prompt: {ix['user_prompt'][:200]}")
    else:
        runs = list_recent_runs(engine, limit=limit)
        if not runs:
            click.echo("No runs found.")
            return

        if output_json:
            click.echo(json_mod.dumps(runs, indent=2, default=str))
        else:
            click.echo(f"Recent runs ({len(runs)}):")
            click.echo("")
            for r in runs:
                click.echo(
                    f"  {r['workflow_id']}  {r['task_id']}  {r['status']}  {r['created_at']}"
                )


# ---------------------------------------------------------------------------
# Extract command (Phase 6)
# ---------------------------------------------------------------------------


def format_extraction_result(result: ExtractionWorkflowResult) -> str:
    """Format an ExtractionWorkflowResult for human-readable output."""
    lines = [
        f"Entries created: {result.entries_created}",
        f"Runs processed: {len(result.source_workflow_ids)}",
    ]
    if result.source_workflow_ids:
        lines.append("")
        lines.append("Source workflows:")
        for wid in result.source_workflow_ids:
            lines.append(f"  {wid}")
    return "\n".join(lines)


async def _submit_extraction(
    temporal_address: str,
    limit: int,
    since_hours: int,
) -> ExtractionWorkflowResult:
    """Submit extraction workflow to Temporal and wait for completion."""
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter

    from forge.extraction_workflow import ForgeExtractionWorkflow
    from forge.models import ExtractionWorkflowInput
    from forge.workflows import FORGE_TASK_QUEUE

    client = await Client.connect(temporal_address, data_converter=pydantic_data_converter)

    result = await client.execute_workflow(
        ForgeExtractionWorkflow.run,
        ExtractionWorkflowInput(limit=limit, since_hours=since_hours),
        id="forge-extraction",
        task_queue=FORGE_TASK_QUEUE,
    )
    return result


@main.command()
@click.option(
    "--limit",
    default=10,
    show_default=True,
    type=int,
    help="Max runs to process.",
)
@click.option(
    "--since-hours",
    default=24,
    show_default=True,
    type=int,
    help="Look-back window in hours.",
)
@click.option("--dry-run", is_flag=True, help="Show unextracted runs without running extraction.")
@click.option("--json", "output_json", is_flag=True, help="Output result as JSON.")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
    help="Temporal server address.",
)
def extract(
    limit: int,
    since_hours: int,
    dry_run: bool,
    output_json: bool,
    temporal_address: str,
) -> None:
    """Extract knowledge from completed runs into playbooks."""
    if dry_run:
        from forge.store import get_db_path, get_engine, get_unextracted_runs

        db_path = get_db_path()
        if db_path is None or not db_path.exists():
            click.echo("No store available.", err=True)
            sys.exit(EXIT_FAILURE)

        engine = get_engine(db_path)
        runs = get_unextracted_runs(engine, limit=limit)

        if not runs:
            click.echo("No unextracted runs found.")
            return

        click.echo(f"Unextracted runs ({len(runs)}):")
        click.echo("")
        for r in runs:
            click.echo(f"  {r['workflow_id']}  {r['task_id']}  {r['status']}  {r['created_at']}")
        return

    try:
        result = asyncio.run(_submit_extraction(temporal_address, limit, since_hours))

        if output_json:
            click.echo(result.model_dump_json(indent=2))
        else:
            click.echo(format_extraction_result(result))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


# ---------------------------------------------------------------------------
# Playbooks command (Phase 6)
# ---------------------------------------------------------------------------


def format_playbook_entry(entry: dict) -> str:
    """Format a playbook entry for human-readable terminal output."""
    import json as json_mod

    tags = json_mod.loads(entry["tags_json"]) if isinstance(entry.get("tags_json"), str) else []
    lines = [
        f"  [{entry['id']}] {entry['title']}",
        f"    Tags: {', '.join(tags)}",
        f"    Source: {entry['source_task_id']} ({entry['source_workflow_id']})",
        f"    Created: {entry['created_at']}",
    ]
    return "\n".join(lines)


@main.command()
@click.option("--tag", multiple=True, help="Filter by tag (repeatable).")
@click.option("--task-id", "filter_task_id", default=None, help="Filter by source task ID.")
@click.option(
    "--limit",
    default=20,
    show_default=True,
    type=int,
    help="Max entries to show.",
)
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def playbooks(
    tag: tuple[str, ...],
    filter_task_id: str | None,
    limit: int,
    output_json: bool,
) -> None:
    """List and inspect playbook entries."""
    import json as json_mod

    from forge.store import (
        get_db_path,
        get_engine,
        get_playbooks_by_tags,
        list_recent_playbooks,
    )

    db_path = get_db_path()
    if db_path is None or not db_path.exists():
        click.echo("No store available. Run a workflow first.", err=True)
        sys.exit(EXIT_FAILURE)

    engine = get_engine(db_path)

    if tag:
        entries = get_playbooks_by_tags(engine, list(tag), limit=limit)
    else:
        entries = list_recent_playbooks(engine, limit=limit)

    if filter_task_id:
        entries = [e for e in entries if e.get("source_task_id") == filter_task_id]

    if not entries:
        click.echo("No playbooks found.")
        return

    if output_json:
        click.echo(json_mod.dumps(entries, indent=2, default=str))
    else:
        click.echo(f"Playbooks ({len(entries)}):")
        click.echo("")
        for entry in entries:
            click.echo(format_playbook_entry(entry))
            click.echo("")


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
                    click.echo(f"Warning: failed to parse {json_file.name}, skipping.", err=True)

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
