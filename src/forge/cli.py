"""CLI entry point for Forge.

Provides ``forge run``, ``forge worker``, ``forge status``,
``forge eval-planner``, ``forge playbooks``,
and ``forge start`` subcommands.

Follows Function Core / Imperative Shell:
- Pure functions: format_task_result, format_validation_results,
  build_task_definition, load_task_definition, format_eval_result,
  format_deterministic_result, format_playbook_entry,
  load_workflow_input
- Async shell: _submit, _run_eval, _start_workflow,
  _start_workflow_and_wait
- Click commands: main, run, worker, status, eval_planner, playbooks, start
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import click
from click.core import ParameterSource

from forge.domains import get_domain_config
from forge.git import RepoDiscoveryError, discover_repo_root
from forge.models import (
    ContextConfig,
    Effort,
    ForgeTaskInput,
    ModelConfig,
    TaskDefinition,
    TaskDomain,
    TaskResult,
    ThinkingPolicy,
    TransitionSignal,
    ValidationConfig,
    derive_execution_timeout,
)
from forge.temporal_client import connect_temporal

if TYPE_CHECKING:
    from pbook.transcript import SessionInfo
    from sax_platform.llm import AnthropicLLM
    from sqlalchemy import Engine
    from temporalio.client import Client

    from forge.eval.models import DeterministicResult, EvalCase, PlanEvalResult
    from forge.models import (
        ExportPlaybookResult,
        LLMStats,
        ManualPlaybookResult,
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
# EX_CONFIG (sysexits.h, 78): the environment guard refused to run because
# FORGE_ENV was unset or invalid. Deliberately outside every other command exit
# code so an operator or monitoring harness never confuses "environment not
# declared" with a real command outcome.
EXIT_CONFIG_ERROR = 78


def _require_forge_env() -> None:
    """Refuse to run any command without an explicitly declared environment.

    The composition-root guard for the CLI shell: it reads the process
    environment through the pure ``resolve_forge_env`` (sax_platform.config),
    which refuses to invent a default so reaching the production store is always
    an explicit act. On failure it prints the guard's complete, actionable
    message to stderr and exits ``EXIT_CONFIG_ERROR``; on success it returns
    silently and the command proceeds.
    """
    import os

    from sax_platform.config import ForgeEnvError, resolve_forge_env

    try:
        resolve_forge_env(os.environ)
    except ForgeEnvError as exc:
        click.echo(str(exc), err=True)
        sys.exit(EXIT_CONFIG_ERROR)


def _apply_env_profile(env_value: str) -> None:
    """Load a ``--env`` profile into the process environment before the guard.

    The pure parsing (``parse_env_profile``/``resolve_env_profile_path``) lives
    in ``sax_platform.config``; this shell reads the resolved file, applies the
    parsed ``KEY=VALUE`` pairs over ``os.environ`` (an explicit flag overrides
    ambient values; keys the file omits are left untouched), and declares
    ``FORGE_ENV`` — the profile *name* for a name value, or the file's
    ``FORGE_ENV_TAG`` for a path value. It never sets ``FORGE_PROD_ACK``, so
    ``--env prod`` still fails unless the ack is exported separately. A missing
    file, a malformed profile line, or a path-form profile with no
    ``FORGE_ENV_TAG`` exits ``EXIT_CONFIG_ERROR``.
    """
    import os

    from sax_platform.config import (
        ForgeEnvError,
        parse_env_profile,
        resolve_env_profile_path,
    )

    is_path = "/" in env_value or env_value.endswith(".env")
    path = resolve_env_profile_path(env_value, xdg_config_home=os.environ.get("XDG_CONFIG_HOME"))
    try:
        text = path.read_text()
    except OSError:
        click.echo(f"--env profile not found: {path}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)
    try:
        values = parse_env_profile(text, expand_from=os.environ)
    except ForgeEnvError as exc:
        click.echo(str(exc), err=True)
        sys.exit(EXIT_CONFIG_ERROR)

    for key, value in values.items():
        os.environ[key] = value

    if is_path:
        tag = values.get("FORGE_ENV_TAG", "")
        if not tag:
            click.echo(
                f"--env profile {path} declares no FORGE_ENV_TAG; a path-form "
                "profile must name its environment (add FORGE_ENV_TAG=<prod|dev|test>).",
                err=True,
            )
            sys.exit(EXIT_CONFIG_ERROR)
        os.environ["FORGE_ENV"] = tag
    else:
        os.environ["FORGE_ENV"] = env_value


def _require_store_engine() -> Engine:
    """Resolve the store engine for a CLI command, exiting if it is unconfigured.

    Builds the ``DbSettings`` group at command start (the composition-root
    convention: each command constructs exactly the settings it needs) and
    threads ``settings.url`` into the engine factory. An unset ``FORGE_DB_URL``
    surfaces as a pydantic ``ValidationError``, which is translated into the
    same clean, exit-code-1 message the store used to raise itself.
    """
    from pydantic import ValidationError
    from sax_platform.config import DbSettings

    from forge.store import get_store_engine

    try:
        settings = DbSettings()  # type: ignore[call-arg]  # url comes from FORGE_DB_URL
    except ValidationError:
        click.echo(
            "FORGE_DB_URL is not set. Set it to a 'sqlite:///<path>' URL for "
            "development and tests, or a 'postgresql+psycopg2://...' URL for "
            "production.",
            err=True,
        )
        sys.exit(EXIT_FAILURE)
    return get_store_engine(settings.url)


async def _connect_temporal_checked(temporal_address: str) -> Client:
    """Connect to Temporal after enforcing env/namespace coherence.

    The group callback already ran ``_require_forge_env`` (so FORGE_ENV is
    valid); re-resolving it here — purely, from ``os.environ`` — pairs it with
    the namespace from :class:`TemporalSettings` (the sole FORGE_TEMPORAL_*
    reader) and refuses to connect a dev/test process to production's namespace,
    or a prod process to any other, before the connection opens. An incoherent
    pairing prints the fix and exits ``EXIT_CONFIG_ERROR``. Commands that never
    reach a connect (e.g. ``status``, which reads the store directly) are
    unaffected by the namespace entirely — this only runs on the connect path.
    """
    import os

    from sax_platform.config import (
        ForgeEnvError,
        TemporalSettings,
        require_namespace_coherence,
        resolve_forge_env,
    )

    settings = TemporalSettings()
    try:
        require_namespace_coherence(resolve_forge_env(os.environ), settings.namespace)
    except ForgeEnvError as exc:
        click.echo(str(exc), err=True)
        sys.exit(EXIT_CONFIG_ERROR)
    return await connect_temporal(temporal_address, namespace=settings.namespace, settings=settings)


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
        from sax_platform.config import DbSettings

        from forge.store import get_interactions, get_store_engine

        engine = get_store_engine(DbSettings().url)  # type: ignore[call-arg]  # url from env
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


def load_workflow_input(
    input_json: str | None,
    input_file: str | None,
) -> dict[str, Any]:
    """Parse workflow input from a JSON string or file path.

    Returns an empty dict when neither *input_json* nor *input_file* is
    provided.

    Raises:
        click.UsageError: If both sources are provided.
        click.BadParameter: If the JSON is invalid or not an object.
    """
    if input_json and input_file:
        msg = "Provide either INPUT_JSON or --input-file, not both."
        raise click.UsageError(msg)

    raw: str | None = None
    if input_file:
        try:
            raw = Path(input_file).read_text()
        except OSError as e:
            msg = f"Cannot read input file: {e}"
            raise click.BadParameter(msg, param_hint="'--input-file'") from e
    elif input_json:
        raw = input_json

    if raw is None:
        return {}

    import json

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON: {e}"
        raise click.BadParameter(msg, param_hint="INPUT_JSON") from e

    if not isinstance(parsed, dict):
        msg = f"Expected a JSON object, got {type(parsed).__name__}."
        raise click.BadParameter(msg, param_hint="INPUT_JSON")

    return parsed


# ---------------------------------------------------------------------------
# Async shell
# ---------------------------------------------------------------------------


@overload
async def _submit(
    task_input: ForgeTaskInput, temporal_address: str, *, wait: Literal[True]
) -> TaskResult: ...


@overload
async def _submit(
    task_input: ForgeTaskInput, temporal_address: str, *, wait: Literal[False]
) -> str: ...


async def _submit(
    task_input: ForgeTaskInput, temporal_address: str, *, wait: bool
) -> TaskResult | str:
    """Submit ``ForgeTaskWorkflow`` to Temporal.

    A single helper for both submission modes: with ``wait=True`` it runs the
    workflow to completion and returns its :class:`TaskResult`; with
    ``wait=False`` it starts the workflow and returns its ID immediately. The
    caller builds the :class:`ForgeTaskInput` (the workflow's whole argument),
    so there is no per-field parameter mirror to drift against the model.
    """
    from sax_platform.contracts.constants import FORGE_TASK_QUEUE

    from forge.workflows import ForgeTaskWorkflow

    client = await _connect_temporal_checked(temporal_address)
    workflow_id = f"forge-task-{task_input.task.task_id}"

    # Derive the execution timeout from the permitted batch-wait budget so a
    # legitimately slow batch is never killed (T4.1 ST3c). Sync mode stays flat 48h.
    execution_timeout = derive_execution_timeout(task_input)

    if wait:
        result: TaskResult = await client.execute_workflow(
            ForgeTaskWorkflow.run,
            task_input,
            id=workflow_id,
            task_queue=FORGE_TASK_QUEUE,
            execution_timeout=execution_timeout,
        )
        return result

    handle = await client.start_workflow(
        ForgeTaskWorkflow.run,
        task_input,
        id=workflow_id,
        task_queue=FORGE_TASK_QUEUE,
        execution_timeout=execution_timeout,
    )
    return handle.id


# ---------------------------------------------------------------------------
# Click commands
# ---------------------------------------------------------------------------

DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"


def configure_logging(verbosity: int, *, log_name: str = "forge") -> None:
    """Set up root logger with console + optional file handler.

    Console level is controlled by *verbosity*: 0=WARNING, 1=INFO, 2+=DEBUG.
    A ``RotatingFileHandler`` at DEBUG level is added via
    :func:`forge.logging_config.configure_file_handler` (best-effort).

    When *verbosity* is 0, the console stream handler is omitted if file
    logging is available — this keeps the worker silent on stdout by default.

    The log directory is resolved through the ``LogSettings`` group (the
    composition-root convention) rather than by ``logging_config`` reading the
    environment at point of use: this shell builds ``LogSettings()`` and hands
    the resolved values to :func:`~forge.logging_config.configure_file_handler`.
    """
    from sax_platform.config import LogSettings

    from forge.logging_config import configure_file_handler

    level_map = {0: logging.WARNING, 1: logging.INFO}
    console_level = level_map.get(verbosity, logging.DEBUG)

    root = logging.getLogger()
    root.handlers.clear()

    log_settings = LogSettings()
    file_handler = configure_file_handler(
        log_name=log_name,
        log_dir_override=log_settings.log_dir,
        xdg_state_home=log_settings.xdg_state_home,
    )

    # Only add a console handler when the user explicitly asked for verbosity
    # or when file logging is unavailable (so messages aren't lost).
    if verbosity > 0 or file_handler is None:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(console_level)
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(stream_handler)

    # Level policy (T0.1): the file handler sits at DEBUG, and Python re-checks
    # only *handler* levels — not ancestor *logger* levels — on records that
    # propagate up from a child logger. So keep the ROOT logger at INFO (which
    # gates any NOTSET third-party logger's effective level, dropping its DEBUG
    # before it can reach the file — SDK payloads at DEBUG can carry prompts)
    # while raising the ``forge`` logger to DEBUG so forge's own DEBUG records
    # still reach the file. Without a file handler there is nothing to leak
    # into, so keep the prior console-level behavior and let ``forge`` inherit.
    forge_logger = logging.getLogger("forge")
    if file_handler is not None:
        root.setLevel(logging.INFO)
        forge_logger.setLevel(logging.DEBUG)
    else:
        root.setLevel(console_level)
        forge_logger.setLevel(logging.NOTSET)


# ---------------------------------------------------------------------------
# Position-independent --env plumbing (the environment guard seam)
# ---------------------------------------------------------------------------

#: Shared help for the ``--env`` option, mounted at both the group level and on
#: every command (so it reads the same wherever ``--help`` surfaces it).
_ENV_OPTION_HELP = (
    "Load an env profile before the environment guard runs. A bare NAME "
    "resolves to $XDG_CONFIG_HOME/forge/envs/<NAME>.env and sets FORGE_ENV; "
    "a path (or a value ending in .env) is read verbatim and takes FORGE_ENV "
    "from its FORGE_ENV_TAG. Never supplies FORGE_PROD_ACK. Valid before or "
    "after the subcommand; given at both, the subcommand value wins."
)


class _EnvCommand(click.Command):
    """A ``click.Command`` that accepts ``--env`` in the subcommand position.

    Every command in this CLI is built from this class (via ``_EnvGroup``'s
    ``command_class``), so ``--env`` parses both before the subcommand
    (``forge --env dev run …``, consumed by the group) and after it
    (``forge run --env dev …``, consumed here) with identical semantics.

    ``__init__`` appends the shared ``--env`` option; ``invoke`` applies that
    profile (when it was given at this level) and then runs the ``FORGE_ENV``
    guard, immediately before the command body. The captured value is popped
    out of ``ctx.params`` so the command's own callback signature is untouched.
    Because the guard lives here — at the command seam, not in the group
    callback — ``--help`` and other parse-only paths short-circuit ahead of it
    and need no declared environment. A command-level ``--env`` is applied last,
    so it wins over a group-level one.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.params.append(
            click.Option(
                ["--env", "env_profile"], default=None, metavar="NAME|PATH", help=_ENV_OPTION_HELP
            )
        )

    def invoke(self, ctx: click.Context) -> Any:
        env_profile: str | None = ctx.params.pop("env_profile", None)
        if env_profile is not None:
            _apply_env_profile(env_profile)
        _require_forge_env()
        return super().invoke(ctx)


class _EnvGroup(click.Group):
    """A ``click.Group`` whose commands (and nested groups) are ``_EnvCommand``s.

    ``command_class`` makes every ``@group.command()`` an ``_EnvCommand`` (the
    ``--env``-in-either-position behavior and the guard attach automatically,
    with no per-command decorator), and ``group_class = type`` propagates the
    same group class to nested groups so their subcommands inherit it too. The
    group callback applies a group-level ``--env`` eagerly; the guard itself
    lives on the commands, so a nested group invoked without a subcommand guards
    in its own body.
    """

    command_class = _EnvCommand
    group_class = type


@click.group(cls=_EnvGroup)
@click.version_option(package_name="forge")
@click.option(
    "-v", "log_verbosity", count=True, help="Increase log verbosity (-v INFO, -vv DEBUG)."
)
@click.option("--env", "env_profile", default=None, metavar="NAME|PATH", help=_ENV_OPTION_HELP)
def main(log_verbosity: int, env_profile: str | None) -> None:
    """Forge — LLM task orchestrator."""
    # Apply a group-level --env eagerly so its vars are in place before the
    # subcommand parses; the FORGE_ENV guard runs per-command (see _EnvCommand),
    # not here, so --help stays usable without a declared environment.
    if env_profile is not None:
        _apply_env_profile(env_profile)
    configure_logging(log_verbosity)


# ---------------------------------------------------------------------------
# Model provider validation (CLI parse-time)
# ---------------------------------------------------------------------------

# The set of LLM providers sax_llm.registry.get_provider_by_name actually
# resolves today — anthropic only. T3.3 deleted Mistral chat/registry support;
# Mistral survives only as MistralOcr in sax_platform, a separate OCR pipeline
# forge's LLM call paths (planner/sanity-check/conflict-resolution/generation)
# never touch. Single source of truth so every tier-override flag below
# validates against the same set instead of scattered literals.
_SUPPORTED_MODEL_PROVIDERS: frozenset[str] = frozenset({"anthropic"})


def _validate_model_provider(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    """Validate the optional ``provider:`` prefix of a tier-override model string.

    A bare model name (no ``:``) defaults to anthropic (mirrors
    ``sax_llm.parse_model_id``) and always passes. Without this, an
    unsupported provider prefix (e.g. ``mistral:foo``) surfaces only once the
    value reaches ``sax_llm.registry.get_provider_by_name`` — deep inside a
    retried Temporal activity, several attempts and minutes later. Reject it
    here instead, at CLI parse time, with a clear message naming the
    supported set.
    """
    if value is None or ":" not in value:
        return value
    provider, _, _ = value.partition(":")
    if provider not in _SUPPORTED_MODEL_PROVIDERS:
        supported = ", ".join(sorted(_SUPPORTED_MODEL_PROVIDERS))
        msg = f"Unsupported provider {provider!r} in {value!r}. Supported providers: {supported}."
        raise click.BadParameter(msg, ctx=ctx, param=param)
    return value


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
    callback=_validate_model_provider,
    help="Override the model used for REASONING tier (planning).",
)
@click.option(
    "--generation-model",
    default=None,
    callback=_validate_model_provider,
    help="Override the model used for GENERATION tier (code gen).",
)
@click.option(
    "--summarization-model",
    default=None,
    callback=_validate_model_provider,
    help="Override the model used for SUMMARIZATION tier (extraction).",
)
@click.option(
    "--classification-model",
    default=None,
    callback=_validate_model_provider,
    help="Override the model used for CLASSIFICATION tier (exploration).",
)
@click.option(
    "--effort",
    type=click.Choice(["low", "medium", "high", "xhigh", "max"]),
    default="high",
    show_default=True,
    help=(
        "Extended-thinking effort for planner/sanity-check/conflict-resolution "
        "calls in planning mode (--plan). Single-step mode has no "
        "thinking-configurable LLM call, so this has no effect there."
    ),
)
@click.option(
    "--no-thinking",
    is_flag=True,
    help=(
        "Disable extended thinking for planner/sanity-check/conflict-resolution "
        "calls in planning mode (--plan). Has no effect in single-step mode."
    ),
)
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
    "--log-messages",
    is_flag=True,
    help="Save full API request/response JSON to messages/ in the worktree.",
)
@click.option(
    "--batch-poll-interval",
    type=click.IntRange(min=300),
    default=600,
    show_default=True,
    help="Seconds between batch status polls (min 300, D88). Batch mode only.",
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
    effort: Effort,
    no_thinking: bool,
    sanity_check_interval: int,
    no_resolve_conflicts: bool,
    sync_mode: bool,
    log_messages: bool,
    batch_poll_interval: int,
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

    # --- Build thinking policy ---
    thinking = ThinkingPolicy(enabled=not no_thinking, effort=effort)

    # --- Warn when --effort/--no-thinking are explicitly passed but inert ---
    # Single-step mode (plan=False) has no thinking-configurable LLM call: the
    # sole call on that path is generation, which always runs thinking-disabled
    # (blocks.dispatch's generation arm). Only warn when the user explicitly
    # passed one of these flags — the flags' own defaults must stay silent.
    if not use_plan:
        ctx = click.get_current_context()
        explicit_sources = (ParameterSource.COMMANDLINE, ParameterSource.ENVIRONMENT)
        effort_explicit = ctx.get_parameter_source("effort") in explicit_sources
        no_thinking_explicit = ctx.get_parameter_source("no_thinking") in explicit_sources
        if effort_explicit or no_thinking_explicit:
            click.echo(
                "Warning: --effort/--no-thinking have no effect without --plan — "
                "single-step mode has no thinking-configurable LLM call "
                "(generation runs thinking-disabled).",
                err=True,
            )

    # --- Build the workflow input (the whole ForgeTaskWorkflow argument) ---
    task_input = ForgeTaskInput(
        task=task_def,
        repo_root=repo_root,
        max_attempts=max_attempts,
        plan=use_plan,
        max_step_attempts=max_step_attempts,
        max_sub_task_attempts=max_sub_task_attempts,
        max_fan_out_depth=max_fan_out_depth,
        max_exploration_rounds=effective_exploration_rounds,
        sanity_check_interval=sanity_check_interval,
        resolve_conflicts=not no_resolve_conflicts,
        model_routing=model_routing or ModelConfig(),
        thinking=thinking,
        sync_mode=sync_mode,
        log_messages=log_messages,
        batch_poll_interval_seconds=batch_poll_interval,
    )

    # --- Submit ---
    try:
        if no_wait:
            workflow_id = asyncio.run(_submit(task_input, temporal_address, wait=False))
            click.echo(workflow_id)
        else:
            result = asyncio.run(_submit(task_input, temporal_address, wait=True))

            # The run result is persisted survivably inside ForgeTaskWorkflow.

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
    "--worker-identity",
    envvar="FORGE_WORKER_IDENTITY",
    default=None,
    help=(
        "Base worker identity reported to Temporal (default: {pid}@{hostname}); "
        "the launch-time git version is appended when known."
    ),
)
def worker(
    temporal_address: str,
    worker_identity: str | None,
) -> None:
    """Start the Temporal worker."""
    from forge.worker import run_worker

    # Reconfigure logging so the worker writes to worker.log instead of forge.log.
    ctx = click.get_current_context()
    verbosity = ctx.parent.params.get("log_verbosity", 0) if ctx.parent else 0
    configure_logging(verbosity, log_name="worker")

    asyncio.run(
        run_worker(
            address=temporal_address,
            identity=worker_identity,
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

    from forge.store import get_interactions, get_run, list_recent_runs

    engine = _require_store_engine()

    if workflow_id:
        run_data = get_run(engine, workflow_id)
        if run_data is None:
            click.echo(f"No run found for workflow ID: {workflow_id}", err=True)
            sys.exit(EXIT_FAILURE)

        if output_json:
            click.echo(json_mod.dumps(run_data, indent=2, default=str))
        else:
            click.echo(f"Workflow: {run_data['workflow_id']}")
            click.echo(f"Run: {run_data['run_id']}")
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
                    f"  {r['workflow_id']}  {r['run_id']}  {r['task_id']}  "
                    f"{r['status']}  {r['created_at']}"
                )


# ---------------------------------------------------------------------------
# Ingest command — submit Claude Code transcripts to BatchIngestionWorkflow
# ---------------------------------------------------------------------------


def format_ingest_dry_run(sessions: list[SessionInfo]) -> str:
    """Format a list of pbook SessionInfo objects for dry-run display.

    Groups by project and shows counts + total size. For small groups
    (<=3 sessions) each session is listed with its id prefix and size.
    """
    total_mb = sum(s.size_bytes for s in sessions) / 1024 / 1024
    lines = [f"Found {len(sessions)} session(s) to ingest ({total_mb:.1f} MB):", ""]

    by_project: dict[str, list[SessionInfo]] = {}
    for s in sessions:
        by_project.setdefault(s.project_name, []).append(s)

    for proj_name, proj_sessions in sorted(by_project.items()):
        proj_mb = sum(s.size_bytes for s in proj_sessions) / 1024 / 1024
        lines.append(f"  {proj_name}: {len(proj_sessions)} session(s), {proj_mb:.1f} MB")
        if len(proj_sessions) <= 3:
            for s in proj_sessions:
                size_kb = s.size_bytes / 1024
                lines.append(f"    {s.session_id[:12]}...  {size_kb:.0f} KB")

    return "\n".join(lines)


def format_ingest_result(result: dict[str, Any]) -> str:
    """Format a BatchIngestionWorkflow result dict for human-readable output."""
    return (
        f"Ingestion complete: "
        f"{result.get('sessions_processed', 0)} sessions processed, "
        f"{result.get('total_experiences', 0)} experiences found, "
        f"{result.get('total_entries_created', 0)} entries created."
    )


async def _submit_ingestion(
    temporal_address: str,
    session_dicts: list[dict[str, str]],
) -> dict[str, Any]:
    """Submit BatchIngestionWorkflow to Temporal and wait for completion."""
    import time

    from sax_platform.contracts.constants import FORGE_TASK_QUEUE

    client = await _connect_temporal_checked(temporal_address)

    import json as json_mod

    result: dict[str, Any] = await client.execute_workflow(
        "BatchIngestionWorkflow",
        json_mod.dumps({"sessions": session_dicts}),
        id=f"forge-batch-ingest-{int(time.time())}",
        task_queue=FORGE_TASK_QUEUE,
    )
    return result


@main.command()
@click.argument(
    "transcript_path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--all",
    "ingest_all",
    is_flag=True,
    help="Discover and ingest all sessions from ~/.claude/projects/.",
)
@click.option(
    "--project",
    default="",
    help="Filter discovered sessions by project name (with --all).",
)
@click.option(
    "--min-size",
    default=10240,
    show_default=True,
    type=int,
    help="Minimum session file size in bytes (discovery only).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show sessions that would be ingested without submitting.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Reprocess sessions that pbook has already recorded as ingested.",
)
@click.option("--json", "output_json", is_flag=True, help="Output result as JSON.")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
    help="Temporal server address.",
)
def ingest(
    transcript_path: Path | None,
    ingest_all: bool,
    project: str,
    min_size: int,
    dry_run: bool,
    force: bool,
    output_json: bool,
    temporal_address: str,
) -> None:
    """Ingest Claude Code conversation transcripts into pbook.

    Submits Claude Code JSONL session files to the BatchIngestionWorkflow,
    which uses forge's batch LLM path to analyze each transcript and
    forwards extracted experiences to pbook's ExtractionWorkflow.

    \b
    Single session:
        forge ingest ~/.claude/projects/<id>/session.jsonl

    \b
    All sessions (discovered from ~/.claude/projects/):
        forge ingest --all
        forge ingest --all --project forge
        forge ingest --all --dry-run
    """
    # pbook is an optional dependency for ingestion; fail fast with a
    # clear message rather than an ImportError stack trace.
    try:
        from pbook.transcript import (
            SessionInfo,
            discover_sessions,
            infer_project_name,
        )
    except ImportError:
        click.echo(
            "Error: pbook is not installed. Install it to use 'forge ingest'.",
            err=True,
        )
        sys.exit(EXIT_FAILURE)

    if not transcript_path and not ingest_all:
        click.echo("Error: provide a TRANSCRIPT_PATH or use --all.", err=True)
        sys.exit(EXIT_FAILURE)

    # Build the list of sessions to ingest
    if ingest_all:
        sessions = discover_sessions(min_size=min_size)
        if project:
            sessions = [s for s in sessions if s.project_name == project]
    else:
        assert transcript_path is not None
        path = transcript_path
        proj = project or infer_project_name(path.parent.name)
        sessions = [
            SessionInfo(
                path=str(path),
                session_id=path.stem,
                project_dir_name=path.parent.name,
                project_name=proj,
                size_bytes=path.stat().st_size,
            )
        ]

    if not sessions:
        click.echo("No sessions found.")
        return

    # Filter out already-ingested sessions by querying pbook's store
    # directly. This skips the Temporal round-trip for sessions we
    # already have results for. If pbook's store is unavailable we
    # assume nothing is ingested rather than failing.
    if not force:
        try:
            from pbook.settings import PbookDbSettings
            from pbook.store import build_engine, get_ingested_session_ids

            # build_engine returns None when pbook's store is disabled
            # (PBOOK_DATABASE_URL unset); otherwise a connected engine.
            engine = build_engine(PbookDbSettings())
            if engine is not None:
                ingested_ids = get_ingested_session_ids(engine)
                before = len(sessions)
                sessions = [s for s in sessions if s.session_id not in ingested_ids]
                skipped = before - len(sessions)
                if skipped:
                    click.echo(f"Skipping {skipped} already-ingested session(s).")
        except Exception as exc:
            click.echo(
                f"Warning: could not query pbook for ingested sessions: {exc}",
                err=True,
            )

    if not sessions:
        click.echo("All sessions have been ingested. Use --force to reprocess.")
        return

    # Dry-run path: describe what would happen and exit
    if dry_run:
        click.echo(format_ingest_dry_run(sessions))
        return

    # Build the payload and submit
    session_dicts = [
        {"path": s.path, "project": s.project_name, "session_id": s.session_id} for s in sessions
    ]

    try:
        click.echo(f"Submitting {len(sessions)} session(s) for ingestion...")
        result = asyncio.run(_submit_ingestion(temporal_address, session_dicts))
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)

    if output_json:
        import json as json_mod

        click.echo(json_mod.dumps(result, indent=2))
    else:
        click.echo(format_ingest_result(result))


# ---------------------------------------------------------------------------
# Playbooks command (Phase 6)
# ---------------------------------------------------------------------------


def format_playbook_entry(entry: dict[str, Any]) -> str:
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


@main.group(invoke_without_command=True)
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
@click.pass_context
def playbooks(
    ctx: click.Context,
    tag: tuple[str, ...],
    filter_task_id: str | None,
    limit: int,
    output_json: bool,
) -> None:
    """List and inspect playbook entries."""
    if ctx.invoked_subcommand is not None:
        return

    # Terminal group path (no subcommand): the per-command guard seam does not
    # cover a group's own body, so run the environment guard here before any
    # store access — matching what every _EnvCommand does at invoke.
    _require_forge_env()

    import json as json_mod

    from forge.store import (
        get_playbooks_by_tags,
        list_recent_playbooks,
    )

    engine = _require_store_engine()

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


async def _submit_manual_playbook(
    temporal_address: str,
    raw_json: str,
) -> ManualPlaybookResult:
    """Submit manual playbook workflow to Temporal and wait for completion."""
    from uuid import uuid4

    from sax_platform.contracts.constants import FORGE_TASK_QUEUE

    from forge.manual_playbook_workflow import ManualPlaybookWorkflow
    from forge.models import ManualPlaybookInput

    client = await _connect_temporal_checked(temporal_address)

    result: ManualPlaybookResult = await client.execute_workflow(
        ManualPlaybookWorkflow.run,
        ManualPlaybookInput(raw_json=raw_json),
        id=f"forge-manual-playbook-{uuid4().hex[:8]}",
        task_queue=FORGE_TASK_QUEUE,
    )
    return result


@playbooks.command(name="add")
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a JSON file matching the PlaybookEntry schema.",
)
@click.option(
    "--schema",
    "show_schema",
    is_flag=True,
    help="Print PlaybookEntry JSON schema and exit.",
)
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
    help="Temporal server address.",
)
def playbooks_add(file_path: Path | None, show_schema: bool, temporal_address: str) -> None:
    """Add a playbook entry with LLM review."""
    import json as json_mod

    from forge.models import PlaybookEntry

    if show_schema:
        click.echo(json_mod.dumps(PlaybookEntry.model_json_schema(), indent=2))
        return

    if file_path is None:
        click.echo("Either --file or --schema is required.", err=True)
        sys.exit(EXIT_FAILURE)

    raw_json = file_path.read_text()

    try:
        result = asyncio.run(_submit_manual_playbook(temporal_address, raw_json))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)

    if result.validation_error:
        click.echo(f"Invalid input: {result.validation_error}", err=True)
        sys.exit(EXIT_FAILURE)

    if not result.approved:
        click.echo(f"Rejected: {result.rejection_reason}", err=True)
        sys.exit(EXIT_FAILURE)

    click.echo("")
    click.echo("Playbook entry saved:")
    if result.entry:
        click.echo(f"  {result.entry.title}")
        click.echo(f"    Tags: {', '.join(result.entry.tags)}")
        click.echo(f"    Source: {result.entry.source_task_id}")


async def _submit_export_playbooks(
    temporal_address: str,
    *,
    tags: list[str],
    source_task_id: str,
    limit: int,
) -> ExportPlaybookResult:
    """Submit export playbook workflow to Temporal and wait for completion."""
    from uuid import uuid4

    from sax_platform.contracts.constants import FORGE_TASK_QUEUE

    from forge.export_playbook_workflow import ExportPlaybookWorkflow
    from forge.models import ExportPlaybookInput

    client = await _connect_temporal_checked(temporal_address)

    result: ExportPlaybookResult = await client.execute_workflow(
        ExportPlaybookWorkflow.run,
        ExportPlaybookInput(tags=tags, source_task_id=source_task_id, limit=limit),
        id=f"forge-export-playbooks-{uuid4().hex[:8]}",
        task_queue=FORGE_TASK_QUEUE,
    )
    return result


@playbooks.command(name="export")
@click.option("--tag", multiple=True, help="Filter by tag (repeatable, OR match).")
@click.option("--task-id", "source_task_id", default="", help="Filter by source task ID.")
@click.option("--limit", default=0, type=int, help="Max entries to export (0 = all).")
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write to file instead of stdout.",
)
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
    help="Temporal server address.",
)
def playbooks_export(
    tag: tuple[str, ...],
    source_task_id: str,
    limit: int,
    output_path: Path | None,
    temporal_address: str,
) -> None:
    """Export playbook entries as PlaybookEntry-compatible JSON."""
    import json as json_mod

    try:
        result = asyncio.run(
            _submit_export_playbooks(
                temporal_address,
                tags=list(tag),
                source_task_id=source_task_id,
                limit=limit,
            )
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)

    entries_json = [entry.model_dump() for entry in result.entries]
    output_text = json_mod.dumps(entries_json, indent=2)

    if output_path:
        output_path.write_text(output_text + "\n")
        click.echo(f"Exported {result.count} playbook(s) to {output_path}", err=True)
    else:
        click.echo(output_text)


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
    cases: list[EvalCase],
    plans_dir: str | None,
    *,
    run_judge: bool,
    judge_model: str | None,
) -> list[PlanEvalResult]:
    """Discover plans and run evaluation over already-loaded *cases*.

    The corpus is discovered once by ``eval_planner`` and the case list is
    passed in here — no second ``discover_eval_cases`` scan (T0.2).
    """
    from forge.eval.corpus import list_repo_files
    from forge.eval.deterministic import run_deterministic_checks
    from forge.eval.judge import format_repo_context
    from forge.eval.runner import build_eval_result
    from forge.models import Plan

    if not cases:
        return []

    # Build the judge LLM client ONCE for the whole run (only when judging),
    # then thread it through every judge call — no per-case client and no
    # module-global cache (T3.6 composition root).
    llm: AnthropicLLM | None = None
    if run_judge:
        from sax_platform.llm import AnthropicLLM, make_client

        llm = AnthropicLLM(make_client())

    # Load plans from plans_dir if provided, otherwise use reference plans from cases
    plans: dict[str, Plan] = {}
    if plans_dir:
        plans_path = Path(plans_dir)
        if plans_path.is_dir():
            for json_file in sorted(plans_path.glob("*.json")):
                try:
                    content = json_file.read_text()
                    loaded_plan = Plan.model_validate_json(content)
                    plans[loaded_plan.task_id] = loaded_plan
                except Exception:
                    click.echo(f"Warning: failed to parse {json_file.name}, skipping.", err=True)

    results: list[PlanEvalResult] = []
    for case in cases:
        # Try to find a plan: by task_id from plans_dir, or reference_plan from case
        plan: Plan | None = plans.get(case.task.task_id) or case.reference_plan
        if plan is None:
            click.echo(f"Warning: no plan for case {case.case_id}, skipping.", err=True)
            continue

        repo_root = Path(case.repo_root)
        known_files = list_repo_files(repo_root) if repo_root.is_dir() else None

        det = run_deterministic_checks(plan, case.task, known_files)

        verdict = None
        if run_judge:
            from forge.eval.judge import judge_plan

            assert llm is not None  # built above whenever run_judge is True
            # Give the judge the same repo file listing the deterministic
            # checks used, so its completeness/context criteria are scored
            # against what the repo actually contains (T0.6).
            repo_context = format_repo_context(known_files) if known_files else None
            verdict = await judge_plan(
                case, plan, llm, repo_context=repo_context, model_name=judge_model
            )

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
    help="Model to use as judge (default: the REASONING tier's registry pin).",
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

    results = asyncio.run(_run_eval(cases, plans_dir, run_judge=judge, judge_model=judge_model))

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


# ---------------------------------------------------------------------------
# Start command — generic workflow launcher
# ---------------------------------------------------------------------------


async def _start_workflow(
    workflow_name: str,
    workflow_input: dict[str, Any],
    *,
    workflow_id: str,
    task_queue: str,
    temporal_address: str,
    timeout_hours: float,
) -> str:
    """Start a Temporal workflow by string name and return its ID."""

    client = await _connect_temporal_checked(temporal_address)
    handle = await client.start_workflow(
        workflow_name,
        workflow_input,
        id=workflow_id,
        task_queue=task_queue,
        execution_timeout=timedelta(hours=timeout_hours),
    )
    workflow_id_result: str = handle.id
    return workflow_id_result


async def _start_workflow_and_wait(
    workflow_name: str,
    workflow_input: dict[str, Any],
    *,
    workflow_id: str,
    task_queue: str,
    temporal_address: str,
    timeout_hours: float,
) -> object:
    """Start a Temporal workflow by string name and wait for its result."""

    client = await _connect_temporal_checked(temporal_address)
    result = await client.execute_workflow(
        workflow_name,
        workflow_input,
        id=workflow_id,
        task_queue=task_queue,
        execution_timeout=timedelta(hours=timeout_hours),
    )
    return result


@main.command()
@click.argument("workflow", required=True)
@click.argument("input_json", required=False, default=None)
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    default=None,
    help="Read JSON input from file instead of argument.",
)
@click.option(
    "--id",
    "workflow_id",
    default=None,
    help="Custom workflow ID (default: auto-generated).",
)
@click.option(
    "--task-queue",
    default="forge-task-queue",
    show_default=True,
    help="Temporal task queue.",
)
@click.option(
    "--wait",
    "wait_for_result",
    is_flag=True,
    help="Wait for result and print it as JSON.",
)
@click.option(
    "--timeout",
    "timeout_hours",
    default=48.0,
    show_default=True,
    type=float,
    help="Execution timeout in hours.",
)
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
    help="Temporal server address.",
)
def start(
    workflow: str,
    input_json: str | None,
    input_file: str | None,
    workflow_id: str | None,
    task_queue: str,
    wait_for_result: bool,
    timeout_hours: float,
    temporal_address: str,
) -> None:
    """Start an arbitrary Temporal workflow by name.

    WORKFLOW is the workflow class name (e.g. ExportPlaybookWorkflow).
    INPUT_JSON is an optional JSON object passed as the workflow argument.
    """
    import json
    import uuid

    wf_input = load_workflow_input(input_json, input_file)

    if workflow_id is None:
        short_uuid = str(uuid.uuid4())[:8]
        workflow_id = f"{workflow.lower()}-{short_uuid}"

    try:
        if wait_for_result:
            result = asyncio.run(
                _start_workflow_and_wait(
                    workflow,
                    wf_input,
                    workflow_id=workflow_id,
                    task_queue=task_queue,
                    temporal_address=temporal_address,
                    timeout_hours=timeout_hours,
                )
            )
            if hasattr(result, "model_dump_json"):
                click.echo(result.model_dump_json(indent=2))
            else:
                click.echo(json.dumps(result, indent=2, default=str))
        else:
            returned_id = asyncio.run(
                _start_workflow(
                    workflow,
                    wf_input,
                    workflow_id=workflow_id,
                    task_queue=task_queue,
                    temporal_address=temporal_address,
                    timeout_hours=timeout_hours,
                )
            )
            click.echo(returned_id)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)
