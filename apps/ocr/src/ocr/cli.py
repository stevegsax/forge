"""CLI for the OCR app: ``ocr worker``, ``ocr submit``, ``ocr list``, ``ocr export``,
``ocr mark``, ``ocr unmark``, ``ocr tracker-status``, ``ocr migrate``,
``ocr db-change``.

Workflows are started on ``ocr-task-queue`` (the OCR worker's queue) so they hit the
OCR-side activities, which now own the Mistral submit + self-polling; the platform
worker on ``forge-task-queue`` only services the cross-queue ``batch_jobs`` ledger
writes. ``tracker-status`` is the exception: it reads the store directly (no Temporal)
so it stays usable as a health probe even when the workers or Temporal are down.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import click
from sax_platform.contracts.constants import OCR_TASK_QUEUE
from sax_platform.temporal.client import connect_temporal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from temporalio.client import Client

EXIT_INFRASTRUCTURE_ERROR = 2
#: ``tracker-status`` exit code reserved for "the probe could not answer" (config
#: unset/invalid or the store unreachable). Kept distinct from the 0/1/2 liveness
#: verdicts so a harness can never confuse a broken probe with a stale tracker.
EXIT_PROBE_ERROR = 3
#: EX_CONFIG (sysexits.h, 78): the environment guard refused to run because
#: FORGE_ENV was unset or invalid. Deliberately outside every command exit-code
#: contract (including tracker-status's 0/1/2/3) so a guard failure — exit 78 with
#: no ``checked_at_gmt`` line — unambiguously means the command never ran.
EXIT_CONFIG_ERROR = 78


def _require_forge_env() -> None:
    """Refuse to run any command without an explicitly declared environment.

    Reads the process environment through the pure ``resolve_forge_env``
    (sax_platform.config), which invents no default so reaching the production
    store is always an explicit act. On failure it prints the guard's complete,
    actionable message to stderr and exits ``EXIT_CONFIG_ERROR``; on success it
    returns silently and the command proceeds. Because this runs in the root
    group callback, a guard failure aborts before any command body — so
    ``tracker-status`` never prints its ``checked_at_gmt`` line.
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


async def _connect_checked(address: str | None) -> Client:
    """Connect to Temporal after enforcing env/namespace coherence.

    The group callback already ran ``_require_forge_env`` (so FORGE_ENV is valid);
    re-resolving it here — purely, from ``os.environ`` — pairs it with the
    namespace from :class:`TemporalSettings` (the sole FORGE_TEMPORAL_* reader) and
    refuses to connect a dev/test process to production's namespace, or a prod
    process to any other, before the connection opens. An incoherent pairing prints
    the fix and exits ``EXIT_CONFIG_ERROR``. The direct-DB commands (``migrate``,
    ``tracker-status``) never reach a connect, so the namespace never affects them.
    """
    import os

    from sax_platform.config import (
        ForgeEnvError,
        TemporalSettings,
        resolve_forge_env,
        resolve_temporal_target,
    )

    settings = TemporalSettings()
    try:
        target = resolve_temporal_target(
            resolve_forge_env(os.environ),
            address_override=address or settings.address,
        )
    except ForgeEnvError as exc:
        click.echo(str(exc), err=True)
        sys.exit(EXIT_CONFIG_ERROR)
    return await connect_temporal(target.address, namespace=target.namespace, settings=settings)


async def _start_and_wait(
    workflow_name: str, arg: object, *, workflow_id: str, address: str | None, timeout_hours: float
) -> object:
    from datetime import timedelta

    client = await _connect_checked(address)
    handle = await client.start_workflow(
        workflow_name,
        arg,
        id=workflow_id,
        task_queue=OCR_TASK_QUEUE,
    )
    return await handle.result(rpc_timeout=timedelta(hours=timeout_hours))


async def _start_submit(arg: object, *, workflow_id: str, address: str | None) -> str:
    """Start OcrSubmitWorkflow and return its id WITHOUT awaiting its full run.

    The submit workflow now awaits its self-polling store children (up to the 25h
    batch ceiling), so the CLI cannot block on the result. It starts the workflow
    with a derived ~26h execution timeout (the ceiling plus reassembly/store
    margins) and returns immediately; progress is tracked via ``ocr list``.
    """
    from datetime import timedelta

    from sax_platform.temporal.polling import BATCH_WAIT_CEILING

    client = await _connect_checked(address)
    handle = await client.start_workflow(
        "OcrSubmitWorkflow",
        arg,
        id=workflow_id,
        task_queue=OCR_TASK_QUEUE,
        execution_timeout=BATCH_WAIT_CEILING + timedelta(hours=1),
    )
    return handle.id


def _auto_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Functional core — tracker-status verdict derivation (pure, no I/O)
# ---------------------------------------------------------------------------

#: Liveness verdict for the stateless OCR batch tracker (T4.4).
type TrackerLiveness = Literal["fresh", "stale", "never-ran"]


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackerStatusReport:
    """Derived tracker-health verdict plus the fields the probe prints.

    ``checked_at_gmt`` is deliberately not carried here: the shell captures ``now``
    and prints it first, before any I/O that might fail, so that field never
    depends on this derivation running. ``last_run_at`` is normalized to aware UTC;
    the ``*_last_cycle``/``cycles_total``/age fields are ``None`` for a tracker that
    has never run.
    """

    last_run_at: datetime | None
    heartbeat_age_seconds: int | None
    cycles_total: int | None
    live_jobs_last_cycle: int | None
    hints_sent_last_cycle: int | None
    live_jobs_now: int
    status: TrackerLiveness
    exit_code: int


def _as_utc(moment: datetime) -> datetime:
    """Normalize a possibly-naive datetime to aware UTC.

    SQLite reads ``last_run_at`` back naive; Postgres reads it aware. A naive value
    is a UTC wall clock (the tracker only ever writes ``datetime.now(UTC)``), so it
    is stamped UTC; an already-aware value is converted to UTC.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def derive_tracker_status(
    heartbeat: Mapping[str, Any] | None,
    *,
    now: datetime,
    stale_after_seconds: int,
    live_jobs_now: int,
) -> TrackerStatusReport:
    """Derive the tracker-health verdict and exit code (pure).

    *heartbeat* is the raw ``get_tracker_heartbeat`` row (a DB boundary payload) or
    ``None`` when no sweep has ever run. *now* is injected (pseudo-I/O) so the age
    is deterministic. Verdict / exit-code table::

        heartbeat present, age <= threshold        -> fresh,          exit 0
        (stale | never-ran) and no live jobs        -> stale/never-ran, exit 1
        (stale | never-ran) and live_jobs_now > 0   -> stale/never-ran, exit 2

    A fresh tracker is healthy regardless of the live-job count. Exit 2 is the
    API-failure signature: work is waiting but the tracker is not completing cycles.
    """
    if heartbeat is None:
        return TrackerStatusReport(
            last_run_at=None,
            heartbeat_age_seconds=None,
            cycles_total=None,
            live_jobs_last_cycle=None,
            hints_sent_last_cycle=None,
            live_jobs_now=live_jobs_now,
            status="never-ran",
            exit_code=2 if live_jobs_now > 0 else 1,
        )

    last_run_at = _as_utc(heartbeat["last_run_at"])
    age = now - last_run_at
    is_fresh = age.total_seconds() <= stale_after_seconds
    status: TrackerLiveness = "fresh" if is_fresh else "stale"
    exit_code = 0 if is_fresh else (2 if live_jobs_now > 0 else 1)
    return TrackerStatusReport(
        last_run_at=last_run_at,
        heartbeat_age_seconds=int(age.total_seconds()),
        cycles_total=heartbeat["cycles_total"],
        live_jobs_last_cycle=heartbeat["live_jobs"],
        hints_sent_last_cycle=heartbeat["hints_sent"],
        live_jobs_now=live_jobs_now,
        status=status,
        exit_code=exit_code,
    )


def _iso_z(moment: datetime) -> str:
    """Render a datetime as ISO-8601 UTC with a ``Z`` suffix."""
    return _as_utc(moment).isoformat().replace("+00:00", "Z")


def _fmt_optional(value: int | None) -> str:
    """Render an optional int field for the report, ``None`` -> ``none``."""
    return "none" if value is None else str(value)


def tracker_status_lines(report: TrackerStatusReport) -> tuple[str, ...]:
    """The report body lines (pure), in fixed order, excluding ``checked_at_gmt``.

    ``checked_at_gmt`` is printed by the shell before this runs (so it survives a
    later I/O failure), so it is not repeated here.
    """
    last_run = _iso_z(report.last_run_at) if report.last_run_at is not None else "none"
    return (
        f"last_run_at: {last_run}",
        f"heartbeat_age_seconds: {_fmt_optional(report.heartbeat_age_seconds)}",
        f"cycles_total: {_fmt_optional(report.cycles_total)}",
        f"live_jobs_last_cycle: {_fmt_optional(report.live_jobs_last_cycle)}",
        f"hints_sent_last_cycle: {_fmt_optional(report.hints_sent_last_cycle)}",
        f"live_jobs_now: {report.live_jobs_now}",
        f"status: {report.status}",
    )


def format_migration_target(url: str) -> str:
    """Render a credential-free ``chain -> host/database`` summary line (pure).

    Parses *url* with SQLAlchemy and reconstructs only the host/port/database (or,
    for SQLite, the file path), deliberately dropping the username and password so
    a connection string with embedded credentials is never echoed to the terminal.
    """
    from sqlalchemy.engine import make_url

    parsed = make_url(url)
    if parsed.host:
        host = parsed.host if parsed.port is None else f"{parsed.host}:{parsed.port}"
        target = f"{host}/{parsed.database or ''}"
    else:  # SQLite and other host-less URLs: the database is the file path.
        target = parsed.database or "(in-memory)"
    return f"alembic_version_ocr -> {target}"


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
    (``ocr --env dev tracker-status``, consumed by the group) and after it
    (``ocr tracker-status --env dev``, consumed here) with identical semantics.

    ``__init__`` appends the shared ``--env`` option; ``invoke`` applies that
    profile (when it was given at this level) and then runs the ``FORGE_ENV``
    guard, immediately before the command body. The captured value is popped out
    of ``ctx.params`` so the command's own callback signature is untouched.
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
    """A ``click.Group`` whose commands are ``_EnvCommand``s.

    ``command_class`` makes every ``@group.command()`` an ``_EnvCommand`` (the
    ``--env``-in-either-position behavior and the guard attach automatically,
    with no per-command decorator). The group callback applies a group-level
    ``--env`` eagerly; the guard itself lives on the commands.
    """

    command_class = _EnvCommand


@click.group(cls=_EnvGroup)
@click.option("--env", "env_profile", default=None, metavar="NAME|PATH", help=_ENV_OPTION_HELP)
def main(env_profile: str | None) -> None:
    """OCR app commands."""
    # Apply a group-level --env eagerly so its vars are in place before the
    # subcommand parses; the FORGE_ENV guard runs per-command (see _EnvCommand),
    # not here, so --help stays usable without a declared environment.
    if env_profile is not None:
        _apply_env_profile(env_profile)

    from sax_platform.logging import setup_logging

    setup_logging("ocr", console=True)


@main.command("worker")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=None,
    help="Temporal server address. Derived from FORGE_ENV when unset; an "
    "override that is not that environment's server is refused.",
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
def worker_cmd(temporal_address: str | None, worker_identity: str | None) -> None:
    """Run the OCR Temporal worker (ocr-task-queue)."""
    from ocr.worker import run_worker

    asyncio.run(run_worker(temporal_address, identity=worker_identity))


#: Config fail-fast message for ``ocr migrate`` when ``FORGE_DB_URL`` is unset.
#: Guard-style (stderr + exit 78): there is no 0/1/2/3 verdict contract here.
_FORGE_DB_URL_MIGRATE_UNSET_MESSAGE = (
    "FORGE_DB_URL is not set — ocr migrate applies the ocr Alembic chain "
    "(alembic_version_ocr) to the shared forge database; source an env profile "
    "(e.g. `ocr migrate --env dev`, reading ~/.config/forge/envs/dev.env) "
    "or export FORGE_DB_URL."
)


@main.command("migrate")
def migrate_cmd() -> None:
    """Run the OCR Alembic chain (``alembic_version_ocr``) against ``FORGE_DB_URL``.

    Standalone parity with ``pbook migrate``: the OCR worker also runs this chain
    at startup, but ``ocr migrate`` applies it ahead of time (schema bootstrap,
    dev setup). Prints a single credential-free ``chain -> host/database`` line
    on success. An unset ``FORGE_DB_URL`` is a config error — stderr + exit 78.
    """
    from pydantic import ValidationError
    from sax_platform.config import DbSettings

    from ocr.store import run_migrations

    try:
        # DbSettings reads FORGE_DB_URL from the env (its ``url`` field's alias);
        # the pydantic mypy plugin can't see that, so it flags the required arg.
        db_url = DbSettings().url  # type: ignore[call-arg]
    except ValidationError:
        click.echo(_FORGE_DB_URL_MIGRATE_UNSET_MESSAGE, err=True)
        sys.exit(EXIT_CONFIG_ERROR)

    run_migrations(db_url)
    click.echo(format_migration_target(db_url))


@main.command("db-change", cls=click.Command)
@click.option(
    "--from",
    "from_revision",
    required=True,
    help="Revision the production database is already stamped with.",
)
@click.option(
    "--to",
    "to_revision",
    default=None,
    help="Last revision of the request (default: the chain head).",
)
@click.option("--title", required=True, help="Kebab-case slug naming the request directory.")
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Where request directories live (default: <repo>/datastore-changes).",
)
@click.option(
    "--no-lint",
    is_flag=True,
    help="Skip Squawk; the request is stamped NOT LINTED (run `make lint-sql` before committing).",
)
def db_change_cmd(
    from_revision: str,
    to_revision: str | None,
    title: str,
    output_root: Path | None,
    no_lint: bool,
) -> None:
    """Generate a sax-datastores change request for the OCR chain.

    OCR's tables live in forge's database and OCR is not a registered
    sax-datastores product of its own, so a request from this chain is filed as
    product ``forge`` against ``forge_prod`` and takes the next id from the
    same repo-root ``datastore-changes/`` sequence forge uses — one id sequence
    per product, and this chain's product is forge. Only the version table
    (``alembic_version_ocr``) distinguishes it.

    Deliberately outside the ``FORGE_ENV`` guard (hence ``cls=click.Command``):
    it opens no database and no Temporal connection — it reads the Alembic
    chain on disk and writes files into the repo.
    """
    # Imported here, not at module scope: sax_platform.db pulls in SQLAlchemy,
    # and every other ocr command would pay for it at startup.
    from sax_platform.db.change_request import (
        ChainSpec,
        ChangeRequestError,
        describe_generated_request,
        find_repo_root,
        generate_change_request,
        squawk_linter,
    )

    chain = ChainSpec(
        product="forge",
        database="forge_prod",
        schema="public",
        version_table="alembic_version_ocr",
        script_location=Path(__file__).resolve().parent / "alembic",
    )

    try:
        repo_root = find_repo_root(Path(__file__).resolve())
        result = generate_change_request(
            chain=chain,
            output_root=output_root or repo_root / "datastore-changes",
            from_revision=from_revision,
            to_revision=to_revision,
            title=title,
            linter=None if no_lint else squawk_linter(repo_root / ".squawk.toml"),
        )
    except ChangeRequestError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    for warning in result.warnings:
        click.echo(f"warning: {warning}", err=True)
    click.echo(describe_generated_request(result))


@main.command("submit")
@click.argument("file_path")
@click.option("--model", default="mistral:mistral-ocr-latest", show_default=True)
@click.option("--skip-duplicate-detection", is_flag=True)
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=None,
    help="Temporal server address. Derived from FORGE_ENV when unset; an "
    "override that is not that environment's server is refused.",
)
def submit_cmd(
    file_path: str, model: str, skip_duplicate_detection: bool, temporal_address: str | None
) -> None:
    """Submit a document for OCR (starts the workflow; does not wait for it)."""
    from ocr.models import OcrSubmitInput

    wf_input = OcrSubmitInput(
        file_path=file_path,
        model_name=model,
        skip_duplicate_detection=skip_duplicate_detection,
    )
    workflow_id = _auto_id("ocr-submit")
    try:
        started_id = asyncio.run(
            _start_submit(wf_input, workflow_id=workflow_id, address=temporal_address)
        )
        _echo({"workflow_id": started_id, "status": "started"})
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


@main.command("list")
@click.option("--limit", default=50, show_default=True, type=int)
@click.option(
    "--status",
    "status_filter",
    default="",
    help="Filter by derived status: processing|succeeded|errored|unknown",
)
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=None,
    help="Temporal server address. Derived from FORGE_ENV when unset; an "
    "override that is not that environment's server is refused.",
)
def list_cmd(limit: int, status_filter: str, temporal_address: str | None) -> None:
    """List OCR submissions with status (ocr_job_status ⋈ batch_jobs)."""
    from ocr.models import OcrListJobsInput

    wf_input = OcrListJobsInput(limit=limit, status_filter=status_filter)
    try:
        result = asyncio.run(
            _start_and_wait(
                "OcrListJobsWorkflow",
                wf_input,
                workflow_id=_auto_id("ocr-list"),
                address=temporal_address,
                timeout_hours=1.0,
            )
        )
        _echo(result)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


@main.command("export")
@click.argument("document_id")
@click.option("--output-dir", default="", help="Defaults to $XDG_DATA_HOME/ocr/export/<id>.")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=None,
    help="Temporal server address. Derived from FORGE_ENV when unset; an "
    "override that is not that environment's server is refused.",
)
def export_cmd(document_id: str, output_dir: str, temporal_address: str | None) -> None:
    """Export OCR text + images for a document to the filesystem."""
    from ocr.models import OcrExportInput

    wf_input = OcrExportInput(document_id=document_id, output_dir=output_dir)
    try:
        result = asyncio.run(
            _start_and_wait(
                "OcrExportWorkflow",
                wf_input,
                workflow_id=_auto_id("ocr-export"),
                address=temporal_address,
                timeout_hours=1.0,
            )
        )
        _echo(result)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


@main.command("mark")
@click.argument("document_id")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=None,
    help="Temporal server address. Derived from FORGE_ENV when unset; an "
    "override that is not that environment's server is refused.",
)
def mark_cmd(document_id: str, temporal_address: str | None) -> None:
    """Mark a document for removal (soft-delete; a periodic workflow deletes it)."""
    from ocr.models import OcrMarkInput

    wf_input = OcrMarkInput(document_id=document_id)
    try:
        result = asyncio.run(
            _start_and_wait(
                "OcrMarkForRemovalWorkflow",
                wf_input,
                workflow_id=_auto_id("ocr-mark"),
                address=temporal_address,
                timeout_hours=1.0,
            )
        )
        _echo(result)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


@main.command("unmark")
@click.argument("document_id")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=None,
    help="Temporal server address. Derived from FORGE_ENV when unset; an "
    "override that is not that environment's server is refused.",
)
def unmark_cmd(document_id: str, temporal_address: str | None) -> None:
    """Clear the removal mark on a document."""
    from ocr.models import OcrMarkInput

    wf_input = OcrMarkInput(document_id=document_id)
    try:
        result = asyncio.run(
            _start_and_wait(
                "OcrClearRemovalMarkWorkflow",
                wf_input,
                workflow_id=_auto_id("ocr-unmark"),
                address=temporal_address,
                timeout_hours=1.0,
            )
        )
        _echo(result)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


#: Actionable one-liner for the config fail-fast path — matches the tone of the
#: worker's MISTRAL_API_KEY message: what is required, and how to supply it.
_FORGE_DB_URL_UNSET_MESSAGE = (
    "FORGE_DB_URL is not set — the probe reads the shared forge database; "
    "source an env profile (~/.config/forge/envs/<env>.env, with FORGE_ENV "
    "exported to match) or export FORGE_DB_URL."
)


def _probe_error(message: str) -> NoReturn:
    """Fail the probe: ``status: error`` on stdout, *message* on stderr, exit 3.

    Shell-only error handling. ``checked_at_gmt`` has already been printed by the
    caller, so stdout stays "``checked_at_gmt`` first, then ``status``". Exit 3 is
    reserved here (never the 0/1/2 liveness verdicts) so "the probe could not
    answer" is unambiguous to a monitoring harness.
    """
    click.echo("status: error")
    click.echo(message, err=True)
    sys.exit(EXIT_PROBE_ERROR)


@main.command("tracker-status")
@click.option(
    "--stale-after",
    "stale_after",
    default=300,
    show_default=True,
    type=click.IntRange(min=1),
    help="Heartbeat age (seconds) past which the tracker is stale (2-3 of its 120s cycles).",
)
def tracker_status_cmd(stale_after: int) -> None:
    """Health probe for the stateless OCR batch tracker (direct DB read; no Temporal).

    Works even when Temporal or the workers are down: it reads the tracker
    heartbeat and the live-job count straight from ``FORGE_DB_URL``. Prints one
    field per line (``checked_at_gmt`` first, always) and exits 0 (fresh),
    1 (stale / never-ran with no live jobs), 2 (stale / never-ran with live jobs
    still waiting — the API-failure signature), or 3 (the probe could not answer:
    FORGE_DB_URL unset/invalid or the store unreachable — prints ``status: error``
    with the reason on stderr; never collides with the 0/1/2 verdicts).
    """
    from pydantic import ValidationError
    from sax_platform.config import DbSettings
    from sax_platform.db import get_store_engine

    from ocr.activities import execute_list_live_ocr_jobs, tracker_created_after
    from ocr.store import get_tracker_heartbeat

    now = datetime.now(UTC)
    # Emit when the probe ran first, before any I/O that could fail.
    click.echo(f"checked_at_gmt: {_iso_z(now)}")

    try:
        # DbSettings reads FORGE_DB_URL from the env (its ``url`` field's alias);
        # the pydantic mypy plugin can't see that, so it flags the required arg.
        db_url = DbSettings().url  # type: ignore[call-arg]
    except ValidationError:
        _probe_error(_FORGE_DB_URL_UNSET_MESSAGE)

    try:
        engine = get_store_engine(db_url)
        try:
            heartbeat = get_tracker_heartbeat(engine)
            live_jobs_now = len(
                execute_list_live_ocr_jobs(engine, min_created_at=tracker_created_after(now))
            )
        finally:
            engine.dispose()
    except Exception as e:
        _probe_error(str(e))

    report = derive_tracker_status(
        heartbeat,
        now=now,
        stale_after_seconds=stale_after,
        live_jobs_now=live_jobs_now,
    )
    for line in tracker_status_lines(report):
        click.echo(line)
    sys.exit(report.exit_code)


def _echo(result: object) -> None:
    if hasattr(result, "model_dump_json"):
        click.echo(result.model_dump_json(indent=2))
    else:
        click.echo(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
