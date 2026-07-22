"""CLI for the OCR app: ``ocr worker``, ``ocr submit``, ``ocr list``, ``ocr export``,
``ocr mark``, ``ocr unmark``, ``ocr tracker-status``.

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
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import click
from sax_platform.contracts.constants import OCR_TASK_QUEUE
from sax_platform.temporal.client import connect_temporal

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"
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


async def _start_and_wait(
    workflow_name: str, arg: object, *, workflow_id: str, address: str, timeout_hours: float
) -> object:
    from datetime import timedelta

    client = await connect_temporal(address)
    handle = await client.start_workflow(
        workflow_name,
        arg,
        id=workflow_id,
        task_queue=OCR_TASK_QUEUE,
    )
    return await handle.result(rpc_timeout=timedelta(hours=timeout_hours))


async def _start_submit(arg: object, *, workflow_id: str, address: str) -> str:
    """Start OcrSubmitWorkflow and return its id WITHOUT awaiting its full run.

    The submit workflow now awaits its self-polling store children (up to the 25h
    batch ceiling), so the CLI cannot block on the result. It starts the workflow
    with a derived ~26h execution timeout (the ceiling plus reassembly/store
    margins) and returns immediately; progress is tracked via ``ocr list``.
    """
    from datetime import timedelta

    from sax_platform.temporal.polling import BATCH_WAIT_CEILING

    client = await connect_temporal(address)
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


@click.group()
def main() -> None:
    """OCR app commands."""
    _require_forge_env()

    from sax_platform.logging import setup_logging

    setup_logging("ocr", console=True)


@main.command("worker")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def worker_cmd(temporal_address: str) -> None:
    """Run the OCR Temporal worker (ocr-task-queue)."""
    from ocr.worker import run_worker

    asyncio.run(run_worker(temporal_address))


@main.command("submit")
@click.argument("file_path")
@click.option("--model", default="mistral:mistral-ocr-latest", show_default=True)
@click.option("--skip-duplicate-detection", is_flag=True)
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def submit_cmd(
    file_path: str, model: str, skip_duplicate_detection: bool, temporal_address: str
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
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def list_cmd(limit: int, status_filter: str, temporal_address: str) -> None:
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
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def export_cmd(document_id: str, output_dir: str, temporal_address: str) -> None:
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
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def mark_cmd(document_id: str, temporal_address: str) -> None:
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
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def unmark_cmd(document_id: str, temporal_address: str) -> None:
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
    "source ~/.config/forge/forge.env (the worker env file) or export FORGE_DB_URL; "
    "for the local stack use the port-5434 override."
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
