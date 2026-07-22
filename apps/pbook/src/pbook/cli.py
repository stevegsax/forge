"""Click CLI for the playbook service.

Thin Temporal client. Every direct-DB command (except ``migrate``)
submits a workflow that runs against the worker's configured DB. The
worker process is the only one that opens the database — its
``PBOOK_DATABASE_URL`` is the single source of truth for which DB any
operation hits.

``migrate`` and ``skill-prompt`` are the lone exceptions: ``migrate``
must precede the worker (schema bootstrap), and ``skill-prompt`` is
pure static config with no DB access.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import click

# Module-level so the connect chokepoint is patchable in tests (as in the forge
# and ocr CLIs). Importing this module does NOT pull in temporalio — the SDK
# imports inside ``connect_temporal`` stay lazy — so the pure commands (migrate,
# skill-prompt) pay nothing for it.
from sax_platform.temporal.client import connect_temporal

if TYPE_CHECKING:
    from typing import Any, NoReturn

    from temporalio.client import Client
    from temporalio.types import MethodAsyncSingleParam

    from pbook.models import RetrievalResult
    from pbook.settings import PbookSettings
    from pbook.transcript import SessionInfo

from pbook.models import PlaybookEntry
from pbook.tags import validate_tags

logger = logging.getLogger(__name__)


_TEMPORAL_ADDRESS = "localhost:7233"
_PBOOK_TASK_QUEUE = "pbook-task-queue"

# EX_CONFIG (sysexits.h, 78): the environment guard refused to run because
# FORGE_ENV was unset or invalid. Deliberately outside every other command exit
# code so an operator or monitoring harness never confuses "environment not
# declared" with a real command outcome.
EXIT_CONFIG_ERROR = 78


def _require_forge_env() -> None:
    """Refuse to run any command without an explicitly declared environment.

    Reads the process environment through the pure ``resolve_forge_env``
    (sax_platform.config), which invents no default so reaching the production
    store is always an explicit act. On failure it prints the guard's complete,
    actionable message to stderr and exits ``EXIT_CONFIG_ERROR``; on success it
    returns silently and the command proceeds.
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


async def _connect_checked(temporal_address: str) -> Client:
    """Connect to Temporal after enforcing env/namespace coherence.

    The single connect chokepoint for the pbook CLI. It routes through the shared
    ``sax_platform`` ``connect_temporal`` (data converter, TLS) and, first,
    re-resolves the already-guarded FORGE_ENV — purely, from ``os.environ`` — to
    pair it with the namespace from :class:`TemporalSettings` (the sole
    FORGE_TEMPORAL_* reader), refusing to connect a dev/test process to
    production's namespace, or a prod process to any other, before the connection
    opens. An incoherent pairing prints the fix and exits ``EXIT_CONFIG_ERROR``.
    ``migrate`` and ``skill-prompt`` never reach a connect, so the namespace never
    affects them.
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


def _execute_workflow[ParamType, ReturnType](
    workflow_fn: MethodAsyncSingleParam[Any, ParamType, ReturnType],
    arg: ParamType,
    *,
    id_prefix: str = "pbook",
    temporal_address: str = _TEMPORAL_ADDRESS,
) -> ReturnType:
    """Submit a pbook workflow and wait for the result.

    All direct-DB commands route through here. The worker is the only
    process that opens the DB; the CLI is purely a Temporal client. If
    the worker isn't running, the connect/submit will fail and the
    caller surfaces the error.
    """

    async def _submit() -> ReturnType:
        client = await _connect_checked(temporal_address)
        result: ReturnType = await client.execute_workflow(
            workflow_fn,
            arg,
            id=f"{id_prefix}-{int(time.time())}-{uuid.uuid4().hex[:8]}",
            task_queue=_PBOOK_TASK_QUEUE,
        )
        return result

    return asyncio.run(_submit())


# ---------------------------------------------------------------------------
# JSON output helpers
#
# Skills calling pbook programmatically rely on a stable contract:
# - Binary BLOB columns are stripped (they don't round-trip through JSON).
# - tags_json is parsed back into a `tags` list (the on-disk shape leaks
#   otherwise — emitting a JSON-string-in-JSON forces consumers to parse twice).
# - datetimes are ISO 8601 with timezone via _json_default below.
# - On `--json` failure paths, _emit_error writes the envelope to stdout
#   and exits non-zero, so the caller has a single parseable stream.
# ---------------------------------------------------------------------------

_BINARY_FIELDS = ("embedding", "source_context_embedding")


def _strip_embedding(row: dict[str, Any]) -> dict[str, Any]:
    """Drop binary embedding columns from a row dict for JSON output."""
    return {k: v for k, v in row.items() if k not in _BINARY_FIELDS}


def _reshape_entry(row: dict[str, Any]) -> dict[str, Any]:
    """Strip binary fields and ensure a `tags` list is present.

    Store read helpers already attach a `tags` list to each entry row;
    this drops binary vector columns and guarantees the key exists so the
    skill consumer sees a consistent shape regardless of which command
    produced the row.
    """
    cleaned = _strip_embedding(row)
    cleaned.setdefault("tags", [])
    return cleaned


def _json_default(obj: object) -> str:
    """JSON encoder for datetimes (ISO 8601) and any other non-serializable type."""
    from datetime import UTC, datetime

    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=UTC)
        return obj.isoformat()
    return str(obj)


def _emit_json(payload: object, *, indent: int = 2) -> None:
    """Print JSON to stdout using the canonical encoder."""
    click.echo(json.dumps(payload, default=_json_default, indent=indent))


def _emit_error(
    code: str,
    message: str,
    *,
    json_mode: bool,
    exit_code: int = 1,
) -> NoReturn:
    """Emit a structured error and exit non-zero.

    When `json_mode` is True, the error envelope goes to **stdout** as JSON
    (so the caller has a single parseable stream). Otherwise we write to
    stderr in the existing human-readable form.
    """
    if json_mode:
        _emit_json({"error": message, "code": code})
    else:
        click.echo(f"Error: {message}", err=True)
    sys.exit(exit_code)


def _format_entry(entry: dict[str, Any]) -> str:
    """Format an entry dict for human-readable terminal output."""
    tags = entry.get("tags", [])
    review = " [needs-review]" if entry.get("needs_review") else ""

    lines = [
        f"[{entry['id']}] {entry['title']}{review}",
        f"  Type: {entry.get('entry_type', 'curated')}",
        f"  Tags: {', '.join(tags) if tags else '(none)'}",
    ]

    project = entry.get("source_project", "")
    if project:
        lines.append(f"  Project: {project}")

    lines.append(f"  {entry['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option(
    "--env",
    "env_profile",
    default=None,
    metavar="NAME|PATH",
    help=(
        "Load an env profile before the environment guard runs. A bare NAME "
        "resolves to $XDG_CONFIG_HOME/forge/envs/<NAME>.env and sets FORGE_ENV; "
        "a path (or a value ending in .env) is read verbatim and takes FORGE_ENV "
        "from its FORGE_ENV_TAG. Never supplies FORGE_PROD_ACK."
    ),
)
@click.pass_context
def main(ctx: click.Context, verbose: bool, env_profile: str | None) -> None:
    """pbook — Knowledge playbook service."""
    if env_profile is not None:
        _apply_env_profile(env_profile)
    _require_forge_env()

    from sax_platform.logging import setup_logging

    from pbook.settings import PbookSettings

    settings = PbookSettings()
    ctx.obj = settings
    setup_logging(
        "pbook",
        log_path=settings.log_path,
        level=logging.DEBUG if verbose else logging.INFO,
        console=True,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--temporal-address",
    default="localhost:7233",
    help="Temporal server address.",
)
def worker(temporal_address: str) -> None:
    """Start the pbook Temporal worker."""
    import asyncio

    from pbook.worker import run_worker

    asyncio.run(run_worker(address=temporal_address))


@main.command(name="list")
@click.option("--tag", multiple=True, help="Filter by tag (repeatable, OR match).")
@click.option("--type", "entry_type", default="", help="Filter by entry type.")
@click.option("--project", default="", help="Filter by source project.")
@click.option("--needs-review", is_flag=True, help="Only show entries flagged for review.")
@click.option(
    "--include-rejected",
    is_flag=True,
    help="Include entries that have been rejected (excluded by default).",
)
@click.option("--limit", default=20, help="Maximum entries to return.")
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def list_entries(
    tag: tuple[str, ...],
    entry_type: str,
    project: str,
    needs_review: bool,
    include_rejected: bool,
    limit: int,
    output_json: bool,
) -> None:
    """List playbook entries."""
    from pbook.models import ListEntriesInput
    from pbook.workflows.cli_ops import ListEntriesWorkflow

    entries = _execute_workflow(
        ListEntriesWorkflow.run,
        ListEntriesInput(
            tags=list(tag),
            entry_type=entry_type or None,
            project=project or None,
            needs_review_only=needs_review,
            include_rejected=include_rejected,
            limit=limit,
        ),
        id_prefix="pbook-list",
    )

    if not entries:
        if output_json:
            _emit_json([])
        else:
            click.echo("No entries found.")
        return

    if output_json:
        _emit_json([_reshape_entry(e) for e in entries])
    else:
        for entry in entries:
            click.echo(_format_entry(entry))
            click.echo("")


@main.command()
@click.argument("entry_id", type=int)
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def get(entry_id: int, output_json: bool) -> None:
    """Get a single entry by ID."""
    from pbook.models import GetEntryInput
    from pbook.workflows.cli_ops import GetEntryWorkflow

    entry = _execute_workflow(
        GetEntryWorkflow.run,
        GetEntryInput(entry_id=entry_id),
        id_prefix="pbook-get",
    )

    if entry is None:
        _emit_error(
            "not_found",
            f"Entry {entry_id} not found.",
            json_mode=output_json,
        )

    if output_json:
        _emit_json(_reshape_entry(entry))
    else:
        click.echo(_format_entry(entry))


@main.command()
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, path_type=Path),
    help="JSON file containing PlaybookEntry. If omitted, JSON is read from stdin.",
)
@click.option("--schema", "show_schema", is_flag=True, help="Print JSON schema.")
@click.option(
    "--needs-review",
    is_flag=True,
    help="Flag the entry as needing review (default: stored as approved).",
)
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def add(
    file_path: Path | None,
    show_schema: bool,
    needs_review: bool,
    output_json: bool,
) -> None:
    """Add a playbook entry.

    Reads JSON from stdin by default, or from --file if given.
    """
    if show_schema:
        click.echo(json.dumps(PlaybookEntry.model_json_schema(), indent=2))
        return

    if file_path is not None:
        raw_json = file_path.read_text()
    else:
        if sys.stdin.isatty():
            _emit_error(
                "validation_error",
                "Provide JSON on stdin, --file <path>, or use --schema to see the format.",
                json_mode=output_json,
            )
        raw_json = sys.stdin.read()

    try:
        entry = PlaybookEntry.model_validate_json(raw_json)
    except Exception as exc:
        _emit_error(
            "validation_error",
            f"Validation error: {exc}",
            json_mode=output_json,
        )

    # Tag pre-validation in the CLI gives a fast, friendly error before
    # we round-trip through Temporal. The activity validates again as
    # belt-and-suspenders.
    tag_errors = validate_tags(entry.tags)
    if tag_errors:
        _emit_error(
            "tag_invalid",
            "; ".join(tag_errors),
            json_mode=output_json,
        )

    from pbook.models import AddEntryInput
    from pbook.workflows.cli_ops import AddEntryWorkflow

    result = _execute_workflow(
        AddEntryWorkflow.run,
        AddEntryInput(entry=entry, needs_review=needs_review),
        id_prefix="pbook-add",
    )

    if result.get("error") == "tag_invalid":
        _emit_error(
            "tag_invalid",
            "; ".join(result.get("messages", [])),
            json_mode=output_json,
        )

    if output_json:
        _emit_json(
            {
                "id": result["id"],
                "title": result["title"],
                "approved": not result["needs_review"],
                "needs_review": result["needs_review"],
                "rejected": result["rejected"],
            }
        )
    else:
        click.echo(f"Added entry {result['id']}: {result['title']}")


@main.command()
@click.argument("entry_id", type=int)
@click.option(
    "--file",
    "file_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="JSON file with updated fields.",
)
def update(entry_id: int, file_path: Path) -> None:
    """Update an entry by ID."""
    from pbook.models import UpdateEntryInput
    from pbook.workflows.cli_ops import UpdateEntryWorkflow

    updates = json.loads(file_path.read_text())

    if "tags" in updates:
        tag_errors = validate_tags(updates["tags"])
        if tag_errors:
            for err in tag_errors:
                click.echo(f"Tag error: {err}", err=True)
            sys.exit(1)
        # `tags` passes through as a list; store.update_entry replaces the
        # entry's tag rows (pbk_entry_tags) rather than a column.

    result = _execute_workflow(
        UpdateEntryWorkflow.run,
        UpdateEntryInput(entry_id=entry_id, updates=updates),
        id_prefix="pbook-update",
    )
    if result.get("error") == "not_found":
        click.echo(f"Entry {entry_id} not found.", err=True)
        sys.exit(1)
    click.echo(f"Updated entry {entry_id}.")


@main.command()
@click.argument("entry_id", type=int)
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def approve(entry_id: int, output_json: bool) -> None:
    """Clear the needs-review flag on an entry."""
    from pbook.models import ApproveEntryInput
    from pbook.workflows.cli_ops import ApproveEntryWorkflow

    result = _execute_workflow(
        ApproveEntryWorkflow.run,
        ApproveEntryInput(entry_id=entry_id),
        id_prefix="pbook-approve",
    )
    if result.get("error") == "not_found":
        _emit_error(
            "not_found",
            f"Entry {entry_id} not found.",
            json_mode=output_json,
        )

    if output_json:
        _emit_json(result)
    else:
        click.echo(f"Approved entry {entry_id}: {result['title']}")


@main.command()
@click.argument("entry_id", type=int)
@click.option("--reason", default="", help="Why the entry is being rejected (optional).")
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def reject(entry_id: int, reason: str, output_json: bool) -> None:
    """Mark an entry as rejected (soft-mark; the row is preserved for audit)."""
    from pbook.models import RejectEntryInput
    from pbook.workflows.cli_ops import RejectEntryWorkflow

    reason_value: str | None = reason if reason else None
    result = _execute_workflow(
        RejectEntryWorkflow.run,
        RejectEntryInput(entry_id=entry_id, reason=reason_value),
        id_prefix="pbook-reject",
    )
    if result.get("error") == "not_found":
        _emit_error(
            "not_found",
            f"Entry {entry_id} not found.",
            json_mode=output_json,
        )

    if output_json:
        _emit_json(result)
    else:
        suffix = f" — {reason}" if reason else ""
        click.echo(f"Rejected entry {entry_id}: {result['title']}{suffix}")


@main.command(name="check-duplicate")
@click.option("--title", required=True, help="Title to check for duplicates.")
@click.option("--tag", multiple=True, help="Tags to refine duplicate search.")
def check_duplicate_cmd(title: str, tag: tuple[str, ...]) -> None:
    """Check for duplicate entries matching a title."""
    from pbook.models import CheckDuplicateInput
    from pbook.workflows.cli_ops import CheckDuplicateWorkflow

    matches = _execute_workflow(
        CheckDuplicateWorkflow.run,
        CheckDuplicateInput(title=title, tags=list(tag) if tag else None),
        id_prefix="pbook-check-dup",
    )

    if not matches:
        click.echo("No duplicates found.")
        return

    click.echo(f"Found {len(matches)} potential duplicate(s):")
    click.echo("")
    for entry in matches:
        click.echo(_format_entry(entry))
        click.echo("")


@main.command()
@click.option(
    "--file",
    "file_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="JSON file with PushExperienceInput data.",
)
@click.option(
    "--temporal-address",
    default="localhost:7233",
    help="Temporal server address.",
)
def push(file_path: Path, temporal_address: str) -> None:
    """Push experience data for LLM extraction."""
    import asyncio

    raw = file_path.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        click.echo(f"Invalid JSON: {exc}", err=True)
        sys.exit(1)

    # Normalize: accept a single object or a list
    if isinstance(data, dict):
        experiences = [data]
        project = data.get("project", "")
    elif isinstance(data, list):
        experiences = data
        project = experiences[0].get("project", "") if experiences else ""
    else:
        click.echo("Error: expected a JSON object or array.", err=True)
        sys.exit(1)

    async def _submit() -> dict[str, Any]:
        from pbook.worker import PBOOK_TASK_QUEUE
        from pbook.workflows.extraction import ExtractionWorkflow

        client = await _connect_checked(temporal_address)
        result: dict[str, Any] = await client.execute_workflow(
            ExtractionWorkflow.run,
            json.dumps({"experiences": experiences, "project": project}),
            id=f"pbook-extract-{int(time.time())}",
            task_queue=PBOOK_TASK_QUEUE,
        )
        return result

    try:
        result = asyncio.run(_submit())
        count = result.get("entries_created", 0)
        click.echo(f"Extraction complete: {count} entries created.")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@main.command()
@click.argument("query", required=False, default="")
@click.option("--tag", multiple=True, help="Filter by tag (repeatable, AND-merged with query).")
@click.option(
    "--threshold",
    type=float,
    default=0.0,
    help="Drop matches below this cosine similarity (0.0 disables the cutoff).",
)
@click.option(
    "--mode",
    type=click.Choice(["create", "fix"]),
    default="create",
    help="Tiebreaker mode when ranking ties on similarity.",
)
@click.option("--limit", default=20, type=int, help="Cap on returned entries.")
@click.option(
    "--token-budget",
    default=5000,
    type=int,
    help="Token budget for the packed result.",
)
@click.option(
    "--include-rejected",
    is_flag=True,
    help="Include entries that have been rejected.",
)
@click.option(
    "--include-unapproved",
    is_flag=True,
    help="Include entries flagged needs_review (default: only approved entries).",
)
@click.option(
    "--temporal-address",
    default="localhost:7233",
    help="Temporal server address.",
)
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def search(
    query: str,
    tag: tuple[str, ...],
    threshold: float,
    mode: str,
    limit: int,
    token_budget: int,
    include_rejected: bool,
    include_unapproved: bool,
    temporal_address: str,
    output_json: bool,
) -> None:
    """Search the playbook by tag and/or free-text query.

    With a free-text QUERY, results are ranked by semantic similarity
    against entry content. Combine with --tag to AND-merge tag filtering
    with semantic ranking. Submits a RetrievalWorkflow on
    pbook-task-queue — the worker must be running.
    """
    import asyncio

    if not query and not tag:
        _emit_error(
            "validation_error",
            "Provide a QUERY argument or at least one --tag.",
            json_mode=output_json,
        )

    async def _submit() -> RetrievalResult:
        from pbook.models import RetrievalInput, RetrievalMode
        from pbook.worker import PBOOK_TASK_QUEUE
        from pbook.workflows.retrieval import RetrievalWorkflow

        client = await _connect_checked(temporal_address)
        retrieval_mode = RetrievalMode(mode)
        result: RetrievalResult = await client.execute_workflow(
            RetrievalWorkflow.run,
            RetrievalInput(
                tags=list(tag),
                mode=retrieval_mode,
                token_budget=token_budget,
                approved_only=not include_unapproved,
                query=query,
                threshold=threshold,
                include_rejected=include_rejected,
            ),
            id=f"pbook-search-{int(time.time())}",
            task_queue=PBOOK_TASK_QUEUE,
        )
        return result

    try:
        result = asyncio.run(_submit())
    except Exception as exc:
        _emit_error(
            "worker_unavailable",
            f"RetrievalWorkflow failed: {exc}",
            json_mode=output_json,
        )

    entries = result.entries[:limit] if limit else result.entries

    if output_json:
        _emit_json(
            {
                "entries": [_reshape_entry(e) for e in entries],
                "total_candidates": result.total_candidates,
                "token_count": result.token_count,
            }
        )
        return

    if not entries:
        click.echo("No matches found.")
        return

    for entry in entries:
        sim = entry.get("similarity")
        prefix = f"[sim={sim:.3f}] " if sim is not None else ""
        click.echo(f"{prefix}{_format_entry(entry)}")
        click.echo("")


def _group_review_by_experience(
    entries_with_sources: list[dict[str, Any]],
) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    """Backwards-compat thin wrapper: the canonical implementation now
    lives in ``pbook.activities.cli_ops`` (so the activity can call it
    inside the worker process). Kept here so existing tests keep
    importing it from ``pbook.cli`` without churn.
    """
    from pbook.activities.cli_ops import group_review_by_experience

    return group_review_by_experience(entries_with_sources)


@main.command()
@click.option("--limit", default=20, help="Maximum entries to show.")
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
@click.option(
    "--by-experience",
    "by_experience",
    is_flag=True,
    help="Group entries by experience_hash to surface over-extraction clusters.",
)
def review(limit: int, output_json: bool, by_experience: bool) -> None:
    """List entries needing review."""
    from pbook.models import ReviewQueueInput
    from pbook.workflows.cli_ops import ReviewQueueWorkflow

    result = _execute_workflow(
        ReviewQueueWorkflow.run,
        ReviewQueueInput(limit=limit, by_experience=by_experience),
        id_prefix="pbook-review",
    )

    if by_experience:
        clusters = result.get("clusters", [])
        singletons = result.get("singletons", [])

        if output_json:
            _emit_json(
                {
                    "clusters": [
                        {
                            "experience_hash": c["experience_hash"],
                            "project_name": c.get("project_name", ""),
                            "entries": [_reshape_entry(e) for e in c["entries"]],
                        }
                        for c in clusters
                    ],
                    "singletons": [_reshape_entry(e) for e in singletons],
                }
            )
            return

        if not clusters and not singletons:
            click.echo("No entries need review.")
            return

        if clusters:
            click.echo(f"{len(clusters)} cluster(s) — entries sharing an experience_hash:")
            click.echo("")
            for c in clusters:
                h = c["experience_hash"]
                ents = c["entries"]
                click.echo(f"Cluster {h[:12]}… ({len(ents)} entries)")
                for e in ents:
                    click.echo(f"  [{e['id']}] {e.get('title', '')}")
                click.echo("")
        if singletons:
            click.echo(f"{len(singletons)} singleton(s):")
            click.echo("")
            for e in singletons:
                click.echo(f"  [{e['id']}] {e.get('title', '')}")
            click.echo("")
        return

    needs_review = result.get("entries", [])

    if output_json:
        _emit_json([_reshape_entry(e) for e in needs_review])
        return

    if not needs_review:
        click.echo("No entries need review.")
        return

    click.echo(f"{len(needs_review)} entry/entries need review:")
    click.echo("")
    for entry in needs_review:
        click.echo(_format_entry(entry))
        click.echo("")


@main.command()
@click.argument("entry_id", type=int)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=True,
    help="Machine-readable JSON output (default).",
)
def sources(entry_id: int, output_json: bool) -> None:
    """List the entry_sources rows that produced an entry."""
    from pbook.models import ListSourcesInput
    from pbook.workflows.cli_ops import ListSourcesWorkflow

    result = _execute_workflow(
        ListSourcesWorkflow.run,
        ListSourcesInput(entry_id=entry_id),
        id_prefix="pbook-sources",
    )
    if not result.get("found"):
        _emit_error(
            "not_found",
            f"Entry {entry_id} not found.",
            json_mode=output_json,
        )

    rows = result.get("rows", [])
    if output_json:
        _emit_json([_strip_embedding(r) for r in rows])
        return

    if not rows:
        click.echo(f"No sources recorded for entry {entry_id}.")
        return

    for row in rows:
        click.echo(
            f"[{row['id']}] session={row['session_id'] or '-'}  "
            f"project={row['project_name'] or '-'}  "
            f"hash={row['experience_hash'] or '-'}",
        )
        if row.get("source_context"):
            click.echo(f"  {row['source_context']}")
        click.echo("")


@main.command(name="session-text")
@click.argument("session_id")
@click.option(
    "--path",
    "path_override",
    type=click.Path(exists=True, path_type=Path),
    help="Override transcript path (used when the session isn't in ingested_sessions).",
)
@click.option(
    "--raw",
    is_flag=True,
    help="Emit the JSONL bytes verbatim instead of rendered text.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="JSON-wrap the output (otherwise plain text).",
)
def session_text(
    session_id: str,
    path_override: Path | None,
    raw: bool,
    output_json: bool,
) -> None:
    """Render the transcript for a Claude Code session by id.

    Resolves the JSONL path by scanning ~/.claude/projects/ for a file
    whose stem matches SESSION_ID. Use --path to override (manual
    sessions, alternate locations).
    """
    from pbook.models import GetSessionTextInput
    from pbook.workflows.cli_ops import GetSessionTextWorkflow

    result = _execute_workflow(
        GetSessionTextWorkflow.run,
        GetSessionTextInput(
            session_id=session_id,
            path=str(path_override) if path_override is not None else None,
            raw=raw,
        ),
        id_prefix="pbook-session-text",
    )

    if result.get("error") == "session_file_missing":
        _emit_error(
            "session_file_missing",
            f"No transcript found for session {session_id}. Use --path to point at one.",
            json_mode=output_json,
        )

    text = result["text"]
    if output_json:
        _emit_json(
            {
                "session_id": session_id,
                "raw": raw,
                "text": text,
                "project_name": result.get("project_name", ""),
            }
        )
    else:
        click.echo(text)


@main.command()
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=True,
    help="Machine-readable JSON output (default).",
)
def tags(output_json: bool) -> None:
    """Show valid tag namespaces and the values currently in use.

    Combines the canonical namespace list (closed set in pbook.tags)
    with values_in_use, computed across non-rejected entries.
    """
    from pbook.models import ListTagsInput
    from pbook.workflows.cli_ops import ListTagsWorkflow

    payload = _execute_workflow(
        ListTagsWorkflow.run,
        ListTagsInput(),
        id_prefix="pbook-tags",
    )

    if output_json:
        _emit_json(payload)
    else:
        click.echo("Tag namespaces:")
        for kind, names in payload["namespaces"].items():
            click.echo(f"  {kind}: {', '.join(names)}")
        click.echo("")
        click.echo("Values in use:")
        for ns, vals in payload["values_in_use"].items():
            click.echo(f"  {ns}: {', '.join(vals) if vals else '(none)'}")


@main.command()
@click.pass_obj
def migrate(settings: PbookSettings) -> None:
    """Run database migrations.

    The lone CLI command that opens the DB directly — schema bootstrap
    must precede the worker's first connection.
    """
    from pbook.store import run_migrations

    url = settings.db.url
    if not url:
        click.echo("Error: Store is disabled (PBOOK_DATABASE_URL is unset or empty).", err=True)
        sys.exit(1)

    run_migrations(url)
    click.echo("Migrations complete.")


@main.command()
@click.argument("entry_id", type=int)
@click.option(
    "--helpful",
    "is_helpful",
    flag_value=True,
    default=None,
    help="Mark entry as helpful.",
)
@click.option("--harmful", "is_helpful", flag_value=False, help="Mark entry as harmful.")
@click.option("--context", default="", help="Why the entry was helpful or harmful.")
def feedback(entry_id: int, is_helpful: bool | None, context: str) -> None:
    """Record feedback on a retrieved entry."""
    if is_helpful is None:
        click.echo("Error: specify --helpful or --harmful.", err=True)
        sys.exit(1)

    from pbook.models import RecordFeedbackInput
    from pbook.workflows.cli_ops import RecordFeedbackWorkflow

    result = _execute_workflow(
        RecordFeedbackWorkflow.run,
        RecordFeedbackInput(entry_id=entry_id, helpful=is_helpful, context=context),
        id_prefix="pbook-feedback",
    )
    if result.get("error") == "not_found":
        click.echo(f"Entry {entry_id} not found.", err=True)
        sys.exit(1)

    label = "helpful" if is_helpful else "harmful"
    click.echo(f"Recorded {label} feedback for entry {entry_id}: {result['title']}")
    if result.get("below_threshold"):
        click.echo(
            "Note: this entry has been retrieved < 3 times; the signal "
            "won't move ranking until the 3-retrieval threshold is met.",
        )


@main.command()
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List candidates without changing anything.",
)
@click.option("--apply", is_flag=True, default=False, help="Mark candidates for review.")
@click.option("--min-retrievals", default=5, help="Minimum retrievals for harmful ratio check.")
@click.option(
    "--max-harmful-ratio",
    default=0.5,
    type=float,
    help="Harmful ratio threshold.",
)
@click.option(
    "--max-stale-days",
    default=180,
    type=int,
    help="Days before unretrieved entry is stale.",
)
def prune(
    dry_run: bool,
    apply: bool,
    min_retrievals: int,
    max_harmful_ratio: float,
    max_stale_days: int,
) -> None:
    """Identify entries that should be pruned."""
    if not dry_run and not apply:
        click.echo("Error: specify --dry-run or --apply.", err=True)
        sys.exit(1)

    from pbook.models import PruneInput
    from pbook.workflows.cli_ops import PruneWorkflow

    result = _execute_workflow(
        PruneWorkflow.run,
        PruneInput(
            min_retrievals=min_retrievals,
            max_harmful_ratio=max_harmful_ratio,
            max_stale_days=max_stale_days,
            apply=apply,
        ),
        id_prefix="pbook-prune",
    )
    candidates = result["candidates"]

    if not candidates:
        click.echo("No prune candidates found.")
        return

    click.echo(f"Found {len(candidates)} prune candidate(s):")
    click.echo("")
    for entry in candidates:
        click.echo(f"  [{entry['id']}] {entry['title']}")
        click.echo(f"    Reason: {entry['prune_reason']}")

    if apply:
        click.echo(f"\nMarked {result['applied_count']} entry/entries for review.")


@main.command()
@click.argument(
    "transcript_path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--all", "ingest_all", is_flag=True, help="Discover and ingest all sessions.")
@click.option("--project", default="", help="Filter by project (with --all) or override.")
@click.option("--min-size", default=10240, type=int, help="Minimum session file size in bytes.")
@click.option("--dry-run", is_flag=True, help="Show sessions/stats without ingesting.")
@click.option("--force", is_flag=True, help="Reprocess already-ingested sessions.")
@click.option(
    "--temporal-address",
    default="localhost:7233",
    help="Temporal server address.",
)
def ingest(
    transcript_path: Path | None,
    ingest_all: bool,
    project: str,
    min_size: int,
    dry_run: bool,
    force: bool,
    temporal_address: str,
) -> None:
    """Ingest Claude Code conversation transcripts.

    Analyzes JSONL session files to identify unexpected problems and
    their resolutions, then extracts them as playbook entries.

    LLM analysis is routed through forge's batch API.

    \b
    Single session:
        pbook ingest ~/.claude/projects/<id>/session.jsonl

    All sessions:
        pbook ingest --all
        pbook ingest --all --project forge
        pbook ingest --all --dry-run
    """
    import asyncio

    from pbook.transcript import discover_sessions, parse_jsonl_file, render_transcript

    def _emit(payload: dict[str, Any]) -> None:
        click.echo(json.dumps(payload, indent=2))

    if not transcript_path and not ingest_all:
        _emit({"status": "error", "error": "provide a TRANSCRIPT_PATH or use --all"})
        sys.exit(1)

    # Discover sessions
    if ingest_all:
        sessions = discover_sessions(min_size=min_size)
        if project:
            sessions = [s for s in sessions if s.project_name == project]
    else:
        from pbook.transcript import SessionInfo, infer_project_name

        path = transcript_path
        assert path is not None
        session_id = path.stem
        proj = project or infer_project_name(path.parent.name)
        sessions = [
            SessionInfo(
                path=str(path),
                session_id=session_id,
                project_dir_name=path.parent.name,
                project_name=proj,
                size_bytes=path.stat().st_size,
            )
        ]

    if not sessions:
        _emit({"status": "no_sessions", "discovered": 0, "skipped_already_ingested": 0})
        return

    # Filter already-ingested sessions
    skipped = 0
    if not force:
        from pbook.models import FilterAlreadyIngestedInput
        from pbook.workflows.cli_ops import FilterAlreadyIngestedWorkflow

        filter_result = _execute_workflow(
            FilterAlreadyIngestedWorkflow.run,
            FilterAlreadyIngestedInput(
                session_ids=[s.session_id for s in sessions],
            ),
            id_prefix="pbook-ingest-filter",
        )
        fresh_ids = set(filter_result["fresh_ids"])
        before = len(sessions)
        sessions = [s for s in sessions if s.session_id in fresh_ids]
        skipped = before - len(sessions)

    if not sessions:
        _emit(
            {
                "status": "all_ingested",
                "skipped_already_ingested": skipped,
                "hint": "use --force to reprocess",
            }
        )
        return

    # Dry-run: emit session info as JSON
    if dry_run:
        by_project: dict[str, list[SessionInfo]] = {}
        for s in sessions:
            by_project.setdefault(s.project_name, []).append(s)

        projects_payload = []
        for proj_name, proj_sessions in sorted(by_project.items()):
            session_details = []
            for s in proj_sessions:
                detail = {
                    "session_id": s.session_id,
                    "path": s.path,
                    "size_bytes": s.size_bytes,
                }
                if len(proj_sessions) <= 3:
                    transcript = parse_jsonl_file(Path(s.path))
                    rendered = render_transcript(transcript)
                    detail["message_count"] = len(transcript.messages)
                    detail["rendered_chars"] = len(rendered)
                session_details.append(detail)
            projects_payload.append(
                {
                    "project": proj_name,
                    "session_count": len(proj_sessions),
                    "size_bytes": sum(s.size_bytes for s in proj_sessions),
                    "sessions": session_details,
                }
            )

        _emit(
            {
                "status": "dry_run",
                "session_count": len(sessions),
                "skipped_already_ingested": skipped,
                "total_size_bytes": sum(s.size_bytes for s in sessions),
                "projects": projects_payload,
            }
        )
        return

    # Submit to forge's task queue via Temporal
    session_dicts = [
        {"path": s.path, "project": s.project_name, "session_id": s.session_id} for s in sessions
    ]

    workflow_id = f"pbook-batch-ingest-{int(time.time())}"

    async def _submit() -> str:
        client = await _connect_checked(temporal_address)
        handle = await client.start_workflow(
            "BatchIngestionWorkflow",
            json.dumps({"sessions": session_dicts}),
            id=workflow_id,
            task_queue="forge-task-queue",
        )
        return handle.first_execution_run_id or ""

    try:
        run_id = asyncio.run(_submit())
    except Exception as exc:
        _emit({"status": "error", "error": str(exc)})
        sys.exit(1)

    # Seed running rows so `pbook sessions` shows them while the workflow is
    # in flight. The forge-side record_ingested_session callback flips them
    # to completed (or record_ingested_session_error flips them to error).
    from pbook.models import IngestedSessionStub, RecordStartedSessionsInput
    from pbook.workflows.cli_ops import RecordStartedSessionsWorkflow

    _execute_workflow(
        RecordStartedSessionsWorkflow.run,
        RecordStartedSessionsInput(
            sessions=[
                IngestedSessionStub(
                    session_id=s.session_id,
                    project_name=s.project_name,
                )
                for s in sessions
            ],
            workflow_id=workflow_id,
            run_id=run_id,
        ),
        id_prefix="pbook-ingest-seed",
    )

    _emit(
        {
            "status": "submitted",
            "workflow_id": workflow_id,
            "run_id": run_id,
            "task_queue": "forge-task-queue",
            "submitted_sessions": len(sessions),
            "skipped_already_ingested": skipped,
            "session_ids": [s.session_id for s in sessions],
        }
    )


@main.command()
@click.option("--project", default="", help="Filter by source project.")
@click.option("--limit", default=20, type=int, help="Maximum sessions to return.")
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def sessions(project: str, limit: int, output_json: bool) -> None:
    """List sessions ingested by `pbook ingest`, newest first."""
    from pbook.models import ListSessionsInput
    from pbook.workflows.cli_ops import ListSessionsWorkflow

    rows = _execute_workflow(
        ListSessionsWorkflow.run,
        ListSessionsInput(project=project or None, limit=limit),
        id_prefix="pbook-sessions",
    )

    if output_json:
        _emit_json(rows)
        return

    if not rows:
        click.echo("No ingested sessions found.")
        return

    for row in rows:
        status = row.get("status") or "completed"
        line = (
            f"{row['session_id']}  "
            f"status={status}  "
            f"project={row['project_name'] or '-'}  "
            f"experiences={row['experiences_found']}  "
            f"entries={row['entries_created']}  "
            f"at={row['ingested_at']}"
        )
        if status == "error" and row.get("error_message"):
            line += f"  error={row['error_message']}"
        click.echo(line)


@main.command(name="skill-prompt")
@click.option(
    "--operation",
    default="",
    help="Limit output to one workflow (query | discuss | review_queue | add).",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=True,
    help="Machine-readable JSON output (default).",
)
def skill_prompt(operation: str, output_json: bool) -> None:
    """Return editorial guidance for the pbook Claude Code skill.

    The payload contains:
      - commands: per-command description, args, example.
      - workflows: per-workflow markdown guidance (query, discuss,
        review_queue, add).
      - tags: canonical namespaces and notes on the tag system.

    Intended consumption: /skill-creator at build time to populate
    SKILL.md, and the skill agent at runtime to refresh context.
    """
    from pbook.skill_prompts import build_skill_prompt

    try:
        payload = build_skill_prompt(operation)
    except KeyError as exc:
        _emit_error(
            "validation_error",
            str(exc).strip("'"),
            json_mode=output_json,
        )

    if output_json:
        _emit_json(payload)
    else:
        click.echo(json.dumps(payload, indent=2, default=_json_default))
