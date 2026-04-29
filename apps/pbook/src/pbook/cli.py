"""Click CLI for the playbook service.

Thin wrapper over store functions.  Each command resolves the database
path, runs migrations if needed, and delegates to a store function.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from typing import NoReturn

from pbook.models import PlaybookEntry
from pbook.store import (
    build_entry_dict,
    check_duplicate,
    get_db_path,
    get_engine,
    get_entry_by_id,
    list_recent_entries,
    record_feedback,
    run_migrations,
    save_entries,
    update_entry,
)
from pbook.tags import validate_tags

logger = logging.getLogger(__name__)


def _resolve_db() -> tuple:
    """Resolve the database path, run migrations, and return (engine, db_path).

    Exits with an error if the store is disabled.
    """
    db_path = get_db_path()
    if db_path is None:
        click.echo("Error: Store is disabled (PBOOK_DB_PATH is empty).", err=True)
        sys.exit(1)

    run_migrations(db_path)
    engine = get_engine(db_path)
    return engine, db_path


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


def _strip_embedding(row: dict) -> dict:
    """Drop binary embedding columns from a row dict for JSON output."""
    return {k: v for k, v in row.items() if k not in _BINARY_FIELDS}


def _reshape_entry(row: dict) -> dict:
    """Strip binary fields and parse `tags_json` into a `tags` list.

    Used by every entry-shaped JSON output site so the skill consumer sees
    a consistent shape regardless of which command produced the row.
    """
    cleaned = _strip_embedding(row)
    tags_raw = cleaned.pop("tags_json", None)
    if tags_raw is not None:
        if isinstance(tags_raw, str):
            try:
                cleaned["tags"] = json.loads(tags_raw)
            except json.JSONDecodeError:
                cleaned["tags"] = []
        else:
            cleaned["tags"] = list(tags_raw)
    return cleaned


def _json_default(obj):
    """JSON encoder for datetimes (ISO 8601) and any other non-serializable type."""
    from datetime import UTC, datetime

    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=UTC)
        return obj.isoformat()
    return str(obj)


def _emit_json(payload, *, indent: int = 2) -> None:
    """Print JSON to stdout using the canonical encoder."""
    click.echo(json.dumps(payload, default=_json_default, indent=indent))


def _emit_error(
    code: str, message: str, *, json_mode: bool, exit_code: int = 1,
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


def _format_entry(entry: dict) -> str:
    """Format an entry dict for human-readable terminal output."""
    tags_raw = entry.get("tags_json", "[]")
    tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
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
def main(verbose: bool) -> None:
    """pbook — Knowledge playbook service."""
    from pbook.log_config import setup_logging

    setup_logging(level=logging.DEBUG if verbose else logging.INFO, console=True)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--temporal-address", default="localhost:7233",
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
    "--include-rejected", is_flag=True,
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
    engine, _ = _resolve_db()

    if tag:
        from pbook.store import get_entries_by_tags

        entries = get_entries_by_tags(
            engine, list(tag), limit=limit, include_rejected=include_rejected,
        )
    else:
        entries = list_recent_entries(
            engine, limit=limit, include_rejected=include_rejected,
        )

    if entry_type:
        entries = [e for e in entries if e.get("entry_type") == entry_type]
    if project:
        entries = [e for e in entries if e.get("source_project") == project]
    if needs_review:
        entries = [e for e in entries if e.get("needs_review")]

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
    engine, _ = _resolve_db()
    entry = get_entry_by_id(engine, entry_id)

    if entry is None:
        _emit_error(
            "not_found", f"Entry {entry_id} not found.", json_mode=output_json,
        )

    if output_json:
        _emit_json(_reshape_entry(entry))
    else:
        click.echo(_format_entry(entry))


@main.command()
@click.option(
    "--file", "file_path",
    type=click.Path(exists=True, path_type=Path),
    help="JSON file containing PlaybookEntry.",
)
@click.option("--schema", "show_schema", is_flag=True, help="Print JSON schema.")
def add(file_path: Path | None, show_schema: bool) -> None:
    """Add a playbook entry."""
    if show_schema:
        click.echo(json.dumps(PlaybookEntry.model_json_schema(), indent=2))
        return

    if file_path is None:
        click.echo("Error: --file is required (or use --schema to see the format).", err=True)
        sys.exit(1)

    raw_json = file_path.read_text()
    try:
        entry = PlaybookEntry.model_validate_json(raw_json)
    except Exception as exc:
        click.echo(f"Validation error: {exc}", err=True)
        sys.exit(1)

    tag_errors = validate_tags(entry.tags)
    if tag_errors:
        for err in tag_errors:
            click.echo(f"Tag error: {err}", err=True)
        sys.exit(1)

    engine, _ = _resolve_db()
    entry_dict = build_entry_dict(entry)
    save_entries(engine, [entry_dict])
    click.echo(f"Added: {entry.title}")


@main.command()
@click.argument("entry_id", type=int)
@click.option(
    "--file", "file_path", required=True,
    type=click.Path(exists=True, path_type=Path),
    help="JSON file with updated fields.",
)
def update(entry_id: int, file_path: Path) -> None:
    """Update an entry by ID."""
    engine, _ = _resolve_db()
    existing = get_entry_by_id(engine, entry_id)
    if existing is None:
        click.echo(f"Entry {entry_id} not found.", err=True)
        sys.exit(1)

    updates = json.loads(file_path.read_text())

    if "tags" in updates:
        tag_errors = validate_tags(updates["tags"])
        if tag_errors:
            for err in tag_errors:
                click.echo(f"Tag error: {err}", err=True)
            sys.exit(1)
        updates["tags_json"] = json.dumps(updates.pop("tags"))

    update_entry(engine, entry_id, updates)
    click.echo(f"Updated entry {entry_id}.")


@main.command()
@click.argument("entry_id", type=int)
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def approve(entry_id: int, output_json: bool) -> None:
    """Clear the needs-review flag on an entry."""
    engine, _ = _resolve_db()
    existing = get_entry_by_id(engine, entry_id)
    if existing is None:
        _emit_error(
            "not_found", f"Entry {entry_id} not found.", json_mode=output_json,
        )

    update_entry(engine, entry_id, {"needs_review": False})

    if output_json:
        _emit_json({
            "id": entry_id,
            "title": existing["title"],
            "approved": True,
            "needs_review": False,
            "rejected": bool(existing.get("rejected", False)),
            "rejection_reason": existing.get("rejection_reason"),
        })
    else:
        click.echo(f"Approved entry {entry_id}: {existing['title']}")


@main.command()
@click.argument("entry_id", type=int)
@click.option("--reason", default="", help="Why the entry is being rejected (optional).")
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def reject(entry_id: int, reason: str, output_json: bool) -> None:
    """Mark an entry as rejected (soft-mark; the row is preserved for audit)."""
    from pbook.store import mark_rejected

    engine, _ = _resolve_db()
    existing = get_entry_by_id(engine, entry_id)
    if existing is None:
        _emit_error(
            "not_found", f"Entry {entry_id} not found.", json_mode=output_json,
        )

    reason_value: str | None = reason if reason else None
    mark_rejected(engine, entry_id, reason=reason_value)

    if output_json:
        _emit_json({
            "id": entry_id,
            "title": existing["title"],
            "approved": False,
            "rejected": True,
            "rejection_reason": reason_value,
        })
    else:
        suffix = f" — {reason}" if reason else ""
        click.echo(f"Rejected entry {entry_id}: {existing['title']}{suffix}")


@main.command(name="check-duplicate")
@click.option("--title", required=True, help="Title to check for duplicates.")
@click.option("--tag", multiple=True, help="Tags to refine duplicate search.")
def check_duplicate_cmd(title: str, tag: tuple[str, ...]) -> None:
    """Check for duplicate entries matching a title."""
    engine, _ = _resolve_db()
    matches = check_duplicate(engine, title, tags=list(tag) if tag else None)

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
    "--file", "file_path", required=True,
    type=click.Path(exists=True, path_type=Path),
    help="JSON file with PushExperienceInput data.",
)
@click.option(
    "--temporal-address", default="localhost:7233",
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

    async def _submit():
        from temporalio.client import Client

        from pbook.worker import PBOOK_TASK_QUEUE
        from pbook.workflows.extraction import ExtractionWorkflow

        client = await Client.connect(temporal_address)
        result = await client.execute_workflow(
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
    "--threshold", type=float, default=0.0,
    help="Drop matches below this cosine similarity (0.0 disables the cutoff).",
)
@click.option(
    "--mode", type=click.Choice(["create", "fix"]), default="create",
    help="Tiebreaker mode when ranking ties on similarity.",
)
@click.option("--limit", default=20, type=int, help="Cap on returned entries.")
@click.option(
    "--token-budget", default=5000, type=int,
    help="Token budget for the packed result.",
)
@click.option(
    "--include-rejected", is_flag=True,
    help="Include entries that have been rejected.",
)
@click.option(
    "--include-unapproved", is_flag=True,
    help="Include entries flagged needs_review (default: only approved entries).",
)
@click.option(
    "--temporal-address", default="localhost:7233",
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

    async def _submit():
        from temporalio.client import Client

        from pbook.models import RetrievalInput, RetrievalMode
        from pbook.worker import PBOOK_TASK_QUEUE
        from pbook.workflows.retrieval import RetrievalWorkflow

        client = await Client.connect(temporal_address)
        retrieval_mode = RetrievalMode(mode)
        return await client.execute_workflow(
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
        _emit_json({
            "entries": [_reshape_entry(e) for e in entries],
            "total_candidates": result.total_candidates,
            "token_count": result.token_count,
        })
        return

    if not entries:
        click.echo("No matches found.")
        return

    for entry in entries:
        sim = entry.get("similarity")
        prefix = f"[sim={sim:.3f}] " if sim is not None else ""
        click.echo(f"{prefix}{_format_entry(entry)}")
        click.echo("")


@main.command()
@click.option("--limit", default=20, help="Maximum entries to show.")
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def review(limit: int, output_json: bool) -> None:
    """List entries needing review."""
    engine, _ = _resolve_db()
    entries = list_recent_entries(engine, limit=limit)
    needs_review = [e for e in entries if e.get("needs_review")]

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
    "--json", "output_json", is_flag=True, default=True,
    help="Machine-readable JSON output (default).",
)
def sources(entry_id: int, output_json: bool) -> None:
    """List the entry_sources rows that produced an entry."""
    from pbook.store import list_entry_sources_for_entry

    engine, _ = _resolve_db()
    if get_entry_by_id(engine, entry_id) is None:
        _emit_error(
            "not_found", f"Entry {entry_id} not found.", json_mode=output_json,
        )

    rows = list_entry_sources_for_entry(engine, entry_id)
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
    "--path", "path_override",
    type=click.Path(exists=True, path_type=Path),
    help="Override transcript path (used when the session isn't in ingested_sessions).",
)
@click.option(
    "--raw", is_flag=True,
    help="Emit the JSONL bytes verbatim instead of rendered text.",
)
@click.option(
    "--json", "output_json", is_flag=True,
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
    from pbook.transcript import discover_sessions, parse_jsonl_file, render_transcript

    if path_override is not None:
        jsonl_path: Path | None = path_override
    else:
        jsonl_path = None
        # discover_sessions scans ~/.claude/projects/; find the matching session.
        for s in discover_sessions(min_size=0, exclude_subagents=False):
            if s.session_id == session_id:
                jsonl_path = Path(s.path)
                break

    if jsonl_path is None or not jsonl_path.exists():
        _emit_error(
            "session_file_missing",
            f"No transcript found for session {session_id}. Use --path to point at one.",
            json_mode=output_json,
        )

    if raw:
        text = jsonl_path.read_text()
    else:
        transcript = parse_jsonl_file(jsonl_path)
        text = render_transcript(transcript)

    if output_json:
        _emit_json({
            "session_id": session_id,
            "path": str(jsonl_path),
            "raw": raw,
            "text": text,
        })
    else:
        click.echo(text)


@main.command()
@click.option(
    "--json", "output_json", is_flag=True, default=True,
    help="Machine-readable JSON output (default).",
)
def tags(output_json: bool) -> None:
    """Show valid tag namespaces and the values currently in use.

    Combines the canonical namespace list (closed set in pbook.tags)
    with values_in_use, computed across non-rejected entries.
    """
    from pbook.store import list_tag_values_in_use
    from pbook.tags import EXTRACTED_NAMESPACES, GENERAL_NAMESPACES

    engine, _ = _resolve_db()
    payload = {
        "namespaces": {
            "general": sorted(GENERAL_NAMESPACES),
            "extracted": sorted(EXTRACTED_NAMESPACES),
        },
        "values_in_use": list_tag_values_in_use(engine),
    }

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
def migrate() -> None:
    """Run database migrations."""
    db_path = get_db_path()
    if db_path is None:
        click.echo("Error: Store is disabled (PBOOK_DB_PATH is empty).", err=True)
        sys.exit(1)

    run_migrations(db_path)
    click.echo(f"Migrations complete: {db_path}")


@main.command()
@click.argument("entry_id", type=int)
@click.option(
    "--helpful", "is_helpful", flag_value=True, default=None,
    help="Mark entry as helpful.",
)
@click.option("--harmful", "is_helpful", flag_value=False, help="Mark entry as harmful.")
@click.option("--context", default="", help="Why the entry was helpful or harmful.")
def feedback(entry_id: int, is_helpful: bool | None, context: str) -> None:
    """Record feedback on a retrieved entry."""
    if is_helpful is None:
        click.echo("Error: specify --helpful or --harmful.", err=True)
        sys.exit(1)

    engine, _ = _resolve_db()
    existing = get_entry_by_id(engine, entry_id)
    if existing is None:
        click.echo(f"Entry {entry_id} not found.", err=True)
        sys.exit(1)

    record_feedback(engine, entry_id, helpful=is_helpful)
    label = "helpful" if is_helpful else "harmful"
    click.echo(f"Recorded {label} feedback for entry {entry_id}: {existing['title']}")


@main.command()
@click.option(
    "--dry-run", is_flag=True, default=False,
    help="List candidates without changing anything.",
)
@click.option("--apply", is_flag=True, default=False, help="Mark candidates for review.")
@click.option("--min-retrievals", default=5, help="Minimum retrievals for harmful ratio check.")
@click.option(
    "--max-harmful-ratio", default=0.5, type=float,
    help="Harmful ratio threshold.",
)
@click.option(
    "--max-stale-days", default=180, type=int,
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

    engine, _ = _resolve_db()

    from pbook.activities.maintenance import identify_prune_candidates
    from pbook.store import list_all_entries

    all_entries = list_all_entries(engine)
    candidates = identify_prune_candidates(
        all_entries,
        min_retrievals=min_retrievals,
        max_harmful_ratio=max_harmful_ratio,
        max_stale_days=max_stale_days,
    )

    if not candidates:
        click.echo("No prune candidates found.")
        return

    click.echo(f"Found {len(candidates)} prune candidate(s):")
    click.echo("")
    for entry in candidates:
        click.echo(f"  [{entry['id']}] {entry['title']}")
        click.echo(f"    Reason: {entry['prune_reason']}")

    if apply:
        for entry in candidates:
            entry_id = entry["id"]
            existing_tags = json.loads(entry.get("tags_json", "[]"))
            if "pattern:prune-candidate" not in existing_tags:
                existing_tags.append("pattern:prune-candidate")
            update_entry(engine, entry_id, {
                "needs_review": True,
                "tags_json": json.dumps(existing_tags),
            })
        click.echo(f"\nMarked {len(candidates)} entry/entries for review.")


@main.command()
@click.argument(
    "transcript_path", required=False,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--all", "ingest_all", is_flag=True, help="Discover and ingest all sessions.")
@click.option("--project", default="", help="Filter by project (with --all) or override.")
@click.option("--min-size", default=10240, type=int, help="Minimum session file size in bytes.")
@click.option("--dry-run", is_flag=True, help="Show sessions/stats without ingesting.")
@click.option("--force", is_flag=True, help="Reprocess already-ingested sessions.")
@click.option(
    "--temporal-address", default="localhost:7233",
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

    def _emit(payload: dict) -> None:
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
        sessions = [SessionInfo(
            path=str(path),
            session_id=session_id,
            project_dir_name=path.parent.name,
            project_name=proj,
            size_bytes=path.stat().st_size,
        )]

    if not sessions:
        _emit({"status": "no_sessions", "discovered": 0, "skipped_already_ingested": 0})
        return

    # Filter already-ingested sessions
    skipped = 0
    if not force:
        engine, _ = _resolve_db()
        from pbook.store import get_ingested_session_ids

        ingested = get_ingested_session_ids(engine)
        before = len(sessions)
        sessions = [s for s in sessions if s.session_id not in ingested]
        skipped = before - len(sessions)

    if not sessions:
        _emit({
            "status": "all_ingested",
            "skipped_already_ingested": skipped,
            "hint": "use --force to reprocess",
        })
        return

    # Dry-run: emit session info as JSON
    if dry_run:
        by_project: dict[str, list] = {}
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
            projects_payload.append({
                "project": proj_name,
                "session_count": len(proj_sessions),
                "size_bytes": sum(s.size_bytes for s in proj_sessions),
                "sessions": session_details,
            })

        _emit({
            "status": "dry_run",
            "session_count": len(sessions),
            "skipped_already_ingested": skipped,
            "total_size_bytes": sum(s.size_bytes for s in sessions),
            "projects": projects_payload,
        })
        return

    # Submit to forge's task queue via Temporal
    session_dicts = [
        {"path": s.path, "project": s.project_name, "session_id": s.session_id}
        for s in sessions
    ]

    workflow_id = f"pbook-batch-ingest-{int(time.time())}"

    async def _submit() -> str:
        from temporalio.client import Client

        client = await Client.connect(temporal_address)
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
    # in flight. The workflow's record_ingested_session callback flips them
    # to completed (or record_ingested_session_error flips them to error).
    engine_for_seed, _ = _resolve_db()
    from pbook.store import record_ingested_session_started

    for s in sessions:
        record_ingested_session_started(
            engine_for_seed,
            session_id=s.session_id,
            project_name=s.project_name,
            workflow_id=workflow_id,
            run_id=run_id,
        )

    _emit({
        "status": "submitted",
        "workflow_id": workflow_id,
        "run_id": run_id,
        "task_queue": "forge-task-queue",
        "submitted_sessions": len(sessions),
        "skipped_already_ingested": skipped,
        "session_ids": [s.session_id for s in sessions],
    })


@main.command()
@click.option("--project", default="", help="Filter by source project.")
@click.option("--limit", default=20, type=int, help="Maximum sessions to return.")
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def sessions(project: str, limit: int, output_json: bool) -> None:
    """List sessions ingested by `pbook ingest`, newest first."""
    from pbook.store import list_ingested_sessions

    engine, _ = _resolve_db()
    rows = list_ingested_sessions(engine, project=project or None, limit=limit)

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
@click.option("--operation", default="add", help="Operation to get instructions for.")
def skill_prompt(operation: str) -> None:
    """Print server-provided instructions for a skill operation (stub)."""
    click.echo(f"# pbook skill-prompt: {operation}")
    click.echo("")
    click.echo("This command will return LLM instructions for the requested operation.")
    click.echo("Not yet implemented — see Phase 5 of the implementation plan.")
