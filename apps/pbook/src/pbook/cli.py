"""Click CLI for the playbook service.

Thin wrapper over store functions.  Each command resolves the database
path, runs migrations if needed, and delegates to a store function.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from pbook.models import PlaybookEntry
from pbook.store import (
    build_entry_dict,
    check_duplicate,
    delete_entry,
    get_db_path,
    get_engine,
    get_entry_by_id,
    list_recent_entries,
    run_migrations,
    save_entries,
    update_entry,
)
from pbook.tags import validate_tags


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

    lines.append(f"  {entry['content'][:200]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """pbook — Knowledge playbook service."""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@main.command(name="list")
@click.option("--tag", multiple=True, help="Filter by tag (repeatable, OR match).")
@click.option("--type", "entry_type", default="", help="Filter by entry type.")
@click.option("--project", default="", help="Filter by source project.")
@click.option("--limit", default=20, help="Maximum entries to return.")
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def list_entries(
    tag: tuple[str, ...],
    entry_type: str,
    project: str,
    limit: int,
    output_json: bool,
) -> None:
    """List playbook entries."""
    engine, _ = _resolve_db()

    if tag:
        from pbook.store import get_entries_by_tags

        entries = get_entries_by_tags(engine, list(tag), limit=limit)
    else:
        entries = list_recent_entries(engine, limit=limit)

    if entry_type:
        entries = [e for e in entries if e.get("entry_type") == entry_type]
    if project:
        entries = [e for e in entries if e.get("source_project") == project]

    if not entries:
        click.echo("No entries found.")
        return

    if output_json:
        click.echo(json.dumps(entries, default=str, indent=2))
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
        click.echo(f"Entry {entry_id} not found.", err=True)
        sys.exit(1)

    if output_json:
        click.echo(json.dumps(entry, default=str, indent=2))
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
def approve(entry_id: int) -> None:
    """Clear the needs-review flag on an entry."""
    engine, _ = _resolve_db()
    existing = get_entry_by_id(engine, entry_id)
    if existing is None:
        click.echo(f"Entry {entry_id} not found.", err=True)
        sys.exit(1)

    update_entry(engine, entry_id, {"needs_review": False})
    click.echo(f"Approved entry {entry_id}: {existing['title']}")


@main.command()
@click.argument("entry_id", type=int)
def reject(entry_id: int) -> None:
    """Delete an entry (reject it)."""
    engine, _ = _resolve_db()
    existing = get_entry_by_id(engine, entry_id)
    if existing is None:
        click.echo(f"Entry {entry_id} not found.", err=True)
        sys.exit(1)

    delete_entry(engine, entry_id)
    click.echo(f"Rejected and deleted entry {entry_id}: {existing['title']}")


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
def migrate() -> None:
    """Run database migrations."""
    db_path = get_db_path()
    if db_path is None:
        click.echo("Error: Store is disabled (PBOOK_DB_PATH is empty).", err=True)
        sys.exit(1)

    run_migrations(db_path)
    click.echo(f"Migrations complete: {db_path}")


@main.command(name="skill-prompt")
@click.option("--operation", default="add", help="Operation to get instructions for.")
def skill_prompt(operation: str) -> None:
    """Print server-provided instructions for a skill operation (stub)."""
    click.echo(f"# pbook skill-prompt: {operation}")
    click.echo("")
    click.echo("This command will return LLM instructions for the requested operation.")
    click.echo("Not yet implemented — see Phase 5 of the implementation plan.")
