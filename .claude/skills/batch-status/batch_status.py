"""Read-only queries against the Forge batch_jobs ledger and OCR projections.

Backs the ``batch-status`` Claude skill (invoked via ``batch-status.sh``, which
runs this under the repo's own venv with ``uv run python`` so SQLAlchemy —
already a workspace dependency — resolves without any extra install).

The store is Postgres (Supabase in production; the local podman stack's
Postgres, port 5434 on this machine, for local dev) reached via
``FORGE_DB_URL``. Every statement below is a SELECT — this module must never
grow an INSERT/UPDATE/DELETE/DDL statement, because the ambient
``FORGE_DB_URL`` in this environment points at the production database.

Status vocabulary (``sax_platform.contracts.models.BatchJobStatus``):
    submitted   in flight at the provider — the only non-terminal state
    ended       terminal success
    failed      terminal: provider rejected the submission, or reported the
                batch FAILED/CANCELED
    expired     terminal: provider TIMEOUT_EXCEEDED
    missing     terminal: the waiter gave up at its 25h ceiling
    processing  legacy only — written by the retired shared poller
                (pre-T4.1), never by the current timer-loop transport; kept
                read-tolerated so old rows stay legible

``batch_jobs`` (the platform's generic audit/spend ledger) has no domain
fields. Where OCR file/document context is wanted, this module LEFT JOINs
``ocr_job_status`` (apps/ocr/src/ocr/store.py) on
``batch_jobs.id = ocr_job_status.request_id`` — the same join
``ocr.activities.execute_list_ocr_jobs`` uses. LEFT JOIN because non-OCR
batch_jobs rows have no ocr_job_status counterpart at all.
"""

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa

# ---------------------------------------------------------------------------
# Pure core: no I/O below this point in the functions that don't touch
# `engine` or `os.environ`.
# ---------------------------------------------------------------------------

_BUCKET_BY_STATUS: Mapping[str, str] = {
    "submitted": "in flight",
    "ended": "success",
    "failed": "failure",
    "expired": "failure",
    "missing": "failure",
    "processing": "legacy (poller-era)",
}


def bucket_label(status: str) -> str:
    """Map a raw ``batch_jobs.status`` value to a display bucket."""
    return _BUCKET_BY_STATUS.get(status, f"unknown ({status})")


def coerce_datetime(value: object) -> datetime | None:
    """Normalize a timestamp cell to ``datetime | None``.

    Raw ``text()`` queries skip SQLAlchemy's typed result processing, so a
    timestamp column can come back as a native ``datetime`` (psycopg2 against
    Postgres) or a plain ``str`` (sqlite3's DBAPI, e.g. against the
    verification fixture). Both schemes must produce the same downstream
    behavior — that's the entire point of going through SQLAlchemy — so this
    accepts either.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    msg = f"unexpected timestamp value: {value!r}"
    raise TypeError(msg)


def hours_since(ts: datetime | None, now: datetime) -> float | None:
    """Hours elapsed between ``ts`` and ``now``, or ``None`` if ``ts`` is absent.

    ``now`` is injected (pseudo-I/O) rather than read internally, so this stays
    a pure function of its arguments.
    """
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return (now - ts).total_seconds() / 3600.0


def display_value(value: object) -> str:
    """Render one cell for the text table."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def format_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render rows as a simple aligned text table."""
    if not rows:
        return "(no rows)"
    columns = list(rows[0].keys())
    body = [[display_value(row[col]) for col in columns] for row in rows]
    widths = [max(len(col), *(len(cell[i]) for cell in body)) for i, col in enumerate(columns)]

    def render_row(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))

    lines = [render_row(columns), render_row(["-" * width for width in widths])]
    lines.extend(render_row(cell) for cell in body)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Queries — parameterized SELECTs only.
# ---------------------------------------------------------------------------

_RECENT_BATCH_JOBS = sa.text("""
    SELECT b.id, b.batch_id, b.provider, b.status, b.error_message,
           b.created_at, b.updated_at, j.file_path, j.document_id
    FROM batch_jobs b
    LEFT JOIN ocr_job_status j ON b.id = j.request_id
    ORDER BY b.updated_at DESC
    LIMIT 20
""")

_RECENT_OCR_RESULTS = sa.text("""
    SELECT document_id, file_path, page_count, model_name,
           input_tokens, output_tokens, batch_id, created_at
    FROM ocr_results
    ORDER BY created_at DESC
    LIMIT 20
""")

_SUMMARY = sa.text("""
    SELECT provider, status, COUNT(*) AS count
    FROM batch_jobs
    GROUP BY provider, status
    ORDER BY provider, status
""")

# failed = terminal-failure statuses only. 'ended' is terminal *success* and
# must never appear here (this is the bug the skill previously had: the old
# filter was `status NOT IN ('submitted', 'succeeded')`, which listed every
# 'ended' row as a failure).
_FAILED = sa.text("""
    SELECT b.id, b.batch_id, b.provider, b.status, b.error_message, b.updated_at,
           j.document_id, j.file_path
    FROM batch_jobs b
    LEFT JOIN ocr_job_status j ON b.id = j.request_id
    WHERE b.status IN ('failed', 'expired', 'missing')
    ORDER BY b.updated_at DESC
""")

# pending = the only non-terminal current status ('submitted') plus legacy
# 'processing' rows, which are neither hidden nor mislabeled as current —
# hours_ago/bucket distinguish them at display time.
_PENDING = sa.text("""
    SELECT id, batch_id, provider, status, created_at
    FROM batch_jobs
    WHERE status IN ('submitted', 'processing')
    ORDER BY status, created_at
""")

_OCR_BY_BATCH = sa.text("""
    SELECT document_id, file_path, page_count, length(text) AS text_chars,
           input_tokens, output_tokens
    FROM ocr_results
    WHERE batch_id = :batch_id
""")

_CROSS_REF = sa.text("""
    SELECT b.id, b.batch_id, b.provider, b.status,
           j.request_id, j.document_id, j.file_path, j.status AS ocr_status
    FROM batch_jobs b
    LEFT JOIN ocr_job_status j ON b.id = j.request_id
    ORDER BY b.updated_at DESC
    LIMIT 30
""")


# ---------------------------------------------------------------------------
# Shell: I/O.
# ---------------------------------------------------------------------------


def require_db_url() -> str:
    """Read ``FORGE_DB_URL`` or fail fast with a clear message."""
    db_url = os.environ.get("FORGE_DB_URL", "").strip()
    if not db_url:
        print(
            "ERROR: FORGE_DB_URL is not set.\n"
            "Point it at the Forge Postgres store (Supabase in production; the\n"
            "local podman stack's Postgres — port 5434 on this machine — for\n"
            "local dev). SQLAlchemy accepts any of its supported URL schemes.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return db_url


def build_engine(db_url: str) -> sa.Engine:
    """Create the engine. ``postgresql_readonly`` is a defense-in-depth belt on
    top of "every statement here is a SELECT": on Postgres it opens each
    transaction with ``SET TRANSACTION READ ONLY``, so an accidental write
    would be rejected by the server rather than merely by code review. Other
    dialects (sqlite, used by this skill's own fixture-based verification)
    ignore the dialect-namespaced option.
    """
    return sa.create_engine(db_url, execution_options={"postgresql_readonly": True})


def fetch(
    engine: sa.Engine, stmt: sa.TextClause, params: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Execute one read-only SELECT and return rows as plain dicts."""
    with engine.connect() as conn:
        result = conn.execute(stmt, params or {})
        return [dict(row) for row in result.mappings()]


def cmd_default(engine: sa.Engine) -> None:
    jobs = fetch(engine, _RECENT_BATCH_JOBS)
    for row in jobs:
        row["bucket"] = bucket_label(row["status"])
    print("=== Recent Batch Jobs (last 20) ===")
    print(format_table(jobs))
    print()
    results = fetch(engine, _RECENT_OCR_RESULTS)
    print("=== Recent OCR Results (last 20) ===")
    print(format_table(results))


def cmd_summary(engine: sa.Engine) -> None:
    rows = fetch(engine, _SUMMARY)
    for row in rows:
        row["bucket"] = bucket_label(row["status"])
    print(format_table(rows))


def cmd_failed(engine: sa.Engine) -> None:
    print(format_table(fetch(engine, _FAILED)))


def cmd_pending(engine: sa.Engine) -> None:
    rows = fetch(engine, _PENDING)
    now = datetime.now(UTC)
    for row in rows:
        row["bucket"] = bucket_label(row["status"])
        hours = hours_since(coerce_datetime(row["created_at"]), now)
        row["hours_ago"] = round(hours, 1) if hours is not None else None
    print(format_table(rows))


def cmd_ocr(engine: sa.Engine, batch_id: str) -> None:
    print(format_table(fetch(engine, _OCR_BY_BATCH, {"batch_id": batch_id})))


def cmd_cross_ref(engine: sa.Engine) -> None:
    rows = fetch(engine, _CROSS_REF)
    for row in rows:
        row["bucket"] = bucket_label(row["status"])
        row["has_ocr_status"] = "yes" if row["request_id"] is not None else "no"
    print(format_table(rows))


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="batch_status.py", description="Read-only Forge batch/OCR status queries."
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("default", help="Recent batch jobs and OCR results")
    sub.add_parser("summary", help="Batch job counts by provider and status")
    sub.add_parser("failed", help="Terminal-failure batch jobs (failed/expired/missing)")
    sub.add_parser("pending", help="In-flight (submitted) and legacy (processing) batch jobs")
    ocr_parser = sub.add_parser("ocr", help="OCR results for a provider batch_id")
    ocr_parser.add_argument("batch_id")
    sub.add_parser("cross-ref", help="Batch jobs cross-referenced with OCR job status")

    args = parser.parse_args(argv)
    command = args.command or "default"

    engine = build_engine(require_db_url())

    if command == "default":
        cmd_default(engine)
    elif command == "summary":
        cmd_summary(engine)
    elif command == "failed":
        cmd_failed(engine)
    elif command == "pending":
        cmd_pending(engine)
    elif command == "ocr":
        cmd_ocr(engine, args.batch_id)
    elif command == "cross-ref":
        cmd_cross_ref(engine)
    else:
        parser.error(f"unknown command: {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
