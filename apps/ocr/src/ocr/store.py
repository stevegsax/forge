"""OCR store — the OCR app's own tables and data-access functions.

Owns the ``ocr_``-prefixed tables in the shared database (``FORGE_DB_URL``) via its
own SQLAlchemy ``Base`` and its own Alembic chain (added in the migrations
increment). Connection config and the idempotent-insert primitive are shared from
``sax_platform.db``; blob I/O from ``sax_platform.contracts.s3_blobs``; the column
type from ``sax_platform.contracts.types``. This module imports only
``sax_platform`` — never ``forge``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

import sqlalchemy as sa

# Shared connection + insert primitives (re-exported so callers can do
# `from ocr.store import get_store_engine, save_ocr_result, ...`).
from sax_platform.contracts.types import UTCDateTime
from sax_platform.db import (
    get_store_engine as get_store_engine,
)
from sax_platform.db import (
    insert_or_ignore,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ``OcrProcessingStatus`` lives in the pure model module (imports flow inward);
# store.py re-exposes it via this import so ``from ocr.store import
# OcrProcessingStatus`` keeps working.
from ocr.models import OcrProcessingStatus as OcrProcessingStatus

if TYPE_CHECKING:
    from sax_platform.contracts.s3_blobs import S3Blobs
    from sqlalchemy import Engine
    from sqlalchemy.dialects.postgresql import Insert as PostgresInsert
    from sqlalchemy.dialects.sqlite import Insert as SQLiteInsert


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Tables (ocr_-prefixed; FileContentBlob rename to ocr_* deferred to the squash)
# ---------------------------------------------------------------------------


class OcrResult(Base):
    __tablename__ = "ocr_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(sa.String, nullable=False, unique=True, index=True)
    file_path: Mapped[str] = mapped_column(sa.String, nullable=False)
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    page_count: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default="0"
    )
    model_name: Mapped[str] = mapped_column(sa.String, nullable=False)
    input_tokens: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    batch_id: Mapped[str] = mapped_column(sa.String, nullable=False)
    workflow_id: Mapped[str] = mapped_column(sa.String, nullable=False)
    file_hash: Mapped[str | None] = mapped_column(sa.String, nullable=True, index=True)
    marked_for_removal: Mapped[bool] = mapped_column(
        sa.Boolean,
        nullable=False,
        default=False,
        server_default=sa.false(),
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
        server_default=sa.func.now(),
    )


class FileContentBlob(Base):
    __tablename__ = "ocr_file_content_blobs"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    s3_key: Mapped[str] = mapped_column(sa.String, nullable=False)
    mime_type: Mapped[str] = mapped_column(sa.String, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
        server_default=sa.func.now(),
    )


class OcrImage(Base):
    __tablename__ = "ocr_images"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        sa.String, nullable=False, default="", server_default=sa.text("''"), index=True
    )
    page_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    original_image_id: Mapped[str] = mapped_column(sa.String, nullable=False)
    s3_key: Mapped[str] = mapped_column(sa.String, nullable=False)
    mime_type: Mapped[str] = mapped_column(sa.String, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    top_left_x: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    top_left_y: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    bottom_right_x: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    bottom_right_y: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
        server_default=sa.func.now(),
    )


class OcrJobStatus(Base):
    """OCR's own status projection, keyed by the single correlation id
    (``request_id`` == provider custom_id == platform ``batch_jobs`` PK).

    Holds ``document_id``/``file_path`` (which left the generic ``batch_jobs``) and
    the coarse processing status. Wired by the OCR workflow + status query in the
    cross-queue increment.
    """

    __tablename__ = "ocr_job_status"

    request_id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    document_id: Mapped[str] = mapped_column(sa.String, nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(
        sa.String, nullable=False, default="", server_default=sa.text("''")
    )
    status: Mapped[str] = mapped_column(sa.String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
        server_default=sa.func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
        server_default=sa.func.now(),
    )


class OcrTrackerHeartbeat(Base):
    """Single-row liveness/telemetry marker for the stateless status tracker (T4.4).

    The tracker workflow sweeps once every ~2 minutes; this row records the last
    sweep so an operator can see the tracker is alive. It is a singleton — always
    ``id == 1`` — upserted each cycle: ``last_run_at``/``live_jobs``/``hints_sent``
    are overwritten with the current cycle's values and ``cycles_total`` is
    incremented, so the running count survives concurrent or duplicated runs.
    """

    __tablename__ = "ocr_tracker_heartbeat"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=False)
    last_run_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
        server_default=sa.func.now(),
    )
    live_jobs: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default="0"
    )
    hints_sent: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default="0"
    )
    cycles_total: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default="0"
    )


#: The tracker heartbeat is a singleton row keyed on this constant id.
TRACKER_HEARTBEAT_ID: Final = 1


# Typed Table handles. ``Model.__table__`` is typed ``FromClause`` by SQLAlchemy's
# declarative stubs, but ``metadata.tables`` is typed ``Table`` — which is what the
# sax_platform DB helpers (e.g. ``insert_or_ignore``) and ``sa.insert``/``update``/
# ``delete`` require.
_OCR_RESULTS_TABLE: sa.Table = Base.metadata.tables[OcrResult.__tablename__]
_FILE_CONTENT_BLOBS_TABLE: sa.Table = Base.metadata.tables[FileContentBlob.__tablename__]
_OCR_IMAGES_TABLE: sa.Table = Base.metadata.tables[OcrImage.__tablename__]
_OCR_JOB_STATUS_TABLE: sa.Table = Base.metadata.tables[OcrJobStatus.__tablename__]
_OCR_TRACKER_HEARTBEAT_TABLE: sa.Table = Base.metadata.tables[OcrTrackerHeartbeat.__tablename__]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def ocr_image_id(request_id: str, original_image_id: str, page_index: int) -> str:
    """Deterministic ``ocr_images.id`` so re-storing on retry is idempotent.

    Keyed on the submission/request id plus the source image and page, so the same
    extracted image always maps to the same row (insert_or_ignore on the PK).
    """
    import uuid

    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"ocr-image:{request_id}:{original_image_id}:{page_index}")
    )


# ---------------------------------------------------------------------------
# OCR result functions
# ---------------------------------------------------------------------------


def save_ocr_result(
    engine: Engine,
    *,
    document_id: str,
    file_path: str,
    text: str,
    page_count: int = 0,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    batch_id: str,
    workflow_id: str,
    file_hash: str | None = None,
) -> bool:
    """Insert a row into the ocr_results table (idempotent on document_id)."""
    return insert_or_ignore(
        engine,
        _OCR_RESULTS_TABLE,
        {
            "document_id": document_id,
            "file_path": file_path,
            "text": text,
            "page_count": page_count,
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "batch_id": batch_id,
            "workflow_id": workflow_id,
            "file_hash": file_hash,
        },
        index_elements=["document_id"],
    )


def delete_ocr_results(engine: Engine, document_ids: list[str]) -> None:
    """Delete OCR results by document IDs (chunk cleanup after reassembly)."""
    t = _OCR_RESULTS_TABLE
    with engine.begin() as conn:
        conn.execute(sa.delete(t).where(t.c.document_id.in_(document_ids)))


def get_ocr_result(engine: Engine, document_id: str) -> dict[str, Any] | None:
    """Look up an OCR result by document ID."""
    t = _OCR_RESULTS_TABLE
    stmt = t.select().where(t.c.document_id == document_id)
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return dict(row)


def find_ocr_result_by_file_path(engine: Engine, file_path: str) -> dict[str, Any] | None:
    """Find an OCR result by file_path that is not marked for removal."""
    t = _OCR_RESULTS_TABLE
    stmt = (
        t.select()
        .where(t.c.file_path == file_path)
        .where(t.c.marked_for_removal == sa.false())
        .limit(1)
    )
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return dict(row)


def find_ocr_result_by_hash(engine: Engine, file_hash: str) -> dict[str, Any] | None:
    """Find an OCR result by SHA-256 file hash that is not marked for removal."""
    t = _OCR_RESULTS_TABLE
    stmt = (
        t.select()
        .where(t.c.file_hash == file_hash)
        .where(t.c.marked_for_removal == sa.false())
        .limit(1)
    )
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return dict(row)


def mark_ocr_for_removal(engine: Engine, document_id: str) -> bool:
    """Set marked_for_removal=True on an OCR result. Returns True if found."""
    t = _OCR_RESULTS_TABLE
    with engine.begin() as conn:
        result = conn.execute(
            sa.update(t).where(t.c.document_id == document_id).values(marked_for_removal=True)
        )
        return result.rowcount > 0


def clear_ocr_removal_mark(engine: Engine, document_id: str) -> bool:
    """Set marked_for_removal=False on an OCR result. Returns True if found."""
    t = _OCR_RESULTS_TABLE
    with engine.begin() as conn:
        result = conn.execute(
            sa.update(t).where(t.c.document_id == document_id).values(marked_for_removal=False)
        )
        return result.rowcount > 0


def get_ocr_results_missing_hash(engine: Engine) -> list[dict[str, Any]]:
    """Return OCR results that have a file_path but no file_hash."""
    t = _OCR_RESULTS_TABLE
    stmt = t.select().where(t.c.file_hash.is_(None)).where(t.c.file_path.isnot(None))
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(stmt).mappings()]


def update_ocr_file_hash(engine: Engine, document_id: str, file_hash: str) -> bool:
    """Set file_hash on an OCR result. Returns True if the row was updated."""
    t = _OCR_RESULTS_TABLE
    with engine.begin() as conn:
        result = conn.execute(
            sa.update(t).where(t.c.document_id == document_id).values(file_hash=file_hash)
        )
        return result.rowcount > 0


# ---------------------------------------------------------------------------
# File content blob functions
# ---------------------------------------------------------------------------


def save_file_content(
    engine: Engine,
    *,
    content_id: str,
    data: bytes,
    mime_type: str,
    file_size_bytes: int,
    blobs: S3Blobs,
) -> None:
    """Upload file bytes to S3 and record the reference in file_content_blobs."""
    s3_key = blobs.build_key(content_id)
    blobs.put(s3_key, data, mime_type)
    insert_or_ignore(
        engine,
        _FILE_CONTENT_BLOBS_TABLE,
        {
            "id": content_id,
            "s3_key": s3_key,
            "mime_type": mime_type,
            "file_size_bytes": file_size_bytes,
        },
        index_elements=["id"],
    )


def get_file_content(
    engine: Engine,
    content_id: str,
    blobs: S3Blobs,
) -> dict[str, Any] | None:
    """Look up file content by ID, fetching the bytes from S3 under ``data``."""
    t = _FILE_CONTENT_BLOBS_TABLE
    stmt = t.select().where(t.c.id == content_id)
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        result = dict(row)
    result["data"] = blobs.get(result["s3_key"])
    return result


def delete_file_content(engine: Engine, content_id: str, blobs: S3Blobs) -> None:
    """Delete file content by ID, removing both the DB row and the S3 object."""
    t = _FILE_CONTENT_BLOBS_TABLE
    with engine.begin() as conn:
        s3_key = conn.execute(sa.select(t.c.s3_key).where(t.c.id == content_id)).scalar()
        conn.execute(sa.delete(t).where(t.c.id == content_id))
    if s3_key is not None:
        blobs.delete(s3_key)


# ---------------------------------------------------------------------------
# OCR image functions
# ---------------------------------------------------------------------------


def save_ocr_image(
    engine: Engine,
    *,
    image_id: str,
    document_id: str = "",
    page_index: int,
    original_image_id: str,
    data: bytes,
    mime_type: str,
    file_size_bytes: int,
    top_left_x: int | None = None,
    top_left_y: int | None = None,
    bottom_right_x: int | None = None,
    bottom_right_y: int | None = None,
    blobs: S3Blobs,
) -> None:
    """Upload image bytes to S3 and record the reference in ocr_images."""
    s3_key = blobs.build_key(image_id)
    blobs.put(s3_key, data, mime_type)
    insert_or_ignore(
        engine,
        _OCR_IMAGES_TABLE,
        {
            "id": image_id,
            "document_id": document_id,
            "page_index": page_index,
            "original_image_id": original_image_id,
            "s3_key": s3_key,
            "mime_type": mime_type,
            "file_size_bytes": file_size_bytes,
            "top_left_x": top_left_x,
            "top_left_y": top_left_y,
            "bottom_right_x": bottom_right_x,
            "bottom_right_y": bottom_right_y,
        },
        index_elements=["id"],
    )


def update_ocr_images_document_id(
    engine: Engine,
    image_ids: list[str],
    document_id: str,
) -> None:
    """Set document_id on ocr_images rows by image UUIDs."""
    if not image_ids:
        return
    t = _OCR_IMAGES_TABLE
    with engine.begin() as conn:
        conn.execute(sa.update(t).where(t.c.id.in_(image_ids)).values(document_id=document_id))


def reassign_ocr_images_document_id(
    engine: Engine,
    old_document_ids: list[str],
    new_document_id: str,
) -> None:
    """Bulk reassign images from old document_ids to new_document_id (chunk reassembly)."""
    if not old_document_ids:
        return
    t = _OCR_IMAGES_TABLE
    with engine.begin() as conn:
        conn.execute(
            sa.update(t)
            .where(t.c.document_id.in_(old_document_ids))
            .values(document_id=new_document_id)
        )


def get_ocr_images(engine: Engine, document_id: str) -> list[dict[str, Any]]:
    """List images for a document (metadata only, no blob data)."""
    t = _OCR_IMAGES_TABLE
    cols = [
        t.c.id,
        t.c.document_id,
        t.c.page_index,
        t.c.original_image_id,
        t.c.mime_type,
        t.c.file_size_bytes,
        t.c.top_left_x,
        t.c.top_left_y,
        t.c.bottom_right_x,
        t.c.bottom_right_y,
        t.c.created_at,
    ]
    stmt = sa.select(*cols).where(t.c.document_id == document_id).order_by(t.c.page_index)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


def get_ocr_image(
    engine: Engine,
    image_id: str,
    blobs: S3Blobs,
) -> dict[str, Any] | None:
    """Get a single image, fetching its bytes from S3 under the ``data`` key."""
    t = _OCR_IMAGES_TABLE
    stmt = t.select().where(t.c.id == image_id)
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        result = dict(row)
    result["data"] = blobs.get(result["s3_key"])
    return result


def delete_ocr_images_by_document(
    engine: Engine,
    document_ids: list[str],
    blobs: S3Blobs,
) -> None:
    """Delete OCR images by document IDs, removing both rows and S3 objects."""
    if not document_ids:
        return
    t = _OCR_IMAGES_TABLE
    with engine.begin() as conn:
        s3_keys = list(
            conn.execute(sa.select(t.c.s3_key).where(t.c.document_id.in_(document_ids))).scalars()
        )
        conn.execute(sa.delete(t).where(t.c.document_id.in_(document_ids)))
    for s3_key in s3_keys:
        blobs.delete(s3_key)


# ---------------------------------------------------------------------------
# OCR job status projection (single-writer: the OCR workflow). Wired in the
# cross-queue increment; the status view JOINs this to the platform batch_jobs
# read model on request_id.
# ---------------------------------------------------------------------------


def upsert_ocr_job_status(
    engine: Engine,
    *,
    request_id: str,
    document_id: str,
    file_path: str = "",
    status: OcrProcessingStatus | str,
    error_message: str | None = None,
) -> None:
    """Insert or update the coarse processing status for a request.

    Accepts either an ``OcrProcessingStatus`` member or its string value. Unknown
    strings raise ``ValueError`` here — this is the single write-side validation
    point (mirroring the platform's ``update_batch_status``) that keeps garbage
    statuses out of the DB. Reads deliberately stay tolerant: ``_derive_status``
    coerces any legacy/unknown stored value to ``UNKNOWN`` rather than crashing.
    """
    validated = OcrProcessingStatus(status)
    t = _OCR_JOB_STATUS_TABLE
    now = datetime.now(UTC)
    with engine.begin() as conn:
        existing = conn.execute(
            sa.select(t.c.request_id).where(t.c.request_id == request_id)
        ).scalar()
        if existing is None:
            conn.execute(
                sa.insert(t).values(
                    request_id=request_id,
                    document_id=document_id,
                    file_path=file_path,
                    status=validated,
                    error_message=error_message,
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            conn.execute(
                sa.update(t)
                .where(t.c.request_id == request_id)
                .values(status=validated, error_message=error_message, updated_at=now)
            )


def get_ocr_job_status(engine: Engine, request_id: str) -> dict[str, Any] | None:
    """Return the OCR processing-status row for a request, or None."""
    t = _OCR_JOB_STATUS_TABLE
    with engine.connect() as conn:
        row = conn.execute(t.select().where(t.c.request_id == request_id)).mappings().first()
        return dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# Tracker heartbeat (single-row liveness marker for the T4.4 status tracker)
# ---------------------------------------------------------------------------


def record_tracker_heartbeat(
    engine: Engine,
    *,
    now: datetime,
    live_jobs: int,
    hints_sent: int,
) -> None:
    """Upsert the singleton tracker-heartbeat row (``id == 1``).

    ``last_run_at``/``live_jobs``/``hints_sent`` are set to this cycle's values;
    ``cycles_total`` is incremented by 1. The increment is expressed as
    ``cycles_total + 1`` inside the dialect's ``ON CONFLICT DO UPDATE`` so it is
    computed atomically against the stored value — a concurrent or duplicated run
    can never lose a count by read-modify-write. Mirrors the dialect-branch shape
    of ``sax_platform.db.insert_or_ignore`` so it works on both the SQLite test
    store and Postgres.
    """
    t = _OCR_TRACKER_HEARTBEAT_TABLE
    values = {
        "id": TRACKER_HEARTBEAT_ID,
        "last_run_at": now,
        "live_jobs": live_jobs,
        "hints_sent": hints_sent,
        "cycles_total": 1,
    }
    updated = {
        "last_run_at": now,
        "live_jobs": live_jobs,
        "hints_sent": hints_sent,
        "cycles_total": t.c.cycles_total + 1,
    }
    dialect = engine.dialect.name
    stmt: SQLiteInsert | PostgresInsert
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        stmt = (
            sqlite_insert(t)
            .values(**values)
            .on_conflict_do_update(index_elements=["id"], set_=updated)
        )
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as postgres_insert

        stmt = (
            postgres_insert(t)
            .values(**values)
            .on_conflict_do_update(index_elements=["id"], set_=updated)
        )
    else:  # pragma: no cover - only sqlite/postgres are supported stores
        msg = f"record_tracker_heartbeat is unsupported on dialect {dialect!r}"
        raise RuntimeError(msg)

    with engine.begin() as conn:
        conn.execute(stmt)


def get_tracker_heartbeat(engine: Engine) -> dict[str, Any] | None:
    """Return the singleton tracker-heartbeat row, or None if no sweep has run."""
    t = _OCR_TRACKER_HEARTBEAT_TABLE
    with engine.connect() as conn:
        row = conn.execute(t.select().where(t.c.id == TRACKER_HEARTBEAT_ID)).mappings().first()
        return dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# Migrations (OCR's own Alembic chain, isolated by version_table)
# ---------------------------------------------------------------------------

#: OCR's Alembic chain: the version table that isolates it from the other chains
#: sharing this database, and the directory holding its ``env.py``/``versions/``.
#: The apply path (``ocr migrate``, tests) and the verify path (worker startup)
#: both key off these, so they can never read different chains.
VERSION_TABLE = "alembic_version_ocr"


def _script_location() -> str:
    from pathlib import Path

    return str(Path(__file__).parent / "alembic")


def run_migrations(url: str) -> None:
    """Run the OCR Alembic chain against the shared store URL (SQLite or Postgres).

    The OCR chain uses ``version_table=alembic_version_ocr`` and an
    ``include_object`` filter, so it coexists with the platform's chain in the
    same database without either dropping the other's tables. Delegates to the
    shared runner (``sax_platform.db.run_migrations``), which adds a per-chain
    Postgres advisory lock this chain previously ran without.

    This is the dev/test self-service apply path (``ocr migrate``, the test
    suites) only. The worker does NOT call it: since the 2026-08-02
    schema-change agreement, startup verifies (:func:`verify_schema`) and never
    applies DDL.
    """
    from sax_platform.db import run_migrations as _run_migrations

    _run_migrations(url, version_table=VERSION_TABLE, script_location=_script_location())


def verify_schema(url: str) -> str:
    """Verify the deployed OCR schema and return the stamped revision.

    Raises :class:`~sax_platform.db.SchemaVersionError` when the database is
    behind this code's chain head, uninitialized, or ambiguously stamped; a
    database *ahead* of this code is allowed (the expand/contract window) and
    logs a warning. See :mod:`sax_platform.db.verify` for the full contract.
    """
    from sax_platform.db import verify_schema_version

    return verify_schema_version(
        url,
        version_table=VERSION_TABLE,
        script_location=_script_location(),
        migrate_command="ocr migrate",
    )
