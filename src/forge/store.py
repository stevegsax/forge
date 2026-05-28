"""Observability store for Forge.

Persists full LLM interaction data and run results to a local SQLite database.

Design follows Function Core / Imperative Shell:
- Pure functions: build_interaction_dict, build_playbook_dict
- Imperative shell: get_store_url, get_store_engine, run_migrations,
  insert_or_ignore, save_interaction, save_run, get_interactions, get_run,
  list_recent_runs
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from forge.models import BatchJobStatus

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from forge.models import (
        AssembledContext,
        ConflictResolutionCallResult,
        ExtractionCallResult,
        LLMCallResult,
        PlanCallResult,
        PlaybookEntry,
        SanityCheckCallResult,
        TaskResult,
    )

    # Union of all LLM result types that share model_name, input_tokens,
    # output_tokens, latency_ms, and optional cache token fields.
    _AnyLLMResult = (
        LLMCallResult
        | PlanCallResult
        | SanityCheckCallResult
        | ConflictResolutionCallResult
        | ExtractionCallResult
    )

logger = logging.getLogger(__name__)


class StoreConfigError(RuntimeError):
    """The observability store is misconfigured (e.g. ``FORGE_DB_URL`` unset)."""


# ---------------------------------------------------------------------------
# Custom SQLAlchemy types
# ---------------------------------------------------------------------------


class UTCDateTime(sa.types.TypeDecorator):
    """Always-UTC DateTime column type.

    SQLite cannot preserve tzinfo on DateTime columns, so we normalize
    to UTC on the way in (stripping tz for storage) and re-attach UTC
    on the way out. Naive inputs are assumed to already be UTC, per
    project convention. Callers always receive tz-aware UTC datetimes.
    """

    impl = sa.DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


# ---------------------------------------------------------------------------
# SQLAlchemy models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    idempotency_key: Mapped[str | None] = mapped_column(sa.String, nullable=True, unique=True)
    task_id: Mapped[str] = mapped_column(sa.String, nullable=False, index=True)
    step_id: Mapped[str | None] = mapped_column(sa.String, nullable=True)
    sub_task_id: Mapped[str | None] = mapped_column(sa.String, nullable=True)
    role: Mapped[str] = mapped_column(sa.String, nullable=False)
    system_prompt: Mapped[str] = mapped_column(sa.Text, nullable=False)
    user_prompt: Mapped[str] = mapped_column(sa.Text, nullable=False)
    model_name: Mapped[str] = mapped_column(sa.String, nullable=False)
    input_tokens: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(sa.Float, nullable=False)
    explanation: Mapped[str] = mapped_column(sa.Text, default="")
    context_stats_json: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    cache_creation_input_tokens: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    cache_read_input_tokens: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(sa.String, nullable=False, index=True)
    workflow_id: Mapped[str] = mapped_column(sa.String, nullable=False, unique=True)
    status: Mapped[str] = mapped_column(sa.String, nullable=False)
    result_json: Mapped[str] = mapped_column(sa.Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
    )


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    batch_id: Mapped[str | None] = mapped_column(sa.String, nullable=True, index=True)
    workflow_id: Mapped[str] = mapped_column(sa.String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(sa.String, nullable=False)
    provider: Mapped[str] = mapped_column(sa.String, nullable=False, server_default="anthropic")
    file_path: Mapped[str | None] = mapped_column(sa.String, nullable=True)
    document_id: Mapped[str | None] = mapped_column(sa.String, nullable=True, index=True)
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
    )


class Playbook(Base):
    __tablename__ = "playbooks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    idempotency_key: Mapped[str | None] = mapped_column(sa.String, nullable=True, unique=True)
    title: Mapped[str] = mapped_column(sa.String, nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    tags_json: Mapped[str] = mapped_column(sa.Text, nullable=False)
    source_task_id: Mapped[str] = mapped_column(sa.String, nullable=False, index=True)
    source_workflow_id: Mapped[str] = mapped_column(sa.String, nullable=False)
    extraction_workflow_id: Mapped[str] = mapped_column(sa.String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
    )


class OcrResult(Base):
    __tablename__ = "ocr_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(sa.String, nullable=False, unique=True, index=True)
    file_path: Mapped[str] = mapped_column(sa.String, nullable=False)
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    page_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
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
    )


class FileContentBlob(Base):
    __tablename__ = "file_content_blobs"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    s3_key: Mapped[str] = mapped_column(sa.String, nullable=False)
    mime_type: Mapped[str] = mapped_column(sa.String, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(UTC),
    )


class OcrImage(Base):
    __tablename__ = "ocr_images"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    document_id: Mapped[str] = mapped_column(sa.String, nullable=False, default="", index=True)
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
    )


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_interaction_dict(
    *,
    task_id: str,
    step_id: str | None,
    sub_task_id: str | None,
    role: str,
    context: AssembledContext,
    llm_result: _AnyLLMResult,
) -> dict:
    """Assemble a dict from activity data for insertion.

    Works with any LLM result type. Extracts explanation via duck typing:
    response.explanation for LLMCallResult/SanityCheckCallResult,
    plan.explanation for PlanCallResult.
    """
    explanation = ""
    if hasattr(llm_result, "response"):
        explanation = llm_result.response.explanation
    elif hasattr(llm_result, "plan"):
        explanation = llm_result.plan.explanation

    context_stats_json = None
    if context.context_stats is not None:
        context_stats_json = context.context_stats.model_dump_json()

    cache_creation = getattr(llm_result, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(llm_result, "cache_read_input_tokens", 0) or 0

    return {
        "task_id": task_id,
        "step_id": step_id,
        "sub_task_id": sub_task_id,
        "role": role,
        "system_prompt": context.system_prompt,
        "user_prompt": context.user_prompt,
        "model_name": llm_result.model_name,
        "input_tokens": llm_result.input_tokens,
        "output_tokens": llm_result.output_tokens,
        "latency_ms": llm_result.latency_ms,
        "explanation": explanation,
        "context_stats_json": context_stats_json,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
    }


def playbook_idempotency_key(extraction_workflow_id: str, title: str) -> str:
    """Deterministic per-playbook idempotency key, stable across retries."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"playbook:{extraction_workflow_id}:{title}"))


def ocr_image_id(request_id: str, original_image_id: str, page_index: int) -> str:
    """Deterministic ``ocr_images.id`` so re-storing on retry is idempotent.

    Keyed on the submission/request id plus the source image and page, so the same
    extracted image always maps to the same row (insert_or_ignore on the PK).
    """
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"ocr-image:{request_id}:{original_image_id}:{page_index}")
    )


def build_playbook_dict(
    entry: PlaybookEntry,
    extraction_workflow_id: str,
) -> dict:
    """Convert a PlaybookEntry to an insertable dict (with a deterministic key)."""
    return {
        "idempotency_key": playbook_idempotency_key(extraction_workflow_id, entry.title),
        "title": entry.title,
        "content": entry.content,
        "tags_json": json.dumps(entry.tags),
        "source_task_id": entry.source_task_id,
        "source_workflow_id": entry.source_workflow_id,
        "extraction_workflow_id": extraction_workflow_id,
    }


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def get_store_url() -> str:
    """Return the configured store URL from ``FORGE_DB_URL``.

    The store is mandatory infrastructure with no implicit default and no
    runtime failover. A ``sqlite:///<path>`` URL is the dev/test configuration;
    a ``postgresql+psycopg2://...`` URL is production. Unset or empty raises.
    """
    url = os.environ.get("FORGE_DB_URL")
    if not url:
        raise StoreConfigError(
            "FORGE_DB_URL is not set. Set it to a 'sqlite:///<path>' URL for "
            "development and tests, or a 'postgresql+psycopg2://...' URL for "
            "production."
        )
    return url


def _ensure_sqlite_parent(url: str) -> None:
    """Create the parent directory for a file-based SQLite URL."""
    database = make_url(url).database
    if database and database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)


def get_store_engine() -> Engine:
    """Build the store engine from ``FORGE_DB_URL``.

    SQLite URLs get WAL journaling (and the parent directory is created);
    Postgres URLs get connection pre-ping and a small bounded pool to respect
    the managed-database connection caps. Connection errors are not caught here
    — they propagate (no runtime failover).
    """
    url = get_store_url()
    if make_url(url).get_backend_name() == "sqlite":
        _ensure_sqlite_parent(url)
        engine = sa.create_engine(url)

        @sa.event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: object, _connection_record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[union-attr]
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        return engine

    return sa.create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def run_migrations(url: str) -> None:
    """Run Alembic migrations against the store URL (SQLite or Postgres)."""
    from alembic import command
    from alembic.config import Config

    alembic_dir = Path(__file__).parent / "alembic"
    ini_path = alembic_dir / "alembic.ini"

    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(alembic_dir))

    if make_url(url).get_backend_name() == "sqlite":
        _ensure_sqlite_parent(url)
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


def insert_or_ignore(
    engine: Engine,
    table: sa.Table,
    values: dict,
    *,
    index_elements: list[str],
) -> bool:
    """Idempotent insert via the dialect's ``ON CONFLICT DO NOTHING``.

    Returns ``True`` if a new row was written, ``False`` if an existing row on
    ``index_elements`` absorbed the insert (a no-op). This makes every write safe
    to re-apply on a Temporal retry: a duplicate never raises, and the caller can
    tell whether it was the first writer.
    """
    dialect = engine.dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as _dialect_insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as _dialect_insert
    else:  # pragma: no cover - only sqlite/postgres are supported stores
        msg = f"insert_or_ignore is unsupported on dialect {dialect!r}"
        raise StoreConfigError(msg)

    stmt = (
        _dialect_insert(table).values(**values).on_conflict_do_nothing(index_elements=index_elements)
    )
    with engine.begin() as conn:
        result = conn.execute(stmt)
    return bool(result.rowcount)


def save_interaction(engine: Engine, **kwargs: object) -> bool:
    """Insert a row into the interactions table (idempotent on idempotency_key)."""
    return insert_or_ignore(
        engine, Interaction.__table__, dict(kwargs), index_elements=["idempotency_key"]
    )


def save_run(engine: Engine, task_result: TaskResult, workflow_id: str) -> bool:
    """Insert a row into the runs table (idempotent on workflow_id)."""
    return insert_or_ignore(
        engine,
        Run.__table__,
        {
            "task_id": task_result.task_id,
            "workflow_id": workflow_id,
            "status": task_result.status.value,
            "result_json": task_result.model_dump_json(),
        },
        index_elements=["workflow_id"],
    )


def get_interactions(
    engine: Engine,
    task_id: str,
    step_id: str | None = None,
) -> list[dict]:
    """Query interactions for a task, optionally filtered by step."""
    t = Interaction.__table__
    stmt = t.select().where(t.c.task_id == task_id).order_by(t.c.created_at)
    if step_id is not None:
        stmt = stmt.where(t.c.step_id == step_id)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


def get_run(engine: Engine, workflow_id: str) -> dict | None:
    """Query a single run by workflow ID."""
    t = Run.__table__
    stmt = t.select().where(t.c.workflow_id == workflow_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        result = dict(row)
        result["result"] = json.loads(result["result_json"])
        return result


def list_recent_runs(engine: Engine, limit: int = 20) -> list[dict]:
    """Query recent runs ordered by creation time descending."""
    t = Run.__table__
    stmt = t.select().order_by(t.c.created_at.desc()).limit(limit)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Playbook functions (Phase 6)
# ---------------------------------------------------------------------------


def save_playbooks(engine: Engine, entries: list[dict]) -> bool:
    """Insert playbook rows (idempotent per-entry on idempotency_key).

    Returns ``True`` if at least one new row was written.
    """
    applied = False
    for entry in entries:
        if insert_or_ignore(
            engine, Playbook.__table__, dict(entry), index_elements=["idempotency_key"]
        ):
            applied = True
    return applied


def get_playbooks_by_tags(
    engine: Engine,
    tags: list[str],
    limit: int = 10,
) -> list[dict]:
    """Query playbooks matching any of the given tags, ordered by recency.

    Uses SQLite json_each() to unnest the tags_json array and match
    against the input tags.
    """
    if not tags:
        return []

    tag_placeholders = ", ".join(f":tag_{i}" for i in range(len(tags)))
    tag_params = {f"tag_{i}": tag for i, tag in enumerate(tags)}

    query = sa.text(f"""
        SELECT DISTINCT p.*
        FROM playbooks p, json_each(p.tags_json) AS t
        WHERE t.value IN ({tag_placeholders})
        ORDER BY p.created_at DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {**tag_params, "limit": limit}).mappings().all()
        return [dict(row) for row in rows]


def list_recent_playbooks(engine: Engine, limit: int = 20) -> list[dict]:
    """Query recent playbooks ordered by creation time descending."""
    t = Playbook.__table__
    stmt = t.select().order_by(t.c.created_at.desc()).limit(limit)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


def get_playbook_ids(
    engine: Engine,
    tags: list[str] | None = None,
    source_task_id: str = "",
    limit: int = 0,
) -> list[int]:
    """Return matching playbook IDs ordered by recency.

    Filters are AND-combined when both are provided.
    Tags use OR matching (any tag matches).
    """
    t = Playbook.__table__

    if tags:
        tag_placeholders = ", ".join(f":tag_{i}" for i in range(len(tags)))
        tag_params: dict[str, object] = {f"tag_{i}": tag for i, tag in enumerate(tags)}

        base_sql = f"""
            SELECT DISTINCT p.id
            FROM playbooks p, json_each(p.tags_json) AS t
            WHERE t.value IN ({tag_placeholders})
        """
        if source_task_id:
            base_sql += " AND p.source_task_id = :source_task_id"
            tag_params["source_task_id"] = source_task_id
        base_sql += " ORDER BY p.created_at DESC"
        if limit > 0:
            base_sql += " LIMIT :limit"
            tag_params["limit"] = limit

        query = sa.text(base_sql)
        with engine.connect() as conn:
            rows = conn.execute(query, tag_params).all()
            return [row[0] for row in rows]
    else:
        stmt = sa.select(t.c.id)
        if source_task_id:
            stmt = stmt.where(t.c.source_task_id == source_task_id)
        stmt = stmt.order_by(t.c.created_at.desc())
        if limit > 0:
            stmt = stmt.limit(limit)

        with engine.connect() as conn:
            rows = conn.execute(stmt).all()
            return [row[0] for row in rows]


def get_playbook_by_id(engine: Engine, playbook_id: int) -> dict | None:
    """Fetch a single playbook row by primary key."""
    t = Playbook.__table__
    stmt = t.select().where(t.c.id == playbook_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return dict(row)


def get_unextracted_runs(engine: Engine, limit: int = 50) -> list[dict]:
    """Query runs that have no corresponding playbook entries.

    Returns runs whose workflow_id is not in the playbooks table's
    source_workflow_id column.
    """
    runs_t = Run.__table__
    playbooks_t = Playbook.__table__

    extracted_ids = sa.select(playbooks_t.c.source_workflow_id).distinct()
    stmt = (
        runs_t.select()
        .where(runs_t.c.workflow_id.notin_(extracted_ids))
        .order_by(runs_t.c.created_at.desc())
        .limit(limit)
    )

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Batch job functions (Phase 14)
# ---------------------------------------------------------------------------


def record_batch_submission(
    engine: Engine,
    *,
    request_id: str,
    batch_id: str,
    workflow_id: str,
    provider: str = "anthropic",
    file_path: str | None = None,
    document_id: str | None = None,
) -> bool:
    """Insert a new batch job record with status 'submitted'.

    ``document_id`` groups chunks of a single submission together so
    that the list_ocr_jobs query can distinguish resubmissions of the
    same file. Defaults to ``request_id`` when not supplied.

    Idempotent on the ``request_id`` primary key.
    """
    return insert_or_ignore(
        engine,
        BatchJob.__table__,
        {
            "id": request_id,
            "batch_id": batch_id,
            "workflow_id": workflow_id,
            "status": BatchJobStatus.SUBMITTED,
            "provider": provider,
            "file_path": file_path,
            "document_id": document_id or request_id,
        },
        index_elements=["id"],
    )


def record_batch_failure(
    engine: Engine,
    *,
    request_id: str,
    workflow_id: str,
    error_message: str,
    provider: str = "anthropic",
    file_path: str | None = None,
    document_id: str | None = None,
) -> bool:
    """Insert a batch job record with status 'failed' and no batch_id.

    Used when the provider API call fails before returning a batch_id.
    ``document_id`` groups chunks of a single submission together; see
    ``record_batch_submission``. Idempotent on the ``request_id`` primary key.
    """
    return insert_or_ignore(
        engine,
        BatchJob.__table__,
        {
            "id": request_id,
            "batch_id": None,
            "workflow_id": workflow_id,
            "status": BatchJobStatus.FAILED,
            "provider": provider,
            "error_message": error_message,
            "file_path": file_path,
            "document_id": document_id or request_id,
        },
        index_elements=["id"],
    )


def update_batch_status(
    engine: Engine,
    *,
    request_id: str,
    status: BatchJobStatus | str,
    error_message: str | None = None,
) -> None:
    """Update batch job status and timestamp.

    Accepts either a ``BatchJobStatus`` member or its string value. Unknown
    strings raise ``ValueError`` — this is the system-boundary validation
    point that prevents garbage statuses from reaching the DB.
    """
    validated = BatchJobStatus(status)
    t = BatchJob.__table__
    with engine.begin() as conn:
        conn.execute(
            sa.update(t)
            .where(t.c.id == request_id)
            .values(
                status=validated,
                error_message=error_message,
                updated_at=datetime.now(UTC),
            )
        )


def get_pending_batch_jobs(engine: Engine) -> list[dict]:
    """Query batch jobs with status 'submitted', ordered by created_at."""
    t = BatchJob.__table__
    stmt = t.select().where(t.c.status == BatchJobStatus.SUBMITTED).order_by(t.c.created_at)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


def get_batch_job(engine: Engine, request_id: str) -> dict | None:
    """Look up a single batch job by request ID."""
    t = BatchJob.__table__
    stmt = t.select().where(t.c.id == request_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return dict(row)


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
        OcrResult.__table__,
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
    t = OcrResult.__table__
    with engine.begin() as conn:
        conn.execute(sa.delete(t).where(t.c.document_id.in_(document_ids)))


def get_ocr_result(engine: Engine, document_id: str) -> dict | None:
    """Look up an OCR result by document ID."""
    t = OcrResult.__table__
    stmt = t.select().where(t.c.document_id == document_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return dict(row)


def find_ocr_result_by_file_path(engine: Engine, file_path: str) -> dict | None:
    """Find an OCR result by file_path that is not marked for removal.

    Returns the first matching row as a dict, or None if no qualifying
    result exists.
    """
    t = OcrResult.__table__
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


def find_ocr_result_by_hash(engine: Engine, file_hash: str) -> dict | None:
    """Find an OCR result by SHA-256 file hash that is not marked for removal.

    Returns the first matching row as a dict, or None if no qualifying
    result exists.
    """
    t = OcrResult.__table__
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
    t = OcrResult.__table__
    with engine.begin() as conn:
        result = conn.execute(
            sa.update(t).where(t.c.document_id == document_id).values(marked_for_removal=True)
        )
        return result.rowcount > 0


def clear_ocr_removal_mark(engine: Engine, document_id: str) -> bool:
    """Set marked_for_removal=False on an OCR result. Returns True if found."""
    t = OcrResult.__table__
    with engine.begin() as conn:
        result = conn.execute(
            sa.update(t).where(t.c.document_id == document_id).values(marked_for_removal=False)
        )
        return result.rowcount > 0


def get_ocr_results_missing_hash(engine: Engine) -> list[dict]:
    """Return OCR results that have a file_path but no file_hash."""
    t = OcrResult.__table__
    stmt = t.select().where(t.c.file_hash.is_(None)).where(t.c.file_path.isnot(None))
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(stmt).mappings()]


def update_ocr_file_hash(engine: Engine, document_id: str, file_hash: str) -> bool:
    """Set file_hash on an OCR result. Returns True if the row was updated."""
    t = OcrResult.__table__
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
) -> None:
    """Upload file bytes to S3 and record the reference in file_content_blobs.

    Uploads to S3 first (so a DB failure leaves only a harmless orphan object,
    never a row pointing at missing bytes). An unset bucket or S3 error raises.
    """
    from forge.ocr import s3_blobs

    s3_key = s3_blobs.build_key(content_id)
    s3_blobs.put(s3_key, data, mime_type)
    insert_or_ignore(
        engine,
        FileContentBlob.__table__,
        {
            "id": content_id,
            "s3_key": s3_key,
            "mime_type": mime_type,
            "file_size_bytes": file_size_bytes,
        },
        index_elements=["id"],
    )


def get_file_content(engine: Engine, content_id: str) -> dict | None:
    """Look up file content by ID, fetching the bytes from S3.

    Returns a dict with id, s3_key, mime_type, file_size_bytes, created_at and a
    ``data`` key holding the bytes fetched from S3 (preserving the historical
    byte-out shape for callers). An S3 fetch error raises.
    """
    from forge.ocr import s3_blobs

    t = FileContentBlob.__table__
    stmt = t.select().where(t.c.id == content_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        result = dict(row)
    result["data"] = s3_blobs.get(result["s3_key"])
    return result


def delete_file_content(engine: Engine, content_id: str) -> None:
    """Delete file content by ID, removing both the DB row and the S3 object."""
    from forge.ocr import s3_blobs

    t = FileContentBlob.__table__
    with engine.begin() as conn:
        s3_key = conn.execute(sa.select(t.c.s3_key).where(t.c.id == content_id)).scalar()
        conn.execute(sa.delete(t).where(t.c.id == content_id))
    if s3_key is not None:
        s3_blobs.delete(s3_key)


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
) -> None:
    """Upload image bytes to S3 and record the reference in ocr_images.

    Uploads to S3 first (a DB failure leaves only a harmless orphan object). An
    unset bucket or S3 error raises.
    """
    from forge.ocr import s3_blobs

    s3_key = s3_blobs.build_key(image_id)
    s3_blobs.put(s3_key, data, mime_type)
    insert_or_ignore(
        engine,
        OcrImage.__table__,
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
    t = OcrImage.__table__
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
    t = OcrImage.__table__
    with engine.begin() as conn:
        conn.execute(
            sa.update(t)
            .where(t.c.document_id.in_(old_document_ids))
            .values(document_id=new_document_id)
        )


def get_ocr_images(engine: Engine, document_id: str) -> list[dict]:
    """List images for a document (metadata only, no blob data)."""
    t = OcrImage.__table__
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


def get_ocr_image(engine: Engine, image_id: str) -> dict | None:
    """Get a single image, fetching its bytes from S3 under the ``data`` key."""
    from forge.ocr import s3_blobs

    t = OcrImage.__table__
    stmt = t.select().where(t.c.id == image_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        result = dict(row)
    result["data"] = s3_blobs.get(result["s3_key"])
    return result


def delete_ocr_images_by_document(engine: Engine, document_ids: list[str]) -> None:
    """Delete OCR images by document IDs, removing both rows and S3 objects."""
    if not document_ids:
        return
    from forge.ocr import s3_blobs

    t = OcrImage.__table__
    with engine.begin() as conn:
        s3_keys = list(
            conn.execute(sa.select(t.c.s3_key).where(t.c.document_id.in_(document_ids))).scalars()
        )
        conn.execute(sa.delete(t).where(t.c.document_id.in_(document_ids)))
    for s3_key in s3_keys:
        s3_blobs.delete(s3_key)
