"""Observability store for Forge.

Persists full LLM interaction data and run results to a local SQLite database.

Design follows Function Core / Imperative Shell:
- Pure functions: get_db_path, build_interaction_dict
- Imperative shell: get_engine, run_migrations, save_interaction, save_run,
  persist_interaction, get_interactions, get_run, list_recent_runs
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from forge.models import (
        AssembledContext,
        ConflictResolutionCallResult,
        ContextStats,
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


# ---------------------------------------------------------------------------
# SQLAlchemy models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
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
        sa.DateTime,
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
        sa.DateTime,
        default=lambda: datetime.now(UTC),
    )


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    batch_id: Mapped[str | None] = mapped_column(sa.String, nullable=True, index=True)
    workflow_id: Mapped[str] = mapped_column(sa.String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(sa.String, nullable=False)
    provider: Mapped[str] = mapped_column(sa.String, nullable=False, server_default="anthropic")
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime,
        default=lambda: datetime.now(UTC),
    )


class Playbook(Base):
    __tablename__ = "playbooks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(sa.String, nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    tags_json: Mapped[str] = mapped_column(sa.Text, nullable=False)
    source_task_id: Mapped[str] = mapped_column(sa.String, nullable=False, index=True)
    source_workflow_id: Mapped[str] = mapped_column(sa.String, nullable=False)
    extraction_workflow_id: Mapped[str] = mapped_column(sa.String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime,
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
    marked_for_removal: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.text("0"),
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime,
        default=lambda: datetime.now(UTC),
    )


class FileContentBlob(Base):
    __tablename__ = "file_content_blobs"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    data: Mapped[bytes] = mapped_column(sa.LargeBinary, nullable=False)
    mime_type: Mapped[str] = mapped_column(sa.String, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime,
        default=lambda: datetime.now(UTC),
    )


class OcrImage(Base):
    __tablename__ = "ocr_images"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    document_id: Mapped[str] = mapped_column(sa.String, nullable=False, default="", index=True)
    page_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    original_image_id: Mapped[str] = mapped_column(sa.String, nullable=False)
    data: Mapped[bytes] = mapped_column(sa.LargeBinary, nullable=False)
    mime_type: Mapped[str] = mapped_column(sa.String, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    top_left_x: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    top_left_y: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    bottom_right_x: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    bottom_right_y: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime,
        default=lambda: datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def get_db_path() -> Path | None:
    """Resolve the database path.

    Resolution order:
    1. ``FORGE_DB_PATH`` environment variable.
    2. ``$XDG_STATE_HOME/forge/forge.db``
    3. ``~/.local/state/forge/forge.db``

    Returns ``None`` if ``FORGE_DB_PATH`` is set to an empty string (disables store).
    """
    env_value = os.environ.get("FORGE_DB_PATH")
    if env_value is not None:
        if env_value == "":
            return None
        return Path(env_value)

    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / "forge" / "forge.db"

    return Path.home() / ".local" / "state" / "forge" / "forge.db"


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


def build_playbook_dict(
    entry: PlaybookEntry,
    extraction_workflow_id: str,
) -> dict:
    """Convert a PlaybookEntry to an insertable dict."""
    return {
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


def get_engine(db_path: Path) -> Engine:
    """Create a SQLAlchemy engine with WAL mode for the given database path."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sa.create_engine(f"sqlite:///{db_path}")

    @sa.event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: object, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[union-attr]
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return engine


def run_migrations(db_path: Path) -> None:
    """Run Alembic migrations programmatically."""
    from alembic import command
    from alembic.config import Config

    alembic_dir = Path(__file__).parent / "alembic"
    ini_path = alembic_dir / "alembic.ini"

    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(alembic_dir))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")


def save_interaction(engine: Engine, **kwargs: object) -> None:
    """Insert a row into the interactions table."""
    with engine.begin() as conn:
        conn.execute(sa.insert(Interaction.__table__).values(**kwargs))


def persist_interaction(
    *,
    task_id: str,
    role: str,
    system_prompt: str,
    user_prompt: str,
    llm_result: _AnyLLMResult,
    step_id: str | None = None,
    sub_task_id: str | None = None,
    context_stats: ContextStats | None = None,
) -> None:
    """Best-effort persist of an LLM interaction. Never raises (D42).

    Consolidates the get_db_path → get_engine → build_interaction_dict →
    save_interaction pattern used across all activity modules.
    """
    try:
        from forge.models import AssembledContext

        db_path = get_db_path()
        if db_path is None:
            return

        context = AssembledContext(
            task_id=task_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_stats=context_stats,
        )

        engine = get_engine(db_path)
        data = build_interaction_dict(
            task_id=task_id,
            step_id=step_id,
            sub_task_id=sub_task_id,
            role=role,
            context=context,
            llm_result=llm_result,
        )
        save_interaction(engine, **data)
    except Exception:
        logger.warning("Failed to persist %s interaction to store", role, exc_info=True)


def save_run(engine: Engine, task_result: TaskResult, workflow_id: str) -> None:
    """Insert a row into the runs table."""
    result_json = task_result.model_dump_json()
    with engine.begin() as conn:
        conn.execute(
            sa.insert(Run.__table__).values(
                task_id=task_result.task_id,
                workflow_id=workflow_id,
                status=task_result.status.value,
                result_json=result_json,
            )
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


def save_playbooks(engine: Engine, entries: list[dict]) -> None:
    """Bulk insert rows into the playbooks table."""
    if not entries:
        return
    with engine.begin() as conn:
        conn.execute(sa.insert(Playbook.__table__), entries)


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
) -> None:
    """Insert a new batch job record with status 'submitted'."""
    with engine.begin() as conn:
        conn.execute(
            sa.insert(BatchJob.__table__).values(
                id=request_id,
                batch_id=batch_id,
                workflow_id=workflow_id,
                status="submitted",
                provider=provider,
            )
        )


def record_batch_failure(
    engine: Engine,
    *,
    request_id: str,
    workflow_id: str,
    error_message: str,
    provider: str = "anthropic",
) -> None:
    """Insert a batch job record with status 'failed' and no batch_id.

    Used when the provider API call fails before returning a batch_id.
    """
    with engine.begin() as conn:
        conn.execute(
            sa.insert(BatchJob.__table__).values(
                id=request_id,
                batch_id=None,
                workflow_id=workflow_id,
                status="failed",
                provider=provider,
                error_message=error_message,
            )
        )


def update_batch_status(
    engine: Engine,
    *,
    request_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    """Update batch job status and timestamp."""
    t = BatchJob.__table__
    with engine.begin() as conn:
        conn.execute(
            sa.update(t)
            .where(t.c.id == request_id)
            .values(
                status=status,
                error_message=error_message,
                updated_at=datetime.now(UTC),
            )
        )


def get_pending_batch_jobs(engine: Engine) -> list[dict]:
    """Query batch jobs with status 'submitted', ordered by created_at."""
    t = BatchJob.__table__
    stmt = t.select().where(t.c.status == "submitted").order_by(t.c.created_at)

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
) -> None:
    """Insert a row into the ocr_results table."""
    with engine.begin() as conn:
        conn.execute(
            sa.insert(OcrResult.__table__).values(
                document_id=document_id,
                file_path=file_path,
                text=text,
                page_count=page_count,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch_id=batch_id,
                workflow_id=workflow_id,
            )
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


def mark_ocr_for_removal(engine: Engine, document_id: str) -> bool:
    """Set marked_for_removal=True on an OCR result. Returns True if found."""
    t = OcrResult.__table__
    with engine.begin() as conn:
        result = conn.execute(
            sa.update(t)
            .where(t.c.document_id == document_id)
            .values(marked_for_removal=True)
        )
        return result.rowcount > 0


def clear_ocr_removal_mark(engine: Engine, document_id: str) -> bool:
    """Set marked_for_removal=False on an OCR result. Returns True if found."""
    t = OcrResult.__table__
    with engine.begin() as conn:
        result = conn.execute(
            sa.update(t)
            .where(t.c.document_id == document_id)
            .values(marked_for_removal=False)
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
    """Insert a row into the file_content_blobs table."""
    with engine.begin() as conn:
        conn.execute(
            sa.insert(FileContentBlob.__table__).values(
                id=content_id,
                data=data,
                mime_type=mime_type,
                file_size_bytes=file_size_bytes,
            )
        )


def get_file_content(engine: Engine, content_id: str) -> dict | None:
    """Look up file content by ID. Returns dict with id, data, mime_type, file_size_bytes."""
    t = FileContentBlob.__table__
    stmt = t.select().where(t.c.id == content_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return dict(row)


def delete_file_content(engine: Engine, content_id: str) -> None:
    """Delete file content by ID."""
    t = FileContentBlob.__table__
    with engine.begin() as conn:
        conn.execute(sa.delete(t).where(t.c.id == content_id))


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
    """Insert a row into the ocr_images table."""
    with engine.begin() as conn:
        conn.execute(
            sa.insert(OcrImage.__table__).values(
                id=image_id,
                document_id=document_id,
                page_index=page_index,
                original_image_id=original_image_id,
                data=data,
                mime_type=mime_type,
                file_size_bytes=file_size_bytes,
                top_left_x=top_left_x,
                top_left_y=top_left_y,
                bottom_right_x=bottom_right_x,
                bottom_right_y=bottom_right_y,
            )
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
        conn.execute(
            sa.update(t)
            .where(t.c.id.in_(image_ids))
            .values(document_id=document_id)
        )


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
    """Get a single image with blob data."""
    t = OcrImage.__table__
    stmt = t.select().where(t.c.id == image_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return dict(row)


def delete_ocr_images_by_document(engine: Engine, document_ids: list[str]) -> None:
    """Delete OCR images by document IDs."""
    if not document_ids:
        return
    t = OcrImage.__table__
    with engine.begin() as conn:
        conn.execute(sa.delete(t).where(t.c.document_id.in_(document_ids)))
