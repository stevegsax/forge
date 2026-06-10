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
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from forge_contracts.db import (
    StoreConfigError as StoreConfigError,
)
from forge_contracts.db import (
    ensure_sqlite_parent,
    insert_or_ignore,
)
from forge_contracts.db import (
    get_store_engine as get_store_engine,
)
from forge_contracts.db import (
    get_store_url as get_store_url,
)
from forge_contracts.types import UTCDateTime
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


# Typed Table handles. ``Model.__table__`` is typed ``FromClause`` by SQLAlchemy's
# declarative stubs, but ``metadata.tables`` is typed ``Table`` — which is what the
# forge_contracts DB helpers (e.g. ``insert_or_ignore``) require.
_INTERACTIONS_TABLE: sa.Table = Base.metadata.tables[Interaction.__tablename__]
_RUNS_TABLE: sa.Table = Base.metadata.tables[Run.__tablename__]
_BATCH_JOBS_TABLE: sa.Table = Base.metadata.tables[BatchJob.__tablename__]
_PLAYBOOKS_TABLE: sa.Table = Base.metadata.tables[Playbook.__tablename__]


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
) -> dict[str, Any]:
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


def build_playbook_dict(
    entry: PlaybookEntry,
    extraction_workflow_id: str,
) -> dict[str, Any]:
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


def run_migrations(url: str) -> None:
    """Run Alembic migrations against the store URL (SQLite or Postgres)."""
    from alembic import command
    from alembic.config import Config

    alembic_dir = Path(__file__).parent / "alembic"
    ini_path = alembic_dir / "alembic.ini"

    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(alembic_dir))

    if make_url(url).get_backend_name() == "sqlite":
        ensure_sqlite_parent(url)
    # Alembic's Config is backed by configparser with interpolation enabled, so a
    # bare '%' (e.g. URL-encoded password chars %23, %21, %40) is parsed as an
    # interpolation token and raises. Escape as '%%'; get_main_option() in env.py
    # reverses this on read, so the engine still receives the original URL.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(cfg, "head")


def save_interaction(engine: Engine, **kwargs: object) -> bool:
    """Insert a row into the interactions table (idempotent on idempotency_key)."""
    inserted: bool = insert_or_ignore(
        engine, _INTERACTIONS_TABLE, dict(kwargs), index_elements=["idempotency_key"]
    )
    return inserted


def save_run(engine: Engine, task_result: TaskResult, workflow_id: str) -> bool:
    """Insert a row into the runs table (idempotent on workflow_id)."""
    inserted: bool = insert_or_ignore(
        engine,
        _RUNS_TABLE,
        {
            "task_id": task_result.task_id,
            "workflow_id": workflow_id,
            "status": task_result.status.value,
            "result_json": task_result.model_dump_json(),
        },
        index_elements=["workflow_id"],
    )
    return inserted


def get_interactions(
    engine: Engine,
    task_id: str,
    step_id: str | None = None,
) -> list[dict[str, Any]]:
    """Query interactions for a task, optionally filtered by step."""
    t = _INTERACTIONS_TABLE
    stmt = t.select().where(t.c.task_id == task_id).order_by(t.c.created_at)
    if step_id is not None:
        stmt = stmt.where(t.c.step_id == step_id)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


def get_run(engine: Engine, workflow_id: str) -> dict[str, Any] | None:
    """Query a single run by workflow ID."""
    t = _RUNS_TABLE
    stmt = t.select().where(t.c.workflow_id == workflow_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        result = dict(row)
        result["result"] = json.loads(result["result_json"])
        return result


def list_recent_runs(engine: Engine, limit: int = 20) -> list[dict[str, Any]]:
    """Query recent runs ordered by creation time descending."""
    t = _RUNS_TABLE
    stmt = t.select().order_by(t.c.created_at.desc()).limit(limit)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Playbook functions (Phase 6)
# ---------------------------------------------------------------------------


def save_playbooks(engine: Engine, entries: list[dict[str, Any]]) -> bool:
    """Insert playbook rows (idempotent per-entry on idempotency_key).

    Returns ``True`` if at least one new row was written.
    """
    applied = False
    for entry in entries:
        if insert_or_ignore(
            engine, _PLAYBOOKS_TABLE, dict(entry), index_elements=["idempotency_key"]
        ):
            applied = True
    return applied


def tags_overlap(tags_json: str, wanted: set[str]) -> bool:
    """True if the row's stored tags intersect ``wanted`` (pure, dialect-free).

    ``tags_json`` is a JSON-encoded list of strings (see ``build_playbook_dict``).
    Malformed or non-list payloads never match.
    """
    try:
        row_tags = json.loads(tags_json)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(row_tags, list):
        return False
    return any(tag in wanted for tag in row_tags)


def get_playbooks_by_tags(
    engine: Engine,
    tags: list[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Query playbooks matching any of the given tags, ordered by recency.

    Tag matching is done in Python (``tags_overlap``) rather than in SQL so the
    query is portable across SQLite and Postgres — the SQLite ``json_each()``
    table function does not exist on Postgres.
    """
    if not tags:
        return []

    wanted = set(tags)
    t = _PLAYBOOKS_TABLE
    stmt = t.select().order_by(t.c.created_at.desc())

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()

    matched = [dict(row) for row in rows if tags_overlap(row["tags_json"], wanted)]
    return matched[:limit] if limit > 0 else matched


def list_recent_playbooks(engine: Engine, limit: int = 20) -> list[dict[str, Any]]:
    """Query recent playbooks ordered by creation time descending."""
    t = _PLAYBOOKS_TABLE
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
    t = _PLAYBOOKS_TABLE

    # ``source_task_id`` filtering and recency ordering stay in SQL; tag matching
    # is done in Python (``tags_overlap``) to keep the query portable across
    # SQLite and Postgres (no ``json_each``). See ``get_playbooks_by_tags``.
    stmt = sa.select(t.c.id, t.c.tags_json)
    if source_task_id:
        stmt = stmt.where(t.c.source_task_id == source_task_id)
    stmt = stmt.order_by(t.c.created_at.desc())

    with engine.connect() as conn:
        rows = conn.execute(stmt).all()

    if tags:
        wanted = set(tags)
        ids = [row[0] for row in rows if tags_overlap(row[1], wanted)]
    else:
        ids = [row[0] for row in rows]

    return ids[:limit] if limit > 0 else ids


def get_playbook_by_id(engine: Engine, playbook_id: int) -> dict[str, Any] | None:
    """Fetch a single playbook row by primary key."""
    t = _PLAYBOOKS_TABLE
    stmt = t.select().where(t.c.id == playbook_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return dict(row)


def get_unextracted_runs(engine: Engine, limit: int = 50) -> list[dict[str, Any]]:
    """Query runs that have no corresponding playbook entries.

    Returns runs whose workflow_id is not in the playbooks table's
    source_workflow_id column.
    """
    runs_t = _RUNS_TABLE
    playbooks_t = _PLAYBOOKS_TABLE

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
) -> bool:
    """Insert a new batch job record with status 'submitted'.

    ``batch_jobs`` is generic provider-batch coordination — it carries no domain
    fields. A consumer keys its own status/metadata table by the same
    ``request_id``. Idempotent on the ``request_id`` primary key.
    """
    inserted: bool = insert_or_ignore(
        engine,
        _BATCH_JOBS_TABLE,
        {
            "id": request_id,
            "batch_id": batch_id,
            "workflow_id": workflow_id,
            "status": BatchJobStatus.SUBMITTED,
            "provider": provider,
        },
        index_elements=["id"],
    )
    return inserted


def record_batch_failure(
    engine: Engine,
    *,
    request_id: str,
    workflow_id: str,
    error_message: str,
    provider: str = "anthropic",
) -> bool:
    """Insert a batch job record with status 'failed' and no batch_id.

    Used when the provider API call fails before returning a batch_id.
    Idempotent on the ``request_id`` primary key.
    """
    inserted: bool = insert_or_ignore(
        engine,
        _BATCH_JOBS_TABLE,
        {
            "id": request_id,
            "batch_id": None,
            "workflow_id": workflow_id,
            "status": BatchJobStatus.FAILED,
            "provider": provider,
            "error_message": error_message,
        },
        index_elements=["id"],
    )
    return inserted


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
    t = _BATCH_JOBS_TABLE
    with engine.begin() as conn:
        conn.execute(
            sa.update(BatchJob)
            .where(t.c.id == request_id)
            .values(
                status=validated,
                error_message=error_message,
                updated_at=datetime.now(UTC),
            )
        )


def get_pending_batch_jobs(engine: Engine) -> list[dict[str, Any]]:
    """Query batch jobs with status 'submitted', ordered by created_at."""
    t = _BATCH_JOBS_TABLE
    stmt = t.select().where(t.c.status == BatchJobStatus.SUBMITTED).order_by(t.c.created_at)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


def get_batch_job(engine: Engine, request_id: str) -> dict[str, Any] | None:
    """Look up a single batch job by request ID."""
    t = _BATCH_JOBS_TABLE
    stmt = t.select().where(t.c.id == request_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return dict(row)
