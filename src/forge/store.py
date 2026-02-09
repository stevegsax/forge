"""Observability store for Forge.

Persists full LLM interaction data and run results to a local SQLite database.

Design follows Function Core / Imperative Shell:
- Pure functions: get_db_path, build_interaction_dict
- Imperative shell: get_engine, run_migrations, save_interaction, save_run,
  get_interactions, get_run, list_recent_runs
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
        LLMCallResult,
        PlanCallResult,
        PlaybookEntry,
        TaskResult,
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
    llm_result: LLMCallResult | PlanCallResult,
) -> dict:
    """Assemble a dict from activity data for insertion.

    Works with both LLMCallResult (has response.explanation) and
    PlanCallResult (has plan.explanation).
    """
    explanation = ""
    if hasattr(llm_result, "response"):
        explanation = llm_result.response.explanation
    elif hasattr(llm_result, "plan"):
        explanation = llm_result.plan.explanation

    context_stats_json = None
    if context.context_stats is not None:
        context_stats_json = context.context_stats.model_dump_json()

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
