"""SQLAlchemy ORM and database operations for the playbook service.

Design follows Function Core / Imperative Shell:

- Pure functions: get_db_path, build_entry_dict
- Imperative shell: get_engine, run_migrations, save_entries,
  get_entries_by_tags, list_recent_entries, get_entry_by_id,
  update_entry, check_duplicate
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

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from pbook.models import PlaybookEntry


# ---------------------------------------------------------------------------
# ORM
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class IngestedSession(Base):
    __tablename__ = "ingested_sessions"

    session_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    project_name: Mapped[str] = mapped_column(sa.Text, nullable=False, default="")
    ingested_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=lambda: datetime.now(UTC),
    )
    experiences_found: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0"),
    )
    entries_created: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0"),
    )


class Entry(Base):
    __tablename__ = "entries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(sa.String, nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    tags_json: Mapped[str] = mapped_column(sa.Text, nullable=False)
    entry_type: Mapped[str] = mapped_column(sa.String, nullable=False, default="curated")
    source_project: Mapped[str] = mapped_column(sa.String, nullable=False, default="")
    source_task_id: Mapped[str] = mapped_column(sa.String, nullable=False, default="")
    needs_review: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.text("0"),
    )
    helpful_count: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0"),
    )
    harmful_count: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0"),
    )
    retrieval_count: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, default=0, server_default=sa.text("0"),
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    embedding: Mapped[bytes | None] = mapped_column(sa.LargeBinary, nullable=True)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def get_db_path() -> Path | None:
    """Resolve the database path.

    Resolution order:

    1. ``PBOOK_DB_PATH`` environment variable.
    2. ``$XDG_STATE_HOME/pbook/pbook.db``
    3. ``~/.local/state/pbook/pbook.db``

    Returns ``None`` if ``PBOOK_DB_PATH`` is set to an empty string (disables store).
    """
    env_value = os.environ.get("PBOOK_DB_PATH")
    if env_value is not None:
        if env_value == "":
            logger.info("Store disabled (PBOOK_DB_PATH is empty)")
            return None
        path = Path(env_value)
        logger.debug("DB path from PBOOK_DB_PATH: %s", path)
        return path

    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        path = Path(xdg_state) / "pbook" / "pbook.db"
        logger.debug("DB path from XDG_STATE_HOME: %s", path)
        return path

    path = Path.home() / ".local" / "state" / "pbook" / "pbook.db"
    logger.debug("DB path (default): %s", path)
    return path


def build_entry_dict(entry: PlaybookEntry) -> dict:
    """Convert a PlaybookEntry to a dict suitable for database insertion."""
    return {
        "title": entry.title,
        "content": entry.content,
        "tags_json": json.dumps(entry.tags),
        "entry_type": entry.entry_type,
        "source_project": entry.source_project,
        "source_task_id": entry.source_task_id,
        "needs_review": entry.needs_review,
        "helpful_count": entry.helpful_count,
        "harmful_count": entry.harmful_count,
        "retrieval_count": entry.retrieval_count,
        "embedding": entry.embedding,
    }


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def get_engine(db_path: Path) -> Engine:
    """Create a SQLAlchemy engine with WAL mode for the given database path."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Creating engine for %s", db_path)
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
    logger.debug("Running migrations for %s", db_path)
    command.upgrade(cfg, "head")


def save_entries(engine: Engine, entries: list[dict]) -> None:
    """Bulk insert rows into the entries table."""
    if not entries:
        return
    logger.info("Saving %d entries", len(entries))
    with engine.begin() as conn:
        conn.execute(sa.insert(Entry.__table__), entries)


def get_entries_by_tags(
    engine: Engine,
    tags: list[str],
    *,
    limit: int = 10,
    approved_only: bool = False,
) -> list[dict]:
    """Query entries matching any of the given tags, ordered by recency.

    Uses SQLite ``json_each()`` to unnest the ``tags_json`` array and match
    against the input tags.
    """
    if not tags:
        return []

    logger.debug("Querying entries by tags=%s limit=%d approved_only=%s", tags, limit, approved_only)
    tag_placeholders = ", ".join(f":tag_{i}" for i in range(len(tags)))
    tag_params = {f"tag_{i}": tag for i, tag in enumerate(tags)}

    approved_clause = ""
    if approved_only:
        approved_clause = "AND p.needs_review = 0"

    query = sa.text(f"""
        SELECT DISTINCT p.*
        FROM entries p, json_each(p.tags_json) AS t
        WHERE t.value IN ({tag_placeholders})
        {approved_clause}
        ORDER BY p.created_at DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {**tag_params, "limit": limit}).mappings().all()
        return [dict(row) for row in rows]


def list_recent_entries(engine: Engine, *, limit: int = 20) -> list[dict]:
    """Query recent entries ordered by creation time descending."""
    t = Entry.__table__
    stmt = t.select().order_by(t.c.created_at.desc()).limit(limit)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


def get_entry_by_id(engine: Engine, entry_id: int) -> dict | None:
    """Fetch a single entry row by primary key."""
    t = Entry.__table__
    stmt = t.select().where(t.c.id == entry_id)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None


def update_entry(engine: Engine, entry_id: int, updates: dict) -> None:
    """Update an entry by primary key with the given field values."""
    logger.info("Updating entry %d: %s", entry_id, list(updates.keys()))
    t = Entry.__table__
    with engine.begin() as conn:
        conn.execute(t.update().where(t.c.id == entry_id).values(**updates))


def delete_entry(engine: Engine, entry_id: int) -> None:
    """Delete an entry by primary key."""
    logger.info("Deleting entry %d", entry_id)
    t = Entry.__table__
    with engine.begin() as conn:
        conn.execute(t.delete().where(t.c.id == entry_id))


def list_all_entries(engine: Engine) -> list[dict]:
    """Fetch all entries including feedback counters for maintenance analysis."""
    t = Entry.__table__
    stmt = t.select().order_by(t.c.created_at.desc())

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


def record_retrieval(engine: Engine, entry_ids: list[int]) -> None:
    """Bulk increment retrieval_count for the given entry IDs."""
    if not entry_ids:
        return
    logger.info("Recording retrieval for %d entries", len(entry_ids))
    t = Entry.__table__
    with engine.begin() as conn:
        conn.execute(
            t.update()
            .where(t.c.id.in_(entry_ids))
            .values(retrieval_count=t.c.retrieval_count + 1),
        )


def record_feedback(engine: Engine, entry_id: int, *, helpful: bool) -> None:
    """Increment helpful_count or harmful_count for a single entry."""
    t = Entry.__table__
    col = t.c.helpful_count if helpful else t.c.harmful_count
    logger.info("Recording %s feedback for entry %d", "helpful" if helpful else "harmful", entry_id)
    with engine.begin() as conn:
        conn.execute(
            t.update().where(t.c.id == entry_id).values(**{col.name: col + 1}),
        )


def check_duplicate(
    engine: Engine,
    title: str,
    tags: list[str] | None = None,
) -> list[dict]:
    """Find entries with similar titles for duplicate detection.

    Uses case-insensitive LIKE matching on title.  If tags are provided,
    also checks for tag overlap.
    """
    t = Entry.__table__
    stmt = t.select().where(t.c.title.ilike(f"%{title}%")).order_by(t.c.created_at.desc()).limit(10)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        results = [dict(row) for row in rows]

    if tags and results:
        tag_set = set(tags)
        results.sort(
            key=lambda r: len(tag_set & set(json.loads(r.get("tags_json", "[]")))),
            reverse=True,
        )

    return results


def get_ingested_session_ids(engine: Engine) -> set[str]:
    """Return the set of session IDs that have already been ingested."""
    t = IngestedSession.__table__
    with engine.connect() as conn:
        rows = conn.execute(sa.select(t.c.session_id)).all()
        return {row[0] for row in rows}


def record_ingested_session(
    engine: Engine,
    session_id: str,
    project_name: str = "",
    experiences_found: int = 0,
    entries_created: int = 0,
) -> None:
    """Record that a session has been ingested."""
    logger.info(
        "Recording ingested session %s: %d experiences, %d entries",
        session_id, experiences_found, entries_created,
    )
    with engine.begin() as conn:
        conn.execute(
            sa.insert(IngestedSession.__table__).values(
                session_id=session_id,
                project_name=project_name,
                experiences_found=experiences_found,
                entries_created=entries_created,
            ),
        )


def find_semantic_duplicates(
    engine: Engine,
    query_embedding: bytes,
    *,
    threshold: float = 0.85,
    limit: int = 5,
) -> list[dict]:
    """Find entries with high semantic similarity to the given embedding.

    Calculates cosine similarity in Python using the ``embedding`` column.
    """
    from pbook.embeddings import cosine_similarity

    t = Entry.__table__
    stmt = t.select().where(t.c.embedding.is_not(None))

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()

    results: list[dict] = []
    for row in rows:
        sim = cosine_similarity(query_embedding, row["embedding"])
        if sim >= threshold:
            results.append({**dict(row), "similarity": sim})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]


def semantic_search(
    engine: Engine,
    query_embedding: bytes,
    *,
    limit: int = 10,
) -> list[dict]:
    """Rank all entries by semantic similarity to the query embedding."""
    from pbook.embeddings import cosine_similarity

    t = Entry.__table__
    stmt = t.select().where(t.c.embedding.is_not(None))

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()

    scored: list[tuple[float, dict]] = []
    for row in rows:
        sim = cosine_similarity(query_embedding, row["embedding"])
        scored.append((sim, dict(row)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _sim, row in scored[:limit]]
