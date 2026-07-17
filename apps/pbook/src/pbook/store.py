"""SQLAlchemy ORM and database operations for the playbook service.

The store targets PostgreSQL exclusively (Supabase-hosted). All objects
live in the ``pbook`` schema and are prefixed ``pbk_`` so they coexist
cleanly with any other tenant of the same database. Vector columns use
pgvector; semantic search is pushed into the database via the ``<=>``
cosine-distance operator rather than scored row-by-row in Python.

Design follows Function Core / Imperative Shell:

- Pure functions: get_database_url, build_entry_dict
- Imperative shell: get_engine, get_store_engine, run_migrations, and
  every query/mutation helper below.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy import Connection, Engine, RowMapping
    from sqlalchemy.sql import Select

    from pbook.models import PlaybookEntry


# ---------------------------------------------------------------------------
# Schema / naming
# ---------------------------------------------------------------------------

SCHEMA = "pbook"
VERSION_TABLE = "pbk_alembic_version"

# text-embedding-3-small dimensionality. The single source of truth for the
# vector width; the migration and the ORM both key off it.
EMBEDDING_DIM = 1536


class Base(DeclarativeBase):
    metadata = sa.MetaData(schema=SCHEMA)


SESSION_STATUS_RUNNING = "running"
SESSION_STATUS_COMPLETED = "completed"
SESSION_STATUS_ERROR = "error"


class Entry(Base):
    __tablename__ = "pbk_entries"

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    title: Mapped[str] = mapped_column(sa.Text, nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    entry_type: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        default="curated",
        server_default="curated",
    )
    source_project: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        default="",
        server_default="",
    )
    source_task_id: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        default="",
        server_default="",
    )
    needs_review: Mapped[bool] = mapped_column(
        sa.Boolean,
        nullable=False,
        default=False,
        server_default=sa.text("false"),
    )
    helpful_count: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        default=0,
        server_default=sa.text("0"),
    )
    harmful_count: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        default=0,
        server_default=sa.text("0"),
    )
    retrieval_count: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        default=0,
        server_default=sa.text("0"),
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=sa.text("now()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=sa.text("now()"),
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    rejected: Mapped[bool] = mapped_column(
        sa.Boolean,
        nullable=False,
        default=False,
        server_default=sa.text("false"),
    )
    rejection_reason: Mapped[str | None] = mapped_column(sa.Text, nullable=True)


class EntryTag(Base):
    """Normalized tag rows — one per (entry, namespaced tag).

    Replaces the old ``tags_json`` TEXT column. Tag matching is a join
    instead of SQLite's ``json_each``; read helpers re-assemble a
    ``tags`` list onto each entry dict so consumers see one shape.
    """

    __tablename__ = "pbk_entry_tags"

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    entry_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey(f"{SCHEMA}.pbk_entries.id", ondelete="CASCADE"),
        nullable=False,
    )
    tag: Mapped[str] = mapped_column(sa.Text, nullable=False)

    __table_args__ = (sa.UniqueConstraint("entry_id", "tag", name="uq_pbk_entry_tags_entry_tag"),)


# Match-or-attach thresholds. See grill-me-sessions/entry-sources.grill.md
# for the rationale (Branch H decisions).
ENTRY_MATCH_THRESHOLD = 0.85
SOURCE_DEDUP_THRESHOLD = 0.92


class EntrySource(Base):
    __tablename__ = "pbk_entry_sources"

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    entry_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey(f"{SCHEMA}.pbk_entries.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        default="",
        server_default="",
    )
    project_name: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        default="",
        server_default="",
    )
    experience_hash: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    source_context: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        default="",
        server_default="",
    )
    source_context_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=sa.text("now()"),
    )

    __table_args__ = (
        sa.UniqueConstraint(
            "entry_id",
            "session_id",
            "experience_hash",
            name="uq_pbk_entry_sources_entry_session_hash",
        ),
    )


class IngestedSession(Base):
    __tablename__ = "pbk_ingested_sessions"

    session_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    project_name: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        default="",
        server_default="",
    )
    ingested_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=sa.text("now()"),
    )
    experiences_found: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        default=0,
        server_default=sa.text("0"),
    )
    entries_created: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        default=0,
        server_default=sa.text("0"),
    )
    status: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        default=SESSION_STATUS_COMPLETED,
        server_default=SESSION_STATUS_COMPLETED,
    )
    workflow_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    run_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )


# ---------------------------------------------------------------------------
# Typed Table handles. ``Model.__table__`` is typed ``FromClause`` by
# SQLAlchemy's stubs, which lacks ``insert``/``update``/``delete``; look the
# concrete ``sa.Table`` up from metadata once so query helpers below get a
# properly typed handle (schema-qualified, since ``Base.metadata`` sets
# ``schema=SCHEMA``).
# ---------------------------------------------------------------------------

_ENTRIES_TABLE: sa.Table = Base.metadata.tables[f"{SCHEMA}.{Entry.__tablename__}"]
_ENTRY_TAGS_TABLE: sa.Table = Base.metadata.tables[f"{SCHEMA}.{EntryTag.__tablename__}"]
_ENTRY_SOURCES_TABLE: sa.Table = Base.metadata.tables[f"{SCHEMA}.{EntrySource.__tablename__}"]
_INGESTED_SESSIONS_TABLE: sa.Table = Base.metadata.tables[
    f"{SCHEMA}.{IngestedSession.__tablename__}"
]


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def get_database_url() -> str | None:
    """Resolve the store's PostgreSQL connection URL from the environment.

    Reads ``PBOOK_DATABASE_URL``. Returns ``None`` when the variable is
    unset or empty, which disables the store (callers no-op). A bare
    ``postgresql://`` / ``postgres://`` URL is normalized to the psycopg3
    driver so we never fall through to psycopg2.
    """
    raw = os.environ.get("PBOOK_DATABASE_URL")
    if not raw:
        logger.info("Store disabled (PBOOK_DATABASE_URL is unset or empty)")
        return None
    return normalize_url(raw)


def normalize_url(url: str) -> str:
    """Force the psycopg (v3) driver onto a PostgreSQL URL."""
    for prefix in ("postgresql+", "sqlite"):
        if url.startswith(prefix):
            return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    return url


def build_entry_dict(entry: PlaybookEntry) -> dict[str, Any]:
    """Convert a PlaybookEntry to an insertion dict for the entries table.

    The ``tags`` list is carried alongside the entry columns; the write
    helpers (:func:`insert_entry`, :func:`save_entries`) split it out into
    the ``pbk_entry_tags`` child table rather than storing it on the row.
    """
    return {
        "title": entry.title,
        "content": entry.content,
        "entry_type": entry.entry_type,
        "source_project": entry.source_project,
        "source_task_id": entry.source_task_id,
        "needs_review": entry.needs_review,
        "helpful_count": entry.helpful_count,
        "harmful_count": entry.harmful_count,
        "retrieval_count": entry.retrieval_count,
        "embedding": entry.embedding,
        "tags": list(entry.tags),
    }


def _dedup_preserving_order(tags: Sequence[str]) -> list[str]:
    """Unique tags, first occurrence wins — avoids the entry_tags UNIQUE."""
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


# ---------------------------------------------------------------------------
# Imperative shell — engine & migrations
# ---------------------------------------------------------------------------

_engines: dict[str, Engine] = {}


def _is_pooler(url: str) -> bool:
    """Heuristic: are we connecting through Supabase's transaction pooler?

    The transaction-mode pooler (PgBouncer) breaks server-side prepared
    statements; psycopg must be told to skip them. Detected by host or an
    explicit ``PBOOK_DB_POOLER`` override.
    """
    if os.environ.get("PBOOK_DB_POOLER", "").lower() in {"1", "true", "yes"}:
        return True
    return "pooler.supabase.com" in url or ":6543" in url


def get_engine(url: str) -> Engine:
    """Create (and cache) a SQLAlchemy engine for the given URL.

    Engines are cached per normalized URL so the connection pool is
    reused across activity calls rather than rebuilt each time.
    """
    norm = normalize_url(url)
    engine = _engines.get(norm)
    if engine is None:
        connect_args: dict[str, Any] = {}
        if _is_pooler(norm):
            # Disable prepared statements for PgBouncer transaction mode.
            connect_args["prepare_threshold"] = None
        logger.debug("Creating engine for %s (pooler=%s)", norm, _is_pooler(norm))
        engine = sa.create_engine(norm, pool_pre_ping=True, connect_args=connect_args)
        _engines[norm] = engine
    return engine


def get_store_engine() -> Engine | None:
    """Return the cached engine for the configured store, or None if disabled."""
    url = get_database_url()
    if url is None:
        return None
    return get_engine(url)


def run_migrations(url: str | None = None) -> None:
    """Run Alembic migrations to head against the configured database.

    Idempotent and intended to run ONCE per process (worker startup or an
    explicit ``pbook migrate``), never per activity call.
    """
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    if url is None:
        url = get_database_url()
    if url is None:
        msg = "Cannot run migrations: PBOOK_DATABASE_URL is not set."
        raise RuntimeError(msg)

    alembic_dir = Path(__file__).parent / "alembic"
    cfg = Config(str(alembic_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(alembic_dir))
    # Pass the URL via the environment rather than the Alembic config:
    # set_main_option runs the value through ConfigParser's %-interpolation,
    # which would choke on a percent-encoded password (e.g. %40, %23). env.py
    # reads PBOOK_DATABASE_URL when sqlalchemy.url is unset.
    os.environ["PBOOK_DATABASE_URL"] = normalize_url(url)
    logger.info("Running migrations to head")
    command.upgrade(cfg, "head")


# ---------------------------------------------------------------------------
# Imperative shell — row shaping helpers
# ---------------------------------------------------------------------------

_VECTOR_FIELDS = ("embedding", "source_context_embedding")


def _row_to_dict(mapping: RowMapping) -> dict[str, Any]:
    """Materialize a result mapping, coercing pgvector arrays to plain lists.

    pgvector returns ``numpy.ndarray`` for vector columns; converting to a
    list keeps row dicts JSON-serializable for the Temporal boundary.
    """
    row = dict(mapping)
    for field in _VECTOR_FIELDS:
        value = row.get(field)
        if value is not None and not isinstance(value, list):
            row[field] = list(value)
    return row


def _load_tags(conn: Connection, entry_ids: Sequence[int]) -> dict[int, list[str]]:
    """Fetch tag lists for the given entry ids, keyed by entry id."""
    if not entry_ids:
        return {}
    tg = _ENTRY_TAGS_TABLE
    stmt = (
        sa.select(tg.c.entry_id, tg.c.tag)
        .where(tg.c.entry_id.in_(entry_ids))
        .order_by(tg.c.entry_id, tg.c.tag)
    )
    grouped: dict[int, list[str]] = {}
    for entry_id, tag in conn.execute(stmt).all():
        grouped.setdefault(entry_id, []).append(tag)
    return grouped


def _attach_tags(conn: Connection, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate each entry row dict with its ``tags`` list (possibly empty)."""
    tag_map = _load_tags(conn, [r["id"] for r in rows])
    for row in rows:
        row["tags"] = tag_map.get(row["id"], [])
    return rows


# ---------------------------------------------------------------------------
# Imperative shell — writes
# ---------------------------------------------------------------------------


def insert_entry(engine: Engine, entry: dict[str, Any], tags: Sequence[str] | None = None) -> int:
    """Insert one entry plus its tag rows; return the new entry id.

    The single insert primitive — used by the CLI add path, the
    extraction match-or-attach path, and consolidation. ``entry`` is a
    column dict (e.g. from :func:`build_entry_dict`); any ``tags`` key it
    carries is split into the child table. An explicit ``tags`` argument
    overrides the dict's.
    """
    e = _ENTRIES_TABLE
    tg = _ENTRY_TAGS_TABLE
    columns = dict(entry)
    embedded_tags = columns.pop("tags", None)
    effective_tags = tags if tags is not None else (embedded_tags or [])
    with engine.begin() as conn:
        new_id = conn.execute(sa.insert(e).values(**columns).returning(e.c.id)).scalar_one()
        unique_tags = _dedup_preserving_order(effective_tags)
        if unique_tags:
            conn.execute(
                sa.insert(tg),
                [{"entry_id": new_id, "tag": t} for t in unique_tags],
            )
    logger.info("Inserted entry %d with %d tag(s)", new_id, len(set(effective_tags)))
    return int(new_id)


def save_entries(engine: Engine, entries: list[dict[str, Any]]) -> None:
    """Bulk-insert entry dicts (each may carry a ``tags`` list)."""
    for entry in entries:
        insert_entry(engine, entry)


def set_entry_tags(engine: Engine, entry_id: int, tags: Sequence[str]) -> None:
    """Replace all tag rows for an entry with the given set."""
    tg = _ENTRY_TAGS_TABLE
    unique_tags = _dedup_preserving_order(tags)
    with engine.begin() as conn:
        conn.execute(tg.delete().where(tg.c.entry_id == entry_id))
        if unique_tags:
            conn.execute(
                sa.insert(tg),
                [{"entry_id": entry_id, "tag": t} for t in unique_tags],
            )


def add_entry_tag(engine: Engine, entry_id: int, tag: str) -> None:
    """Add a single tag to an entry, ignoring duplicates."""
    tg = _ENTRY_TAGS_TABLE
    stmt = (
        pg_insert(tg)
        .values(entry_id=entry_id, tag=tag)
        .on_conflict_do_nothing(
            constraint="uq_pbk_entry_tags_entry_tag",
        )
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def update_entry(engine: Engine, entry_id: int, updates: dict[str, Any]) -> None:
    """Update an entry by primary key.

    A ``tags`` key (list[str]) in ``updates`` is special-cased: it
    replaces the entry's tag rows rather than updating a column.
    """
    updates = dict(updates)
    tags = updates.pop("tags", None)
    logger.info(
        "Updating entry %d (%d column(s), tags=%s)",
        entry_id,
        len(updates),
        tags is not None,
    )
    e = _ENTRIES_TABLE
    tg = _ENTRY_TAGS_TABLE
    with engine.begin() as conn:
        if updates:
            conn.execute(e.update().where(e.c.id == entry_id).values(**updates))
        if tags is not None:
            conn.execute(tg.delete().where(tg.c.entry_id == entry_id))
            unique_tags = _dedup_preserving_order(tags)
            if unique_tags:
                conn.execute(
                    sa.insert(tg),
                    [{"entry_id": entry_id, "tag": t} for t in unique_tags],
                )


def delete_entry(engine: Engine, entry_id: int) -> None:
    """Delete an entry by primary key (tags/sources cascade)."""
    logger.info("Deleting entry %d", entry_id)
    e = _ENTRIES_TABLE
    with engine.begin() as conn:
        conn.execute(e.delete().where(e.c.id == entry_id))


def mark_rejected(engine: Engine, entry_id: int, *, reason: str | None = None) -> None:
    """Soft-mark an entry as rejected with an optional reason.

    The row stays in the table so the rejection (and its reason) survive
    for audit; default queries hide rejected rows.
    """
    logger.info("Marking entry %d as rejected (reason=%r)", entry_id, reason or "<none>")
    e = _ENTRIES_TABLE
    with engine.begin() as conn:
        conn.execute(
            e.update().where(e.c.id == entry_id).values(rejected=True, rejection_reason=reason),
        )


def record_retrieval(engine: Engine, entry_ids: list[int]) -> None:
    """Bulk increment retrieval_count for the given entry IDs."""
    if not entry_ids:
        return
    logger.info("Recording retrieval for %d entries", len(entry_ids))
    e = _ENTRIES_TABLE
    with engine.begin() as conn:
        conn.execute(
            e.update()
            .where(e.c.id.in_(entry_ids))
            .values(
                retrieval_count=e.c.retrieval_count + 1,
            ),
        )


def record_feedback(engine: Engine, entry_id: int, *, helpful: bool) -> None:
    """Increment helpful_count or harmful_count for a single entry."""
    e = _ENTRIES_TABLE
    col = e.c.helpful_count if helpful else e.c.harmful_count
    logger.info("Recording %s feedback for entry %d", "helpful" if helpful else "harmful", entry_id)
    with engine.begin() as conn:
        conn.execute(e.update().where(e.c.id == entry_id).values(**{col.name: col + 1}))


# ---------------------------------------------------------------------------
# Imperative shell — reads
# ---------------------------------------------------------------------------


def get_entry_by_id(engine: Engine, entry_id: int) -> dict[str, Any] | None:
    """Fetch a single entry row (with tags) by primary key."""
    e = _ENTRIES_TABLE
    with engine.connect() as conn:
        mapping = conn.execute(e.select().where(e.c.id == entry_id)).mappings().first()
        if mapping is None:
            return None
        return _attach_tags(conn, [_row_to_dict(mapping)])[0]


def get_entries_by_ids(engine: Engine, ids: list[int]) -> list[dict[str, Any]]:
    """Bulk-fetch entries (with tags) by primary key, arbitrary order."""
    if not ids:
        return []
    e = _ENTRIES_TABLE
    with engine.connect() as conn:
        rows = [_row_to_dict(m) for m in conn.execute(e.select().where(e.c.id.in_(ids))).mappings()]
        return _attach_tags(conn, rows)


def get_entries_by_tags(
    engine: Engine,
    tags: list[str],
    *,
    limit: int = 10,
    approved_only: bool = False,
    include_rejected: bool = False,
) -> list[dict[str, Any]]:
    """Query entries carrying any of the given tags, newest first.

    Joins ``pbk_entry_tags`` instead of unnesting a JSON column. Rejected
    entries are excluded by default.
    """
    if not tags:
        return []
    logger.debug(
        "Querying entries by tags=%s limit=%d approved_only=%s include_rejected=%s",
        tags,
        limit,
        approved_only,
        include_rejected,
    )
    e = _ENTRIES_TABLE
    tg = _ENTRY_TAGS_TABLE
    stmt = (
        sa.select(e)
        .distinct()
        .select_from(e.join(tg, tg.c.entry_id == e.c.id))
        .where(tg.c.tag.in_(tags))
    )
    if approved_only:
        stmt = stmt.where(e.c.needs_review == sa.false())
    if not include_rejected:
        stmt = stmt.where(e.c.rejected == sa.false())
    stmt = stmt.order_by(e.c.created_at.desc()).limit(limit)

    with engine.connect() as conn:
        rows = [_row_to_dict(m) for m in conn.execute(stmt).mappings()]
        return _attach_tags(conn, rows)


def list_recent_entries(
    engine: Engine,
    *,
    limit: int = 20,
    include_rejected: bool = False,
) -> list[dict[str, Any]]:
    """Query recent entries (with tags) ordered by creation time descending."""
    e = _ENTRIES_TABLE
    stmt = e.select()
    if not include_rejected:
        stmt = stmt.where(e.c.rejected == sa.false())
    stmt = stmt.order_by(e.c.created_at.desc()).limit(limit)
    with engine.connect() as conn:
        rows = [_row_to_dict(m) for m in conn.execute(stmt).mappings()]
        return _attach_tags(conn, rows)


def list_embedded_entries(
    engine: Engine,
    *,
    approved_only: bool = False,
    include_rejected: bool = False,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Fetch recent entries that have an embedding, newest first.

    Backs the free-text retrieval path: a broad candidate pool the
    semantic step can rank. Tags are attached so tag-overlap can serve
    as the ranking tiebreaker.
    """
    e = _ENTRIES_TABLE
    stmt = e.select().where(e.c.embedding.is_not(None))
    if approved_only:
        stmt = stmt.where(e.c.needs_review == sa.false())
    if not include_rejected:
        stmt = stmt.where(e.c.rejected == sa.false())
    stmt = stmt.order_by(e.c.created_at.desc()).limit(limit)
    with engine.connect() as conn:
        rows = [_row_to_dict(m) for m in conn.execute(stmt).mappings()]
        return _attach_tags(conn, rows)


def list_all_entries(engine: Engine) -> list[dict[str, Any]]:
    """Fetch all entries (with tags and embeddings) for maintenance analysis."""
    e = _ENTRIES_TABLE
    stmt = e.select().order_by(e.c.created_at.desc())
    with engine.connect() as conn:
        rows = [_row_to_dict(m) for m in conn.execute(stmt).mappings()]
        return _attach_tags(conn, rows)


def list_review_queue_with_sources(engine: Engine) -> list[dict[str, Any]]:
    """Fetch needs_review entries each annotated with their source rows.

    Each returned dict is an entry (with ``tags``) plus a ``sources`` key
    holding its ``entry_sources`` rows in created_at order. Rejected
    entries are excluded.
    """
    e = _ENTRIES_TABLE
    stmt = (
        e.select()
        .where(e.c.needs_review == sa.true())
        .where(e.c.rejected == sa.false())
        .order_by(e.c.created_at.desc())
    )
    with engine.connect() as conn:
        entries = [_row_to_dict(m) for m in conn.execute(stmt).mappings()]
        if not entries:
            return []
        _attach_tags(conn, entries)

        s = _ENTRY_SOURCES_TABLE
        src_stmt = (
            s.select()
            .where(s.c.entry_id.in_([e_["id"] for e_ in entries]))
            .order_by(s.c.created_at.asc())
        )
        source_rows = [_row_to_dict(m) for m in conn.execute(src_stmt).mappings()]

    by_entry: dict[int, list[dict[str, Any]]] = {}
    for src in source_rows:
        by_entry.setdefault(src["entry_id"], []).append(src)
    for entry in entries:
        entry["sources"] = by_entry.get(entry["id"], [])
    return entries


def list_tag_values_in_use(engine: Engine) -> dict[str, list[str]]:
    """Group distinct in-use tag values by namespace across non-rejected entries."""
    from pbook.tags import VALID_NAMESPACES

    e = _ENTRIES_TABLE
    tg = _ENTRY_TAGS_TABLE
    stmt = (
        sa.select(tg.c.tag)
        .distinct()
        .select_from(tg.join(e, e.c.id == tg.c.entry_id))
        .where(e.c.rejected == sa.false())
    )
    groups: dict[str, set[str]] = {ns: set() for ns in VALID_NAMESPACES}
    with engine.connect() as conn:
        for (tag,) in conn.execute(stmt).all():
            if not isinstance(tag, str) or ":" not in tag:
                continue
            ns, _, value = tag.partition(":")
            if ns in groups and value:
                groups[ns].add(value)
    return {ns: sorted(values) for ns, values in groups.items()}


def check_duplicate(
    engine: Engine,
    title: str,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Find entries with similar titles for duplicate detection.

    Case-insensitive LIKE on title. If tags are provided, results are
    sorted by tag overlap (descending).
    """
    e = _ENTRIES_TABLE
    stmt = e.select().where(e.c.title.ilike(f"%{title}%")).order_by(e.c.created_at.desc()).limit(10)
    with engine.connect() as conn:
        rows = [_row_to_dict(m) for m in conn.execute(stmt).mappings()]
        results = _attach_tags(conn, rows)

    if tags and results:
        tag_set = set(tags)
        results.sort(key=lambda r: len(tag_set & set(r.get("tags", []))), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Imperative shell — semantic search (pgvector)
# ---------------------------------------------------------------------------


def find_semantic_duplicates(
    engine: Engine,
    query_embedding: list[float],
    *,
    threshold: float = 0.85,
    limit: int = 5,
    include_rejected: bool = False,
) -> list[dict[str, Any]]:
    """Find entries within ``threshold`` cosine similarity of the embedding.

    Cosine distance (``<=>``) is computed in PostgreSQL; similarity is
    ``1 - distance``. Rejected entries are excluded by default.
    """
    e = _ENTRIES_TABLE
    distance = e.c.embedding.cosine_distance(query_embedding)
    stmt = (
        sa.select(e, distance.label("_distance"))
        .where(e.c.embedding.is_not(None))
        .where(distance <= (1.0 - threshold))
    )
    if not include_rejected:
        stmt = stmt.where(e.c.rejected == sa.false())
    stmt = stmt.order_by(distance.asc()).limit(limit)
    return _scored_rows(engine, stmt)


def semantic_search(
    engine: Engine,
    query_embedding: list[float],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Rank all embedded entries by cosine similarity to the query."""
    e = _ENTRIES_TABLE
    distance = e.c.embedding.cosine_distance(query_embedding)
    stmt = (
        sa.select(e, distance.label("_distance"))
        .where(e.c.embedding.is_not(None))
        .order_by(distance.asc())
        .limit(limit)
    )
    return _scored_rows(engine, stmt)


def _scored_rows(engine: Engine, stmt: Select[Any]) -> list[dict[str, Any]]:
    """Execute a select that includes a ``_distance`` column and attach
    ``similarity`` (1 - distance) plus tags to each row."""
    with engine.connect() as conn:
        rows: list[dict[str, Any]] = []
        for mapping in conn.execute(stmt).mappings():
            row = _row_to_dict(mapping)
            distance = row.pop("_distance")
            row["similarity"] = 1.0 - float(distance)
            rows.append(row)
        return _attach_tags(conn, rows)


def cosine_similarities_for_ids(
    engine: Engine,
    query_embedding: list[float],
    ids: list[int],
) -> dict[int, float]:
    """Cosine similarity between the query embedding and each named entry.

    Computed in PostgreSQL. Entries with NULL embeddings or ids absent
    from the table are silently omitted.
    """
    if not ids:
        return {}
    e = _ENTRIES_TABLE
    distance = e.c.embedding.cosine_distance(query_embedding)
    stmt = sa.select(e.c.id, distance).where(
        e.c.id.in_(ids),
        e.c.embedding.is_not(None),
    )
    with engine.connect() as conn:
        return {row[0]: 1.0 - float(row[1]) for row in conn.execute(stmt).all()}


# ---------------------------------------------------------------------------
# Imperative shell — entry_sources
# ---------------------------------------------------------------------------


def add_entry_source(
    engine: Engine,
    *,
    entry_id: int,
    session_id: str = "",
    project_name: str = "",
    experience_hash: str | None = None,
    source_context: str = "",
    source_context_embedding: list[float] | None = None,
) -> int | None:
    """Insert an entry_sources row, honoring the UNIQUE constraint.

    Returns the new row id, or ``None`` when the insert was a no-op
    because an identical (entry_id, session_id, experience_hash) row
    already exists.
    """
    s = _ENTRY_SOURCES_TABLE
    stmt = (
        pg_insert(s)
        .values(
            entry_id=entry_id,
            session_id=session_id,
            project_name=project_name,
            experience_hash=experience_hash,
            source_context=source_context,
            source_context_embedding=source_context_embedding,
        )
        .on_conflict_do_nothing(constraint="uq_pbk_entry_sources_entry_session_hash")
        .returning(s.c.id)
    )
    with engine.begin() as conn:
        row = conn.execute(stmt).first()
        return int(row[0]) if row else None


def list_entry_sources_for_entry(engine: Engine, entry_id: int) -> list[dict[str, Any]]:
    """Return all entry_sources rows for an entry, oldest first."""
    s = _ENTRY_SOURCES_TABLE
    stmt = s.select().where(s.c.entry_id == entry_id).order_by(s.c.created_at)
    with engine.connect() as conn:
        return [_row_to_dict(m) for m in conn.execute(stmt).mappings()]


def find_similar_source_contexts(
    engine: Engine,
    entry_id: int,
    query_embedding: list[float],
    *,
    threshold: float = SOURCE_DEDUP_THRESHOLD,
) -> list[dict[str, Any]]:
    """Find this entry's source rows within ``threshold`` similarity.

    Used to skip writing a duplicate justification for an entry.
    """
    s = _ENTRY_SOURCES_TABLE
    distance = s.c.source_context_embedding.cosine_distance(query_embedding)
    stmt = (
        sa.select(s, distance.label("_distance"))
        .where(s.c.entry_id == entry_id)
        .where(s.c.source_context_embedding.is_not(None))
        .where(distance <= (1.0 - threshold))
        .order_by(distance.asc())
    )
    with engine.connect() as conn:
        out: list[dict[str, Any]] = []
        for mapping in conn.execute(stmt).mappings():
            row = _row_to_dict(mapping)
            row["similarity"] = 1.0 - float(row.pop("_distance"))
            out.append(row)
        return out


def reparent_entry_sources(
    engine: Engine,
    *,
    from_entry_ids: list[int],
    to_entry_id: int,
) -> int:
    """Move all entry_sources rows from ``from_entry_ids`` to ``to_entry_id``.

    Used by consolidation when N entries merge into one. UNIQUE conflicts
    are resolved by dropping the losing (already-recorded) source row.
    Returns the number of rows re-parented onto the survivor.
    """
    if not from_entry_ids:
        return 0
    s = _ENTRY_SOURCES_TABLE
    with engine.begin() as conn:
        existing = conn.execute(
            sa.select(s.c.session_id, s.c.experience_hash).where(s.c.entry_id == to_entry_id),
        ).all()
        existing_set = {(sess, h) for sess, h in existing}

        from_rows = conn.execute(
            sa.select(s.c.id, s.c.session_id, s.c.experience_hash).where(
                s.c.entry_id.in_(from_entry_ids),
            ),
        ).all()
        to_delete = [
            row.id for row in from_rows if (row.session_id, row.experience_hash) in existing_set
        ]
        if to_delete:
            conn.execute(s.delete().where(s.c.id.in_(to_delete)))

        result = conn.execute(
            s.update().where(s.c.entry_id.in_(from_entry_ids)).values(entry_id=to_entry_id),
        )
        return result.rowcount or 0


# ---------------------------------------------------------------------------
# Imperative shell — ingested sessions
# ---------------------------------------------------------------------------


def get_ingested_session_ids(engine: Engine) -> set[str]:
    """Session IDs to skip on the next ingest run (completed or running)."""
    t = _INGESTED_SESSIONS_TABLE
    stmt = sa.select(t.c.session_id).where(
        t.c.status.in_([SESSION_STATUS_COMPLETED, SESSION_STATUS_RUNNING]),
    )
    with engine.connect() as conn:
        return {row[0] for row in conn.execute(stmt).all()}


def list_ingested_sessions(
    engine: Engine,
    *,
    project: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List ingested sessions, newest known activity first."""
    t = _INGESTED_SESSIONS_TABLE
    order_key = sa.func.coalesce(t.c.started_at, t.c.ingested_at)
    stmt = t.select()
    if project:
        stmt = stmt.where(t.c.project_name == project)
    stmt = stmt.order_by(order_key.desc()).limit(limit)
    with engine.connect() as conn:
        return [dict(m) for m in conn.execute(stmt).mappings()]


def _upsert_session(engine: Engine, values: dict[str, Any], set_: dict[str, Any]) -> None:
    """Upsert an ingested_sessions row on the session_id primary key."""
    t = _INGESTED_SESSIONS_TABLE
    stmt = (
        pg_insert(t)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["session_id"],
            set_=set_,
        )
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def record_ingested_session_started(
    engine: Engine,
    session_id: str,
    *,
    project_name: str = "",
    workflow_id: str | None = None,
    run_id: str | None = None,
) -> None:
    """Mark a session as in-flight (status='running')."""
    logger.info("Recording ingested session %s as running", session_id)
    now = datetime.now(UTC)
    common = {
        "project_name": project_name,
        "status": SESSION_STATUS_RUNNING,
        "workflow_id": workflow_id,
        "run_id": run_id,
        "started_at": now,
        "ingested_at": now,
        "experiences_found": 0,
        "entries_created": 0,
        "error_message": None,
    }
    _upsert_session(engine, {"session_id": session_id, **common}, common)


def record_ingested_session(
    engine: Engine,
    session_id: str,
    project_name: str = "",
    experiences_found: int = 0,
    entries_created: int = 0,
) -> None:
    """Record that a session has finished ingesting (status='completed')."""
    logger.info(
        "Recording ingested session %s: %d experiences, %d entries",
        session_id,
        experiences_found,
        entries_created,
    )
    now = datetime.now(UTC)
    common = {
        "project_name": project_name,
        "experiences_found": experiences_found,
        "entries_created": entries_created,
        "status": SESSION_STATUS_COMPLETED,
        "ingested_at": now,
        "error_message": None,
    }
    _upsert_session(engine, {"session_id": session_id, **common}, common)


def record_ingested_session_error(
    engine: Engine,
    session_id: str,
    error_message: str,
    *,
    project_name: str = "",
) -> None:
    """Mark a session as failed (status='error')."""
    logger.info("Recording ingested session %s as error: %s", session_id, error_message)
    now = datetime.now(UTC)
    _upsert_session(
        engine,
        {
            "session_id": session_id,
            "project_name": project_name,
            "status": SESSION_STATUS_ERROR,
            "error_message": error_message,
            "ingested_at": now,
            "experiences_found": 0,
            "entries_created": 0,
        },
        {
            "status": SESSION_STATUS_ERROR,
            "error_message": error_message,
            "ingested_at": now,
        },
    )
