"""Verify a deployed Alembic chain against the schema the running code expects.

The counterpart to :mod:`sax_platform.db.migrations`: that module *applies* a
chain (CLI, dev self-service, tests); this one only *reads* the chain's version
table and decides whether the deployed code may run against it.

Why startup stopped migrating (2026-08-02, the schema-change agreement with the
sax-datastores operator, ``sax-datastores/docs/schema-changes.md``): the
application credential cannot run DDL at all under the operator model, and a
migration that runs as a side effect of a process starting turns one failed
migration into a crash-looping fleet. Worker startup therefore compares
``alembic_version`` against the code's expected head and fails closed with a
named error — never DDL.

A database **ahead** of the deployed code is deliberately allowed. Under the
binding expand/contract contract every schema change is applied *before* the
code that uses it deploys, so a worker restart inside that window is normal and
must not brick the lane; it logs a warning and runs.

Function Core / Imperative Shell: :func:`classify_schema_version` and
:func:`describe_verdict` are pure over already-read facts (what the database is
stamped with, what the chain declares), and :func:`verify_schema_version` is the
thin shell that reads those facts and turns the verdict into a return value, a
log line, or a raise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, assert_never

import sqlalchemy as sa
from sqlalchemy.engine import make_url

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from sqlalchemy.sql import Select

logger = logging.getLogger(__name__)

__all__ = [
    "Ahead",
    "AmbiguousStamp",
    "AtHead",
    "Behind",
    "BrokenChain",
    "SchemaVerdict",
    "SchemaVersionError",
    "Uninitialized",
    "classify_schema_version",
    "describe_verdict",
    "verify_schema_version",
    "version_table_select",
]


class SchemaVersionError(RuntimeError):
    """The deployed schema is unusable by this code, and no process may proceed.

    Raised at worker startup, where the message surfaces verbatim to an operator
    in a launchd/tmux log, so — like :class:`~sax_platform.config.ForgeEnvError`
    — it is written to be complete and actionable: which chain, which revision
    the database is at, which revision the code expects, which database (with
    the password masked), and what to do next.
    """


# ---------------------------------------------------------------------------
# Pure core: the verdicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class AtHead:
    """The database is stamped with exactly the revision the code expects."""

    revision: str
    kind: Literal["at-head"] = "at-head"


@dataclass(frozen=True, slots=True, kw_only=True)
class Ahead:
    """The database carries a revision this code's chain has never heard of.

    The expand/contract window: a change was applied before the code that uses
    it deployed. Allowed — the caller logs a warning and proceeds.
    """

    revision: str
    head: str
    kind: Literal["ahead"] = "ahead"


@dataclass(frozen=True, slots=True, kw_only=True)
class Behind:
    """The database is at a known revision of this chain, but not its head."""

    revision: str
    head: str
    kind: Literal["behind"] = "behind"


@dataclass(frozen=True, slots=True, kw_only=True)
class Uninitialized:
    """No revision is stamped: the version table is missing, or it is empty."""

    head: str
    table_present: bool
    kind: Literal["uninitialized"] = "uninitialized"


@dataclass(frozen=True, slots=True, kw_only=True)
class AmbiguousStamp:
    """The version table holds more than one row — a branched/merged state."""

    revisions: tuple[str, ...]
    head: str
    kind: Literal["ambiguous-stamp"] = "ambiguous-stamp"


@dataclass(frozen=True, slots=True, kw_only=True)
class BrokenChain:
    """The code's own chain does not resolve to exactly one head.

    Zero heads (an empty ``versions/``) or several (an unmerged branch): there is
    no single revision to compare against, so no verdict about the database can
    be trusted and the check fails closed on the code, not the data.
    """

    heads: tuple[str, ...]
    kind: Literal["broken-chain"] = "broken-chain"


type SchemaVerdict = AtHead | Ahead | Behind | Uninitialized | AmbiguousStamp | BrokenChain


def classify_schema_version(
    *,
    heads: Sequence[str],
    known_revisions: Collection[str],
    table_present: bool,
    stamped: Sequence[str],
) -> SchemaVerdict:
    """Decide what the stamped revision means for this chain (pure).

    ``heads`` and ``known_revisions`` describe the chain in the deployed code
    (its head, and every revision it can name); ``table_present`` and ``stamped``
    describe the database (whether the version table exists, and the rows it
    holds). Nothing here reads a file or a database — the shell supplies both
    sides, so every branch is testable by table.
    """
    if len(heads) != 1:
        return BrokenChain(heads=tuple(heads))
    head = heads[0]
    if not stamped:
        return Uninitialized(head=head, table_present=table_present)
    if len(stamped) > 1:
        return AmbiguousStamp(revisions=tuple(stamped), head=head)
    revision = stamped[0]
    if revision == head:
        return AtHead(revision=revision)
    if revision in known_revisions:
        return Behind(revision=revision, head=head)
    return Ahead(revision=revision, head=head)


# ---------------------------------------------------------------------------
# Pure core: rendering a verdict for an operator
# ---------------------------------------------------------------------------

#: The canonical process doc in the operator's repo (agreed on sax-datastores
#: issue #2). Named in every failure message so the fix needs no institutional
#: memory.
_PROCESS_DOC: Final = "sax-datastores/docs/schema-changes.md"


def _fix_text(migrate_command: str) -> str:
    """The two-lane remediation sentence shared by every failure message."""
    return (
        f"Fix: dev/test is self-service — apply the chain with `{migrate_command}`. "
        f"Production schema changes go through the sax-datastores change-request "
        f"process ({_PROCESS_DOC}): forge commits the revision, the administrator "
        f"applies it. A worker never applies DDL."
    )


def describe_verdict(
    verdict: SchemaVerdict,
    *,
    version_table: str,
    target: str,
    migrate_command: str,
) -> str:
    """Render a verdict as one operator-facing line (pure).

    ``target`` is the credential-free database description (the caller masks the
    URL before calling); ``migrate_command`` is the owning product's apply
    command, e.g. ``forge migrate``. Success verdicts render a log line, failure
    verdicts render the :class:`SchemaVersionError` message.
    """
    fix = _fix_text(migrate_command)
    match verdict:
        case AtHead(revision=revision):
            return f"Schema verified: {version_table} at {revision} (head) on {target}."
        case Ahead(revision=revision, head=head):
            return (
                f"Schema AHEAD of deployed code: {version_table} is at {revision} on "
                f"{target}, which this code's chain does not know (its head is {head}). "
                f"Proceeding — expected during an expand/contract window, where the "
                f"schema change lands before the code that uses it. If no change is in "
                f"flight, this deployment is running stale code."
            )
        case Behind(revision=revision, head=head):
            return (
                f"Schema behind: {version_table} is at {revision} on {target}, but this "
                f"code expects head {head}. The database has not had this chain's "
                f"pending revision(s) applied, so refusing to start. {fix}"
            )
        case Uninitialized(head=head, table_present=table_present):
            state = (
                f"{version_table} exists but is empty"
                if table_present
                else f"{version_table} does not exist"
            )
            return (
                f"Schema not initialized: {state} on {target}, so no revision is "
                f"stamped and this code's chain (head {head}) has never been applied "
                f"there. Refusing to start. {fix}"
            )
        case AmbiguousStamp(revisions=revisions, head=head):
            stamped = ", ".join(revisions)
            return (
                f"Schema stamp is ambiguous: {version_table} on {target} holds more "
                f"than one revision ({stamped}) where exactly one is expected (head "
                f"{head}). The chain was left branched or half-merged; refusing to "
                f"start. {fix}"
            )
        case BrokenChain(heads=chain_heads):
            found = ", ".join(chain_heads) if chain_heads else "none"
            return (
                f"Migration chain is broken in the deployed code: {version_table} "
                f"resolves to {len(chain_heads)} heads ({found}) where exactly one is "
                f"expected. Nothing can be verified against {target} until the chain is "
                f"repaired (merge the branch, or restore the missing revisions)."
            )
        case _ as unreachable:  # pragma: no cover - exhaustiveness is checked statically
            assert_never(unreachable)


def version_table_select(
    version_table: str, version_table_schema: str | None = None
) -> Select[tuple[str]]:
    """Build the ``SELECT version_num FROM <schema.>version_table`` statement (pure).

    Constructed through SQLAlchemy rather than string-formatted SQL so the table
    and schema names are quoted by the dialect. Both names come from code, never
    from user input, but an unquoted identifier is a habit worth not forming.
    """
    column = sa.Column("version_num", sa.String)
    sa.Table(version_table, sa.MetaData(schema=version_table_schema), column)
    return sa.select(column)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def _load_chain(script_location: str) -> tuple[tuple[str, ...], frozenset[str]]:
    """Read the chain in ``script_location``: its heads and every known revision.

    Mirrors :func:`sax_platform.db.migrations.run_migrations`' Alembic setup (an
    ``alembic.ini`` beside the ``versions/`` directory, with ``script_location``
    overridden programmatically), so verify and apply always read the same chain.
    The alembic import is function-local for the same reason it is there.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    script_dir = Path(script_location)
    cfg = Config(str(script_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(script_dir))

    script = ScriptDirectory.from_config(cfg)
    heads = tuple(script.get_heads())
    if not heads:
        return ((), frozenset())
    return heads, frozenset(revision.revision for revision in script.walk_revisions())


def _read_stamped_revisions(
    url: str, *, version_table: str, version_table_schema: str | None
) -> tuple[bool, tuple[str, ...]]:
    """Read the version table: ``(table_present, stamped_revisions)``.

    Existence is checked through the dialect's inspector rather than by catching
    a failed ``SELECT``, so "the table is not there yet" stays a value the pure
    classifier handles and never an exception to disambiguate. The engine is
    short-lived and unpooled: this runs once, at process start, before the real
    store engine exists.
    """
    engine = sa.create_engine(url, poolclass=sa.pool.NullPool)
    try:
        with engine.connect() as conn:
            if not sa.inspect(conn).has_table(version_table, schema=version_table_schema):
                return False, ()
            rows = conn.execute(version_table_select(version_table, version_table_schema)).scalars()
            return True, tuple(str(row) for row in rows)
    finally:
        engine.dispose()


def verify_schema_version(
    url: str,
    *,
    version_table: str,
    script_location: str,
    version_table_schema: str | None = None,
    migrate_command: str = "alembic upgrade head",
) -> str:
    """Verify the deployed schema for one Alembic chain, or raise.

    Returns the revision the database is stamped with when it is safe to run —
    at head, or ahead of this code (the expand/contract window, which logs a
    warning). Every other state raises :class:`SchemaVersionError` with a
    message naming the chain, both revisions, the masked database, and the fix.

    Applies no DDL and takes no lock: it opens one connection, reads one table,
    and closes it. ``migrate_command`` names the owning product's self-service
    apply command (``forge migrate``, ``ocr migrate``, ``pbook migrate``) so the
    dev-lane fix in the message is exact.
    """
    heads, known_revisions = _load_chain(script_location)
    table_present, stamped = _read_stamped_revisions(
        url, version_table=version_table, version_table_schema=version_table_schema
    )
    verdict = classify_schema_version(
        heads=heads,
        known_revisions=known_revisions,
        table_present=table_present,
        stamped=stamped,
    )
    message = describe_verdict(
        verdict,
        version_table=version_table,
        target=make_url(url).render_as_string(hide_password=True),
        migrate_command=migrate_command,
    )

    match verdict:
        case AtHead(revision=revision):
            return revision
        case Ahead(revision=revision):
            logger.warning("%s", message)
            return revision
        case Behind() | Uninitialized() | AmbiguousStamp() | BrokenChain():
            raise SchemaVersionError(message)
        case _ as unreachable:  # pragma: no cover - exhaustiveness is checked statically
            assert_never(unreachable)
