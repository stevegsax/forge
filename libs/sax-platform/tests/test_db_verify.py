"""Tests for the verify-only schema check.

The classification is pure and gets an exhaustive table here (at head, behind,
uninitialized both ways, ahead, ambiguous stamp, broken chain), plus message
assertions on what an operator actually needs to read. ``verify_schema_version``
itself — the shell — is then run end to end against a real throwaway SQLite
database and a real (tiny) Alembic script directory, so the ``ScriptDirectory``
head computation and the version-table read are exercised for real rather than
mocked.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from sax_platform.db.verify import (
    Ahead,
    AmbiguousStamp,
    AtHead,
    Behind,
    BrokenChain,
    SchemaVerdict,
    SchemaVersionError,
    Uninitialized,
    classify_schema_version,
    describe_verdict,
    verify_schema_version,
    version_table_select,
)

if TYPE_CHECKING:
    from pathlib import Path

VERSION_TABLE = "alembic_version_fixture"


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------


class TestClassifySchemaVersion:
    @pytest.mark.parametrize(
        ("table_present", "stamped", "expected"),
        [
            (True, ("0002",), AtHead(revision="0002")),
            (True, ("0001",), Behind(revision="0001", head="0002")),
            (True, ("ffff",), Ahead(revision="ffff", head="0002")),
            (True, (), Uninitialized(head="0002", table_present=True)),
            (False, (), Uninitialized(head="0002", table_present=False)),
            (
                True,
                ("0001", "0002"),
                AmbiguousStamp(revisions=("0001", "0002"), head="0002"),
            ),
        ],
    )
    def test_verdict_for_each_database_state(
        self, table_present: bool, stamped: tuple[str, ...], expected: SchemaVerdict
    ) -> None:
        assert (
            classify_schema_version(
                heads=("0002",),
                known_revisions=frozenset({"0001", "0002"}),
                table_present=table_present,
                stamped=stamped,
            )
            == expected
        )

    @pytest.mark.parametrize("heads", [(), ("0002", "0003")])
    def test_chain_without_exactly_one_head_is_broken(self, heads: tuple[str, ...]) -> None:
        """A broken chain fails on the code, before any claim about the data.

        Zero heads (empty ``versions/``) or two (an unmerged branch): there is no
        single revision to compare against, so the stamped value is irrelevant.
        """
        assert classify_schema_version(
            heads=heads,
            known_revisions=frozenset(heads),
            table_present=True,
            stamped=("0002",),
        ) == BrokenChain(heads=heads)

    def test_ahead_does_not_depend_on_revision_ordering(self) -> None:
        """ "Ahead" is "unknown to this chain", not "lexically greater".

        Revision ids are opaque hashes in a real chain, so the only usable test
        is membership in what the deployed code can name.
        """
        verdict = classify_schema_version(
            heads=("zzzz",),
            known_revisions=frozenset({"zzzz"}),
            table_present=True,
            stamped=("aaaa",),
        )
        assert verdict == Ahead(revision="aaaa", head="zzzz")


class TestDescribeVerdict:
    def _describe(self, verdict: SchemaVerdict) -> str:
        return describe_verdict(
            verdict,
            version_table=VERSION_TABLE,
            target="postgresql://localhost:5442/forge_prod",
            migrate_command="forge migrate",
        )

    def test_behind_names_both_revisions_the_table_the_target_and_both_lanes(self) -> None:
        message = self._describe(Behind(revision="0001", head="0002"))
        assert "0001" in message
        assert "0002" in message
        assert VERSION_TABLE in message
        assert "localhost:5442/forge_prod" in message
        # Both remediation lanes, so the reader never has to guess which applies.
        assert "forge migrate" in message
        assert "sax-datastores/docs/schema-changes.md" in message

    def test_uninitialized_distinguishes_missing_table_from_empty_table(self) -> None:
        missing = self._describe(Uninitialized(head="0002", table_present=False))
        empty = self._describe(Uninitialized(head="0002", table_present=True))
        assert "does not exist" in missing
        assert "exists but is empty" in empty

    def test_ahead_explains_the_expand_contract_window_and_does_not_read_as_failure(
        self,
    ) -> None:
        message = self._describe(Ahead(revision="ffff", head="0002"))
        assert "expand/contract" in message
        assert "Proceeding" in message

    def test_ambiguous_stamp_lists_every_stamped_revision(self) -> None:
        message = self._describe(AmbiguousStamp(revisions=("0001", "0002"), head="0002"))
        assert "0001, 0002" in message

    def test_broken_chain_blames_the_code_not_the_database(self) -> None:
        message = self._describe(BrokenChain(heads=("0002", "0003")))
        assert "broken in the deployed code" in message
        assert "2 heads" in message
        # No migrate/change-request advice: applying a chain cannot fix a chain
        # that does not resolve.
        assert "forge migrate" not in message

    def test_at_head_renders_a_log_line(self) -> None:
        message = self._describe(AtHead(revision="0002"))
        assert "Schema verified" in message
        assert "0002" in message


class TestVersionTableSelect:
    def test_unqualified_select_names_the_table(self) -> None:
        compiled = str(version_table_select("alembic_version_forge"))
        assert "FROM alembic_version_forge" in compiled
        assert "version_num" in compiled

    def test_schema_qualified_select_is_prefixed(self) -> None:
        """pbook's chain lives in its own schema — the read must follow it there."""
        compiled = str(version_table_select("pbk_alembic_version", "pbook"))
        assert "FROM pbook.pbk_alembic_version" in compiled

    def test_identifiers_needing_quotes_are_quoted(self) -> None:
        compiled = str(version_table_select("select", "order"))
        assert '"order"."select"' in compiled


# ---------------------------------------------------------------------------
# Imperative shell: a real SQLite database and a real Alembic script directory
# ---------------------------------------------------------------------------

_ALEMBIC_INI = """\
[alembic]
script_location = %(here)s
sqlalchemy.url =
"""

_REVISION = '''\
"""fixture revision {revision}"""

revision = "{revision}"
down_revision = {down_revision}
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
'''


def _make_chain(tmp_path: Path, *, revisions: tuple[tuple[str, str | None], ...]) -> str:
    """Scaffold a throwaway Alembic script directory and return its path.

    ``revisions`` is ``(revision, down_revision)`` pairs; two revisions with no
    ``down_revision`` produce the two-head (broken) chain.
    """
    script_dir = tmp_path / "alembic"
    (script_dir / "versions").mkdir(parents=True)
    (script_dir / "alembic.ini").write_text(_ALEMBIC_INI)
    for revision, down_revision in revisions:
        down = "None" if down_revision is None else f'"{down_revision}"'
        (script_dir / "versions" / f"{revision}_fixture.py").write_text(
            _REVISION.format(revision=revision, down_revision=down)
        )
    return str(script_dir)


@pytest.fixture
def chain(tmp_path: Path) -> str:
    """A linear two-revision chain: 0001 -> 0002 (head)."""
    return _make_chain(tmp_path, revisions=(("0001", None), ("0002", "0001")))


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'verify.db'}"


def _stamp(url: str, *revisions: str) -> None:
    """Create the version table and write the given revision rows."""
    engine = sa.create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(sa.text(f"CREATE TABLE {VERSION_TABLE} (version_num VARCHAR(32))"))
            for revision in revisions:
                conn.execute(
                    sa.text(f"INSERT INTO {VERSION_TABLE} (version_num) VALUES (:v)"),
                    {"v": revision},
                )
    finally:
        engine.dispose()


class TestVerifySchemaVersion:
    def test_at_head_returns_the_stamped_revision(self, chain: str, db_url: str) -> None:
        _stamp(db_url, "0002")

        assert (
            verify_schema_version(db_url, version_table=VERSION_TABLE, script_location=chain)
            == "0002"
        )

    def test_behind_raises_with_the_named_error(self, chain: str, db_url: str) -> None:
        _stamp(db_url, "0001")

        with pytest.raises(SchemaVersionError, match="Schema behind") as excinfo:
            verify_schema_version(
                db_url,
                version_table=VERSION_TABLE,
                script_location=chain,
                migrate_command="forge migrate",
            )

        message = str(excinfo.value)
        assert "0001" in message
        assert "0002" in message
        assert "forge migrate" in message

    def test_missing_version_table_raises(self, chain: str, db_url: str) -> None:
        # The database file exists but the chain was never applied.
        sa.create_engine(db_url).connect().close()

        with pytest.raises(SchemaVersionError, match="Schema not initialized"):
            verify_schema_version(db_url, version_table=VERSION_TABLE, script_location=chain)

    def test_empty_version_table_raises(self, chain: str, db_url: str) -> None:
        _stamp(db_url)

        with pytest.raises(SchemaVersionError, match="exists but is empty"):
            verify_schema_version(db_url, version_table=VERSION_TABLE, script_location=chain)

    def test_ahead_is_allowed_and_warns(
        self, chain: str, db_url: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The expand/contract window: schema applied, code not yet deployed."""
        _stamp(db_url, "0003")

        with caplog.at_level(logging.WARNING, logger="sax_platform.db.verify"):
            revision = verify_schema_version(
                db_url, version_table=VERSION_TABLE, script_location=chain
            )

        assert revision == "0003"
        assert "AHEAD of deployed code" in caplog.text

    def test_multiple_stamped_rows_raise(self, chain: str, db_url: str) -> None:
        _stamp(db_url, "0001", "0002")

        with pytest.raises(SchemaVersionError, match="ambiguous"):
            verify_schema_version(db_url, version_table=VERSION_TABLE, script_location=chain)

    def test_two_headed_chain_raises_even_when_the_database_is_fine(
        self, tmp_path: Path, db_url: str
    ) -> None:
        branched = _make_chain(tmp_path, revisions=(("0002", None), ("0003", None)))
        _stamp(db_url, "0002")

        with pytest.raises(SchemaVersionError, match="Migration chain is broken"):
            verify_schema_version(db_url, version_table=VERSION_TABLE, script_location=branched)

    def test_empty_versions_directory_raises(self, tmp_path: Path, db_url: str) -> None:
        """No revisions at all: there is no head to compare against."""
        empty_chain = _make_chain(tmp_path, revisions=())
        _stamp(db_url, "0002")

        with pytest.raises(SchemaVersionError, match="resolves to 0 heads"):
            verify_schema_version(db_url, version_table=VERSION_TABLE, script_location=empty_chain)

    def test_names_the_database_in_the_failure_message(self, chain: str, tmp_path: Path) -> None:
        """The message is read off a terminal, so it must say which database."""
        url = f"sqlite:///{tmp_path / 'named.db'}"
        _stamp(url, "0001")

        with pytest.raises(SchemaVersionError) as excinfo:
            verify_schema_version(url, version_table=VERSION_TABLE, script_location=chain)

        assert "named.db" in str(excinfo.value)

    def test_never_echoes_the_password(self, chain: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A startup failure is pasted into tickets and chat — it must not carry
        the credential. Proving this needs a URL with a password, so the version
        table read (the only part that would need a live server) is substituted;
        everything downstream of it is the real code path."""
        monkeypatch.setattr(
            "sax_platform.db.verify._read_stamped_revisions",
            lambda url, *, version_table, version_table_schema: (True, ("0001",)),
        )

        with pytest.raises(SchemaVersionError) as excinfo:
            verify_schema_version(
                "postgresql+psycopg://forge_prod:sup3rs3cret@127.0.0.1:5442/forge_prod",
                version_table=VERSION_TABLE,
                script_location=chain,
            )

        message = str(excinfo.value)
        assert "sup3rs3cret" not in message
        assert "127.0.0.1:5442/forge_prod" in message

    def test_applies_no_ddl(self, chain: str, db_url: str) -> None:
        """Verify must never create or alter anything — that is the whole point."""
        sa.create_engine(db_url).connect().close()

        with pytest.raises(SchemaVersionError):
            verify_schema_version(db_url, version_table=VERSION_TABLE, script_location=chain)

        engine = sa.create_engine(db_url)
        try:
            assert sa.inspect(engine).get_table_names() == []
        finally:
            engine.dispose()
