"""Tests for the schema-change request generator.

The pure core (stripping, classification, id allocation, rendering) is tested
exhaustively by table — it is where every decision the generator makes lives.
The shell is tested against a synthetic Alembic chain written into ``tmp_path``
(``chain`` fixture below) rather than any app's real chain: this package cannot
import forge/ocr/pbook, and a chain built here can be given the shapes the real
ones do not have — a branched head, a merge revision — that the generator has
to refuse. Squawk is never actually invoked: ``subprocess.run`` is patched, and
the real invocation is covered by ``make lint-sql`` in CI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from sax_platform.db.change_request import (
    SQUAWK_PACKAGE,
    SQUAWK_VERSION,
    ChainSpec,
    ChangeRequestError,
    LintReport,
    NotLinted,
    Phase,
    build_phases,
    classify_phase,
    describe_generated_request,
    find_repo_root,
    generate_change_request,
    next_request_id,
    render_request,
    resumability_warning,
    run_squawk,
    squawk_linter,
    strip_transaction_wrappers,
    validate_title,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# A synthetic Alembic chain: 001 -> 002 (transactional) -> 003 (concurrent)
# ---------------------------------------------------------------------------

_ALEMBIC_INI = """\
[alembic]
script_location = %(here)s
sqlalchemy.url =
"""

_ENV_PY = '''\
"""Offline-only env.py for the test chain (no database is ever opened)."""

from alembic import context

context.configure(
    url=context.config.get_main_option("sqlalchemy.url"),
    literal_binds=True,
    dialect_opts={"paramstyle": "named"},
    version_table="alembic_version_test",
)
with context.begin_transaction():
    context.run_migrations()
'''

_REVISION_TEMPLATE = '''\
"""Test revision {revision}."""

from alembic import op

revision = "{revision}"
down_revision = {down_revision}
branch_labels = None
depends_on = None


def upgrade() -> None:
{body}


def downgrade() -> None:
    pass
'''

_TRANSACTIONAL_BODY = '    op.execute("CREATE TABLE widgets (id integer)")'

_CONCURRENT_BODY = """\
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_widgets_id ON widgets (id)"
        )"""

_UNRESUMABLE_BODY = """\
    with op.get_context().autocommit_block():
        op.execute("CREATE INDEX CONCURRENTLY ix_widgets_id2 ON widgets (id)")"""


def _write_chain(root: Path, revisions: Sequence[tuple[str, str | None, str]]) -> Path:
    """Write a runnable offline Alembic chain under ``root`` and return its path."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "alembic.ini").write_text(_ALEMBIC_INI)
    (root / "env.py").write_text(_ENV_PY)
    versions = root / "versions"
    versions.mkdir(exist_ok=True)
    for revision, down_revision, body in revisions:
        down = "None" if down_revision is None else f'"{down_revision}"'
        (versions / f"{revision}_test.py").write_text(
            _REVISION_TEMPLATE.format(revision=revision, down_revision=down, body=body)
        )
    return root


@pytest.fixture
def chain(tmp_path: Path) -> Path:
    """The default three-revision chain: 001 -> 002 -> 003 (003 is concurrent)."""
    return _write_chain(
        tmp_path / "chain",
        [
            ("001", None, _TRANSACTIONAL_BODY),
            ("002", "001", _TRANSACTIONAL_BODY),
            ("003", "002", _CONCURRENT_BODY),
        ],
    )


@pytest.fixture
def spec(chain: Path) -> ChainSpec:
    return ChainSpec(
        product="widget",
        database="widget_prod",
        schema="public",
        version_table="alembic_version_test",
        script_location=chain,
    )


def _phase(sql: str, *, number: int = 1) -> Phase:
    return Phase(number=number, from_revision="001", to_revision="002", sql=sql)


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------


class TestStripTransactionWrappers:
    def test_removes_wrapper_lines_and_collapses_the_blanks_they_leave(self) -> None:
        raw = (
            "BEGIN;\n\n-- Running upgrade 003 -> 004\n\nCOMMIT;\n\n"
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix ON t (c);\n\n"
            "BEGIN;\n\nUPDATE alembic_version_test SET version_num='004';\n\nCOMMIT;\n\n"
        )
        assert strip_transaction_wrappers(raw) == (
            "\n-- Running upgrade 003 -> 004\n\n"
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix ON t (c);\n\n"
            "UPDATE alembic_version_test SET version_num='004';\n\n"
        )

    def test_keeps_the_version_table_stamp(self) -> None:
        raw = "BEGIN;\n\nUPDATE alembic_version_test SET version_num='002';\n\nCOMMIT;\n\n"
        assert "UPDATE alembic_version_test SET version_num='002';" in strip_transaction_wrappers(
            raw
        )

    def test_keeps_statements_that_merely_mention_the_keywords(self) -> None:
        raw = "INSERT INTO log (msg) VALUES ('BEGIN;');\n"
        assert strip_transaction_wrappers(raw) == raw

    def test_ignores_leading_and_trailing_whitespace_on_a_wrapper_line(self) -> None:
        assert strip_transaction_wrappers("  BEGIN;  \nSELECT 1;\n") == "SELECT 1;\n"

    def test_empty_input_stays_empty(self) -> None:
        assert strip_transaction_wrappers("") == ""

    def test_wrappers_only_input_produces_no_artifact(self) -> None:
        assert strip_transaction_wrappers("BEGIN;\n\nCOMMIT;\n\n") == ""

    def test_output_ends_with_exactly_one_newline_after_the_last_line(self) -> None:
        assert strip_transaction_wrappers("SELECT 1;") == "SELECT 1;\n"


class TestClassifyPhase:
    def test_plain_ddl_is_transactional_with_the_default_timeout(self) -> None:
        classification = classify_phase("CREATE TABLE widgets (id integer);\n")
        assert classification.transactional is True
        assert classification.statement_timeout == "default"
        assert classification.resumable is True

    def test_concurrent_ddl_is_non_transactional_with_timeout_zero(self) -> None:
        classification = classify_phase("CREATE INDEX CONCURRENTLY IF NOT EXISTS ix ON t (c);\n")
        assert classification.transactional is False
        assert classification.statement_timeout == "0"
        assert classification.resumable is True

    def test_concurrent_drop_with_if_exists_is_resumable(self) -> None:
        assert classify_phase("DROP INDEX CONCURRENTLY IF EXISTS ix;\n").resumable is True

    def test_concurrent_without_a_guard_is_not_resumable(self) -> None:
        assert classify_phase("CREATE INDEX CONCURRENTLY ix ON t (c);\n").resumable is False

    def test_classification_is_case_insensitive(self) -> None:
        classification = classify_phase("create index concurrently if not exists ix on t (c);\n")
        assert classification.transactional is False
        assert classification.resumable is True

    def test_a_guarded_neighbour_does_not_excuse_an_unguarded_statement(self) -> None:
        sql = (
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_a ON t (a);\n\n"
            "CREATE INDEX CONCURRENTLY ix_b ON t (b);\n"
        )
        assert classify_phase(sql).resumable is False

    def test_a_transactional_statement_needs_no_guard(self) -> None:
        sql = (
            "CREATE TABLE widgets (id integer);\n\n"
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix ON widgets (id);\n"
        )
        assert classify_phase(sql).resumable is True


class TestResumabilityWarning:
    def test_transactional_phase_has_no_warning(self) -> None:
        assert resumability_warning(_phase("CREATE TABLE widgets (id integer);\n")) is None

    def test_guarded_concurrent_phase_has_no_warning(self) -> None:
        assert (
            resumability_warning(_phase("CREATE INDEX CONCURRENTLY IF NOT EXISTS ix ON t (c);\n"))
            is None
        )

    def test_unguarded_concurrent_phase_names_the_file_and_the_fix(self) -> None:
        warning = resumability_warning(_phase("CREATE INDEX CONCURRENTLY ix ON t (c);\n"))
        assert warning is not None
        assert "change-1.sql" in warning
        assert "IF NOT EXISTS" in warning


class TestNextRequestId:
    @pytest.mark.parametrize(
        ("existing", "expected"),
        [
            ((), "0001"),
            (("0001-first",), "0002"),
            (("0001-first", "0002-second"), "0003"),
            # A gap is never reused: a submitted id is referenced by an issue.
            (("0001-first", "0003-third"), "0004"),
            # Unrelated / malformed names are ignored.
            (("README.md", "draft", "001-too-short"), "0001"),
            (("0009-ninth", "notes"), "0010"),
            (("0099-last",), "0100"),
        ],
    )
    def test_allocation(self, existing: tuple[str, ...], expected: str) -> None:
        assert next_request_id(existing) == expected


class TestValidateTitle:
    @pytest.mark.parametrize(
        "title",
        ["interactions-created-at-index", "a", "add-2fa", "x1-y2-z3"],
    )
    def test_accepts_kebab_case(self, title: str) -> None:
        assert validate_title(title) == title

    @pytest.mark.parametrize(
        "title",
        ["Interactions-Index", "two words", "trailing-", "-leading", "double--hyphen", ""],
    )
    def test_rejects_anything_else(self, title: str) -> None:
        with pytest.raises(ChangeRequestError, match="kebab-case"):
            validate_title(title)


_RENDER_CHAIN = ChainSpec(
    product="widget",
    database="widget_prod",
    schema="public",
    version_table="alembic_version_test",
    script_location=Path("/nowhere"),
)

_RENDER_PHASES = (
    Phase(
        number=1,
        from_revision="001",
        to_revision="002",
        sql="CREATE TABLE widgets (id integer);\n",
    ),
    Phase(
        number=2,
        from_revision="002",
        to_revision="003",
        sql="CREATE INDEX CONCURRENTLY IF NOT EXISTS ix ON widgets (id);\n",
    ),
)


class TestRenderRequest:
    def _render(
        self,
        *,
        phases: Sequence[Phase] = _RENDER_PHASES,
        lint: LintReport | NotLinted | None = None,
        warnings: Sequence[str] = (),
    ) -> str:
        return render_request(
            title="add-widget-index",
            request_id="0007",
            chain=_RENDER_CHAIN,
            phases=phases,
            lint=lint or LintReport(version=SQUAWK_VERSION, output="", clean=True),
            warnings=warnings,
        )

    def test_fills_in_the_field_table(self) -> None:
        rendered = self._render()
        assert "# Change request: `add-widget-index`" in rendered
        assert "| Product | `widget` |" in rendered
        assert "| Id | `0007` |" in rendered
        assert "| Database | `widget_prod` (prod) |" in rendered
        assert "| Schema | `public` |" in rendered
        assert "| Version table | `alembic_version_test` |" in rendered
        assert "| From → to | `001` → `003` (linear; single head verified) |" in rendered
        assert "| Tier claimed | 1 or 2 (operator confirms at intake) |" in rendered
        assert "| Stacks | `prod` only — dev/test are self-service |" in rendered

    def test_renders_one_phases_row_per_phase_with_its_timeout(self) -> None:
        rendered = self._render()
        assert "| `change-1.sql` | `001` → `002` | yes | default |" in rendered
        assert "| `change-2.sql` | `002` → `003` | no (concurrent index) | 0 |" in rendered

    def test_keeps_the_template_guidance_for_the_judgment_sections(self) -> None:
        rendered = self._render()
        for heading in (
            "## What and why",
            "## Risk notes (evidence, not intentions)",
            "## Compatibility (expand/contract)",
            "## Backfill",
            "## Rollback stance",
            "## Lint",
            "## Dev evidence",
        ):
            assert heading in rendered
        assert "<!-- One logical change" in rendered

    def test_clean_lint_states_the_version_and_no_findings(self) -> None:
        rendered = self._render()
        assert f"Squawk `{SQUAWK_VERSION}`" in rendered
        assert "**clean, no findings.**" in rendered

    def test_findings_are_quoted_verbatim_and_flagged(self) -> None:
        rendered = self._render(
            lint=LintReport(
                version=SQUAWK_VERSION,
                output="change-1.sql:0:0: warning: prefer-robust-stmts Missing `IF NOT EXISTS`",
                clean=False,
            )
        )
        assert "**not clean.**" in rendered
        assert "prefer-robust-stmts" in rendered
        assert "```text" in rendered

    def test_skipped_lint_says_so_loudly(self) -> None:
        rendered = self._render(lint=NotLinted())
        assert "**NOT LINTED — run `make lint-sql` before committing.**" in rendered

    def test_warnings_are_written_into_the_request(self) -> None:
        rendered = self._render(warnings=("change-2.sql is not resumable",))
        assert "**Resumability warning (fix before submitting):** change-2.sql" in rendered

    def test_no_phases_is_refused(self) -> None:
        with pytest.raises(ChangeRequestError, match="no phases"):
            self._render(phases=())


class TestDescribeGeneratedRequest:
    def test_summary_names_the_files_and_the_lint_verdict(self, tmp_path: Path) -> None:
        request = generate_change_request(
            chain=ChainSpec(
                product="widget",
                database="widget_prod",
                schema="public",
                version_table="alembic_version_test",
                script_location=_write_chain(tmp_path / "c", [("001", None, _TRANSACTIONAL_BODY)]),
            ),
            output_root=tmp_path / "out",
            from_revision="base",
            to_revision=None,
            title="first",
            linter=None,
        )
        summary = describe_generated_request(request)
        assert "0001" in summary
        assert "change-1.sql" in summary
        assert "request.md" in summary
        assert "skipped (--no-lint)" in summary

    def test_summary_reports_a_clean_lint(self, spec: ChainSpec, tmp_path: Path) -> None:
        request = generate_change_request(
            chain=spec,
            output_root=tmp_path / "out",
            from_revision="001",
            to_revision="002",
            title="second",
            linter=lambda _paths: LintReport(version=SQUAWK_VERSION, output="", clean=True),
        )
        assert f"squawk {SQUAWK_VERSION}: clean" in describe_generated_request(request)

    def test_summary_shouts_about_findings(self, spec: ChainSpec, tmp_path: Path) -> None:
        request = generate_change_request(
            chain=spec,
            output_root=tmp_path / "out",
            from_revision="001",
            to_revision="002",
            title="third",
            linter=lambda _paths: LintReport(version=SQUAWK_VERSION, output="x", clean=False),
        )
        assert "FINDINGS" in describe_generated_request(request)


# ---------------------------------------------------------------------------
# Shell: the revision walk
# ---------------------------------------------------------------------------


class TestBuildPhases:
    def test_one_phase_per_revision_step_in_apply_order(self, chain: Path) -> None:
        phases = build_phases(script_location=chain, from_revision="001", to_revision="003")
        assert [(p.number, p.from_revision, p.to_revision) for p in phases] == [
            (1, "001", "002"),
            (2, "002", "003"),
        ]
        assert [p.file_name for p in phases] == ["change-1.sql", "change-2.sql"]

    def test_to_defaults_to_the_single_head(self, chain: Path) -> None:
        phases = build_phases(script_location=chain, from_revision="002", to_revision=None)
        assert [p.to_revision for p in phases] == ["003"]

    def test_generated_sql_carries_no_transaction_control(self, chain: Path) -> None:
        phases = build_phases(script_location=chain, from_revision="002", to_revision="003")
        lines = phases[0].sql.splitlines()
        assert "BEGIN;" not in lines
        assert "COMMIT;" not in lines

    def test_generated_sql_keeps_its_own_stamp(self, chain: Path) -> None:
        phases = build_phases(script_location=chain, from_revision="001", to_revision="002")
        assert "UPDATE alembic_version_test SET version_num='002'" in phases[0].sql

    def test_unknown_from_revision_is_refused_by_name(self, chain: Path) -> None:
        with pytest.raises(ChangeRequestError, match="Cannot walk nope -> 003"):
            build_phases(script_location=chain, from_revision="nope", to_revision="003")

    def test_unknown_to_revision_is_refused(self, chain: Path) -> None:
        with pytest.raises(ChangeRequestError, match="Cannot walk 001 -> nope"):
            build_phases(script_location=chain, from_revision="001", to_revision="nope")

    def test_from_that_is_not_an_ancestor_is_refused(self, chain: Path) -> None:
        with pytest.raises(ChangeRequestError, match="not an ancestor"):
            build_phases(script_location=chain, from_revision="003", to_revision="001")

    def test_empty_range_is_refused(self, chain: Path) -> None:
        with pytest.raises(ChangeRequestError, match="Nothing to generate"):
            build_phases(script_location=chain, from_revision="003", to_revision="003")

    def test_a_branched_chain_is_refused_before_any_generation(self, tmp_path: Path) -> None:
        branched = _write_chain(
            tmp_path / "branched",
            [
                ("001", None, _TRANSACTIONAL_BODY),
                ("002", "001", _TRANSACTIONAL_BODY),
                ("003", "001", _TRANSACTIONAL_BODY),
            ],
        )
        with pytest.raises(ChangeRequestError, match="resolves to 2 heads"):
            build_phases(script_location=branched, from_revision="001", to_revision=None)

    def test_a_merge_revision_is_refused(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged"
        _write_chain(
            merged,
            [
                ("001", None, _TRANSACTIONAL_BODY),
                ("002", "001", _TRANSACTIONAL_BODY),
                ("003", "001", _TRANSACTIONAL_BODY),
            ],
        )
        (merged / "versions" / "004_test.py").write_text(
            _REVISION_TEMPLATE.format(
                revision="004", down_revision='("002", "003")', body="    pass"
            )
        )
        with pytest.raises(ChangeRequestError, match="is a merge"):
            build_phases(script_location=merged, from_revision="001", to_revision="004")


# ---------------------------------------------------------------------------
# Shell: Squawk
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestRunSquawk:
    def test_clean_run_reports_the_pinned_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_run(argv: list[str], **_kwargs: object) -> _FakeCompleted:
            captured.append(argv)
            return _FakeCompleted(0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        report = run_squawk([tmp_path / "change-1.sql"], config_path=tmp_path / ".squawk.toml")
        assert report.clean is True
        assert report.version == SQUAWK_VERSION
        assert captured[0][:3] == ["npx", "--yes", SQUAWK_PACKAGE]
        assert "--reporter" in captured[0]
        assert "gcc" in captured[0]

    def test_findings_are_a_result_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            subprocess, "run", lambda *_a, **_k: _FakeCompleted(1, stdout="a.sql:0:0: warning: x")
        )
        report = run_squawk([tmp_path / "a.sql"], config_path=tmp_path / ".squawk.toml")
        assert report.clean is False
        assert "warning" in report.output

    def test_a_crash_raises_rather_than_reading_as_clean(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _FakeCompleted(2, stderr="boom"))
        with pytest.raises(ChangeRequestError, match="exited 2"):
            run_squawk([tmp_path / "a.sql"], config_path=tmp_path / ".squawk.toml")

    def test_a_missing_npx_raises_with_the_no_lint_escape_hatch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def boom(*_args: object, **_kwargs: object) -> _FakeCompleted:
            raise OSError("npx: not found")

        monkeypatch.setattr(subprocess, "run", boom)
        with pytest.raises(ChangeRequestError, match="--no-lint"):
            run_squawk([tmp_path / "a.sql"], config_path=tmp_path / ".squawk.toml")

    def test_squawk_linter_binds_the_config_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_run(argv: list[str], **_kwargs: object) -> _FakeCompleted:
            captured.append(argv)
            return _FakeCompleted(0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        squawk_linter(tmp_path / "cfg.toml")([tmp_path / "a.sql"])
        assert str(tmp_path / "cfg.toml") in captured[0]


# ---------------------------------------------------------------------------
# Shell: the filesystem
# ---------------------------------------------------------------------------


class TestFindRepoRoot:
    def test_finds_a_git_directory(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_repo_root(nested) == tmp_path

    def test_finds_a_git_file_as_a_worktree_has(self, tmp_path: Path) -> None:
        (tmp_path / ".git").write_text("gitdir: /elsewhere\n")
        assert find_repo_root(tmp_path) == tmp_path

    def test_raises_when_there_is_no_repository(self, tmp_path: Path) -> None:
        nested = tmp_path / "no" / "repo"
        nested.mkdir(parents=True)
        with pytest.raises(ChangeRequestError, match="--output-root"):
            find_repo_root(nested)


class TestGenerateChangeRequest:
    def test_writes_sql_and_request_into_a_numbered_directory(
        self, spec: ChainSpec, tmp_path: Path
    ) -> None:
        out = tmp_path / "datastore-changes"
        request = generate_change_request(
            chain=spec,
            output_root=out,
            from_revision="001",
            to_revision="003",
            title="add-widget-index",
            linter=lambda _paths: LintReport(version=SQUAWK_VERSION, output="", clean=True),
        )
        assert request.directory == out / "0001-add-widget-index"
        assert [path.name for path in request.sql_files] == ["change-1.sql", "change-2.sql"]
        assert request.request_file.read_text().startswith("<!-- Schema change request")
        assert "CREATE TABLE widgets" in request.sql_files[0].read_text()

    def test_ids_advance_past_existing_requests(self, spec: ChainSpec, tmp_path: Path) -> None:
        out = tmp_path / "datastore-changes"
        (out / "0001-something").mkdir(parents=True)
        request = generate_change_request(
            chain=spec,
            output_root=out,
            from_revision="001",
            to_revision="002",
            title="next-one",
            linter=None,
        )
        assert request.request_id == "0002"

    def test_the_linter_sees_the_written_sql_files(self, spec: ChainSpec, tmp_path: Path) -> None:
        seen: list[tuple[Path, ...]] = []

        def linter(paths: Sequence[Path]) -> LintReport:
            seen.append(tuple(paths))
            return LintReport(version=SQUAWK_VERSION, output="", clean=True)

        generate_change_request(
            chain=spec,
            output_root=tmp_path / "out",
            from_revision="001",
            to_revision="002",
            title="linted",
            linter=linter,
        )
        assert len(seen) == 1
        assert all(path.exists() for path in seen[0])

    def test_no_linter_stamps_the_request_not_linted(self, spec: ChainSpec, tmp_path: Path) -> None:
        request = generate_change_request(
            chain=spec,
            output_root=tmp_path / "out",
            from_revision="001",
            to_revision="002",
            title="unlinted",
            linter=None,
        )
        assert isinstance(request.lint, NotLinted)
        assert "NOT LINTED" in request.request_file.read_text()

    def test_an_unresumable_phase_warns_and_lands_in_the_request(self, tmp_path: Path) -> None:
        chain = _write_chain(
            tmp_path / "risky",
            [("001", None, _TRANSACTIONAL_BODY), ("002", "001", _UNRESUMABLE_BODY)],
        )
        request = generate_change_request(
            chain=ChainSpec(
                product="widget",
                database="widget_prod",
                schema="public",
                version_table="alembic_version_test",
                script_location=chain,
            ),
            output_root=tmp_path / "out",
            from_revision="001",
            to_revision="002",
            title="risky-index",
            linter=None,
        )
        assert len(request.warnings) == 1
        assert "not resumable" in request.warnings[0]
        assert "Resumability warning" in request.request_file.read_text()

    def test_an_existing_path_is_never_overwritten(self, spec: ChainSpec, tmp_path: Path) -> None:
        # Id allocation only counts directories, so a stray *file* named like a
        # request is the reachable collision: the scan skips it, allocation
        # hands back 0001, and the target path is already taken.
        out = tmp_path / "out"
        out.mkdir()
        (out / "0001-collide").write_text("not a request directory\n")
        with pytest.raises(ChangeRequestError, match="refusing to overwrite"):
            generate_change_request(
                chain=spec,
                output_root=out,
                from_revision="001",
                to_revision="002",
                title="collide",
                linter=None,
            )

    def test_a_bad_title_writes_nothing(self, spec: ChainSpec, tmp_path: Path) -> None:
        out = tmp_path / "out"
        with pytest.raises(ChangeRequestError, match="kebab-case"):
            generate_change_request(
                chain=spec,
                output_root=out,
                from_revision="001",
                to_revision="002",
                title="Not Kebab",
                linter=None,
            )
        assert not out.exists()

    def test_a_bad_range_writes_nothing(self, spec: ChainSpec, tmp_path: Path) -> None:
        out = tmp_path / "out"
        with pytest.raises(ChangeRequestError, match="Cannot walk"):
            generate_change_request(
                chain=spec,
                output_root=out,
                from_revision="nope",
                to_revision="002",
                title="doomed",
                linter=None,
            )
        assert not out.exists()
