"""Generate a sax-datastores schema-change request from an Alembic chain.

The mechanization of the process agreed on sax-datastores issue #2 and written
down in ``sax-datastores/docs/schema-changes.md``: a consumer never runs DDL
against a production database. It authors an Alembic revision per phase,
generates the offline SQL for each phase, commits those artifacts under
``datastore-changes/<id>-<title>/`` beside a filled-in ``request.md`` — the
commit *is* the request — and opens an issue; the administrator applies it.

What this module automates, exactly:

* **Offline SQL per phase.** ``alembic upgrade <n-1>:<n> --sql`` against the
  postgres dialect, one file per revision step, numbered in apply order. The
  dummy URL supplies a dialect and nothing else: offline mode never connects.
* **Wrapper stripping.** The apply runbook owns transaction control, so the
  artifact must carry none — see :func:`strip_transaction_wrappers`. The
  per-phase version-table stamp stays: it is the resume marker.
* **Classification.** A phase containing ``CONCURRENTLY`` cannot run inside a
  transaction and is applied with ``statement_timeout`` 0; such a phase must
  also be resumable (``IF NOT EXISTS`` / ``IF EXISTS``), which is a warning
  here rather than a failure because only the author can fix the revision.
* **The request skeleton.** The operator's template with every mechanically
  derivable field filled in (product, id, database, version table, the phases
  table, the lint evidence) and the judgment sections left as the template's
  own guidance for the author to write.

Function Core / Imperative Shell, as in :mod:`sax_platform.db.verify`: the
classification, the id allocation, the stripping, and the whole of the rendered
``request.md`` are pure functions over values, and the shell is the part that
runs Alembic, shells out to Squawk, and writes files.
"""

from __future__ import annotations

import io
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, assert_never

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from typing import TextIO

    from alembic.config import Config

__all__ = [
    "SQUAWK_PACKAGE",
    "SQUAWK_VERSION",
    "ChainSpec",
    "ChangeRequestError",
    "GeneratedRequest",
    "LintReport",
    "NotLinted",
    "Phase",
    "PhaseClassification",
    "SqlLinter",
    "build_phases",
    "classify_phase",
    "describe_generated_request",
    "find_repo_root",
    "generate_change_request",
    "next_request_id",
    "render_request",
    "resumability_warning",
    "run_squawk",
    "squawk_linter",
    "strip_transaction_wrappers",
    "validate_title",
]


class ChangeRequestError(RuntimeError):
    """A change request could not be generated, and nothing was written.

    Raised for every refusal this module makes — an unusable revision range, a
    branched chain, a directory that already exists, a linter that will not
    run. The message surfaces verbatim on a CLI's stderr, so it names what was
    wrong and what to do about it.
    """


#: The Squawk release every lint claim in this repo is made against. Pinned in
#: exactly one place: ``make lint-sql`` reads this constant rather than
#: carrying its own copy, because "clean lint" means clean against a known
#: rule set — a floating ``latest`` makes the claim unfalsifiable.
SQUAWK_VERSION: Final = "2.61.0"

#: The npx package spec built from the pin (``npx --yes <this>``).
SQUAWK_PACKAGE: Final = f"squawk-cli@{SQUAWK_VERSION}"

#: The canonical process doc in the operator's repo (sax-datastores issue #2).
_PROCESS_DOC: Final = "sax-datastores/docs/schema-changes.md"

#: Offline generation needs a dialect, not a database. Alembic's offline mode
#: never opens a connection, so this URL is only ever parsed.
_OFFLINE_URL: Final = "postgresql://x/x"

#: Transaction control the apply runbook owns and the artifact must not carry.
_WRAPPER_LINES: Final = frozenset({"BEGIN;", "COMMIT;"})

_TITLE_RE: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_REQUEST_DIR_RE: Final = re.compile(r"^(\d{4})-")


# ---------------------------------------------------------------------------
# Pure core: values
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ChainSpec:
    """The identity of one Alembic chain, as a change request describes it.

    Everything here is a fact about the chain and the database it targets, not
    about a particular request: the three CLIs each construct their own
    constant instance.
    """

    product: str
    database: str
    schema: str
    version_table: str
    script_location: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class Phase:
    """One revision step of the request: the SQL for exactly one phase.

    ``number`` is 1-based in *apply* order, which is what names the file — the
    administrator applies ``change-1.sql`` then ``change-2.sql``, and a partial
    apply resumes at the first phase whose stamp is not yet in the version
    table.
    """

    number: int
    from_revision: str
    to_revision: str
    sql: str

    @property
    def file_name(self) -> str:
        """The artifact filename for this phase (``change-<n>.sql``)."""
        return f"change-{self.number}.sql"


@dataclass(frozen=True, slots=True, kw_only=True)
class PhaseClassification:
    """How one phase must be applied, derived from its SQL alone."""

    transactional: bool
    statement_timeout: Literal["default", "0"]
    resumable: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class LintReport:
    """Squawk ran over the generated SQL and this is what it said."""

    version: str
    output: str
    clean: bool
    kind: Literal["linted"] = "linted"


@dataclass(frozen=True, slots=True, kw_only=True)
class NotLinted:
    """Generation was asked to skip the linter (``--no-lint``).

    A distinct state rather than an empty :class:`LintReport`, so the rendered
    request says "NOT LINTED" instead of quietly reading as clean.
    """

    kind: Literal["not-linted"] = "not-linted"


type LintOutcome = LintReport | NotLinted

#: The linter seam: takes the written SQL artifacts, returns what a linter said.
#: A plain callable so tests inject a fake and no test run touches the network.
type SqlLinter = Callable[[Sequence[Path]], LintReport]


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratedRequest:
    """What :func:`generate_change_request` wrote, and what the author must see."""

    directory: Path
    product: str
    request_id: str
    title: str
    request_file: Path
    sql_files: tuple[Path, ...]
    phases: tuple[Phase, ...]
    lint: LintOutcome
    warnings: tuple[str, ...]


# ---------------------------------------------------------------------------
# Pure core: the SQL
# ---------------------------------------------------------------------------


def strip_transaction_wrappers(sql: str) -> str:
    """Remove Alembic's ``BEGIN;``/``COMMIT;`` lines from offline SQL (pure).

    The process forbids transaction control inside an artifact: ``apply-change``
    wraps each phase itself (and cannot wrap a ``CONCURRENTLY`` phase at all),
    and it refuses artifacts that carry their own. Everything else is kept
    verbatim — the ``-- Running upgrade`` header and, crucially, the version
    table ``UPDATE``, which is what makes a partial apply resumable.

    Alembic emits every statement as ``<text>\\n\\n``, so dropping a wrapper
    line on its own would leave the blank line that separated it behind and
    the artifact would accumulate double blank lines around each removal.
    Runs of blank lines are therefore collapsed to one (``cat -s`` semantics),
    which is what makes the output byte-identical to a hand-built artifact.

    A line is a wrapper only if the whole line is exactly ``BEGIN;`` or
    ``COMMIT;``; a statement that merely contains those words is untouched.
    """
    kept: list[str] = []
    for line in sql.splitlines():
        if line.strip() in _WRAPPER_LINES:
            continue
        if not line.strip() and kept and not kept[-1].strip():
            continue
        kept.append(line)
    if not any(line.strip() for line in kept):
        return ""
    return "\n".join(kept) + "\n"


def classify_phase(sql: str) -> PhaseClassification:
    """Decide how a phase must be applied, from its SQL alone (pure).

    ``CONCURRENTLY`` is the whole signal: Postgres refuses a concurrent index
    build inside a transaction block, so such a phase is applied unwrapped and
    with ``statement_timeout`` 0 (a concurrent build waits on other sessions
    and a timeout would kill it mid-flight). Every other phase runs inside one
    transaction under the runbook's default timeout.

    Resumability is checked per statement, split on ``;``: a non-transactional
    phase leaves no transaction to roll back, so a statement that fails part
    way through must be safe to re-run — which for DDL means ``IF NOT EXISTS``
    or ``IF EXISTS``. Only statements that are themselves concurrent are
    checked; the rest are covered by their transaction.
    """
    upper = sql.upper()
    transactional = "CONCURRENTLY" not in upper
    unresumable = [
        statement
        for statement in upper.split(";")
        if "CONCURRENTLY" in statement
        and "IF NOT EXISTS" not in statement
        and "IF EXISTS" not in statement
    ]
    return PhaseClassification(
        transactional=transactional,
        statement_timeout="default" if transactional else "0",
        resumable=not unresumable,
    )


def resumability_warning(phase: Phase) -> str | None:
    """Warn when a non-transactional phase is not re-runnable (pure).

    Returns ``None`` when the phase is fine. A warning rather than a refusal:
    the fix is in the Alembic revision, which this module does not own, and an
    artifact the author can see and correct beats a generator that hides the
    problem or blocks on it.
    """
    classification = classify_phase(phase.sql)
    if classification.transactional or classification.resumable:
        return None
    return (
        f"{phase.file_name} ({phase.from_revision} -> {phase.to_revision}) runs outside a "
        f"transaction (CONCURRENTLY) but is not resumable: a concurrent statement lacks "
        f"IF NOT EXISTS / IF EXISTS. A non-transactional phase that fails part way through "
        f"leaves nothing to roll back, so the process requires it to be safe to re-run. "
        f"Fix the revision and regenerate."
    )


# ---------------------------------------------------------------------------
# Pure core: naming
# ---------------------------------------------------------------------------


def validate_title(title: str) -> str:
    """Return ``title`` if it is a kebab-case slug, else raise (pure).

    The title becomes a directory name that shell globs (``make lint-sql``) and
    an issue title both have to survive, so it is restricted to lowercase
    alphanumerics separated by single hyphens.
    """
    if not _TITLE_RE.match(title):
        msg = (
            f"Invalid --title {title!r}: use a kebab-case slug — lowercase letters and "
            f"digits separated by single hyphens (e.g. interactions-created-at-index). "
            f"It becomes the request directory name."
        )
        raise ChangeRequestError(msg)
    return title


def next_request_id(existing: Iterable[str]) -> str:
    """Allocate the next zero-padded request id from existing directory names (pure).

    Ids are per-product and sequential; a gap is never reused (``max + 1``, not
    "first free"), because a submitted request is identified by its id in an
    issue and in the operator's records, and reusing one would collide with
    something already reviewed. Names that do not start with four digits and a
    hyphen are ignored.
    """
    ids = [int(match.group(1)) for name in existing if (match := _REQUEST_DIR_RE.match(name))]
    return f"{max(ids, default=0) + 1:04d}"


# ---------------------------------------------------------------------------
# Pure core: rendering the request
# ---------------------------------------------------------------------------


def _phase_row(phase: Phase) -> str:
    classification = classify_phase(phase.sql)
    transactional = "yes" if classification.transactional else "no (concurrent index)"
    return (
        f"| `{phase.file_name}` | `{phase.from_revision}` → `{phase.to_revision}` | "
        f"{transactional} | {classification.statement_timeout} |"
    )


def _lint_section(lint: LintOutcome) -> str:
    match lint:
        case NotLinted():
            return (
                "**NOT LINTED — run `make lint-sql` before committing.** This request "
                "was generated with `--no-lint`, so no Squawk evidence exists for it "
                "yet. Clean lint is the price of Tier-1 treatment; replace this "
                "paragraph with the tool output and its version before submitting."
            )
        case LintReport(version=version, output=output, clean=clean):
            header = (
                f"Squawk `{version}` (`--reporter gcc`), run over the generated SQL with "
                f"the repo-root `.squawk.toml` — the byte-copy of the operator's canonical "
                f"config, so these exclusions are the ones the process assumes:"
            )
            if clean:
                return f"{header} **clean, no findings.**"
            body = output.strip() or "(squawk reported a non-zero exit with no output)"
            return (
                f"{header} **not clean.** Every finding below must be justified here, "
                f"or fixed in the revision and the request regenerated.\n\n"
                f"```text\n{body}\n```"
            )
        case _ as unreachable:  # pragma: no cover - exhaustiveness is checked statically
            assert_never(unreachable)


def render_request(
    *,
    title: str,
    request_id: str,
    chain: ChainSpec,
    phases: Sequence[Phase],
    lint: LintOutcome,
    warnings: Sequence[str] = (),
) -> str:
    """Render ``request.md`` for one change request (pure).

    Mirrors the operator's template (``sax-datastores/templates/``) heading for
    heading. Everything derivable from the chain and the generated SQL is
    filled in; every section that is a judgment — why, risk, compatibility,
    backfill, rollback, dev evidence — keeps the template's own HTML-comment
    guidance, because those are the sections the reviewer actually reads and
    nothing here can write them.
    """
    if not phases:
        msg = "Cannot render a change request with no phases."
        raise ChangeRequestError(msg)
    first, last = phases[0], phases[-1]
    lines = [
        "<!-- Schema change request — generated by `make db-change`. Fill in the",
        "     prose sections below, COMMIT (the commit is the request; a new",
        "     commit voids any prior approval), then open an issue on",
        f"     sax-datastores titled `change: {chain.product} {request_id} {title}`.",
        f"     Full guide: {_PROCESS_DOC} -->",
        "",
        f"# Change request: `{title}`",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Product | `{chain.product}` |",
        f"| Id | `{request_id}` |",
        f"| Database | `{chain.database}` (prod) |",
        f"| Schema | `{chain.schema}` |",
        f"| Version table | `{chain.version_table}` |",
        (
            f"| From → to | `{first.from_revision}` → `{last.to_revision}` "
            f"(linear; single head verified) |"
        ),
        "| Tier claimed | 1 or 2 (operator confirms at intake) |",
        "| Stacks | `prod` only — dev/test are self-service |",
        "",
        "## What and why",
        "",
        "<!-- One logical change: one expand step OR one contract step. What",
        "     it does, and the product motivation in a sentence or two. -->",
        "",
        "## Phases",
        "",
        "| File | From → to | Transactional | statement_timeout |",
        "| --- | --- | --- | --- |",
        *[_phase_row(phase) for phase in phases],
        "",
        (
            f"Generated with `alembic upgrade {first.from_revision}:{last.to_revision} --sql` "
            f"(postgres dialect, offline);"
        ),
        "`BEGIN`/`COMMIT` lines stripped per the process; each phase keeps its own",
        f"`{chain.version_table}` stamp as its final statement, which is what makes a",
        "partial apply resumable.",
    ]
    for warning in warnings:
        lines += ["", f"**Resumability warning (fix before submitting):** {warning}"]
    lines += [
        "",
        "## Risk notes (evidence, not intentions)",
        "",
        "<!-- Per affected table: current row count and size; per statement:",
        '     expected lock level and duration. Say "empty table" when true —',
        "     it makes the review trivial. -->",
        "",
        "## Compatibility (expand/contract)",
        "",
        "<!-- Why currently-deployed code keeps working against the",
        "     post-change schema. For a contract step: which release stopped",
        "     depending on what is being dropped, and how that was verified. -->",
        "",
        "## Backfill",
        "",
        '<!-- "None", or: what DML runs consumer-side under the app',
        "     credential, rough row count, batch size, and where it sits in",
        "     the expand → migrate → contract sequence. -->",
        "",
        "## Rollback stance",
        "",
        "<!-- Forward-fix is acceptable; silence is not. If rollback SQL",
        "     exists, where; if not, why forward-fix suffices. -->",
        "",
        "## Lint",
        "",
        _lint_section(lint),
        "",
        "## Dev evidence",
        "",
        "<!-- How the same revision was applied and verified on the dev stack",
        "     (`<product> migrate`), and what was checked afterwards. -->",
        "",
    ]
    return "\n".join(lines)


def describe_generated_request(result: GeneratedRequest) -> str:
    """Render the CLI's success summary for a generated request (pure)."""
    files = "\n".join(f"  {path.name}" for path in (*result.sql_files, result.request_file))
    match result.lint:
        case NotLinted():
            lint_line = "skipped (--no-lint) — run `make lint-sql` before committing"
        case LintReport(version=version, clean=clean):
            lint_line = f"squawk {version}: {'clean' if clean else 'FINDINGS — see request.md'}"
        case _ as unreachable:  # pragma: no cover - exhaustiveness is checked statically
            assert_never(unreachable)
    return (
        f"Wrote change request {result.request_id} to {result.directory}\n"
        f"{files}\n"
        f"lint: {lint_line}\n"
        f"Next: fill in the prose sections of request.md, commit (the commit is the "
        f"request), then open an issue on sax-datastores titled "
        f"`change: {result.product} {result.request_id} {result.title}` — {_PROCESS_DOC}."
    )


# ---------------------------------------------------------------------------
# Imperative shell: Alembic
# ---------------------------------------------------------------------------


def _alembic_config(script_location: Path, output_buffer: TextIO | None = None) -> Config:
    """Build an Alembic ``Config`` for ``script_location`` (offline, dialect-only).

    Mirrors :func:`sax_platform.db.migrations.run_migrations`' setup — the
    ``alembic.ini`` beside ``versions/``, with ``script_location`` overridden
    programmatically — so generation always reads the same chain the runner
    applies. The URL supplies the postgres dialect and nothing else; the
    ``%`` -> ``%%`` escape that matters for real credentials is moot here but
    kept so the constant can never break configparser interpolation.
    """
    from alembic.config import Config

    script_dir = script_location.resolve()
    cfg = Config(str(script_dir / "alembic.ini"), output_buffer=output_buffer)
    cfg.set_main_option("script_location", str(script_dir))
    cfg.set_main_option("sqlalchemy.url", _OFFLINE_URL.replace("%", "%%"))
    return cfg


def _revision_steps(
    script_location: Path, from_revision: str, to_revision: str | None
) -> tuple[tuple[str, str], ...]:
    """Resolve the chain walk FROM -> TO into ordered ``(from, to)`` pairs.

    Fails closed, and before anything is written: a chain without exactly one
    head has no unambiguous target (the same reasoning as
    :class:`sax_platform.db.verify.BrokenChain`), an unknown revision or a
    FROM that is not an ancestor of TO cannot describe a linear request, and a
    merge revision is outside the process (one revision per phase, linear).
    """
    from alembic.script import ScriptDirectory
    from alembic.script.revision import RevisionError

    script = ScriptDirectory.from_config(_alembic_config(script_location))
    heads = tuple(script.get_heads())
    if len(heads) != 1:
        found = ", ".join(heads) if heads else "none"
        msg = (
            f"The chain at {script_location} resolves to {len(heads)} heads ({found}) where "
            f"exactly one is expected. A change request describes a linear range; merge the "
            f"branch (or restore the missing revisions) before generating one."
        )
        raise ChangeRequestError(msg)

    target = to_revision or heads[0]
    try:
        walked = tuple(script.iterate_revisions(target, from_revision))
    except RevisionError as exc:
        msg = (
            f"Cannot walk {from_revision} -> {target} in the chain at {script_location}: {exc}. "
            f"Pass --from as the revision the database is already stamped with, and --to as a "
            f"descendant of it (default: the chain head, {heads[0]})."
        )
        raise ChangeRequestError(msg) from exc

    if not walked:
        msg = (
            f"Nothing to generate: {from_revision} and {target} are the same revision, so the "
            f"range is empty. Author the Alembic revision first, then generate the request."
        )
        raise ChangeRequestError(msg)

    steps: list[tuple[str, str]] = []
    for revision in reversed(walked):
        down = revision.down_revision
        if down is not None and not isinstance(down, str):
            msg = (
                f"Revision {revision.revision} in {script_location} is a merge revision: its "
                f"down_revision is {down!r}, not a single parent. The process takes one "
                f"revision per phase along a linear chain."
            )
            raise ChangeRequestError(msg)
        # The chain's first revision has no parent; Alembic names that point
        # "base", and `upgrade base:<rev> --sql` is a valid offline range (a
        # product bootstrapping a database it has never had).
        steps.append((down or "base", revision.revision))
    return tuple(steps)


def _offline_sql(script_location: Path, from_revision: str, to_revision: str) -> str:
    """Run ``alembic upgrade <from>:<to> --sql`` and return the raw output.

    Offline mode opens no connection; the output is captured through the
    ``Config``'s ``output_buffer`` (not ``stdout``, which only carries a
    command's own printed chatter) so nothing reaches the terminal.
    """
    from alembic import command

    buffer = io.StringIO()
    cfg = _alembic_config(script_location, output_buffer=buffer)
    command.upgrade(cfg, f"{from_revision}:{to_revision}", sql=True)
    return buffer.getvalue()


def build_phases(
    *, script_location: Path, from_revision: str, to_revision: str | None
) -> tuple[Phase, ...]:
    """Generate one stripped-SQL :class:`Phase` per revision step, in apply order."""
    steps = _revision_steps(script_location, from_revision, to_revision)
    return tuple(
        Phase(
            number=number,
            from_revision=step_from,
            to_revision=step_to,
            sql=strip_transaction_wrappers(_offline_sql(script_location, step_from, step_to)),
        )
        for number, (step_from, step_to) in enumerate(steps, start=1)
    )


# ---------------------------------------------------------------------------
# Imperative shell: Squawk
# ---------------------------------------------------------------------------


def run_squawk(paths: Sequence[Path], *, config_path: Path) -> LintReport:
    """Lint the generated SQL with the pinned Squawk, or raise.

    ``npx --yes squawk-cli@<pin>`` — the pin is :data:`SQUAWK_VERSION`, so a
    request's lint claim names a known rule set. Findings are a *result*
    (``clean=False``), not an error; only a linter that could not run at all
    (no node, no network on a cold npx cache, a crash) raises, because a
    silently skipped lint would let an unlinted request read as a clean one.

    The ``gcc`` reporter is deliberate: one plain line per finding, no ANSI
    escapes, so the output can be pasted into the committed request.
    """
    argv = [
        "npx",
        "--yes",
        SQUAWK_PACKAGE,
        "--config",
        str(config_path),
        "--reporter",
        "gcc",
        *(str(path) for path in paths),
    ]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    except OSError as exc:
        msg = (
            f"Could not run Squawk ({SQUAWK_PACKAGE}): {exc}. It is invoked through `npx`, so "
            f"node must be installed and able to fetch the package. Generate without lint "
            f"evidence using --no-lint, then run `make lint-sql` before committing."
        )
        raise ChangeRequestError(msg) from exc

    output = f"{completed.stdout}{completed.stderr}".strip()
    if completed.returncode not in (0, 1):
        msg = (
            f"Squawk ({SQUAWK_PACKAGE}) exited {completed.returncode} without producing a "
            f"verdict:\n{output}"
        )
        raise ChangeRequestError(msg)
    return LintReport(version=SQUAWK_VERSION, output=output, clean=completed.returncode == 0)


def squawk_linter(config_path: Path) -> SqlLinter:
    """Bind :func:`run_squawk` to one config file, producing a :data:`SqlLinter`."""

    def lint(paths: Sequence[Path]) -> LintReport:
        return run_squawk(paths, config_path=config_path)

    return lint


# ---------------------------------------------------------------------------
# Imperative shell: the filesystem
# ---------------------------------------------------------------------------


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the directory holding ``.git``.

    Used to resolve a CLI's default ``--output-root`` from the module's own
    location rather than the cwd, so ``forge db-change`` writes to the repo it
    was installed from no matter where it is run. ``.git`` may be a directory
    or a file (a worktree, as the pinned prod checkout is), so existence is
    what is checked.
    """
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    msg = (
        f"Could not find a repository root above {start}: no .git entry in any parent. "
        f"Pass --output-root explicitly."
    )
    raise ChangeRequestError(msg)


def _existing_request_ids(output_root: Path) -> tuple[str, ...]:
    """List the request directory names already under ``output_root``."""
    if not output_root.is_dir():
        return ()
    return tuple(sorted(entry.name for entry in output_root.iterdir() if entry.is_dir()))


def generate_change_request(
    *,
    chain: ChainSpec,
    output_root: Path,
    from_revision: str,
    to_revision: str | None,
    title: str,
    linter: SqlLinter | None,
) -> GeneratedRequest:
    """Write a complete change request directory, or raise having written nothing.

    Order is load-bearing: the title is validated and the SQL is generated
    *before* any directory is created, so a bad range or a broken chain leaves
    no half-written request behind. An existing directory is never overwritten
    — a submitted request is identified by its id, and rewriting one silently
    would void an approval without anybody noticing.

    ``linter`` is the injection seam: pass :func:`squawk_linter` for real lint
    evidence, or ``None`` for ``--no-lint`` (which stamps the request NOT
    LINTED rather than leaving the section looking clean).
    """
    validate_title(title)
    phases = build_phases(
        script_location=chain.script_location,
        from_revision=from_revision,
        to_revision=to_revision,
    )

    request_id = next_request_id(_existing_request_ids(output_root))
    directory = output_root / f"{request_id}-{title}"
    if directory.exists():
        msg = (
            f"{directory} already exists — refusing to overwrite a change request. "
            f"Rename or delete it if it was never submitted; otherwise generate the next "
            f"one under a different title."
        )
        raise ChangeRequestError(msg)

    directory.mkdir(parents=True)
    sql_files = tuple(directory / phase.file_name for phase in phases)
    for phase, path in zip(phases, sql_files, strict=True):
        path.write_text(phase.sql)

    lint: LintOutcome = NotLinted() if linter is None else linter(sql_files)
    warnings = tuple(
        warning for phase in phases if (warning := resumability_warning(phase)) is not None
    )

    request_file = directory / "request.md"
    request_file.write_text(
        render_request(
            title=title,
            request_id=request_id,
            chain=chain,
            phases=phases,
            lint=lint,
            warnings=warnings,
        )
    )
    return GeneratedRequest(
        directory=directory,
        product=chain.product,
        request_id=request_id,
        title=title,
        request_file=request_file,
        sql_files=sql_files,
        phases=phases,
        lint=lint,
        warnings=warnings,
    )
