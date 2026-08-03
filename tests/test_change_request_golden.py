"""The change-request generator reproduces the maiden request, byte for byte.

``datastore-changes/0001-interactions-created-at-index/`` was built by hand on
2026-08-02 under the process agreed on sax-datastores issue #2, and submitted.
The generator's whole claim is that it mechanizes exactly that, so the test is
an equality against the committed artifact rather than a shape assertion: if
the generator's output drifts — a stray blank line, a lost stamp, a header —
this fails, and the artifact is never the thing that gets "fixed" (it is a
submitted request, identified by its id in an issue).

The other tests here pin the wiring that makes the lint claim in a request
mean something: one Squawk version pin, ``lint-sql`` inside ``gates``, and the
same target in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sax_platform.db.change_request import (
    SQUAWK_VERSION,
    ChainSpec,
    build_phases,
    generate_change_request,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FORGE_CHAIN = REPO_ROOT / "src" / "forge" / "alembic"
MAIDEN_REQUEST = REPO_ROOT / "datastore-changes" / "0001-interactions-created-at-index"


@pytest.fixture
def forge_spec() -> ChainSpec:
    return ChainSpec(
        product="forge",
        database="forge_prod",
        schema="public",
        version_table="alembic_version_forge",
        script_location=FORGE_CHAIN,
    )


class TestMaidenRequestIsReproducible:
    def test_generated_sql_matches_the_committed_artifact_byte_for_byte(self) -> None:
        phases = build_phases(script_location=FORGE_CHAIN, from_revision="003", to_revision="004")
        assert len(phases) == 1
        assert phases[0].sql == (MAIDEN_REQUEST / "change-1.sql").read_text()

    def test_the_whole_request_directory_regenerates(
        self, forge_spec: ChainSpec, tmp_path: Path
    ) -> None:
        result = generate_change_request(
            chain=forge_spec,
            output_root=tmp_path,
            from_revision="003",
            to_revision="004",
            title="interactions-created-at-index",
            linter=None,
        )
        assert result.directory.name == MAIDEN_REQUEST.name
        assert result.sql_files[0].read_text() == (MAIDEN_REQUEST / "change-1.sql").read_text()
        # The phase is classified the way the submitted request describes it.
        rendered = result.request_file.read_text()
        assert "| `change-1.sql` | `003` → `004` | no (concurrent index) | 0 |" in rendered
        assert "| Version table | `alembic_version_forge` |" in rendered


class TestLintWiring:
    def test_the_makefile_reads_the_version_pin_rather_than_copying_it(self) -> None:
        makefile = (REPO_ROOT / "Makefile").read_text()
        assert "squawk-cli@$$ver" in makefile
        assert "from sax_platform.db.change_request import SQUAWK_VERSION" in makefile
        # A second copy of the pin would drift from the one a request records.
        assert SQUAWK_VERSION not in makefile

    def test_lint_sql_is_a_gate(self) -> None:
        makefile = (REPO_ROOT / "Makefile").read_text()
        assert "gates: lint typecheck lint-imports lint-sql test" in makefile

    def test_ci_runs_the_same_target(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "make lint-sql" in workflow

    def test_the_committed_artifact_is_covered_by_the_lint_globs(self) -> None:
        # `make lint-sql` globs datastore-changes/*/change-*.sql; if a request
        # ever lands outside that shape it would silently stop being linted.
        matched = set((REPO_ROOT / "datastore-changes").glob("*/change-*.sql"))
        assert MAIDEN_REQUEST / "change-1.sql" in matched
