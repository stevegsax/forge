"""CI's Postgres service runs the operator's canonical image, same as the tests.

sax-datastores publishes one Postgres image — PG17 + ``rum`` + ``pgvector`` — and
the dev stack, the prod stack, and every product's CI run it, so the extension
surface is identical everywhere. Forge consumes that image; it never names one
of its own. Python provisioning sites read
``sax_platform.testing.CANONICAL_POSTGRES_IMAGE``, but ``ci.yml`` cannot read a
Python constant, so its service image is a second copy of the string and these
tests are what keeps the copy honest — the same shape as
``test_change_request_golden.py``'s check that the Makefile reads the Squawk pin
rather than restating it.
"""

from __future__ import annotations

from pathlib import Path

from sax_platform.testing import CANONICAL_POSTGRES_IMAGE

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# Images forge provisioned before the 2026-08-02 consolidation onto the
# operator's image. Their absence is what proves the swap is complete rather
# than partial: a site still on one of these is a lane running a different
# extension surface from dev and prod.
RETIRED_IMAGES = ("postgres:16-alpine", "pgvector/pgvector:pg17")

PROVISIONING_SITES = (
    CI_WORKFLOW,
    REPO_ROOT / "tests" / "test_migrations_postgres.py",
    REPO_ROOT / "apps" / "ocr" / "tests" / "test_migrations.py",
    REPO_ROOT / "apps" / "pbook" / "tests" / "conftest.py",
)


def _postgres_service_image(workflow_text: str) -> str | None:
    """Return the ``image:`` of the workflow's ``postgres:`` service, if any.

    ``ci.yml`` is scanned as text rather than parsed: PyYAML is not a declared
    dependency of this workspace (it is only ever present transitively), and a
    gate must not rest on a transitive import. The scan walks the indented block
    under ``postgres:`` and stops at the first dedent, so an ``image:`` key
    belonging to some other service can never be picked up by accident.
    """
    lines = workflow_text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "postgres:":
            continue
        indent = len(line) - len(line.lstrip())
        for later in lines[index + 1 :]:
            stripped = later.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if len(later) - len(later.lstrip()) <= indent:
                break
            if stripped.startswith("image:"):
                return stripped.removeprefix("image:").strip()
    return None


class TestCanonicalPostgresImage:
    def test_ci_service_image_equals_the_python_constant(self) -> None:
        image = _postgres_service_image(CI_WORKFLOW.read_text())
        assert image == CANONICAL_POSTGRES_IMAGE

    def test_no_provisioning_site_still_names_a_retired_image(self) -> None:
        offenders = {
            str(path.relative_to(REPO_ROOT)): retired
            for path in PROVISIONING_SITES
            for retired in RETIRED_IMAGES
            if retired in path.read_text()
        }
        assert offenders == {}

    def test_the_scan_ignores_an_image_outside_the_postgres_service(self) -> None:
        # Guards the helper itself: a sibling service's image must not satisfy
        # the equality test above once the postgres block carries no image key.
        workflow = "services:\n  postgres:\n    ports:\n      - 5432:5432\n  other:\n    image: x\n"
        assert _postgres_service_image(workflow) is None
