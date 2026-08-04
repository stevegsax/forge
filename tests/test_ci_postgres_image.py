"""CI's Postgres service runs the operator's canonical image, same as the tests.

sax-datastores publishes one Postgres image — PG17 + ``rum`` + ``pgvector`` — and
the dev stack, the prod stack, and every product's CI run it, so the extension
surface is identical everywhere. Forge consumes that image; it never names one
of its own.

Since the trust-path repoint, ``ci.yml`` is the *only* place forge provisions a
Postgres at all — the agent-run suites connect to the already-running shared dev
stack instead (sax-datastores rationale §22). But YAML cannot read a Python
constant, so each of its ``services:`` blocks carries its own copy of the image
string, and these tests are what keeps every copy honest — the same shape as
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

# The three suites that used to provision their own Postgres, plus ci.yml.
# They now connect to the trust path instead, so the check over them is purely
# that no retired image string crept back in.
PROVISIONING_SITES = (
    CI_WORKFLOW,
    REPO_ROOT / "tests" / "test_migrations_postgres.py",
    REPO_ROOT / "apps" / "ocr" / "tests" / "test_migrations.py",
    REPO_ROOT / "apps" / "pbook" / "tests" / "conftest.py",
)


def _postgres_service_images(workflow_text: str) -> tuple[str | None, ...]:
    """Return the ``image:`` of EVERY ``postgres:`` service block, in order.

    One entry per block, so a block declaring no image shows up as ``None``
    rather than vanishing — an equality check over the collected values would
    otherwise pass by finding nothing. ``ci.yml`` is scanned as text rather than
    parsed: PyYAML is not a declared dependency of this workspace (it is only
    ever present transitively), and a gate must not rest on a transitive import.
    The scan walks the indented block under each ``postgres:`` and stops at the
    first dedent, so an ``image:`` key belonging to some other service can never
    be picked up by accident.
    """
    lines = workflow_text.splitlines()
    images: list[str | None] = []
    for index, line in enumerate(lines):
        if line.strip() != "postgres:":
            continue
        indent = len(line) - len(line.lstrip())
        found: str | None = None
        for later in lines[index + 1 :]:
            stripped = later.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if len(later) - len(later.lstrip()) <= indent:
                break
            if stripped.startswith("image:"):
                found = stripped.removeprefix("image:").strip()
                break
        images.append(found)
    return tuple(images)


class TestCanonicalPostgresImage:
    def test_every_ci_service_image_equals_the_python_constant(self) -> None:
        images = _postgres_service_images(CI_WORKFLOW.read_text())
        # Two service containers today: test-pbook and test-postgres-migrations.
        assert len(images) >= 2
        assert set(images) == {CANONICAL_POSTGRES_IMAGE}

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
        assert _postgres_service_images(workflow) == (None,)

    def test_the_scan_reports_one_entry_per_postgres_block(self) -> None:
        # Two jobs, two services: both copies must be visible to the check above.
        workflow = (
            "a:\n  services:\n    postgres:\n      image: one\n"
            "b:\n  services:\n    postgres:\n      image: two\n"
        )
        assert _postgres_service_images(workflow) == ("one", "two")
