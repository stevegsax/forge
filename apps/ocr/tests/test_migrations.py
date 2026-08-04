"""Migration isolation/coexistence tests for the OCR Alembic chain.

The OCR app shares one database with the platform but owns an isolated chain. These
prove the OCR chain (a) manages only OCR tables, and (b) coexists with the
platform-owned ``batch_jobs`` table under a distinct ``version_table`` without
dropping it. The sqlite tests run by default; the same coexistence is re-checked on
real Postgres under the opt-in ``postgres`` marker.
"""

from __future__ import annotations

import os

import pytest
import sqlalchemy as sa
from sax_platform.testing import (
    FORGE_TEST_DB_URL_ENV,
    FORGE_TRUST_TEST_DB_URL,
    reset_public_schema,
    resolve_test_database_url,
)

from ocr.store import Base, run_migrations

_OCR_TABLES = {
    "ocr_results",
    "ocr_images",
    "ocr_file_content_blobs",
    "ocr_job_status",
    "ocr_tracker_heartbeat",
}
_FORGE_VERSION_DDL = "CREATE TABLE alembic_version_forge (version_num varchar PRIMARY KEY)"


def test_metadata_owns_only_ocr_tables() -> None:
    """OCR's Base.metadata (what include_object filters on) is OCR tables only."""
    assert set(Base.metadata.tables) == _OCR_TABLES


def test_chain_creates_ocr_tables_with_own_version_table(forge_db_url: str) -> None:
    run_migrations(forge_db_url)
    engine = sa.create_engine(forge_db_url)
    try:
        names = set(sa.inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert names >= _OCR_TABLES
    assert "alembic_version_ocr" in names
    assert "alembic_version" not in names  # not the default table name


def test_verify_schema_passes_on_a_migrated_store(forge_db_url: str) -> None:
    """The startup check over OCR's real chain: apply, then verify.

    Also the guard against a two-headed OCR chain — an unmerged branch fails
    here rather than at a production worker boot.
    """
    from ocr.store import verify_schema

    run_migrations(forge_db_url)

    assert verify_schema(forge_db_url)


def test_verify_schema_refuses_an_unmigrated_store(forge_db_url: str) -> None:
    """Fail closed: the worker never applies the chain itself (2026-08-02 agreement)."""
    from sax_platform.db import SchemaVersionError

    from ocr.store import verify_schema

    sa.create_engine(forge_db_url).connect().close()

    with pytest.raises(SchemaVersionError, match="Schema not initialized") as excinfo:
        verify_schema(forge_db_url)

    assert "ocr migrate" in str(excinfo.value)


def test_coexists_with_platform_batch_jobs(forge_db_url: str) -> None:
    """The OCR chain coexists with a pre-existing platform batch_jobs + version table."""
    from sax_platform.contracts.batch_jobs import metadata as bj_metadata

    engine = sa.create_engine(forge_db_url)
    try:
        # Simulate the platform side already present in the shared DB.
        bj_metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(sa.text(_FORGE_VERSION_DDL))

        run_migrations(forge_db_url)
        run_migrations(forge_db_url)  # rerun is a clean no-op

        names = set(sa.inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert "batch_jobs" in names  # platform table not dropped
    assert names >= _OCR_TABLES
    assert {"alembic_version_forge", "alembic_version_ocr"} <= names


@pytest.fixture
def postgres_url() -> str:
    """The shared test database, emptied for this test.

    ocr shares forge's database in production, so it shares ``forge_test`` here —
    the credential-free trust role on the running sax-datastores dev stack
    (rationale §22: an agent session provisions no containers). CI overrides it
    with ``FORGE_TEST_DATABASE_URL`` pointing at its own service container.
    Unreachable is an error, never a skip.
    """
    url = resolve_test_database_url(
        os.environ,
        env_var=FORGE_TEST_DB_URL_ENV,
        default=FORGE_TRUST_TEST_DB_URL,
    )
    reset_public_schema(url, env_var=FORGE_TEST_DB_URL_ENV)
    return url


@pytest.mark.postgres
def test_coexists_on_postgres(postgres_url: str) -> None:
    from sax_platform.contracts.batch_jobs import metadata as bj_metadata

    engine = sa.create_engine(postgres_url)
    bj_metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(sa.text(_FORGE_VERSION_DDL))

    run_migrations(postgres_url)
    run_migrations(postgres_url)

    names = set(sa.inspect(engine).get_table_names())
    assert "batch_jobs" in names
    assert names >= _OCR_TABLES
    assert {"alembic_version_forge", "alembic_version_ocr"} <= names
    engine.dispose()
