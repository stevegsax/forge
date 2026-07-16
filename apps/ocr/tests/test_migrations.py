"""Migration isolation/coexistence tests for the OCR Alembic chain.

The OCR app shares one database with the platform but owns an isolated chain. These
prove the OCR chain (a) manages only OCR tables, and (b) coexists with the
platform-owned ``batch_jobs`` table under a distinct ``version_table`` without
dropping it. The sqlite tests run by default; the same coexistence is re-checked on
real Postgres under the opt-in ``postgres`` marker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from ocr.store import Base, run_migrations

if TYPE_CHECKING:
    from collections.abc import Iterator

_OCR_TABLES = {"ocr_results", "ocr_images", "ocr_file_content_blobs", "ocr_job_status"}
_FORGE_VERSION_DDL = "CREATE TABLE alembic_version_forge (version_num varchar PRIMARY KEY)"


def test_metadata_owns_only_ocr_tables() -> None:
    """OCR's Base.metadata (what include_object filters on) is OCR tables only."""
    assert set(Base.metadata.tables) == _OCR_TABLES


def test_chain_creates_ocr_tables_with_own_version_table(forge_db_url: str) -> None:
    run_migrations(forge_db_url)
    insp = sa.inspect(sa.create_engine(forge_db_url))
    names = set(insp.get_table_names())
    assert names >= _OCR_TABLES
    assert "alembic_version_ocr" in names
    assert "alembic_version" not in names  # not the default table name


def test_coexists_with_platform_batch_jobs(forge_db_url: str) -> None:
    """The OCR chain coexists with a pre-existing platform batch_jobs + version table."""
    from forge_contracts.batch_jobs import metadata as bj_metadata

    engine = sa.create_engine(forge_db_url)
    # Simulate the platform side already present in the shared DB.
    bj_metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(sa.text(_FORGE_VERSION_DDL))

    run_migrations(forge_db_url)
    run_migrations(forge_db_url)  # rerun is a clean no-op

    names = set(sa.inspect(engine).get_table_names())
    assert "batch_jobs" in names  # platform table not dropped
    assert names >= _OCR_TABLES
    assert {"alembic_version_forge", "alembic_version_ocr"} <= names


@pytest.fixture
def postgres_url() -> Iterator[str]:
    pytest.importorskip("testcontainers.postgres")
    from testcontainers.postgres import PostgresContainer

    try:
        container = PostgresContainer("postgres:16-alpine", driver="psycopg2")
        container.start()
    except Exception as exc:
        pytest.skip(f"Postgres testcontainer unavailable: {exc}")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.mark.postgres
def test_coexists_on_postgres(postgres_url: str) -> None:
    from forge_contracts.batch_jobs import metadata as bj_metadata

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
