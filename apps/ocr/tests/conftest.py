"""Shared test fixtures for the OCR app."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

if TYPE_CHECKING:
    from pathlib import Path

    from temporalio.testing import WorkflowEnvironment


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def env() -> WorkflowEnvironment:
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import WorkflowEnvironment

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        yield env


@pytest.fixture(autouse=True)
def forge_db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point every test at an isolated SQLite store via ``FORGE_DB_URL``."""
    url = f"sqlite:///{tmp_path / 'ocr.db'}"
    monkeypatch.setenv("FORGE_DB_URL", url)
    return url


@pytest.fixture
def migrated(forge_db_url: str) -> str:
    """A migrated OCR store (OCR Alembic chain applied)."""
    from ocr.store import run_migrations

    run_migrations(forge_db_url)
    return forge_db_url


@pytest.fixture(scope="session", autouse=True)
def _s3_backend():
    """In-memory moto S3 bucket for OCR blobs (file content, images, request blobs)."""
    import os

    for key, value in (
        ("AWS_ACCESS_KEY_ID", "testing"),
        ("AWS_SECRET_ACCESS_KEY", "testing"),
        ("AWS_SESSION_TOKEN", "testing"),
        ("AWS_DEFAULT_REGION", "us-east-1"),
    ):
        os.environ.setdefault(key, value)

    import boto3
    from moto import mock_aws

    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="ocr-test-blobs")
        yield "ocr-test-blobs"


@pytest.fixture(autouse=True)
def ocr_s3(monkeypatch: pytest.MonkeyPatch, _s3_backend: str) -> str:
    """Point every test at the mocked blob bucket via ``FORGE_OCR_S3_BUCKET``."""
    monkeypatch.setenv("FORGE_OCR_S3_BUCKET", _s3_backend)
    monkeypatch.delenv("FORGE_OCR_S3_PREFIX", raising=False)
    return _s3_backend


@pytest.fixture(autouse=True)
def dispose_store_engines(monkeypatch: pytest.MonkeyPatch):
    """Dispose SQLAlchemy engines created during a test."""
    import sqlalchemy as sa

    original_create_engine = sa.create_engine
    created = []

    def tracking_create_engine(*args, **kwargs):
        engine = original_create_engine(*args, **kwargs)
        created.append(engine)
        return engine

    monkeypatch.setattr(sa, "create_engine", tracking_create_engine)
    yield
    for engine in created:
        engine.dispose()
