"""Shared test fixtures for the OCR app.

Unit/integration tests run against an in-memory moto S3 bucket and a throwaway
SQLite store. The e2e suite (``-m e2e``) instead needs **real** services (Mistral,
S3, a shared DB, a running platform worker); set ``OCR_E2E_PLATFORM=1`` to put the
autouse fixtures into pass-through mode so they don't mock/override the real env.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

# The session-scoped, pydantic-aware time-skipping Temporal environment is shared
# workspace-wide (D93); this suite requests it under the name ``env``.
from sax_platform.testing import temporal_env

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sax_platform.contracts.s3_blobs import S3Blobs
    from sqlalchemy import Engine

    from ocr.activities import OcrStoreActivities

env = temporal_env

# In e2e/real-services mode the autouse fixtures defer to the operator's real env
# (no moto, no sqlite override) so tests reach live Mistral / S3 / DB / platform.
_REAL_SERVICES = bool(os.environ.get("OCR_E2E_PLATFORM"))


@pytest.fixture(autouse=True)
def forge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare ``FORGE_ENV=test`` so the ST-G2 environment guard passes.

    Every ocr CLI command and worker startup now resolves ``FORGE_ENV`` and
    refuses to run without it. This one central fixture satisfies the guard for
    the whole suite; the guard's own tests override it with ``monkeypatch.delenv``
    or a different value. In real-services mode the operator's declared
    ``FORGE_ENV`` is left untouched.
    """
    if _REAL_SERVICES:
        return
    monkeypatch.setenv("FORGE_ENV", "test")


@pytest.fixture(autouse=True)
def forge_db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point every test at an isolated SQLite store via ``FORGE_DB_URL``.

    In real-services mode, defer to the operator's ``FORGE_DB_URL`` (the shared DB
    the platform worker also uses) instead of a throwaway SQLite file.
    """
    if _REAL_SERVICES:
        return os.environ["FORGE_DB_URL"]
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
    """In-memory moto S3 bucket for OCR blobs (file content, images, request blobs).

    In real-services mode, yield the operator's real bucket without entering moto
    (a session-wide ``mock_aws`` would otherwise intercept real S3 calls).
    """
    if _REAL_SERVICES:
        yield os.environ["FORGE_OCR_S3_BUCKET"]
        return

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
    """Point every test at the mocked blob bucket via ``FORGE_OCR_S3_BUCKET``.

    In real-services mode, leave the operator's bucket env untouched.
    """
    if _REAL_SERVICES:
        return _s3_backend
    monkeypatch.setenv("FORGE_OCR_S3_BUCKET", _s3_backend)
    monkeypatch.delenv("FORGE_OCR_S3_PREFIX", raising=False)
    return _s3_backend


@pytest.fixture
def store_engine(migrated: str) -> Iterator[Engine]:
    """One store engine per test, built from the migrated URL and disposed on teardown.

    Replaces the former global engine-disposal monkeypatch (which wrapped
    ``sa.create_engine`` to catch the fresh engine each ``get_store_engine()``
    built per call): T3.6 builds the engine once and injects it, so tests own the
    engine lifecycle explicitly instead.
    """
    from sax_platform.db import get_store_engine

    engine = get_store_engine(migrated)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def blobs(ocr_s3: str) -> S3Blobs:
    """A moto-backed ``S3Blobs`` bound to the test bucket.

    Injected into the ``execute_*`` cores and ``ocr.store`` helpers, which take a
    required ``S3Blobs`` as of T3.6 (the env fallback is gone).
    """
    from sax_platform.contracts.s3_blobs import S3Blobs

    return S3Blobs(ocr_s3)


@pytest.fixture
def store_activities(store_engine: Engine, ocr_s3: str) -> OcrStoreActivities:
    """An ``OcrStoreActivities`` bound to the test engine + a moto-backed ``S3Blobs``.

    Used by the activity-wrapper tests, which exercise the bound ``@activity.defn``
    methods (formerly module-level functions) with real dependencies injected. The
    Mistral capability is a default ``FakeMistralOcr``; tests that exercise the
    submit/poll/fetch activities construct ``OcrStoreActivities`` directly with a
    configured fake.
    """
    from sax_platform.contracts.s3_blobs import S3Blobs
    from sax_platform.testing import FakeMistralOcr

    from ocr.activities import OcrStoreActivities

    return OcrStoreActivities(store_engine, S3Blobs(ocr_s3), FakeMistralOcr())
