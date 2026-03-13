"""End-to-end tests for the OCR pipeline with real Mistral API.

These tests send real images through the synchronous OCR workflow
using the Mistral OCR API and verify that the correct text is extracted.

Gated by the ``e2e`` marker and the ``MISTRAL_API_KEY`` env var.
Run with::

    MISTRAL_API_KEY=<key> uv run python -m pytest -m e2e tests/test_ocr_e2e.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from temporalio.worker import Worker

from forge.ocr.activities import (
    call_ocr_sync,
    check_ocr_duplicate,
    read_and_store_file_content,
    reassemble_ocr_chunks,
    split_file_into_chunks,
)
from forge.ocr.models import OcrSyncInput
from forge.ocr.workflow_sync import OcrSyncWorkflow
from forge.workflows import FORGE_TASK_QUEUE

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ocr"

# Skip the entire module if MISTRAL_API_KEY is not set
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY"),
        reason="MISTRAL_API_KEY not set",
    ),
]

# Workflows and activities needed for the synchronous OCR pipeline
_OCR_SYNC_WORKFLOWS = [OcrSyncWorkflow]
_OCR_SYNC_ACTIVITIES = [
    call_ocr_sync,
    check_ocr_duplicate,
    read_and_store_file_content,
    split_file_into_chunks,
    reassemble_ocr_chunks,
]


@pytest.fixture()
async def local_env():
    """Temporal environment with real wall-clock time (not time-skipping).

    Time-skipping environments auto-advance time, which breaks real API
    calls that need wall-clock time to complete.
    """
    from temporalio.testing import WorkflowEnvironment

    async with await WorkflowEnvironment.start_local() as env:
        yield env


@pytest.fixture()
def test_db(tmp_path, monkeypatch):
    """Set up a test database with migrations and point FORGE_DB_PATH to it."""
    from forge.store import run_migrations

    db_path = tmp_path / "test_forge.db"
    run_migrations(db_path)
    monkeypatch.setenv("FORGE_DB_PATH", str(db_path))
    return db_path


@pytest.mark.e2e
class TestOcrSyncE2E:
    """End-to-end synchronous OCR tests with real Mistral API."""

    async def test_jpeg_sync_ocr(self, local_env, test_db):
        """JPEG image -> real synchronous Mistral OCR -> verify extracted text."""
        fixture_path = FIXTURES_DIR / "hello_jpeg.jpg"
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

        await self._run_sync_ocr_and_verify(
            local_env,
            test_db,
            file_path=str(fixture_path),
            expected_words=["Hello", "JPEG"],
        )

    async def test_png_sync_ocr(self, local_env, test_db):
        """PNG image -> real synchronous Mistral OCR -> verify extracted text."""
        fixture_path = FIXTURES_DIR / "hello_png.png"
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

        await self._run_sync_ocr_and_verify(
            local_env,
            test_db,
            file_path=str(fixture_path),
            expected_words=["Hello", "PNG"],
        )

    async def _run_sync_ocr_and_verify(
        self,
        env,
        db_path: Path,
        *,
        file_path: str,
        expected_words: list[str],
    ) -> None:
        """Core test logic: submit sync OCR workflow, await result, verify."""
        from forge.store import get_engine, get_ocr_result

        ocr_input = OcrSyncInput(file_path=file_path, force=True)

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=_OCR_SYNC_WORKFLOWS,
            activities=_OCR_SYNC_ACTIVITIES,
        ):
            # Sync workflow — no polling needed, just await the result
            result = await env.client.execute_workflow(
                OcrSyncWorkflow.run,
                ocr_input,
                id=f"e2e-ocr-sync-{Path(file_path).stem}",
                task_queue=FORGE_TASK_QUEUE,
            )

        # -- Assertions on workflow result --
        assert result.document_id, "document_id should be set"
        assert result.text_length > 0, "text_length should be > 0"
        assert result.stored is True, "result should be stored"

        # -- Assertions on database row --
        engine = get_engine(db_path)
        row = get_ocr_result(engine, result.document_id)
        assert row is not None, f"No OCR result in DB for document_id={result.document_id}"

        extracted_text = row["text"]
        assert len(extracted_text) > 0, "Extracted text should not be empty"

        # Fuzzy match: OCR output may vary in casing or spacing
        text_lower = extracted_text.lower()
        for word in expected_words:
            assert word.lower() in text_lower, (
                f"Expected word '{word}' not found in extracted text: {extracted_text!r}"
            )
