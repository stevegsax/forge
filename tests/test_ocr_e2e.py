"""End-to-end tests for the OCR pipeline with real Mistral API.

These tests send real images through the full Temporal workflow pipeline
(OcrSubmitWorkflow -> OcrStoreWorkflow) using the Mistral OCR API and
verify that the correct text is extracted.

Gated by the ``e2e`` marker and the ``MISTRAL_API_KEY`` env var.
Run with::

    MISTRAL_API_KEY=<key> uv run python -m pytest -m e2e tests/test_ocr_e2e.py -v --timeout=7200
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from pathlib import Path

import pytest
from temporalio.worker import Worker

from forge.activities.batch_poll import set_temporal_client
from forge.models import BatchPollerInput
from forge.ocr.activities import (
    check_ocr_duplicate,
    parse_ocr_result,
    read_and_store_file_content,
    reassemble_ocr_chunks,
    split_file_into_chunks,
    store_ocr_result,
    submit_ocr_batch,
)
from forge.ocr.models import OcrSubmitInput
from forge.ocr.workflow_gather import OcrGatherWorkflow
from forge.ocr.workflow_store import OcrStoreWorkflow
from forge.ocr.workflow_submit import OcrSubmitWorkflow
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

# Poll interval and timeout for batch completion
_POLL_INTERVAL = timedelta(minutes=2)
_POLL_TIMEOUT = timedelta(hours=2)

# Workflows and activities needed for OCR pipeline
_OCR_WORKFLOWS = [OcrSubmitWorkflow, OcrStoreWorkflow, OcrGatherWorkflow]
_OCR_ACTIVITIES = [
    check_ocr_duplicate,
    read_and_store_file_content,
    split_file_into_chunks,
    submit_ocr_batch,
    parse_ocr_result,
    store_ocr_result,
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


async def _poll_until_complete(timeout: timedelta) -> None:
    """Poll for batch results until all pending jobs are resolved or timeout."""
    from forge.activities.batch_poll import poll_batch_results

    deadline = asyncio.get_event_loop().time() + timeout.total_seconds()

    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(_POLL_INTERVAL.total_seconds())

        # Run the poll activity directly (not via workflow) for simplicity
        try:
            result = await poll_batch_results(BatchPollerInput())
        except RuntimeError:
            # poll_batch_results raises RuntimeError on errors — may be
            # transient (e.g. batch still processing). Continue polling.
            continue

        if result.signals_sent > 0:
            return

    pytest.fail("Timed out waiting for OCR batch to complete")


@pytest.mark.e2e
class TestOcrImageE2E:
    """End-to-end OCR tests with real Mistral API and Temporal workflows."""

    async def test_jpeg_ocr(self, local_env, test_db):
        """JPEG image -> real Mistral OCR -> verify extracted text."""
        fixture_path = FIXTURES_DIR / "hello_jpeg.jpg"
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

        await self._run_ocr_and_verify(
            local_env,
            test_db,
            file_path=str(fixture_path),
            expected_words=["Hello", "JPEG"],
        )

    async def test_png_ocr(self, local_env, test_db):
        """PNG image -> real Mistral OCR -> verify extracted text."""
        fixture_path = FIXTURES_DIR / "hello_png.png"
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

        await self._run_ocr_and_verify(
            local_env,
            test_db,
            file_path=str(fixture_path),
            expected_words=["Hello", "PNG"],
        )

    async def _run_ocr_and_verify(
        self,
        env,
        db_path: Path,
        *,
        file_path: str,
        expected_words: list[str],
    ) -> None:
        """Core test logic: submit OCR workflow, poll, verify results."""
        from forge.store import get_engine, get_ocr_result

        # Inject the Temporal client for signal delivery in poll_batch_results
        set_temporal_client(env.client)

        ocr_input = OcrSubmitInput(file_path=file_path, force=True)

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=_OCR_WORKFLOWS,
            activities=_OCR_ACTIVITIES,
        ):
            # Start the OCR submit workflow
            handle = await env.client.start_workflow(
                OcrSubmitWorkflow.run,
                ocr_input,
                id=f"e2e-ocr-{Path(file_path).stem}",
                task_queue=FORGE_TASK_QUEUE,
            )

            # Poll for batch completion (calls real Mistral API)
            await _poll_until_complete(_POLL_TIMEOUT)

            # Await the workflow result
            result = await handle.result()

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
