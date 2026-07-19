"""End-to-end OCR test against the real platform + Mistral batch API.

This is a CONSUMER-SIDE e2e: OCR consumes the platform, so the platform must be
running. It runs the OCR worker in-process and submits a real document; the running
forge platform worker services the opaque-blob submit SPI, the batch poller, and the
batch_jobs record. The full path exercised is:

    OcrSubmitWorkflow (build request -> S3, submit SPI cross-queue, record batch_jobs)
      -> Mistral batch API -> platform poller -> batch_result_received signal
      -> OcrStoreWorkflow (store images + text, write ocr_job_status='stored')

Prerequisites (the test SKIPs unless all are present):
  - MISTRAL_API_KEY            real Mistral key (provider submit happens platform-side)
  - FORGE_OCR_S3_BUCKET        real S3 bucket both workers can read/write
  - FORGE_DB_URL               the shared DB both workers use
  - OCR_E2E_PLATFORM=1         operator confirms a forge platform worker is running on
                               forge-task-queue against the SAME Temporal / DB / S3
                               (start it with e.g. `forge worker --batch-poll-interval 30`
                               so the poll cycle is fast enough for the test)

Optional:
  - FORGE_TEMPORAL_ADDRESS     default localhost:7233
  - OCR_E2E_TIMEOUT            seconds to wait for the batch to land (default 1800)

Run with::

    OCR_E2E_PLATFORM=1 MISTRAL_API_KEY=... FORGE_OCR_S3_BUCKET=... \
      FORGE_DB_URL=postgresql+psycopg2://... uv run pytest -m e2e -v
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest
from sax_platform.contracts.constants import OCR_TASK_QUEUE
from sax_platform.contracts.s3_blobs import S3Blobs
from sax_platform.db import get_store_engine
from sax_platform.temporal.client import connect_temporal
from temporalio.worker import Worker

from ocr.activities import OcrStoreActivities
from ocr.models import OcrSubmitInput
from ocr.store import get_ocr_job_status, get_ocr_result, run_migrations
from ocr.worker import activity_methods
from ocr.worker import workflows as ocr_workflows
from ocr.workflow_submit import OcrSubmitWorkflow

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ocr"
_DEFAULT_TIMEOUT = 1800
_POLL_INTERVAL = 10

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not os.environ.get("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not set"),
    pytest.mark.skipif(
        not os.environ.get("FORGE_OCR_S3_BUCKET"), reason="FORGE_OCR_S3_BUCKET not set"
    ),
    pytest.mark.skipif(not os.environ.get("FORGE_DB_URL"), reason="FORGE_DB_URL not set"),
    pytest.mark.skipif(
        not os.environ.get("OCR_E2E_PLATFORM"),
        reason="set OCR_E2E_PLATFORM=1 to confirm a forge platform worker is running",
    ),
]


@pytest.fixture(scope="module")
def fixtures_ready() -> Path:
    """Ensure the fixture images exist (regenerate via PyMuPDF if missing)."""
    jpeg = FIXTURES_DIR / "hello_jpeg.jpg"
    png = FIXTURES_DIR / "hello_png.png"
    if not (jpeg.exists() and png.exists()):
        import fitz

        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        for text, path in (("Hello from JPEG", jpeg), ("Hello from PNG", png)):
            doc = fitz.open()
            page = doc.new_page(width=400, height=100)
            page.insert_text(fitz.Point(20, 50 + 8), text, fontsize=24, color=(0, 0, 0))
            page.get_pixmap(dpi=150).save(str(path))
            doc.close()
    return FIXTURES_DIR


@pytest.fixture
def migrated_real_store() -> str:
    """Apply the OCR Alembic chain to the shared DB (platform owns its own chain)."""
    url = os.environ["FORGE_DB_URL"]
    run_migrations(url)
    return url


async def _submit_and_await_stored(file_path: str, expected_words: list[str]) -> None:
    address = os.environ.get("FORGE_TEMPORAL_ADDRESS", "localhost:7233")
    timeout = int(os.environ.get("OCR_E2E_TIMEOUT", _DEFAULT_TIMEOUT))
    client = await connect_temporal(address)

    engine = get_store_engine(os.environ["FORGE_DB_URL"])
    blobs = S3Blobs(os.environ["FORGE_OCR_S3_BUCKET"], os.environ.get("FORGE_OCR_S3_PREFIX", ""))
    store = OcrStoreActivities(engine, blobs)

    async with Worker(
        client,
        task_queue=OCR_TASK_QUEUE,
        workflows=ocr_workflows(),
        activities=activity_methods(store),
    ):
        result = await client.execute_workflow(
            OcrSubmitWorkflow.run,
            OcrSubmitInput(file_path=file_path, skip_duplicate_detection=True),
            id=f"e2e-ocr-submit-{Path(file_path).stem}-{int(time.time())}",
            task_queue=OCR_TASK_QUEUE,
        )
        assert result.chunk_count >= 1
        assert result.batch_refs, "submit returned no batch refs"
        request_id = result.batch_refs[0].request_id
        document_id = result.document_id

        # The store child waits for the platform poller's signal; poll the OCR-owned
        # status projection until it reaches a terminal state (or we time out).
        deadline = time.monotonic() + timeout
        row = None
        while time.monotonic() < deadline:
            row = get_ocr_job_status(engine, request_id)
            if row is not None and row["status"] in ("stored", "failed"):
                break
            await asyncio.sleep(_POLL_INTERVAL)

        assert row is not None, f"no ocr_job_status row for request_id={request_id}"
        assert row["status"] == "stored", f"terminal status was {row['status']}: {row}"

        stored = get_ocr_result(engine, document_id)
        assert stored is not None, f"no ocr_results row for document_id={document_id}"
        text_lower = stored["text"].lower()
        for word in expected_words:
            assert word.lower() in text_lower, (
                f"expected '{word}' not in extracted text: {stored['text']!r}"
            )


class TestOcrBatchE2E:
    @pytest.mark.asyncio
    async def test_jpeg_batch_ocr(self, fixtures_ready: Path, migrated_real_store: str) -> None:
        await _submit_and_await_stored(
            str(fixtures_ready / "hello_jpeg.jpg"), expected_words=["hello", "jpeg"]
        )

    @pytest.mark.asyncio
    async def test_png_batch_ocr(self, fixtures_ready: Path, migrated_real_store: str) -> None:
        await _submit_and_await_stored(
            str(fixtures_ready / "hello_png.png"), expected_words=["hello", "png"]
        )
