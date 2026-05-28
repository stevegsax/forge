"""Tests for synchronous OCR path — call_ocr provider method, activities, and workflow."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from sax_llm.models import ExtractedImage
from temporalio import activity
from temporalio.worker import Worker

from forge.ocr.activities import execute_call_ocr_sync
from forge.ocr.models import (
    ChunkRef,
    FileContentRef,
    OcrDuplicateCheckResult,
    OcrStoreResult,
    OcrSyncInput,
    SplitResult,
)
from forge.ocr.workflow_sync import OcrSyncWorkflow
from forge.store import run_migrations
from forge.workflows import FORGE_TASK_QUEUE

if TYPE_CHECKING:
    from pathlib import Path

    from temporalio.testing import WorkflowEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_db(tmp_path: Path):
    """Create a test database with migrations applied."""
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    run_migrations(url)
    return sa.create_engine(url), db_path


def _build_ocr_response(
    text: str = "Extracted text",
    model: str = "mistral-ocr-latest",
    pages_processed: int = 1,
    doc_size_bytes: int = 1000,
    images: list | None = None,
) -> dict:
    """Build a mock OCR response dict matching Mistral output shape."""
    page = {"index": 0, "markdown": text, "images": images or []}
    return {
        "pages": [page],
        "model": model,
        "usage_info": {
            "pages_processed": pages_processed,
            "doc_size_bytes": doc_size_bytes,
        },
    }


def _build_mock_provider(response_body: dict | None = None) -> MagicMock:
    """Build a mock LLMProvider with call_ocr returning the given body."""
    provider = MagicMock()
    provider.supports_sync_ocr = True
    body = response_body or _build_ocr_response()
    provider.call_ocr = AsyncMock(return_value=body)
    return provider


# ---------------------------------------------------------------------------
# MistralProvider.call_ocr
# ---------------------------------------------------------------------------


class TestMistralProviderCallOcr:
    """Test MistralProvider.call_ocr wraps the SDK correctly."""

    @pytest.mark.asyncio
    async def test_calls_process_async_with_document_url(self) -> None:
        """PDF documents use DocumentURLChunk."""
        from sax_llm.mistral import MistralProvider

        provider = MistralProvider.__new__(MistralProvider)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = _build_ocr_response()
        mock_client.ocr.process_async = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.call_ocr(
            document_data_uri="data:application/pdf;base64,dGVzdA==",
            model="mistral-ocr-latest",
        )

        assert result == _build_ocr_response()
        mock_client.ocr.process_async.assert_called_once()
        call_kwargs = mock_client.ocr.process_async.call_args.kwargs
        assert call_kwargs["model"] == "mistral-ocr-latest"
        assert call_kwargs["include_image_base64"] is True

    @pytest.mark.asyncio
    async def test_calls_process_async_with_image_url(self) -> None:
        """Image files use ImageURLChunk."""
        from sax_llm.mistral import MistralProvider

        provider = MistralProvider.__new__(MistralProvider)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = _build_ocr_response()
        mock_client.ocr.process_async = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        await provider.call_ocr(
            document_data_uri="data:image/png;base64,dGVzdA==",
            model="mistral-ocr-latest",
        )

        call_kwargs = mock_client.ocr.process_async.call_args.kwargs
        # Should pass an ImageURLChunk, not DocumentURLChunk
        doc = call_kwargs["document"]
        assert hasattr(doc, "image_url")


# ---------------------------------------------------------------------------
# execute_call_ocr_sync (testable function)
# ---------------------------------------------------------------------------


class TestExecuteCallOcrSync:
    """Test the synchronous OCR testable function."""

    @pytest.mark.asyncio
    async def test_basic_ocr_and_store(self, tmp_path: Path) -> None:
        """Calls provider, parses result, stores in DB."""
        engine, _ = _setup_db(tmp_path)
        provider = _build_mock_provider()

        import forge.store as store_module

        original_get_store_engine = store_module.get_store_engine
        try:
            store_module.get_store_engine = lambda: engine

            result = await execute_call_ocr_sync(
                base64_data="dGVzdA==",
                mime_type="application/pdf",
                model_name="mistral:mistral-ocr-latest",
                document_id="doc-sync-1",
                file_path="/tmp/test.pdf",
                workflow_id="wf-sync-1",
                provider=provider,
            )

            assert result.document_id == "doc-sync-1"
            assert result.stored is True
            assert result.text_length == len("Extracted text")

            # Verify provider was called correctly
            provider.call_ocr.assert_called_once()
            call_kwargs = provider.call_ocr.call_args.kwargs
            assert call_kwargs["model"] == "mistral-ocr-latest"
            assert call_kwargs["document_data_uri"] == "data:application/pdf;base64,dGVzdA=="
        finally:
            store_module.get_store_engine = original_get_store_engine

    @pytest.mark.asyncio
    async def test_image_extraction_and_storage(self, tmp_path: Path) -> None:
        """Images in the OCR response are extracted, stored, and referenced."""
        engine, _ = _setup_db(tmp_path)

        # OCR response with an image
        response_body = _build_ocr_response(
            text="See ![chart](img-0.jpeg) for details.",
            images=[{
                "id": "img-0.jpeg",
                "image_base64": "data:image/jpeg;base64,"
                + base64.b64encode(b"\xff\xd8\xff\xe0test-jpeg").decode(),
                "top_left_x": 0,
                "top_left_y": 0,
                "bottom_right_x": 100,
                "bottom_right_y": 100,
            }],
        )
        provider = _build_mock_provider(response_body)

        # Track stored images
        stored_images: list[ExtractedImage] = []

        def mock_store_images(images: list[ExtractedImage]) -> dict[str, str]:
            stored_images.extend(images)
            return {"img-0.jpeg": "uuid-0001"}

        import forge.store as store_module

        original_get_store_engine = store_module.get_store_engine
        try:
            store_module.get_store_engine = lambda: engine

            result = await execute_call_ocr_sync(
                base64_data="dGVzdA==",
                mime_type="application/pdf",
                model_name="mistral:mistral-ocr-latest",
                document_id="doc-img-1",
                file_path="/tmp/test.pdf",
                workflow_id="wf-img-1",
                provider=provider,
                store_images_fn=mock_store_images,
            )

            assert result.stored is True
            # Verify image was extracted
            assert len(stored_images) == 1
            assert stored_images[0].original_image_id == "img-0.jpeg"
        finally:
            store_module.get_store_engine = original_get_store_engine

    @pytest.mark.asyncio
    async def test_no_store_images_fn_skips_images(self, tmp_path: Path) -> None:
        """When store_images_fn is None, images are extracted but not stored."""
        engine, _ = _setup_db(tmp_path)

        response_body = _build_ocr_response(
            text="Has image ![x](img-0.jpeg).",
            images=[{
                "id": "img-0.jpeg",
                "image_base64": "data:image/jpeg;base64,/9j/4AAQ",
                "top_left_x": None,
                "top_left_y": None,
                "bottom_right_x": None,
                "bottom_right_y": None,
            }],
        )
        provider = _build_mock_provider(response_body)

        import forge.store as store_module

        original_get_store_engine = store_module.get_store_engine
        try:
            store_module.get_store_engine = lambda: engine

            result = await execute_call_ocr_sync(
                base64_data="dGVzdA==",
                mime_type="application/pdf",
                model_name="mistral:mistral-ocr-latest",
                document_id="doc-no-store",
                file_path="/tmp/test.pdf",
                workflow_id="wf-1",
                provider=provider,
                store_images_fn=None,
            )

            # Should still succeed — just no image mapping in output
            assert result.stored is True
            # Image refs are NOT rewritten (no mapping)
            assert result.text_length > 0
        finally:
            store_module.get_store_engine = original_get_store_engine


# ---------------------------------------------------------------------------
# OcrSyncWorkflow
# ---------------------------------------------------------------------------


class TestOcrSyncWorkflow:
    """Workflow-level tests for OcrSyncWorkflow."""

    @pytest.mark.asyncio
    async def test_single_chunk_workflow(self, env: WorkflowEnvironment) -> None:
        """Single-chunk document goes through read → split → sync OCR → done."""

        @activity.defn(name="check_ocr_duplicate")
        async def mock_check_dup(_file_path: str) -> OcrDuplicateCheckResult:
            return OcrDuplicateCheckResult(is_duplicate=False)

        @activity.defn(name="read_and_store_file_content")
        async def mock_read(_file_path: str) -> FileContentRef:
            return FileContentRef(
                content_id="content-1",
                mime_type="application/pdf",
                file_size_bytes=1000,
            )

        @activity.defn(name="split_file_into_chunks")
        async def mock_split(_input_json: str) -> SplitResult:
            return SplitResult(
                chunks=[ChunkRef(
                    content_id="content-1",
                    mime_type="application/pdf",
                    file_size_bytes=1000,
                    chunk_index=0,
                    page_start=1,
                    page_end=5,
                )],
                total_pages=5,
                original_content_id="content-1",
            )

        @activity.defn(name="call_ocr_sync")
        async def mock_call_ocr_sync(_input_json: str) -> OcrStoreResult:
            data = json.loads(_input_json)
            return OcrStoreResult(
                document_id=data["document_id"],
                text_length=100,
                page_count=5,
            )

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrSyncWorkflow],
            activities=[
                mock_check_dup,
                mock_read,
                mock_split,
                mock_call_ocr_sync,
            ],
        ):
            result = await env.client.execute_workflow(
                OcrSyncWorkflow.run,
                OcrSyncInput(file_path="/tmp/test.pdf"),
                id="test-ocr-sync-single",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.text_length == 100
        assert result.page_count == 5
        assert result.stored is True

    @pytest.mark.asyncio
    async def test_duplicate_skips(self, env: WorkflowEnvironment) -> None:
        """Duplicate document is detected and skipped."""

        @activity.defn(name="check_ocr_duplicate")
        async def mock_check_dup(_file_path: str) -> OcrDuplicateCheckResult:
            return OcrDuplicateCheckResult(
                is_duplicate=True,
                existing_document_id="existing-doc",
            )

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrSyncWorkflow],
            activities=[mock_check_dup],
        ):
            result = await env.client.execute_workflow(
                OcrSyncWorkflow.run,
                OcrSyncInput(file_path="/tmp/test.pdf"),
                id="test-ocr-sync-dup",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.skipped is True
        assert result.document_id == "existing-doc"

    @pytest.mark.asyncio
    async def test_multi_chunk_reassembles(self, env: WorkflowEnvironment) -> None:
        """Multi-chunk documents call OCR per chunk then reassemble."""
        call_count = 0

        @activity.defn(name="check_ocr_duplicate")
        async def mock_check_dup(_file_path: str) -> OcrDuplicateCheckResult:
            return OcrDuplicateCheckResult(is_duplicate=False)

        @activity.defn(name="read_and_store_file_content")
        async def mock_read(_file_path: str) -> FileContentRef:
            return FileContentRef(
                content_id="content-big",
                mime_type="application/pdf",
                file_size_bytes=20_000_000,
            )

        @activity.defn(name="split_file_into_chunks")
        async def mock_split(_input_json: str) -> SplitResult:
            return SplitResult(
                chunks=[
                    ChunkRef(
                        content_id="chunk-0",
                        mime_type="application/pdf",
                        file_size_bytes=5_000_000,
                        chunk_index=0,
                        page_start=1,
                        page_end=25,
                    ),
                    ChunkRef(
                        content_id="chunk-1",
                        mime_type="application/pdf",
                        file_size_bytes=5_000_000,
                        chunk_index=1,
                        page_start=26,
                        page_end=50,
                    ),
                ],
                total_pages=50,
                original_content_id="content-big",
            )

        @activity.defn(name="call_ocr_sync")
        async def mock_call_ocr_sync(_input_json: str) -> OcrStoreResult:
            nonlocal call_count
            call_count += 1
            data = json.loads(_input_json)
            return OcrStoreResult(
                document_id=data["document_id"],
                text_length=500,
                page_count=25,
            )

        @activity.defn(name="reassemble_ocr_chunks")
        async def mock_reassemble(_input_json: str) -> OcrStoreResult:
            data = json.loads(_input_json)
            return OcrStoreResult(
                document_id=data["document_id"],
                text_length=1000,
                page_count=data["total_pages"],
            )

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrSyncWorkflow],
            activities=[
                mock_check_dup,
                mock_read,
                mock_split,
                mock_call_ocr_sync,
                mock_reassemble,
            ],
        ):
            result = await env.client.execute_workflow(
                OcrSyncWorkflow.run,
                OcrSyncInput(file_path="/tmp/large.pdf"),
                id="test-ocr-sync-multi",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert call_count == 2
        assert result.text_length == 1000
        assert result.page_count == 50

    @pytest.mark.asyncio
    async def test_skip_duplicate_detection_bypasses_check(self, env: WorkflowEnvironment) -> None:
        """skip_duplicate_detection=True bypasses duplicate detection."""

        @activity.defn(name="read_and_store_file_content")
        async def mock_read(_file_path: str) -> FileContentRef:
            return FileContentRef(
                content_id="content-1",
                mime_type="application/pdf",
                file_size_bytes=1000,
            )

        @activity.defn(name="split_file_into_chunks")
        async def mock_split(_input_json: str) -> SplitResult:
            return SplitResult(
                chunks=[ChunkRef(
                    content_id="content-1",
                    mime_type="application/pdf",
                    file_size_bytes=1000,
                    chunk_index=0,
                    page_start=1,
                    page_end=1,
                )],
                total_pages=1,
                original_content_id="content-1",
            )

        @activity.defn(name="call_ocr_sync")
        async def mock_call_ocr_sync(_input_json: str) -> OcrStoreResult:
            data = json.loads(_input_json)
            return OcrStoreResult(
                document_id=data["document_id"],
                text_length=50,
                page_count=1,
            )

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrSyncWorkflow],
            activities=[mock_read, mock_split, mock_call_ocr_sync],
        ):
            result = await env.client.execute_workflow(
                OcrSyncWorkflow.run,
                OcrSyncInput(file_path="/tmp/test.pdf", skip_duplicate_detection=True),
                id="test-ocr-sync-skip-dup",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.stored is True
        assert result.text_length == 50
