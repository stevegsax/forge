"""Tests for forge.ocr — OCR models, activities, store functions, and workflow."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.client import WorkflowFailureError
from temporalio.worker import Worker

from forge.models import BatchResult
from forge.ocr.activities import (
    CHUNK_SIZE_PAGES,
    MAX_FILE_SIZE_BYTES,
    MAX_PAGES,
    build_ocr_batch_body,
    build_ocr_messages,
    detect_mime_type,
    execute_parse_ocr_result,
    execute_read_and_store_file,
    execute_read_file,
    execute_reassemble_ocr_chunks,
    execute_split_file_into_chunks,
    execute_store_ocr_result,
    execute_submit_ocr_batch,
    validate_file_size,
)
from forge.ocr.models import (
    FileContentRef,
    FileContentResult,
    OcrParseResult,
    OcrStoreInput,
    OcrStoreResult,
    OcrSubmitInput,
    OcrSubmitResult,
)
from forge.ocr.workflow_store import OcrStoreWorkflow
from forge.store import (
    delete_ocr_results,
    get_engine,
    get_file_content,
    get_ocr_result,
    run_migrations,
    save_file_content,
    save_ocr_result,
)
from forge.workflows import FORGE_TASK_QUEUE

if TYPE_CHECKING:
    from pathlib import Path

    from temporalio.testing import WorkflowEnvironment


def _setup_db(tmp_path: Path):
    """Create a test database with migrations applied."""
    db_path = tmp_path / "test.db"
    run_migrations(db_path)
    return get_engine(db_path), db_path


# ---------------------------------------------------------------------------
# OCR Models
# ---------------------------------------------------------------------------


class TestOcrModels:
    def test_submit_input_defaults(self) -> None:
        inp = OcrSubmitInput(file_path="/tmp/test.pdf")
        assert inp.model_name == "mistral:mistral-ocr-latest"
        assert inp.max_tokens == 16384
        assert inp.document_id == ""

    def test_submit_result_defaults(self) -> None:
        result = OcrSubmitResult(
            batch_id="b-1",
            request_id="r-1",
            document_id="d-1",
            workflow_id="wf-1",
        )
        assert result.chunk_count == 1
        assert result.total_pages == 0

    def test_submit_input_custom(self) -> None:
        inp = OcrSubmitInput(
            file_path="/tmp/test.pdf",
            model_name="mistral:custom-model",
            max_tokens=8192,
            document_id="doc-123",
        )
        assert inp.model_name == "mistral:custom-model"
        assert inp.max_tokens == 8192
        assert inp.document_id == "doc-123"

    def test_submit_result(self) -> None:
        result = OcrSubmitResult(
            batch_id="batch-1",
            request_id="req-1",
            document_id="doc-1",
            workflow_id="wf-1",
        )
        assert result.batch_id == "batch-1"
        assert result.workflow_id == "wf-1"

    def test_store_input(self) -> None:
        inp = OcrStoreInput(
            batch_id="b-1",
            request_id="r-1",
            document_id="d-1",
            file_path="/tmp/test.pdf",
        )
        assert inp.document_id == "d-1"

    def test_store_result_defaults(self) -> None:
        result = OcrStoreResult(document_id="d-1", text_length=100)
        assert result.page_count == 0
        assert result.stored is True

    def test_store_result_not_stored(self) -> None:
        result = OcrStoreResult(document_id="d-1", text_length=0, stored=False)
        assert result.stored is False

    def test_file_content_result(self) -> None:
        result = FileContentResult(
            base64_data="dGVzdA==",
            mime_type="application/pdf",
            file_size_bytes=4,
        )
        assert result.base64_data == "dGVzdA=="
        assert result.mime_type == "application/pdf"

    def test_parse_result(self) -> None:
        result = OcrParseResult(
            text="Hello world",
            model_name="pixtral-large",
            input_tokens=50,
            output_tokens=10,
        )
        assert result.text == "Hello world"


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestDetectMimeType:
    def test_pdf(self) -> None:
        assert detect_mime_type("/tmp/test.pdf") == "application/pdf"

    def test_png(self) -> None:
        assert detect_mime_type("/tmp/test.png") == "image/png"

    def test_jpeg(self) -> None:
        assert detect_mime_type("/tmp/test.jpg") == "image/jpeg"

    def test_unknown(self) -> None:
        assert detect_mime_type("/tmp/test.xyz123") == "application/octet-stream"


class TestBuildOcrMessages:
    def test_image_content(self) -> None:
        messages = build_ocr_messages("base64data", "image/png")
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        # User message should have image + text content blocks
        assert isinstance(messages[1].content, list)
        assert len(messages[1].content) == 2
        assert messages[1].content[0].type == "image"
        assert messages[1].content[0].media_type == "image/png"

    def test_document_content(self) -> None:
        messages = build_ocr_messages("base64data", "application/pdf")
        assert len(messages) == 2
        assert isinstance(messages[1].content, list)
        assert messages[1].content[0].type == "document"
        assert messages[1].content[0].media_type == "application/pdf"

    def test_custom_instruction(self) -> None:
        messages = build_ocr_messages("data", "image/png", instruction="Summarize this.")
        text_block = messages[1].content[1]
        assert text_block.text == "Summarize this."


class TestBuildOcrBatchBody:
    def test_image_type(self) -> None:
        body = build_ocr_batch_body("abc123", "image/png")
        assert body == {
            "document": {
                "type": "image_url",
                "image_url": "data:image/png;base64,abc123",
            }
        }

    def test_document_type(self) -> None:
        body = build_ocr_batch_body("abc123", "application/pdf")
        assert body == {
            "document": {
                "type": "document_url",
                "document_url": "data:application/pdf;base64,abc123",
            }
        }

    def test_image_jpeg(self) -> None:
        body = build_ocr_batch_body("data", "image/jpeg")
        assert body["document"]["type"] == "image_url"
        assert "data:image/jpeg;base64,data" in body["document"]["image_url"]


# ---------------------------------------------------------------------------
# Testable functions
# ---------------------------------------------------------------------------


class TestExecuteReadFile:
    def test_reads_and_encodes(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, OCR!")

        result = execute_read_file(str(test_file))
        assert result.mime_type == "text/plain"
        assert result.file_size_bytes == len(b"Hello, OCR!")
        decoded = base64.b64decode(result.base64_data)
        assert decoded == b"Hello, OCR!"

    def test_binary_file(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.pdf"
        content = b"%PDF-1.4 fake content"
        test_file.write_bytes(content)

        result = execute_read_file(str(test_file))
        assert result.mime_type == "application/pdf"
        assert result.file_size_bytes == len(content)
        assert base64.b64decode(result.base64_data) == content

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            execute_read_file("/tmp/nonexistent_file_12345.pdf")


class TestExecuteSubmitOcrBatch:
    @pytest.mark.asyncio
    async def test_submits_batch(self) -> None:
        mock_provider = MagicMock()
        mock_provider.submit_batch = AsyncMock(return_value="batch-123")

        inp = OcrSubmitInput(
            file_path="/tmp/test.pdf",
            model_name="mistral:mistral-ocr-latest",
            document_id="doc-1",
        )
        file_content = FileContentResult(
            base64_data="dGVzdA==",
            mime_type="application/pdf",
            file_size_bytes=4,
        )

        result = await execute_submit_ocr_batch(
            inp, file_content, mock_provider, "wf-store-1"
        )

        assert result.batch_id == "batch-123"
        assert result.document_id == "doc-1"
        assert result.workflow_id == "wf-store-1"
        assert result.request_id  # non-empty UUID

        # Verify OCR body format and endpoint
        mock_provider.submit_batch.assert_called_once()
        call_args = mock_provider.submit_batch.call_args
        requests = call_args.args[0]
        assert len(requests) == 1
        assert "body" in requests[0]
        assert "document" in requests[0]["body"]
        assert requests[0]["body"]["document"]["type"] == "document_url"
        assert call_args.kwargs.get("endpoint") == "/v1/ocr"

    @pytest.mark.asyncio
    async def test_auto_generates_document_id(self) -> None:
        mock_provider = MagicMock()
        mock_provider.submit_batch = AsyncMock(return_value="batch-456")

        inp = OcrSubmitInput(
            file_path="/tmp/test.pdf",
            document_id="",  # empty = auto-generate
        )
        file_content = FileContentResult(
            base64_data="dGVzdA==",
            mime_type="application/pdf",
            file_size_bytes=4,
        )

        result = await execute_submit_ocr_batch(
            inp, file_content, mock_provider, "wf-store-2"
        )
        assert result.document_id  # auto-generated UUID
        assert result.document_id != ""

    @pytest.mark.asyncio
    async def test_image_uses_image_url_type(self) -> None:
        mock_provider = MagicMock()
        mock_provider.submit_batch = AsyncMock(return_value="batch-789")

        inp = OcrSubmitInput(
            file_path="/tmp/test.png",
            model_name="mistral:mistral-ocr-latest",
            document_id="doc-img",
        )
        file_content = FileContentResult(
            base64_data="aW1hZ2U=",
            mime_type="image/png",
            file_size_bytes=5,
        )

        result = await execute_submit_ocr_batch(
            inp, file_content, mock_provider, "wf-store-img"
        )
        assert result.batch_id == "batch-789"

        requests = mock_provider.submit_batch.call_args.args[0]
        assert requests[0]["body"]["document"]["type"] == "image_url"


class TestExecuteParseOcrResult:
    def test_parses_ocr_response(self) -> None:
        raw_json = json.dumps({
            "pages": [
                {"markdown": "Page one text."},
                {"markdown": "Page two text."},
            ],
            "model": "mistral-ocr-latest",
            "usage_info": {
                "pages_processed": 2,
                "doc_size_bytes": 50000,
            },
        })

        result = execute_parse_ocr_result(raw_json)
        assert result.text == "Page one text.\n\nPage two text."
        assert result.model_name == "mistral-ocr-latest"
        assert result.input_tokens == 2
        assert result.output_tokens == 50000
        assert result.page_count == 2

    def test_single_page(self) -> None:
        raw_json = json.dumps({
            "pages": [{"markdown": "Only page."}],
            "model": "mistral-ocr-latest",
            "usage_info": {"pages_processed": 1, "doc_size_bytes": 1000},
        })

        result = execute_parse_ocr_result(raw_json)
        assert result.text == "Only page."
        assert result.input_tokens == 1

    def test_empty_pages_returns_empty_string(self) -> None:
        raw_json = json.dumps({
            "pages": [],
            "model": "mistral-ocr-latest",
            "usage_info": {},
        })

        result = execute_parse_ocr_result(raw_json)
        assert result.text == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_missing_usage_info(self) -> None:
        raw_json = json.dumps({
            "pages": [{"markdown": "Some text."}],
            "model": "mistral-ocr-latest",
        })

        result = execute_parse_ocr_result(raw_json)
        assert result.text == "Some text."
        assert result.input_tokens == 0
        assert result.output_tokens == 0


class TestExecuteStoreOcrResult:
    def test_stores_result(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        import forge.store as store_module

        original_get_db_path = store_module.get_db_path
        original_get_engine = store_module.get_engine
        try:
            store_module.get_db_path = lambda: db_path
            store_module.get_engine = lambda _path: engine

            result = execute_store_ocr_result(
                document_id="doc-123",
                file_path="/tmp/test.pdf",
                text="Extracted text content",
                model_name="pixtral-large",
                input_tokens=500,
                output_tokens=100,
                batch_id="batch-1",
                workflow_id="wf-1",
            )

            assert result.document_id == "doc-123"
            assert result.text_length == len("Extracted text content")
            assert result.stored is True

            # Verify the data was actually stored
            stored = get_ocr_result(engine, "doc-123")
            assert stored is not None
            assert stored["text"] == "Extracted text content"
            assert stored["model_name"] == "pixtral-large"
        finally:
            store_module.get_db_path = original_get_db_path
            store_module.get_engine = original_get_engine

    def test_returns_not_stored_when_no_db(self) -> None:
        import forge.store as store_module

        original = store_module.get_db_path
        try:
            store_module.get_db_path = lambda: None

            result = execute_store_ocr_result(
                document_id="doc-no-db",
                file_path="/tmp/test.pdf",
                text="Some text",
                model_name="model",
                input_tokens=10,
                output_tokens=5,
                batch_id="b-1",
                workflow_id="wf-1",
            )

            assert result.stored is False
            assert result.text_length == len("Some text")
        finally:
            store_module.get_db_path = original


# ---------------------------------------------------------------------------
# Store functions
# ---------------------------------------------------------------------------


class TestOcrStoreRoundtrip:
    def test_save_and_get(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        save_ocr_result(
            engine,
            document_id="doc-001",
            file_path="/data/sample.pdf",
            text="The quick brown fox jumps over the lazy dog.",
            model_name="pixtral-large-latest",
            input_tokens=1000,
            output_tokens=200,
            batch_id="batch-abc",
            workflow_id="wf-store-001",
        )

        result = get_ocr_result(engine, "doc-001")
        assert result is not None
        assert result["document_id"] == "doc-001"
        assert result["file_path"] == "/data/sample.pdf"
        assert result["text"] == "The quick brown fox jumps over the lazy dog."
        assert result["model_name"] == "pixtral-large-latest"
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 200
        assert result["batch_id"] == "batch-abc"
        assert result["workflow_id"] == "wf-store-001"
        assert result["page_count"] == 0  # default

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        result = get_ocr_result(engine, "nonexistent")
        assert result is None

    def test_save_with_page_count(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        save_ocr_result(
            engine,
            document_id="doc-002",
            file_path="/data/multi.pdf",
            text="Page 1 text. Page 2 text.",
            page_count=2,
            model_name="pixtral-large",
            input_tokens=2000,
            output_tokens=400,
            batch_id="batch-def",
            workflow_id="wf-store-002",
        )

        result = get_ocr_result(engine, "doc-002")
        assert result is not None
        assert result["page_count"] == 2

    def test_unique_document_id(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        save_ocr_result(
            engine,
            document_id="doc-unique",
            file_path="/data/a.pdf",
            text="First",
            model_name="model",
            input_tokens=10,
            output_tokens=5,
            batch_id="b-1",
            workflow_id="wf-1",
        )

        # Inserting duplicate document_id should raise IntegrityError
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            save_ocr_result(
                engine,
                document_id="doc-unique",
                file_path="/data/b.pdf",
                text="Second",
                model_name="model",
                input_tokens=10,
                output_tokens=5,
                batch_id="b-2",
                workflow_id="wf-2",
            )


# ---------------------------------------------------------------------------
# Migration 006
# ---------------------------------------------------------------------------


class TestMigration006:
    def test_creates_ocr_results_table(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        # Should be able to insert and query
        save_ocr_result(
            engine,
            document_id="mig-test",
            file_path="/tmp/test.pdf",
            text="Migration test text",
            model_name="test-model",
            input_tokens=10,
            output_tokens=5,
            batch_id="b-mig",
            workflow_id="wf-mig",
        )

        result = get_ocr_result(engine, "mig-test")
        assert result is not None
        assert result["text"] == "Migration test text"


# ---------------------------------------------------------------------------
# Read and store file content
# ---------------------------------------------------------------------------


class TestReadAndStoreFileContent:
    def test_execute_read_and_store_file(self, tmp_path: Path) -> None:
        # Create a test file
        test_file = tmp_path / "test.pdf"
        content = b"%PDF-1.4 fake pdf content for testing"
        test_file.write_bytes(content)

        # Set up test database
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        ref = execute_read_and_store_file(str(test_file), engine)

        # Verify the returned ref
        assert isinstance(ref, FileContentRef)
        assert ref.mime_type == "application/pdf"
        assert ref.file_size_bytes == len(content)
        assert ref.content_id  # non-empty UUID

        # Verify the bytes were stored in the database
        blob = get_file_content(engine, ref.content_id)
        assert blob is not None
        assert blob["data"] == content
        assert blob["mime_type"] == "application/pdf"
        assert blob["file_size_bytes"] == len(content)

    def test_execute_read_and_store_file_not_found(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        with pytest.raises(FileNotFoundError):
            execute_read_and_store_file("/tmp/nonexistent_file_12345.pdf", engine)

    @pytest.mark.asyncio
    async def test_submit_ocr_batch_loads_from_db(self, tmp_path: Path) -> None:
        """Verify submit_ocr_batch loads content from DB using file_content_ref."""
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        # Pre-store file content
        test_data = b"fake pdf bytes"
        content_id = "test-content-id"
        save_file_content(
            engine,
            content_id=content_id,
            data=test_data,
            mime_type="application/pdf",
            file_size_bytes=len(test_data),
        )

        # Build the input JSON with file_content_ref instead of file_content
        submit_input = OcrSubmitInput(
            file_path="/tmp/test.pdf",
            model_name="mistral:mistral-ocr-latest",
            document_id="doc-1",
        )
        input_json = json.dumps({
            "submit_input": submit_input.model_dump(),
            "file_content_ref": {
                "content_id": content_id,
                "mime_type": "application/pdf",
                "file_size_bytes": len(test_data),
            },
            "store_workflow_id": "wf-store-1",
        })

        # Mock the provider and store functions
        mock_provider = MagicMock()
        mock_provider.submit_batch = AsyncMock(return_value="batch-123")

        import forge.llm_providers as llm_mod
        import forge.store as store_mod

        original_get_provider = llm_mod.get_provider
        original_get_db_path = store_mod.get_db_path
        original_get_engine = store_mod.get_engine
        try:
            llm_mod.get_provider = MagicMock(return_value=mock_provider)
            store_mod.get_db_path = lambda: db_path
            store_mod.get_engine = lambda _path: engine

            from forge.ocr.activities import submit_ocr_batch

            result = await submit_ocr_batch(input_json)

            assert result.batch_id == "batch-123"
            assert result.document_id == "doc-1"

            # Verify OCR body format and endpoint were used
            call_args = mock_provider.submit_batch.call_args
            requests = call_args.args[0]
            assert "document" in requests[0]["body"]
            assert call_args.kwargs.get("endpoint") == "/v1/ocr"
        finally:
            llm_mod.get_provider = original_get_provider
            store_mod.get_db_path = original_get_db_path
            store_mod.get_engine = original_get_engine

    @pytest.mark.asyncio
    async def test_submit_ocr_batch_cleans_up_blob(self, tmp_path: Path) -> None:
        """Verify BLOB is deleted after successful submission."""
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        # Pre-store file content
        test_data = b"cleanup test bytes"
        content_id = "cleanup-content-id"
        save_file_content(
            engine,
            content_id=content_id,
            data=test_data,
            mime_type="application/pdf",
            file_size_bytes=len(test_data),
        )

        # Verify blob exists before submission
        assert get_file_content(engine, content_id) is not None

        submit_input = OcrSubmitInput(
            file_path="/tmp/test.pdf",
            model_name="mistral:mistral-ocr-latest",
            document_id="doc-cleanup",
        )
        input_json = json.dumps({
            "submit_input": submit_input.model_dump(),
            "file_content_ref": {
                "content_id": content_id,
                "mime_type": "application/pdf",
                "file_size_bytes": len(test_data),
            },
            "store_workflow_id": "wf-store-cleanup",
        })

        mock_provider = MagicMock()
        mock_provider.submit_batch = AsyncMock(return_value="batch-456")

        import forge.llm_providers as llm_mod
        import forge.store as store_mod

        original_get_provider = llm_mod.get_provider
        original_get_db_path = store_mod.get_db_path
        original_get_engine = store_mod.get_engine
        try:
            llm_mod.get_provider = MagicMock(return_value=mock_provider)
            store_mod.get_db_path = lambda: db_path
            store_mod.get_engine = lambda _path: engine

            from forge.ocr.activities import submit_ocr_batch

            await submit_ocr_batch(input_json)

            # Verify the BLOB was deleted after successful submission
            assert get_file_content(engine, content_id) is None
        finally:
            llm_mod.get_provider = original_get_provider
            store_mod.get_db_path = original_get_db_path
            store_mod.get_engine = original_get_engine


# ---------------------------------------------------------------------------
# submit_ocr_batch — failure recording
# ---------------------------------------------------------------------------


class TestSubmitOcrBatchFailureRecording:
    """Tests that failed API calls are recorded in the database."""

    def _build_input_json(self, content_id: str) -> str:
        submit_input = OcrSubmitInput(
            file_path="/tmp/test.pdf",
            model_name="mistral:mistral-ocr-latest",
            document_id="doc-fail",
        )
        return json.dumps({
            "submit_input": submit_input.model_dump(),
            "file_content_ref": {
                "content_id": content_id,
                "mime_type": "application/pdf",
                "file_size_bytes": 14,
            },
            "store_workflow_id": "wf-store-fail",
        })

    @pytest.mark.asyncio
    async def test_records_failure_on_api_error(self, tmp_path: Path) -> None:
        """When execute_submit_ocr_batch raises, record_batch_failure is called."""
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        # Pre-store file content
        content_id = "fail-content-id"
        save_file_content(
            engine,
            content_id=content_id,
            data=b"fake pdf bytes",
            mime_type="application/pdf",
            file_size_bytes=14,
        )

        mock_provider = MagicMock()
        mock_provider.submit_batch = AsyncMock(
            side_effect=RuntimeError("400 Bad Request")
        )

        import forge.llm_providers as llm_mod
        import forge.store as store_mod

        original_get_provider = llm_mod.get_provider
        original_get_db_path = store_mod.get_db_path
        original_get_engine = store_mod.get_engine
        try:
            llm_mod.get_provider = MagicMock(return_value=mock_provider)
            store_mod.get_db_path = lambda: db_path
            store_mod.get_engine = lambda _path: engine

            from forge.ocr.activities import submit_ocr_batch

            input_json = self._build_input_json(content_id)

            with pytest.raises(RuntimeError, match="400 Bad Request"):
                await submit_ocr_batch(input_json)

            # Verify a failed record was inserted
            from forge.store import BatchJob

            t = BatchJob.__table__
            with engine.connect() as conn:
                rows = conn.execute(
                    t.select().where(t.c.status == "failed")
                ).mappings().all()

            assert len(rows) == 1
            row = dict(rows[0])
            assert row["batch_id"] is None
            assert row["status"] == "failed"
            assert "400 Bad Request" in row["error_message"]
            assert row["workflow_id"] == "wf-store-fail"
            assert row["provider"] == "mistral"
        finally:
            llm_mod.get_provider = original_get_provider
            store_mod.get_db_path = original_get_db_path
            store_mod.get_engine = original_get_engine

    @pytest.mark.asyncio
    async def test_original_exception_propagates_when_recording_fails(
        self, tmp_path: Path
    ) -> None:
        """When record_batch_failure itself raises, the original exception still propagates."""
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        content_id = "fail-record-content-id"
        save_file_content(
            engine,
            content_id=content_id,
            data=b"fake pdf bytes",
            mime_type="application/pdf",
            file_size_bytes=14,
        )

        mock_provider = MagicMock()
        mock_provider.submit_batch = AsyncMock(
            side_effect=RuntimeError("API exploded")
        )

        import forge.llm_providers as llm_mod
        import forge.store as store_mod

        original_get_provider = llm_mod.get_provider
        original_get_db_path = store_mod.get_db_path
        original_get_engine = store_mod.get_engine
        original_record_failure = store_mod.record_batch_failure
        try:
            llm_mod.get_provider = MagicMock(return_value=mock_provider)
            store_mod.get_db_path = lambda: db_path
            store_mod.get_engine = lambda _path: engine
            store_mod.record_batch_failure = MagicMock(
                side_effect=RuntimeError("DB write failed")
            )

            from forge.ocr.activities import submit_ocr_batch

            input_json = self._build_input_json(content_id)

            with pytest.raises(RuntimeError, match="API exploded"):
                await submit_ocr_batch(input_json)
        finally:
            llm_mod.get_provider = original_get_provider
            store_mod.get_db_path = original_get_db_path
            store_mod.get_engine = original_get_engine
            store_mod.record_batch_failure = original_record_failure


# ---------------------------------------------------------------------------
# OcrStoreWorkflow — workflow-level tests
# ---------------------------------------------------------------------------


class TestOcrStoreWorkflow:
    """Workflow-level tests for OcrStoreWorkflow signal handling and error paths."""

    @pytest.mark.asyncio
    async def test_batch_id_from_signal_not_input(self, env: WorkflowEnvironment) -> None:
        """batch_id in store_data must come from the BatchResult signal, not OcrStoreInput."""
        from temporalio import activity

        captured_store_data: list[str] = []

        @activity.defn(name="parse_ocr_result")
        async def mock_parse_ocr_result(_raw_json: str) -> OcrParseResult:
            return OcrParseResult(
                text="Extracted text",
                model_name="pixtral-large-latest",
                input_tokens=500,
                output_tokens=100,
            )

        @activity.defn(name="store_ocr_result")
        async def mock_store_ocr_result(store_data: str) -> OcrStoreResult:
            captured_store_data.append(store_data)
            return OcrStoreResult(document_id="doc-1", text_length=14)

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrStoreWorkflow],
            activities=[mock_parse_ocr_result, mock_store_ocr_result],
        ):
            handle = await env.client.start_workflow(
                OcrStoreWorkflow.run,
                OcrStoreInput(
                    batch_id="",  # empty — the bug scenario
                    request_id="req-1",
                    document_id="doc-1",
                    file_path="/tmp/test.pdf",
                ),
                id="test-ocr-store-batch-id",
                task_queue=FORGE_TASK_QUEUE,
            )

            await handle.signal(
                OcrStoreWorkflow.batch_result_received,
                BatchResult(
                    request_id="req-1",
                    batch_id="real-batch-id-from-poller",
                    raw_response_json='{"choices": []}',
                    result_type="succeeded",
                ),
            )

            result = await handle.result()

        assert result.document_id == "doc-1"
        assert result.text_length == 14

        # Core assertion: batch_id must be from the signal, not the empty input
        assert len(captured_store_data) == 1
        stored = json.loads(captured_store_data[0])
        assert stored["batch_id"] == "real-batch-id-from-poller"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("signal_kwargs", "expected_message"),
        [
            pytest.param(
                {"error": "something went wrong", "result_type": "errored"},
                "something went wrong",
                id="error-signal",
            ),
            pytest.param(
                {"raw_response_json": None, "result_type": "succeeded"},
                "no response JSON",
                id="missing-response-json",
            ),
        ],
    )
    async def test_error_paths(
        self,
        env: WorkflowEnvironment,
        signal_kwargs: dict,
        expected_message: str,
    ) -> None:
        """BatchResult with error or missing response JSON raises ApplicationError."""
        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrStoreWorkflow],
            activities=[],
        ):
            handle = await env.client.start_workflow(
                OcrStoreWorkflow.run,
                OcrStoreInput(
                    batch_id="",
                    request_id="req-err",
                    document_id="doc-err",
                    file_path="/tmp/test.pdf",
                ),
                id=f"test-ocr-store-{signal_kwargs.get('error', 'no-json')}",
                task_queue=FORGE_TASK_QUEUE,
            )

            await handle.signal(
                OcrStoreWorkflow.batch_result_received,
                BatchResult(
                    request_id="req-err",
                    batch_id="batch-err",
                    **signal_kwargs,
                ),
            )

            with pytest.raises(WorkflowFailureError) as exc_info:
                await handle.result()
            assert expected_message in str(exc_info.value.cause)


# ---------------------------------------------------------------------------
# validate_file_size
# ---------------------------------------------------------------------------


class TestValidateFileSize:
    def test_pdf_not_rejected_for_size(self) -> None:
        """PDFs are never rejected by validate_file_size (they get split instead)."""
        # Should not raise even for huge size
        validate_file_size(100 * 1024 * 1024, "application/pdf")

    def test_non_pdf_under_limit_passes(self) -> None:
        validate_file_size(MAX_FILE_SIZE_BYTES, "image/png")

    def test_non_pdf_over_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="Non-PDF file"):
            validate_file_size(MAX_FILE_SIZE_BYTES + 1, "image/png")


# ---------------------------------------------------------------------------
# execute_split_file_into_chunks
# ---------------------------------------------------------------------------


def _create_test_pdf(page_count: int) -> bytes:
    """Create a minimal PDF with the given number of pages using PyMuPDF."""
    import fitz

    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page(width=200, height=200)
        page.insert_text((10, 50), f"Page {i + 1}")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestExecuteSplitFileIntoChunks:
    def test_non_pdf_single_chunk(self, tmp_path: Path) -> None:
        """Non-PDF files produce a single chunk reusing the original blob."""
        engine, _ = _setup_db(tmp_path)
        content_id = "img-content"
        data = b"fake image bytes"
        save_file_content(
            engine,
            content_id=content_id,
            data=data,
            mime_type="image/png",
            file_size_bytes=len(data),
        )

        result = execute_split_file_into_chunks(
            content_id, "image/png", len(data), engine
        )

        assert len(result.chunks) == 1
        assert result.chunks[0].content_id == content_id  # reuses original
        assert result.chunks[0].chunk_index == 0
        assert result.chunks[0].page_start == 1
        assert result.chunks[0].page_end == 1
        assert result.total_pages == 1
        assert result.original_content_id == content_id

    def test_non_pdf_over_size_raises(self, tmp_path: Path) -> None:
        """Non-PDF files exceeding size limit are rejected."""
        engine, _ = _setup_db(tmp_path)
        content_id = "big-img"
        data = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        save_file_content(
            engine,
            content_id=content_id,
            data=data,
            mime_type="image/png",
            file_size_bytes=len(data),
        )

        with pytest.raises(ValueError, match="Non-PDF file"):
            execute_split_file_into_chunks(
                content_id, "image/png", len(data), engine
            )

    def test_small_pdf_single_chunk(self, tmp_path: Path) -> None:
        """PDFs under cutoffs produce a single chunk reusing the original blob."""
        engine, _ = _setup_db(tmp_path)
        pdf_data = _create_test_pdf(10)
        content_id = "small-pdf"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine
        )

        assert len(result.chunks) == 1
        assert result.chunks[0].content_id == content_id  # reuses original
        assert result.chunks[0].page_start == 1
        assert result.chunks[0].page_end == 10
        assert result.total_pages == 10
        # Original blob should still exist
        assert get_file_content(engine, content_id) is not None

    def test_boundary_30_pages_single_chunk(self, tmp_path: Path) -> None:
        """Exactly MAX_PAGES pages stays as a single chunk."""
        engine, _ = _setup_db(tmp_path)
        pdf_data = _create_test_pdf(MAX_PAGES)
        content_id = "boundary-30"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine
        )

        assert len(result.chunks) == 1
        assert result.total_pages == MAX_PAGES

    def test_boundary_31_pages_splits(self, tmp_path: Path) -> None:
        """MAX_PAGES + 1 pages triggers split: 25 + 6."""
        engine, _ = _setup_db(tmp_path)
        pdf_data = _create_test_pdf(MAX_PAGES + 1)
        content_id = "boundary-31"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine
        )

        assert len(result.chunks) == 2
        assert result.total_pages == MAX_PAGES + 1

        # First chunk: pages 1-25
        assert result.chunks[0].page_start == 1
        assert result.chunks[0].page_end == CHUNK_SIZE_PAGES
        assert result.chunks[0].chunk_index == 0

        # Second chunk: pages 26-31
        assert result.chunks[1].page_start == CHUNK_SIZE_PAGES + 1
        assert result.chunks[1].page_end == MAX_PAGES + 1
        assert result.chunks[1].chunk_index == 1

    def test_large_pdf_60_pages_3_chunks(self, tmp_path: Path) -> None:
        """60-page PDF splits into 3 chunks: 25 + 25 + 10."""
        engine, _ = _setup_db(tmp_path)
        pdf_data = _create_test_pdf(60)
        content_id = "large-60"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine
        )

        assert len(result.chunks) == 3
        assert result.total_pages == 60

        # Chunk 0: pages 1-25
        assert result.chunks[0].page_start == 1
        assert result.chunks[0].page_end == 25

        # Chunk 1: pages 26-50
        assert result.chunks[1].page_start == 26
        assert result.chunks[1].page_end == 50

        # Chunk 2: pages 51-60
        assert result.chunks[2].page_start == 51
        assert result.chunks[2].page_end == 60

    def test_original_blob_deleted_after_split(self, tmp_path: Path) -> None:
        """After multi-chunk split, the original blob is deleted."""
        engine, _ = _setup_db(tmp_path)
        pdf_data = _create_test_pdf(MAX_PAGES + 1)
        content_id = "delete-test"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine
        )

        # Original blob should be deleted
        assert get_file_content(engine, content_id) is None
        # Chunk blobs should exist
        for chunk in result.chunks:
            assert get_file_content(engine, chunk.content_id) is not None

    def test_original_blob_kept_for_single_chunk(self, tmp_path: Path) -> None:
        """Single-chunk case preserves the original blob."""
        engine, _ = _setup_db(tmp_path)
        pdf_data = _create_test_pdf(5)
        content_id = "keep-test"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
        )

        execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine
        )

        assert get_file_content(engine, content_id) is not None

    def test_chunk_pdfs_are_valid(self, tmp_path: Path) -> None:
        """Each chunk blob is a valid PDF with correct page count."""
        import fitz

        engine, _ = _setup_db(tmp_path)
        pdf_data = _create_test_pdf(60)
        content_id = "valid-chunks"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine
        )

        expected_pages = [25, 25, 10]
        for chunk, expected in zip(result.chunks, expected_pages, strict=True):
            blob = get_file_content(engine, chunk.content_id)
            assert blob is not None
            doc = fitz.open(stream=blob["data"], filetype="pdf")
            assert len(doc) == expected
            doc.close()

    def test_missing_content_raises(self, tmp_path: Path) -> None:
        """Raises RuntimeError if content_id not found in DB."""
        engine, _ = _setup_db(tmp_path)

        with pytest.raises(RuntimeError, match="File content not found"):
            execute_split_file_into_chunks(
                "nonexistent", "application/pdf", 1000, engine
            )


# ---------------------------------------------------------------------------
# execute_reassemble_ocr_chunks
# ---------------------------------------------------------------------------


class TestExecuteReassembleOcrChunks:
    def _store_chunk_result(
        self, engine, document_id: str, text: str, input_tokens: int, output_tokens: int
    ) -> None:
        save_ocr_result(
            engine,
            document_id=document_id,
            file_path="/tmp/test.pdf",
            text=text,
            model_name="mistral-ocr-latest",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            batch_id="batch-1",
            workflow_id=f"wf-{document_id}",
        )

    def test_combines_text_in_order(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        self._store_chunk_result(engine, "doc__chunk_0", "First chunk.", 10, 100)
        self._store_chunk_result(engine, "doc__chunk_1", "Second chunk.", 20, 200)
        self._store_chunk_result(engine, "doc__chunk_2", "Third chunk.", 30, 300)

        result = execute_reassemble_ocr_chunks(
            document_id="doc-combined",
            chunk_document_ids=["doc__chunk_0", "doc__chunk_1", "doc__chunk_2"],
            file_path="/tmp/test.pdf",
            total_pages=60,
            engine=engine,
        )

        assert result.text_length == len("First chunk.\n\nSecond chunk.\n\nThird chunk.")
        assert result.page_count == 60
        assert result.document_id == "doc-combined"

    def test_sums_tokens(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        self._store_chunk_result(engine, "tok__chunk_0", "A", 10, 100)
        self._store_chunk_result(engine, "tok__chunk_1", "B", 20, 200)

        execute_reassemble_ocr_chunks(
            document_id="tok-combined",
            chunk_document_ids=["tok__chunk_0", "tok__chunk_1"],
            file_path="/tmp/test.pdf",
            total_pages=50,
            engine=engine,
        )

        stored = get_ocr_result(engine, "tok-combined")
        assert stored is not None
        assert stored["input_tokens"] == 30
        assert stored["output_tokens"] == 300

    def test_stores_combined_result(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        self._store_chunk_result(engine, "st__chunk_0", "Hello", 5, 50)
        self._store_chunk_result(engine, "st__chunk_1", "World", 5, 50)

        execute_reassemble_ocr_chunks(
            document_id="st-combined",
            chunk_document_ids=["st__chunk_0", "st__chunk_1"],
            file_path="/tmp/doc.pdf",
            total_pages=40,
            engine=engine,
        )

        stored = get_ocr_result(engine, "st-combined")
        assert stored is not None
        assert stored["text"] == "Hello\n\nWorld"
        assert stored["page_count"] == 40
        assert stored["file_path"] == "/tmp/doc.pdf"

    def test_cleans_up_chunk_rows(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        self._store_chunk_result(engine, "cl__chunk_0", "A", 1, 1)
        self._store_chunk_result(engine, "cl__chunk_1", "B", 1, 1)

        execute_reassemble_ocr_chunks(
            document_id="cl-combined",
            chunk_document_ids=["cl__chunk_0", "cl__chunk_1"],
            file_path="/tmp/test.pdf",
            total_pages=50,
            engine=engine,
        )

        # Chunk rows should be deleted
        assert get_ocr_result(engine, "cl__chunk_0") is None
        assert get_ocr_result(engine, "cl__chunk_1") is None
        # Combined row should exist
        assert get_ocr_result(engine, "cl-combined") is not None

    def test_missing_chunk_raises(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        self._store_chunk_result(engine, "miss__chunk_0", "A", 1, 1)

        with pytest.raises(RuntimeError, match="OCR result not found"):
            execute_reassemble_ocr_chunks(
                document_id="miss-combined",
                chunk_document_ids=["miss__chunk_0", "miss__chunk_1"],
                file_path="/tmp/test.pdf",
                total_pages=50,
                engine=engine,
            )


# ---------------------------------------------------------------------------
# delete_ocr_results store helper
# ---------------------------------------------------------------------------


class TestDeleteOcrResults:
    def test_deletes_specified_rows(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        for i in range(3):
            save_ocr_result(
                engine,
                document_id=f"del-{i}",
                file_path="/tmp/test.pdf",
                text=f"Text {i}",
                model_name="model",
                input_tokens=1,
                output_tokens=1,
                batch_id="b-1",
                workflow_id="wf-1",
            )

        delete_ocr_results(engine, ["del-0", "del-2"])

        assert get_ocr_result(engine, "del-0") is None
        assert get_ocr_result(engine, "del-1") is not None
        assert get_ocr_result(engine, "del-2") is None

    def test_empty_list_no_op(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        # Should not raise
        delete_ocr_results(engine, [])
