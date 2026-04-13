"""Tests for forge.ocr — OCR models, activities, store functions, and workflow."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError
from temporalio.worker import Worker

from forge.models import BatchResult
from forge.ocr.activities import (
    CHUNK_SIZE_PAGES,
    MAX_FILE_SIZE_BYTES,
    MAX_PAGES,
    _mime_to_extension,
    _strip_image_prefix,
    build_ocr_batch_body,
    build_ocr_messages,
    detect_mime_type,
    execute_check_ocr_duplicate,
    execute_export_ocr_document,
    execute_list_ocr_jobs,
    execute_parse_ocr_result,
    execute_read_and_store_file,
    execute_read_file,
    execute_reassemble_ocr_chunks,
    execute_split_file_into_chunks,
    execute_store_ocr_result,
    execute_submit_ocr_batch,
    rewrite_image_references,
    rewrite_ocr_uris_to_local,
    validate_file_size,
)
from forge.ocr.models import (
    ChunkRef,
    FileContentRef,
    FileContentResult,
    OcrBatchRef,
    OcrDuplicateCheckResult,
    OcrExportInput,
    OcrExportResult,
    OcrGatherInput,
    OcrMarkInput,
    OcrMarkResult,
    OcrParseResult,
    OcrStoreInput,
    OcrStoreResult,
    OcrSubmitInput,
    SplitResult,
)
from forge.ocr.workflow_gather import OcrGatherWorkflow
from forge.ocr.workflow_store import OcrStoreWorkflow
from forge.ocr.workflow_submit import OcrSubmitWorkflow
from forge.store import (
    clear_ocr_removal_mark,
    delete_ocr_results,
    find_ocr_result_by_file_path,
    get_engine,
    get_file_content,
    get_ocr_image,
    get_ocr_images,
    get_ocr_result,
    get_ocr_results_missing_hash,
    mark_ocr_for_removal,
    reassign_ocr_images_document_id,
    run_migrations,
    save_file_content,
    save_ocr_image,
    save_ocr_result,
    update_ocr_file_hash,
    update_ocr_images_document_id,
)
from forge.workflows import FORGE_TASK_QUEUE

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.engine import Engine
    from temporalio.testing import WorkflowEnvironment


def _setup_db(tmp_path: Path):
    """Create a test database with migrations applied."""
    db_path = tmp_path / "test.db"
    run_migrations(db_path)
    return get_engine(db_path), db_path


@workflow.defn(name="OcrStoreWorkflow")
class MockOcrStoreWorkflow:
    @workflow.run
    async def run(self, _input: OcrStoreInput) -> OcrStoreResult:
        return OcrStoreResult(document_id="unused", text_length=0)


@workflow.defn(name="OcrGatherWorkflow")
class MockOcrGatherWorkflow:
    @workflow.run
    async def run(self, _input: OcrGatherInput) -> OcrStoreResult:
        return OcrStoreResult(document_id="unused", text_length=0)


# ---------------------------------------------------------------------------
# OCR Models
# ---------------------------------------------------------------------------


class TestOcrModels:
    def test_submit_input_defaults(self) -> None:
        inp = OcrSubmitInput(file_path="/tmp/test.pdf")
        assert inp.model_name == "mistral:mistral-ocr-latest"
        assert inp.max_tokens == 16384
        assert inp.document_id == ""

    def test_batch_ref(self) -> None:
        ref = OcrBatchRef(batch_id="b-1", request_id="r-1")
        assert ref.batch_id == "b-1"
        assert ref.request_id == "r-1"

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

    def test_submit_input_strips_whitespace(self) -> None:
        inp = OcrSubmitInput(file_path="  /tmp/test.pdf\n")
        assert inp.file_path == "/tmp/test.pdf"

    def test_submit_input_rejects_empty_path(self) -> None:
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            OcrSubmitInput(file_path="")

    def test_submit_input_rejects_whitespace_only_path(self) -> None:
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            OcrSubmitInput(file_path="\n")

    def test_batch_ref_fields(self) -> None:
        ref = OcrBatchRef(batch_id="batch-1", request_id="req-1")
        assert ref.batch_id == "batch-1"
        assert ref.request_id == "req-1"

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
            },
            "include_image_base64": True,
        }

    def test_document_type(self) -> None:
        body = build_ocr_batch_body("abc123", "application/pdf")
        assert body == {
            "document": {
                "type": "document_url",
                "document_url": "data:application/pdf;base64,abc123",
            },
            "include_image_base64": True,
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

        result = await execute_submit_ocr_batch(inp, file_content, mock_provider)

        assert result.batch_id == "batch-123"
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

        result = await execute_submit_ocr_batch(inp, file_content, mock_provider)
        assert result.batch_id == "batch-456"
        assert result.request_id  # non-empty UUID

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

        result = await execute_submit_ocr_batch(inp, file_content, mock_provider)
        assert result.batch_id == "batch-789"

        requests = mock_provider.submit_batch.call_args.args[0]
        assert requests[0]["body"]["document"]["type"] == "image_url"


class TestExecuteParseOcrResult:
    def test_parses_ocr_response(self) -> None:
        raw_json = json.dumps(
            {
                "pages": [
                    {"markdown": "Page one text."},
                    {"markdown": "Page two text."},
                ],
                "model": "mistral-ocr-latest",
                "usage_info": {
                    "pages_processed": 2,
                    "doc_size_bytes": 50000,
                },
            }
        )

        result = execute_parse_ocr_result(raw_json)
        assert result.text == "Page one text.\n\nPage two text."
        assert result.model_name == "mistral-ocr-latest"
        assert result.input_tokens == 2
        assert result.output_tokens == 50000
        assert result.page_count == 2

    def test_single_page(self) -> None:
        raw_json = json.dumps(
            {
                "pages": [{"markdown": "Only page."}],
                "model": "mistral-ocr-latest",
                "usage_info": {"pages_processed": 1, "doc_size_bytes": 1000},
            }
        )

        result = execute_parse_ocr_result(raw_json)
        assert result.text == "Only page."
        assert result.input_tokens == 1

    def test_empty_pages_returns_empty_string(self) -> None:
        raw_json = json.dumps(
            {
                "pages": [],
                "model": "mistral-ocr-latest",
                "usage_info": {},
            }
        )

        result = execute_parse_ocr_result(raw_json)
        assert result.text == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_missing_usage_info(self) -> None:
        raw_json = json.dumps(
            {
                "pages": [{"markdown": "Some text."}],
                "model": "mistral-ocr-latest",
            }
        )

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
        input_json = json.dumps(
            {
                "submit_input": submit_input.model_dump(),
                "file_content_ref": {
                    "content_id": content_id,
                    "mime_type": "application/pdf",
                    "file_size_bytes": len(test_data),
                },
                "store_workflow_id": "wf-store-1",
            }
        )

        # Mock the provider and store functions
        mock_provider = MagicMock()
        mock_provider.submit_batch = AsyncMock(return_value="batch-123")

        import forge.store as store_mod

        original_get_db_path = store_mod.get_db_path
        original_get_engine = store_mod.get_engine
        try:
            store_mod.get_db_path = lambda: db_path
            store_mod.get_engine = lambda _path: engine

            from forge.ocr.activities import submit_ocr_batch

            with patch("sax_llm.get_provider", return_value=mock_provider):
                result = await submit_ocr_batch(input_json)

            assert result.batch_id == "batch-123"
            assert result.request_id  # non-empty UUID

            # Verify OCR body format and endpoint were used
            call_args = mock_provider.submit_batch.call_args
            requests = call_args.args[0]
            assert "document" in requests[0]["body"]
            assert call_args.kwargs.get("endpoint") == "/v1/ocr"
        finally:
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
        input_json = json.dumps(
            {
                "submit_input": submit_input.model_dump(),
                "file_content_ref": {
                    "content_id": content_id,
                    "mime_type": "application/pdf",
                    "file_size_bytes": len(test_data),
                },
                "store_workflow_id": "wf-store-cleanup",
            }
        )

        mock_provider = MagicMock()
        mock_provider.submit_batch = AsyncMock(return_value="batch-456")

        import forge.store as store_mod

        original_get_db_path = store_mod.get_db_path
        original_get_engine = store_mod.get_engine
        try:
            store_mod.get_db_path = lambda: db_path
            store_mod.get_engine = lambda _path: engine

            from forge.ocr.activities import submit_ocr_batch

            with patch("sax_llm.get_provider", return_value=mock_provider):
                await submit_ocr_batch(input_json)

            # Verify the BLOB was deleted after successful submission
            assert get_file_content(engine, content_id) is None
        finally:
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
        return json.dumps(
            {
                "submit_input": submit_input.model_dump(),
                "file_content_ref": {
                    "content_id": content_id,
                    "mime_type": "application/pdf",
                    "file_size_bytes": 14,
                },
                "store_workflow_id": "wf-store-fail",
            }
        )

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
        mock_provider.submit_batch = AsyncMock(side_effect=RuntimeError("400 Bad Request"))

        import forge.store as store_mod

        original_get_db_path = store_mod.get_db_path
        original_get_engine = store_mod.get_engine
        try:
            store_mod.get_db_path = lambda: db_path
            store_mod.get_engine = lambda _path: engine

            from forge.ocr.activities import submit_ocr_batch

            input_json = self._build_input_json(content_id)

            with (
                patch("sax_llm.get_provider", return_value=mock_provider),
                pytest.raises(RuntimeError, match="400 Bad Request"),
            ):
                await submit_ocr_batch(input_json)

            # Verify a failed record was inserted
            from forge.store import BatchJob

            t = BatchJob.__table__
            with engine.connect() as conn:
                rows = conn.execute(t.select().where(t.c.status == "failed")).mappings().all()

            assert len(rows) == 1
            row = dict(rows[0])
            assert row["batch_id"] is None
            assert row["status"] == "failed"
            assert "400 Bad Request" in row["error_message"]
            assert row["workflow_id"] == "wf-store-fail"
            assert row["provider"] == "mistral"
            # document_id flows through either from root_document_id in the
            # submit JSON, or (as here) from ocr_input.document_id as fallback.
            assert row["document_id"] == "doc-fail"
        finally:
            store_mod.get_db_path = original_get_db_path
            store_mod.get_engine = original_get_engine

    @pytest.mark.asyncio
    async def test_original_exception_propagates_when_recording_fails(self, tmp_path: Path) -> None:
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
        mock_provider.submit_batch = AsyncMock(side_effect=RuntimeError("API exploded"))

        import forge.store as store_mod

        original_get_db_path = store_mod.get_db_path
        original_get_engine = store_mod.get_engine
        original_record_failure = store_mod.record_batch_failure
        try:
            store_mod.get_db_path = lambda: db_path
            store_mod.get_engine = lambda _path: engine
            store_mod.record_batch_failure = MagicMock(
                side_effect=RuntimeError("DB write failed")
            )

            from forge.ocr.activities import submit_ocr_batch

            input_json = self._build_input_json(content_id)

            with (
                patch("sax_llm.get_provider", return_value=mock_provider),
                pytest.raises(RuntimeError, match="API exploded"),
            ):
                await submit_ocr_batch(input_json)
        finally:
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
        captured_status_updates: list[dict] = []

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

        @activity.defn(name="update_batch_job_status")
        async def mock_update_batch_job_status(input_json: str) -> None:
            captured_status_updates.append(json.loads(input_json))

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrStoreWorkflow],
            activities=[
                mock_parse_ocr_result,
                mock_store_ocr_result,
                mock_update_batch_job_status,
            ],
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

        # After store succeeds, the workflow promotes the batch_jobs row
        # from STORING to SUCCEEDED via update_batch_job_status.
        assert len(captured_status_updates) == 1
        assert captured_status_updates[0]["request_id"] == "req-1"
        assert captured_status_updates[0]["status"] == "succeeded"

    @pytest.mark.asyncio
    async def test_store_failure_marks_batch_job_errored(self, env: WorkflowEnvironment) -> None:
        """If the store activity raises, the workflow must mark the batch_jobs
        row ERRORED before propagating, so the list view doesn't leave it
        stuck in STORING."""
        from temporalio import activity

        captured_status_updates: list[dict] = []

        @activity.defn(name="parse_ocr_result")
        async def mock_parse_ocr_result(_raw_json: str) -> OcrParseResult:
            return OcrParseResult(
                text="text",
                model_name="pixtral-large-latest",
                input_tokens=10,
                output_tokens=5,
            )

        @activity.defn(name="store_ocr_result")
        async def mock_store_ocr_result(_store_data: str) -> OcrStoreResult:
            msg = "disk full"
            raise RuntimeError(msg)

        @activity.defn(name="update_batch_job_status")
        async def mock_update_batch_job_status(input_json: str) -> None:
            captured_status_updates.append(json.loads(input_json))

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrStoreWorkflow],
            activities=[
                mock_parse_ocr_result,
                mock_store_ocr_result,
                mock_update_batch_job_status,
            ],
        ):
            handle = await env.client.start_workflow(
                OcrStoreWorkflow.run,
                OcrStoreInput(
                    batch_id="",
                    request_id="req-err",
                    document_id="doc-err",
                    file_path="/tmp/test.pdf",
                ),
                id="test-ocr-store-errored",
                task_queue=FORGE_TASK_QUEUE,
            )

            await handle.signal(
                OcrStoreWorkflow.batch_result_received,
                BatchResult(
                    request_id="req-err",
                    batch_id="batch-err",
                    raw_response_json='{"choices": []}',
                    result_type="succeeded",
                ),
            )

            with pytest.raises(WorkflowFailureError):
                await handle.result()

        assert len(captured_status_updates) == 1
        assert captured_status_updates[0]["request_id"] == "req-err"
        assert captured_status_updates[0]["status"] == "errored"
        # Temporal wraps the underlying RuntimeError in an ActivityError,
        # so the message we record is the outer wrapper — we just want a
        # non-empty error_message captured on the row.
        assert captured_status_updates[0]["error_message"]

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
        """BatchResult with error or missing response JSON raises ApplicationError
        and marks the batch_jobs row ERRORED."""
        from temporalio import activity

        captured_status_updates: list[dict] = []

        @activity.defn(name="update_batch_job_status")
        async def mock_update_batch_job_status(input_json: str) -> None:
            captured_status_updates.append(json.loads(input_json))

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrStoreWorkflow],
            activities=[mock_update_batch_job_status],
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

        # The workflow must promote STORING -> ERRORED before re-raising,
        # so the list view doesn't leave this row stuck.
        assert len(captured_status_updates) == 1
        assert captured_status_updates[0]["request_id"] == "req-err"
        assert captured_status_updates[0]["status"] == "errored"


# ---------------------------------------------------------------------------
# OcrSubmitWorkflow / OcrGatherWorkflow — workflow-level tests
# ---------------------------------------------------------------------------


class TestOcrSubmitWorkflow:
    @pytest.mark.asyncio
    async def test_duplicate_short_circuits_before_submission(
        self, env: WorkflowEnvironment
    ) -> None:
        @activity.defn(name="check_ocr_duplicate")
        async def mock_check_duplicate(_file_path: str) -> OcrDuplicateCheckResult:
            return OcrDuplicateCheckResult(
                is_duplicate=True,
                existing_document_id="existing-doc-123",
            )

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrSubmitWorkflow],
            activities=[mock_check_duplicate],
        ):
            result = await env.client.execute_workflow(
                OcrSubmitWorkflow.run,
                OcrSubmitInput(file_path="/tmp/report.pdf"),
                id="test-ocr-submit-duplicate",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.document_id == "existing-doc-123"
        assert result.skipped is True
        assert result.skip_reason == "Duplicate document"
        assert result.batch_refs == []
        assert result.chunk_count == 0

    @pytest.mark.asyncio
    async def test_multi_chunk_starts_gather_and_store_children(
        self, env: WorkflowEnvironment
    ) -> None:
        submitted_payloads: list[dict[str, object]] = []

        @activity.defn(name="check_ocr_duplicate")
        async def mock_check_duplicate(_file_path: str) -> OcrDuplicateCheckResult:
            return OcrDuplicateCheckResult(is_duplicate=False)

        @activity.defn(name="read_and_store_file_content")
        async def mock_read_and_store_file_content(_file_path: str) -> FileContentRef:
            return FileContentRef(
                content_id="blob-original",
                mime_type="application/pdf",
                file_size_bytes=1024,
            )

        @activity.defn(name="split_file_into_chunks")
        async def mock_split_file_into_chunks(_input_json: str) -> SplitResult:
            return SplitResult(
                chunks=[
                    ChunkRef(
                        content_id="blob-chunk-0",
                        mime_type="application/pdf",
                        file_size_bytes=600,
                        chunk_index=0,
                        page_start=1,
                        page_end=25,
                    ),
                    ChunkRef(
                        content_id="blob-chunk-1",
                        mime_type="application/pdf",
                        file_size_bytes=424,
                        chunk_index=1,
                        page_start=26,
                        page_end=40,
                    ),
                ],
                total_pages=40,
                original_content_id="blob-original",
            )

        @activity.defn(name="submit_ocr_batch")
        async def mock_submit_ocr_batch(input_json: str) -> OcrBatchRef:
            payload = json.loads(input_json)
            submitted_payloads.append(payload)
            doc_id = payload["submit_input"]["document_id"]
            return OcrBatchRef(
                batch_id=f"batch-{doc_id}",
                request_id=f"req-{doc_id}",
            )

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrSubmitWorkflow, MockOcrStoreWorkflow, MockOcrGatherWorkflow],
            activities=[
                mock_check_duplicate,
                mock_read_and_store_file_content,
                mock_split_file_into_chunks,
                mock_submit_ocr_batch,
            ],
        ):
            result = await env.client.execute_workflow(
                OcrSubmitWorkflow.run,
                OcrSubmitInput(file_path="/tmp/report.pdf", document_id="doc-abc"),
                id="test-ocr-submit-multi-chunk",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result.document_id == "doc-abc"
        assert result.chunk_count == 2
        assert [ref.batch_id for ref in result.batch_refs] == [
            "batch-doc-abc__chunk_0",
            "batch-doc-abc__chunk_1",
        ]
        assert len(submitted_payloads) == 2
        assert submitted_payloads[0]["submit_input"]["document_id"] == "doc-abc__chunk_0"
        assert submitted_payloads[0]["store_workflow_id"] == "ocr-store-doc-abc__chunk_0"
        assert submitted_payloads[1]["submit_input"]["document_id"] == "doc-abc__chunk_1"
        assert submitted_payloads[1]["store_workflow_id"] == "ocr-store-doc-abc__chunk_1"


class TestOcrGatherWorkflow:
    @pytest.mark.asyncio
    async def test_waits_for_all_chunks_before_reassembly(self, env: WorkflowEnvironment) -> None:
        captured_reassemble_data: list[dict[str, object]] = []

        @activity.defn(name="reassemble_ocr_chunks")
        async def mock_reassemble_ocr_chunks(input_json: str) -> OcrStoreResult:
            captured_reassemble_data.append(json.loads(input_json))
            return OcrStoreResult(
                document_id="doc-final",
                text_length=1234,
                page_count=40,
            )

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[OcrGatherWorkflow],
            activities=[mock_reassemble_ocr_chunks],
        ):
            handle = await env.client.start_workflow(
                OcrGatherWorkflow.run,
                OcrGatherInput(
                    document_id="doc-final",
                    chunk_document_ids=["doc-final__chunk_0", "doc-final__chunk_1"],
                    store_workflow_ids=[],
                    file_path="/tmp/report.pdf",
                    total_pages=40,
                ),
                id="test-ocr-gather-complete",
                task_queue=FORGE_TASK_QUEUE,
            )

            await handle.signal(OcrGatherWorkflow.chunk_completed, "doc-final__chunk_0")
            await handle.signal(OcrGatherWorkflow.chunk_completed, "doc-final__chunk_1")
            result = await handle.result()

        assert result.document_id == "doc-final"
        assert result.text_length == 1234
        assert captured_reassemble_data == [
            {
                "document_id": "doc-final",
                "chunk_document_ids": ["doc-final__chunk_0", "doc-final__chunk_1"],
                "file_path": "/tmp/report.pdf",
                "total_pages": 40,
            }
        ]


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

        result = execute_split_file_into_chunks(content_id, "image/png", len(data), engine)

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
            execute_split_file_into_chunks(content_id, "image/png", len(data), engine)

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

        execute_split_file_into_chunks(content_id, "application/pdf", len(pdf_data), engine)

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
            execute_split_file_into_chunks("nonexistent", "application/pdf", 1000, engine)


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


# ---------------------------------------------------------------------------
# rewrite_image_references (pure function)
# ---------------------------------------------------------------------------


class TestRewriteImageReferences:
    def test_single_image(self) -> None:
        md = "Some text\n![img-0.jpeg](img-0.jpeg)\nMore text"
        mapping = {"img-0.jpeg": "uuid-aaa"}
        result = rewrite_image_references(md, mapping)
        assert "ocr-image://uuid-aaa" in result
        assert "![img-0.jpeg](ocr-image://uuid-aaa)" in result

    def test_multiple_images(self) -> None:
        md = "![img-0.jpeg](img-0.jpeg)\n![img-1.png](img-1.png)"
        mapping = {"img-0.jpeg": "uuid-111", "img-1.png": "uuid-222"}
        result = rewrite_image_references(md, mapping)
        assert "![img-0.jpeg](ocr-image://uuid-111)" in result
        assert "![img-1.png](ocr-image://uuid-222)" in result

    def test_no_images(self) -> None:
        md = "Plain text with no images."
        result = rewrite_image_references(md, {"img-0.jpeg": "uuid-aaa"})
        assert result == md

    def test_empty_mapping(self) -> None:
        md = "![img-0.jpeg](img-0.jpeg)"
        result = rewrite_image_references(md, {})
        assert result == md

    def test_partial_mapping(self) -> None:
        md = "![img-0.jpeg](img-0.jpeg)\n![img-1.jpeg](img-1.jpeg)"
        mapping = {"img-0.jpeg": "uuid-only"}
        result = rewrite_image_references(md, mapping)
        assert "ocr-image://uuid-only" in result
        assert "![img-1.jpeg](img-1.jpeg)" in result

    def test_preserves_alt_text(self) -> None:
        md = "![Chart showing revenue](img-0.jpeg)"
        mapping = {"img-0.jpeg": "uuid-chart"}
        result = rewrite_image_references(md, mapping)
        assert result == "![Chart showing revenue](ocr-image://uuid-chart)"


# ---------------------------------------------------------------------------
# build_ocr_batch_body — include_image_base64
# ---------------------------------------------------------------------------


class TestBuildOcrBatchBodyImageFlag:
    def test_includes_image_base64_flag(self) -> None:
        body = build_ocr_batch_body("abc123", "application/pdf")
        assert body["include_image_base64"] is True

    def test_includes_image_base64_flag_for_images(self) -> None:
        body = build_ocr_batch_body("abc123", "image/png")
        assert body["include_image_base64"] is True


# ---------------------------------------------------------------------------
# execute_parse_ocr_result — with image mapping
# ---------------------------------------------------------------------------


class TestExecuteParseOcrResultWithImages:
    def test_rewrites_image_references(self) -> None:
        raw_json = json.dumps(
            {
                "pages": [
                    {
                        "markdown": "Text ![img-0.jpeg](img-0.jpeg) more",
                        "images": [{"id": "img-0.jpeg"}],
                    },
                ],
                "model": "mistral-ocr-latest",
                "usage_info": {"pages_processed": 1, "doc_size_bytes": 1000},
                "_image_mapping": {"img-0.jpeg": "uuid-abc"},
            }
        )

        result = execute_parse_ocr_result(raw_json)
        assert "ocr-image://uuid-abc" in result.text
        assert "img-0.jpeg)" not in result.text
        assert result.image_count == 1
        assert result.image_ids == ["uuid-abc"]

    def test_backward_compat_without_image_mapping(self) -> None:
        raw_json = json.dumps(
            {
                "pages": [{"markdown": "No images here."}],
                "model": "mistral-ocr-latest",
                "usage_info": {"pages_processed": 1, "doc_size_bytes": 500},
            }
        )

        result = execute_parse_ocr_result(raw_json)
        assert result.text == "No images here."
        assert result.image_count == 0
        assert result.image_ids == []

    def test_multiple_pages_with_images(self) -> None:
        raw_json = json.dumps(
            {
                "pages": [
                    {"markdown": "![img-0.jpeg](img-0.jpeg)"},
                    {"markdown": "![img-0.jpeg](img-0.jpeg)"},
                ],
                "model": "mistral-ocr-latest",
                "usage_info": {"pages_processed": 2, "doc_size_bytes": 2000},
                "_image_mapping": {"img-0.jpeg": "uuid-shared"},
            }
        )

        result = execute_parse_ocr_result(raw_json)
        # Both pages should be rewritten with the same UUID
        assert result.text.count("ocr-image://uuid-shared") == 2


# ---------------------------------------------------------------------------
# execute_store_ocr_result — with image_ids
# ---------------------------------------------------------------------------


class TestExecuteStoreOcrResultWithImages:
    def test_updates_image_document_ids(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        # Pre-store an image with empty document_id
        save_ocr_image(
            engine,
            image_id="img-uuid-1",
            page_index=0,
            original_image_id="img-0.jpeg",
            data=b"fake-image-bytes",
            mime_type="image/jpeg",
            file_size_bytes=16,
        )

        import forge.store as store_module

        original_get_db_path = store_module.get_db_path
        original_get_engine = store_module.get_engine
        try:
            store_module.get_db_path = lambda: db_path
            store_module.get_engine = lambda _path: engine

            execute_store_ocr_result(
                document_id="doc-with-images",
                file_path="/tmp/test.pdf",
                text="Some text",
                model_name="model",
                input_tokens=10,
                output_tokens=5,
                batch_id="b-1",
                workflow_id="wf-1",
                image_ids=["img-uuid-1"],
            )

            # Verify image document_id was updated
            img = get_ocr_image(engine, "img-uuid-1")
            assert img is not None
            assert img["document_id"] == "doc-with-images"
        finally:
            store_module.get_db_path = original_get_db_path
            store_module.get_engine = original_get_engine

    def test_no_image_ids_is_backward_compatible(self, tmp_path: Path) -> None:
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
                document_id="doc-no-images",
                file_path="/tmp/test.pdf",
                text="Plain text",
                model_name="model",
                input_tokens=10,
                output_tokens=5,
                batch_id="b-1",
                workflow_id="wf-1",
            )
            assert result.stored is True
        finally:
            store_module.get_db_path = original_get_db_path
            store_module.get_engine = original_get_engine


# ---------------------------------------------------------------------------
# execute_reassemble_ocr_chunks — image reassignment
# ---------------------------------------------------------------------------


class TestExecuteReassembleOcrChunksWithImages:
    def test_reassigns_images_to_final_document(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        # Create chunk OCR results
        for i in range(2):
            save_ocr_result(
                engine,
                document_id=f"chunk-{i}",
                file_path="/tmp/test.pdf",
                text=f"Chunk {i} text",
                model_name="model",
                input_tokens=10,
                output_tokens=5,
                batch_id="b-1",
                workflow_id="wf-1",
            )
            # Pre-store image for each chunk
            save_ocr_image(
                engine,
                image_id=f"img-chunk-{i}",
                document_id=f"chunk-{i}",
                page_index=0,
                original_image_id="img-0.jpeg",
                data=b"image-bytes",
                mime_type="image/jpeg",
                file_size_bytes=11,
            )

        result = execute_reassemble_ocr_chunks(
            document_id="final-doc",
            chunk_document_ids=["chunk-0", "chunk-1"],
            file_path="/tmp/test.pdf",
            total_pages=2,
            engine=engine,
        )
        assert result.document_id == "final-doc"

        # Verify images were reassigned
        images = get_ocr_images(engine, "final-doc")
        assert len(images) == 2


# ---------------------------------------------------------------------------
# OCR image store CRUD
# ---------------------------------------------------------------------------


class TestOcrImageStore:
    def test_save_and_get_roundtrip(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        save_ocr_image(
            engine,
            image_id="img-001",
            document_id="doc-x",
            page_index=0,
            original_image_id="img-0.jpeg",
            data=b"jpeg-bytes-here",
            mime_type="image/jpeg",
            file_size_bytes=15,
            top_left_x=10,
            top_left_y=20,
            bottom_right_x=100,
            bottom_right_y=200,
        )

        img = get_ocr_image(engine, "img-001")
        assert img is not None
        assert img["id"] == "img-001"
        assert img["document_id"] == "doc-x"
        assert img["page_index"] == 0
        assert img["original_image_id"] == "img-0.jpeg"
        assert img["data"] == b"jpeg-bytes-here"
        assert img["mime_type"] == "image/jpeg"
        assert img["file_size_bytes"] == 15
        assert img["top_left_x"] == 10
        assert img["bottom_right_y"] == 200

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        assert get_ocr_image(engine, "nonexistent") is None

    def test_update_document_id(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        save_ocr_image(
            engine,
            image_id="img-upd-1",
            page_index=0,
            original_image_id="img-0.jpeg",
            data=b"bytes",
            mime_type="image/jpeg",
            file_size_bytes=5,
        )
        save_ocr_image(
            engine,
            image_id="img-upd-2",
            page_index=1,
            original_image_id="img-1.jpeg",
            data=b"bytes",
            mime_type="image/jpeg",
            file_size_bytes=5,
        )

        update_ocr_images_document_id(engine, ["img-upd-1", "img-upd-2"], "doc-final")

        img1 = get_ocr_image(engine, "img-upd-1")
        img2 = get_ocr_image(engine, "img-upd-2")
        assert img1["document_id"] == "doc-final"
        assert img2["document_id"] == "doc-final"

    def test_update_empty_list_no_op(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        # Should not raise
        update_ocr_images_document_id(engine, [], "doc-x")

    def test_reassign_document_id(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        save_ocr_image(
            engine,
            image_id="img-ra-1",
            document_id="old-doc-1",
            page_index=0,
            original_image_id="img-0.jpeg",
            data=b"bytes",
            mime_type="image/jpeg",
            file_size_bytes=5,
        )
        save_ocr_image(
            engine,
            image_id="img-ra-2",
            document_id="old-doc-2",
            page_index=0,
            original_image_id="img-0.jpeg",
            data=b"bytes",
            mime_type="image/jpeg",
            file_size_bytes=5,
        )

        reassign_ocr_images_document_id(engine, ["old-doc-1", "old-doc-2"], "new-doc")

        img1 = get_ocr_image(engine, "img-ra-1")
        img2 = get_ocr_image(engine, "img-ra-2")
        assert img1["document_id"] == "new-doc"
        assert img2["document_id"] == "new-doc"

    def test_get_ocr_images_metadata_only(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        save_ocr_image(
            engine,
            image_id="img-list-1",
            document_id="doc-list",
            page_index=0,
            original_image_id="img-0.jpeg",
            data=b"bytes",
            mime_type="image/jpeg",
            file_size_bytes=5,
        )
        save_ocr_image(
            engine,
            image_id="img-list-2",
            document_id="doc-list",
            page_index=1,
            original_image_id="img-1.jpeg",
            data=b"bytes2",
            mime_type="image/png",
            file_size_bytes=6,
        )

        images = get_ocr_images(engine, "doc-list")
        assert len(images) == 2
        # Should be ordered by page_index
        assert images[0]["page_index"] == 0
        assert images[1]["page_index"] == 1
        # Metadata only — no data column
        assert "data" not in images[0]

    def test_get_ocr_images_empty(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        assert get_ocr_images(engine, "nonexistent-doc") == []

    def test_default_empty_document_id(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        save_ocr_image(
            engine,
            image_id="img-default",
            page_index=0,
            original_image_id="img-0.jpeg",
            data=b"bytes",
            mime_type="image/jpeg",
            file_size_bytes=5,
        )

        img = get_ocr_image(engine, "img-default")
        assert img["document_id"] == ""


# ---------------------------------------------------------------------------
# OCR Export — pure functions
# ---------------------------------------------------------------------------


class TestMimeToExtension:
    def test_jpeg(self) -> None:
        assert _mime_to_extension("image/jpeg") == ".jpeg"

    def test_png(self) -> None:
        assert _mime_to_extension("image/png") == ".png"

    def test_unknown(self) -> None:
        assert _mime_to_extension("application/x-unknown-thing") == ".bin"


class TestStripImagePrefix:
    def test_valid_jpeg_unchanged(self) -> None:
        data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        assert _strip_image_prefix(data) == data

    def test_valid_png_unchanged(self) -> None:
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        assert _strip_image_prefix(data) == data

    def test_strips_data_uri_prefix_from_jpeg(self) -> None:
        import base64

        prefix_garbage = base64.b64decode("data:image/jpeg;base64,")
        real_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF rest of image"
        corrupt = prefix_garbage + real_jpeg
        assert _strip_image_prefix(corrupt) == real_jpeg

    def test_unknown_data_returned_as_is(self) -> None:
        data = b"not an image at all"
        assert _strip_image_prefix(data) == data


class TestRewriteOcrUrisToLocal:
    def test_rewrites_matching_uris(self) -> None:
        md = "![chart](ocr-image://abc-123) and ![logo](ocr-image://def-456)"
        mapping = {"abc-123": "abc-123.jpeg", "def-456": "def-456.png"}
        result = rewrite_ocr_uris_to_local(md, mapping)
        assert result == "![chart](abc-123.jpeg) and ![logo](def-456.png)"

    def test_leaves_unknown_uris_unchanged(self) -> None:
        md = "![x](ocr-image://unknown-id)"
        result = rewrite_ocr_uris_to_local(md, {})
        assert result == md

    def test_empty_mapping(self) -> None:
        md = "![x](ocr-image://abc)"
        result = rewrite_ocr_uris_to_local(md, {})
        assert result == md

    def test_no_ocr_uris(self) -> None:
        md = "![x](https://example.com/img.png)"
        result = rewrite_ocr_uris_to_local(md, {"abc": "abc.jpeg"})
        assert result == md


# ---------------------------------------------------------------------------
# OCR Export — testable function
# ---------------------------------------------------------------------------


class TestExecuteExportOcrDocument:
    def test_exports_text_and_images(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        doc_id = "test-doc-export"
        image_id = "img-uuid-001"

        # Save OCR result with an ocr-image:// reference
        save_ocr_result(
            engine,
            document_id=doc_id,
            file_path="/tmp/test.pdf",
            text=f"Hello\n\n![fig](ocr-image://{image_id})",
            page_count=1,
            model_name="mistral-ocr",
            input_tokens=10,
            output_tokens=20,
            batch_id="b-1",
            workflow_id="w-1",
        )

        # Save an image with corrupt data-URI prefix (matches existing DB data)
        import base64 as _b64

        prefix_garbage = _b64.b64decode("data:image/jpeg;base64,")
        real_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF real image data"
        save_ocr_image(
            engine,
            image_id=image_id,
            document_id=doc_id,
            page_index=0,
            original_image_id="img-0.jpeg",
            data=prefix_garbage + real_jpeg,
            mime_type="image/jpeg",
            file_size_bytes=len(prefix_garbage) + len(real_jpeg),
        )

        export_dir = tmp_path / "export"
        result = execute_export_ocr_document(
            document_id=doc_id,
            output_dir=str(export_dir),
            engine=engine,
        )

        assert result.document_id == doc_id
        assert result.export_dir == str(export_dir)
        assert result.image_count == 1

        # Check markdown file — named after original file stem, not document_id
        md_path = export_dir / "test.md"
        assert md_path.exists()
        md_text = md_path.read_text()
        assert f"![fig]({image_id}.jpeg)" in md_text
        assert "ocr-image://" not in md_text

        # Check image file — prefix should be stripped
        img_path = export_dir / f"{image_id}.jpeg"
        assert img_path.exists()
        assert img_path.read_bytes() == real_jpeg

    def test_exports_text_only_no_images(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        doc_id = "text-only-doc"
        save_ocr_result(
            engine,
            document_id=doc_id,
            file_path="/tmp/test.pdf",
            text="Just text, no images.",
            page_count=1,
            model_name="mistral-ocr",
            input_tokens=5,
            output_tokens=10,
            batch_id="b-2",
            workflow_id="w-2",
        )

        export_dir = tmp_path / "export2"
        result = execute_export_ocr_document(
            document_id=doc_id,
            output_dir=str(export_dir),
            engine=engine,
        )

        assert result.image_count == 0
        md_path = export_dir / "test.md"
        assert md_path.read_text() == "Just text, no images."

    def test_returns_not_found_for_missing_document(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        result = execute_export_ocr_document(
            document_id="nonexistent",
            output_dir=str(tmp_path / "out"),
            engine=engine,
        )

        assert result.status == "not_found"
        assert result.document_id == "nonexistent"
        assert result.export_dir == ""
        assert result.markdown_path == ""
        assert result.image_count == 0

    def test_default_xdg_export_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = _setup_db(tmp_path)

        doc_id = "xdg-test-doc"
        xdg_data = tmp_path / "xdg_data"
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))

        save_ocr_result(
            engine,
            document_id=doc_id,
            file_path="/tmp/test.pdf",
            text="XDG test",
            page_count=1,
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="w",
        )

        result = execute_export_ocr_document(
            document_id=doc_id,
            output_dir="",
            engine=engine,
        )

        expected_dir = xdg_data / "forge" / "ocr-export" / doc_id
        assert result.export_dir == str(expected_dir)
        assert (expected_dir / "test.md").exists()


# ---------------------------------------------------------------------------
# OCR Export — model tests
# ---------------------------------------------------------------------------


class TestOcrExportModels:
    def test_export_input_defaults(self) -> None:
        inp = OcrExportInput(document_id="doc-1")
        assert inp.document_id == "doc-1"
        assert inp.output_dir == ""

    def test_export_result(self) -> None:
        result = OcrExportResult(
            document_id="doc-1",
            export_dir="/tmp/out",
            markdown_path="/tmp/out/doc-1.md",
            image_count=3,
        )
        assert result.document_id == "doc-1"
        assert result.image_count == 3


# ---------------------------------------------------------------------------
# OCR Mark for Removal — store functions
# ---------------------------------------------------------------------------


def _save_test_ocr(engine, doc_id: str) -> None:
    """Helper to save a minimal OCR result for testing."""
    save_ocr_result(
        engine,
        document_id=doc_id,
        file_path="/tmp/test.pdf",
        text="test",
        page_count=1,
        model_name="m",
        input_tokens=0,
        output_tokens=0,
        batch_id="b",
        workflow_id="w",
    )


class TestMarkOcrForRemoval:
    def test_marks_existing_document(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        _save_test_ocr(engine, "doc-mark-1")

        result = mark_ocr_for_removal(engine, "doc-mark-1")
        assert result is True

        row = get_ocr_result(engine, "doc-mark-1")
        assert row is not None
        assert row["marked_for_removal"] is True

    def test_returns_false_for_missing_document(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        result = mark_ocr_for_removal(engine, "nonexistent")
        assert result is False

    def test_idempotent(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        _save_test_ocr(engine, "doc-mark-2")

        mark_ocr_for_removal(engine, "doc-mark-2")
        mark_ocr_for_removal(engine, "doc-mark-2")

        row = get_ocr_result(engine, "doc-mark-2")
        assert row["marked_for_removal"] is True


class TestClearOcrRemovalMark:
    def test_clears_marked_document(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        _save_test_ocr(engine, "doc-clear-1")

        mark_ocr_for_removal(engine, "doc-clear-1")
        result = clear_ocr_removal_mark(engine, "doc-clear-1")
        assert result is True

        row = get_ocr_result(engine, "doc-clear-1")
        assert row["marked_for_removal"] is False

    def test_returns_false_for_missing_document(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        result = clear_ocr_removal_mark(engine, "nonexistent")
        assert result is False

    def test_default_is_not_marked(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        _save_test_ocr(engine, "doc-default")

        row = get_ocr_result(engine, "doc-default")
        assert row["marked_for_removal"] is False


# ---------------------------------------------------------------------------
# OCR Duplicate Detection — store function
# ---------------------------------------------------------------------------


class TestFindOcrResultByFilePath:
    def test_finds_existing_result(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        save_ocr_result(
            engine,
            document_id="dup-doc-1",
            file_path="/data/report.pdf",
            text="some text",
            page_count=1,
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="w",
        )

        row = find_ocr_result_by_file_path(engine, "/data/report.pdf")
        assert row is not None
        assert row["document_id"] == "dup-doc-1"

    def test_returns_none_when_no_match(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        row = find_ocr_result_by_file_path(engine, "/data/nonexistent.pdf")
        assert row is None

    def test_excludes_marked_for_removal(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        save_ocr_result(
            engine,
            document_id="dup-doc-2",
            file_path="/data/removed.pdf",
            text="text",
            page_count=1,
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="w",
        )
        mark_ocr_for_removal(engine, "dup-doc-2")

        row = find_ocr_result_by_file_path(engine, "/data/removed.pdf")
        assert row is None


# ---------------------------------------------------------------------------
# OCR Duplicate Detection — activity function
# ---------------------------------------------------------------------------


def _test_input_file() -> str:
    """Return path to a real test file for hash-based duplicate detection tests."""
    import pathlib

    return str(pathlib.Path(__file__).resolve().parent.parent / "test-inputs" / "2311.06440v1.pdf")


class TestExecuteCheckOcrDuplicate:
    def test_detects_duplicate(self, tmp_path: Path) -> None:
        from forge.ocr.activities import compute_file_hash

        test_file = _test_input_file()
        engine, _ = _setup_db(tmp_path)
        file_hash = compute_file_hash(test_file)
        save_ocr_result(
            engine,
            document_id="dup-act-1",
            file_path="/data/dup.pdf",
            text="text",
            page_count=1,
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="w",
            file_hash=file_hash,
        )

        result = execute_check_ocr_duplicate(str(_test_input_file()), engine)
        assert result.is_duplicate is True
        assert result.existing_document_id == "dup-act-1"

    def test_no_duplicate(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        new_file = tmp_path / "new.pdf"
        new_file.write_bytes(b"unique content that has no match")
        result = execute_check_ocr_duplicate(str(new_file), engine)
        assert result.is_duplicate is False
        assert result.existing_document_id == ""

    def test_ignores_marked_for_removal(self, tmp_path: Path) -> None:
        from forge.ocr.activities import compute_file_hash

        engine, _ = _setup_db(tmp_path)
        file_hash = compute_file_hash(str(_test_input_file()))
        save_ocr_result(
            engine,
            document_id="dup-act-2",
            file_path="/data/removed.pdf",
            text="text",
            page_count=1,
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="w",
            file_hash=file_hash,
        )
        mark_ocr_for_removal(engine, "dup-act-2")

        result = execute_check_ocr_duplicate(str(_test_input_file()), engine)
        assert result.is_duplicate is False

    def test_detects_duplicate_different_filename(self, tmp_path: Path) -> None:
        """Same content under a different name is still a duplicate."""
        import shutil

        from forge.ocr.activities import compute_file_hash

        engine, _ = _setup_db(tmp_path)
        file_hash = compute_file_hash(str(_test_input_file()))
        save_ocr_result(
            engine,
            document_id="dup-act-3",
            file_path=str(_test_input_file()),
            text="text",
            page_count=1,
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="w",
            file_hash=file_hash,
        )

        renamed = tmp_path / "renamed_copy.pdf"
        shutil.copy2(_test_input_file(), renamed)
        result = execute_check_ocr_duplicate(str(renamed), engine)
        assert result.is_duplicate is True
        assert result.existing_document_id == "dup-act-3"


# ---------------------------------------------------------------------------
# OCR Duplicate Detection — model tests
# ---------------------------------------------------------------------------


class TestOcrDuplicateCheckResult:
    def test_defaults(self) -> None:
        result = OcrDuplicateCheckResult(is_duplicate=False)
        assert result.existing_document_id == ""

    def test_duplicate(self) -> None:
        result = OcrDuplicateCheckResult(is_duplicate=True, existing_document_id="doc-1")
        assert result.is_duplicate is True
        assert result.existing_document_id == "doc-1"


# ---------------------------------------------------------------------------
# OCR Mark for Removal — model tests
# ---------------------------------------------------------------------------


class TestOcrMarkModels:
    def test_mark_input(self) -> None:
        inp = OcrMarkInput(document_id="doc-1")
        assert inp.document_id == "doc-1"

    def test_mark_result(self) -> None:
        result = OcrMarkResult(document_id="doc-1", found=True)
        assert result.found is True


# ---------------------------------------------------------------------------
# Backfill hash store functions
# ---------------------------------------------------------------------------


class TestBackfillHashFunctions:
    def _save_result(
        self,
        engine: Engine,
        doc_id: str,
        file_hash: str | None = None,
    ) -> None:
        save_ocr_result(
            engine,
            document_id=doc_id,
            file_path=f"/data/{doc_id}.pdf",
            text="text",
            page_count=1,
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="w",
            file_hash=file_hash,
        )

    def test_get_results_missing_hash(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        self._save_result(engine, "has-hash", file_hash="abc123")
        self._save_result(engine, "no-hash-1")
        self._save_result(engine, "no-hash-2")

        missing = get_ocr_results_missing_hash(engine)
        doc_ids = {r["document_id"] for r in missing}
        assert doc_ids == {"no-hash-1", "no-hash-2"}

    def test_get_results_missing_hash_empty(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        self._save_result(engine, "has-hash", file_hash="abc123")
        assert get_ocr_results_missing_hash(engine) == []

    def test_update_ocr_file_hash(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        self._save_result(engine, "doc-1")

        assert update_ocr_file_hash(engine, "doc-1", "newhash") is True

        result = get_ocr_result(engine, "doc-1")
        assert result["file_hash"] == "newhash"

    def test_update_ocr_file_hash_nonexistent(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        assert update_ocr_file_hash(engine, "no-such-doc", "hash") is False


# ---------------------------------------------------------------------------
# execute_list_ocr_jobs
# ---------------------------------------------------------------------------


class TestExecuteListOcrJobs:
    """Tests for the OCR job list query — grouping, sort, filter, join."""

    def _insert_batch_row(
        self,
        engine: Engine,
        *,
        request_id: str,
        document_id: str,
        file_path: str | None,
        status: str,
        created_at,
        provider: str = "mistral",
        batch_id: str | None = "b-1",
    ) -> None:
        """Directly insert a batch_jobs row with explicit timestamp."""
        import sqlalchemy as sa

        from forge.store import BatchJob

        with engine.begin() as conn:
            conn.execute(
                sa.insert(BatchJob.__table__).values(
                    id=request_id,
                    batch_id=batch_id,
                    workflow_id=f"wf-{request_id}",
                    status=status,
                    provider=provider,
                    file_path=file_path,
                    document_id=document_id,
                    created_at=created_at,
                    updated_at=created_at,
                )
            )

    def test_single_submission_single_chunk_succeeded(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        ts = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-1",
            document_id="doc-1",
            file_path="/data/a.pdf",
            status="succeeded",
            created_at=ts,
        )
        save_ocr_result(
            engine,
            document_id="doc-1",
            file_path="/data/a.pdf",
            text="content",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b-1",
            workflow_id="wf-req-1",
        )

        result = execute_list_ocr_jobs(engine)

        assert result.total == 1
        assert len(result.jobs) == 1
        job = result.jobs[0]
        assert job.file_path == "/data/a.pdf"
        assert job.document_id == "doc-1"
        assert job.status == "succeeded"
        assert job.chunk_count == 1

    def test_multi_chunk_single_submission(self, tmp_path: Path) -> None:
        """All chunks of one submission collapse to a single row."""
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        base = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        for i in range(3):
            self._insert_batch_row(
                engine,
                request_id=f"req-{i}",
                document_id="doc-multi",
                file_path="/data/multi.pdf",
                status="succeeded",
                created_at=base,
            )

        result = execute_list_ocr_jobs(engine)

        assert result.total == 1
        assert result.jobs[0].chunk_count == 3
        assert result.jobs[0].status == "succeeded"

    def test_resubmission_appears_as_distinct_row(self, tmp_path: Path) -> None:
        """Errored original and its resubmission each occupy their own row."""
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)

        # Original errored submission
        self._insert_batch_row(
            engine,
            request_id="req-old",
            document_id="doc-old",
            file_path="/data/same.pdf",
            status="errored",
            created_at=datetime(2026, 4, 13, 9, 0, 0, tzinfo=UTC),
        )
        # Resubmission — still processing
        self._insert_batch_row(
            engine,
            request_id="req-new",
            document_id="doc-new",
            file_path="/data/same.pdf",
            status="submitted",
            created_at=datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC),
        )

        result = execute_list_ocr_jobs(engine)

        assert result.total == 2
        # Newest first (sorted by MAX(created_at) DESC)
        assert result.jobs[0].document_id == "doc-new"
        assert result.jobs[0].status == "processing"
        assert result.jobs[1].document_id == "doc-old"
        assert result.jobs[1].status == "errored"

    def test_resubmission_old_errored_unaffected_by_new_success(self, tmp_path: Path) -> None:
        """A successful resubmission does NOT mark the old errored row as succeeded."""
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)

        self._insert_batch_row(
            engine,
            request_id="req-old",
            document_id="doc-old",
            file_path="/data/same.pdf",
            status="errored",
            created_at=datetime(2026, 4, 13, 9, 0, 0, tzinfo=UTC),
        )
        self._insert_batch_row(
            engine,
            request_id="req-new",
            document_id="doc-new",
            file_path="/data/same.pdf",
            status="succeeded",
            created_at=datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC),
        )
        save_ocr_result(
            engine,
            document_id="doc-new",
            file_path="/data/same.pdf",
            text="ok",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b-new",
            workflow_id="wf-new",
        )

        result = execute_list_ocr_jobs(engine)

        doc_ids_by_status = {job.status: job.document_id for job in result.jobs}
        assert doc_ids_by_status["errored"] == "doc-old"
        assert doc_ids_by_status["succeeded"] == "doc-new"
        assert result.total == 2

    def test_status_filter_errored_only(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        ts = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-ok",
            document_id="doc-ok",
            file_path="/data/a.pdf",
            status="succeeded",
            created_at=ts,
        )
        self._insert_batch_row(
            engine,
            request_id="req-err",
            document_id="doc-err",
            file_path="/data/b.pdf",
            status="errored",
            created_at=ts,
        )

        result = execute_list_ocr_jobs(engine, status_filter="errored")

        assert result.total == 1
        assert result.jobs[0].document_id == "doc-err"
        assert result.jobs[0].status == "errored"

    def test_sort_order_newest_first(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)

        self._insert_batch_row(
            engine,
            request_id="req-old",
            document_id="doc-old",
            file_path="/data/old.pdf",
            status="succeeded",
            created_at=datetime(2026, 4, 10, 10, 0, 0, tzinfo=UTC),
        )
        self._insert_batch_row(
            engine,
            request_id="req-new",
            document_id="doc-new",
            file_path="/data/new.pdf",
            status="succeeded",
            created_at=datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC),
        )
        self._insert_batch_row(
            engine,
            request_id="req-mid",
            document_id="doc-mid",
            file_path="/data/mid.pdf",
            status="succeeded",
            created_at=datetime(2026, 4, 12, 10, 0, 0, tzinfo=UTC),
        )

        result = execute_list_ocr_jobs(engine)

        ids = [job.document_id for job in result.jobs]
        assert ids == ["doc-new", "doc-mid", "doc-old"]

    def test_limit_respected(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        base = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        for i in range(5):
            self._insert_batch_row(
                engine,
                request_id=f"req-{i}",
                document_id=f"doc-{i}",
                file_path=f"/data/f{i}.pdf",
                status="succeeded",
                created_at=base,
            )

        result = execute_list_ocr_jobs(engine, limit=2)

        assert len(result.jobs) == 2
        assert result.total == 2

    def test_excludes_marked_for_removal_results(self, tmp_path: Path) -> None:
        """ocr_results rows marked for removal are excluded from the join.

        The batch_jobs row still shows up (so the user can see the submission
        happened), but the document_id field falls back to the batch_jobs
        document_id since the ocr_results join is filtered out.
        """
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        ts = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-1",
            document_id="doc-removed",
            file_path="/data/x.pdf",
            status="succeeded",
            created_at=ts,
        )
        save_ocr_result(
            engine,
            document_id="doc-removed",
            file_path="/data/x.pdf",
            text="content",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b-1",
            workflow_id="wf-req-1",
        )
        mark_ocr_for_removal(engine, "doc-removed")

        result = execute_list_ocr_jobs(engine)

        # The batch_jobs row is still listed (submission history)
        assert result.total == 1
        # document_id falls back to the batch_jobs submission_document_id
        # since the ocr_results row was filtered out by marked_for_removal.
        assert result.jobs[0].document_id == "doc-removed"

    def test_excludes_non_mistral_provider(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        ts = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-mistral",
            document_id="doc-m",
            file_path="/data/m.pdf",
            status="succeeded",
            created_at=ts,
        )
        self._insert_batch_row(
            engine,
            request_id="req-anthropic",
            document_id="doc-a",
            file_path="/data/a.pdf",
            status="succeeded",
            created_at=ts,
            provider="anthropic",
        )

        result = execute_list_ocr_jobs(engine)

        assert result.total == 1
        assert result.jobs[0].document_id == "doc-m"

    def test_excludes_rows_without_file_path(self, tmp_path: Path) -> None:
        """Generic batch submissions (no file_path) must not appear."""
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        ts = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-ocr",
            document_id="doc-ocr",
            file_path="/data/x.pdf",
            status="succeeded",
            created_at=ts,
        )
        self._insert_batch_row(
            engine,
            request_id="req-generic",
            document_id="doc-generic",
            file_path=None,
            status="succeeded",
            created_at=ts,
        )

        result = execute_list_ocr_jobs(engine)

        assert result.total == 1
        assert result.jobs[0].document_id == "doc-ocr"

    def test_mixed_chunk_statuses_any_errored(self, tmp_path: Path) -> None:
        """If any chunk errored, the submission is 'errored'."""
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        base = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-0",
            document_id="doc-mix",
            file_path="/data/mix.pdf",
            status="succeeded",
            created_at=base,
        )
        self._insert_batch_row(
            engine,
            request_id="req-1",
            document_id="doc-mix",
            file_path="/data/mix.pdf",
            status="errored",
            created_at=base,
        )

        result = execute_list_ocr_jobs(engine)

        assert result.total == 1
        assert result.jobs[0].status == "errored"
        assert result.jobs[0].chunk_count == 2

    def test_mixed_chunk_statuses_any_submitted_is_processing(self, tmp_path: Path) -> None:
        """If any chunk is still submitted (and none errored), status is 'processing'."""
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        base = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-0",
            document_id="doc-proc",
            file_path="/data/proc.pdf",
            status="succeeded",
            created_at=base,
        )
        self._insert_batch_row(
            engine,
            request_id="req-1",
            document_id="doc-proc",
            file_path="/data/proc.pdf",
            status="submitted",
            created_at=base,
        )

        result = execute_list_ocr_jobs(engine)

        assert result.total == 1
        assert result.jobs[0].status == "processing"

    def test_storing_chunk_derives_to_processing(self, tmp_path: Path) -> None:
        """A chunk in STORING (post-Mistral, pre-store) surfaces as 'processing'."""
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        ts = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-storing",
            document_id="doc-storing",
            file_path="/data/storing.pdf",
            status="storing",
            created_at=ts,
        )

        result = execute_list_ocr_jobs(engine)

        assert result.total == 1
        assert result.jobs[0].status == "processing"

    def test_mixed_succeeded_and_storing_is_processing(self, tmp_path: Path) -> None:
        """One chunk SUCCEEDED + another STORING -> still 'processing'."""
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        base = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-0",
            document_id="doc-half",
            file_path="/data/half.pdf",
            status="succeeded",
            created_at=base,
        )
        self._insert_batch_row(
            engine,
            request_id="req-1",
            document_id="doc-half",
            file_path="/data/half.pdf",
            status="storing",
            created_at=base,
        )

        result = execute_list_ocr_jobs(engine)

        assert result.total == 1
        assert result.jobs[0].status == "processing"

    def test_failed_chunk_derives_to_errored(self, tmp_path: Path) -> None:
        """A chunk in FAILED (submit refused by provider) surfaces as 'errored'."""
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        ts = datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-failed",
            document_id="doc-failed",
            file_path="/data/failed.pdf",
            status="failed",
            created_at=ts,
            batch_id=None,
        )

        result = execute_list_ocr_jobs(engine)

        assert result.total == 1
        assert result.jobs[0].status == "errored"

    def test_created_at_is_tz_aware_utc(self, tmp_path: Path) -> None:
        """created_at must be emitted with an explicit UTC tz suffix.

        Without this, downstream tools parsing the naive ISO string
        (e.g. nushell `into datetime`) interpret the value as local
        time and shift it by the local UTC offset, causing timestamps
        to appear hours in the future.
        """
        from datetime import UTC, datetime

        engine, _ = _setup_db(tmp_path)
        ts = datetime(2026, 4, 13, 17, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-tz",
            document_id="doc-tz",
            file_path="/data/tz.pdf",
            status="succeeded",
            created_at=ts,
        )

        result = execute_list_ocr_jobs(engine)

        assert result.total == 1
        emitted = result.jobs[0].created_at
        # Must carry explicit UTC offset so clients parse correctly.
        assert emitted.endswith("+00:00") or emitted.endswith("Z"), (
            f"created_at lacks tz suffix: {emitted!r}"
        )
        # And must round-trip back to the same instant.
        parsed = datetime.fromisoformat(emitted)
        assert parsed.tzinfo is not None
        assert parsed == ts

    def test_batch_job_row_read_is_tz_aware_utc(self, tmp_path: Path) -> None:
        """Direct BatchJob row reads return tz-aware UTC datetimes.

        This validates the UTCDateTime TypeDecorator at the column level,
        not just through the list_ocr_jobs serialization layer.
        """
        from datetime import UTC, datetime

        from forge.store import get_batch_job

        engine, _ = _setup_db(tmp_path)
        ts = datetime(2026, 4, 13, 17, 0, 0, tzinfo=UTC)

        self._insert_batch_row(
            engine,
            request_id="req-direct",
            document_id="doc-direct",
            file_path="/data/direct.pdf",
            status="succeeded",
            created_at=ts,
        )

        job = get_batch_job(engine, "req-direct")
        assert job is not None
        assert job["created_at"].tzinfo == UTC
        assert job["created_at"] == ts
        assert job["updated_at"].tzinfo == UTC
