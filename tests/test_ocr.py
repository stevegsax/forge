"""Tests for forge.ocr — OCR models, activities, and store functions."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.ocr.activities import (
    build_ocr_messages,
    detect_mime_type,
    execute_parse_ocr_result,
    execute_read_file,
    execute_store_ocr_result,
    execute_submit_ocr_batch,
)
from forge.ocr.models import (
    FileContentResult,
    OcrParseResult,
    OcrStoreInput,
    OcrStoreResult,
    OcrSubmitInput,
    OcrSubmitResult,
)
from forge.store import (
    get_engine,
    get_ocr_result,
    run_migrations,
    save_ocr_result,
)

if TYPE_CHECKING:
    from pathlib import Path


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
        assert inp.model_name == "mistral:pixtral-large-latest"
        assert inp.max_tokens == 16384
        assert inp.document_id == ""

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
        mock_provider.build_request_params = MagicMock(return_value={"model": "test"})
        mock_provider.build_batch_request = MagicMock(
            return_value={"custom_id": "req-1", "params": {}}
        )
        mock_provider.submit_batch = AsyncMock(return_value="batch-123")

        inp = OcrSubmitInput(
            file_path="/tmp/test.pdf",
            model_name="mistral:pixtral-large-latest",
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

        # Verify provider methods were called
        mock_provider.build_request_params.assert_called_once()
        call_kwargs = mock_provider.build_request_params.call_args
        assert call_kwargs.kwargs.get("output_type") is None
        mock_provider.build_batch_request.assert_called_once()
        mock_provider.submit_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_generates_document_id(self) -> None:
        mock_provider = MagicMock()
        mock_provider.build_request_params = MagicMock(return_value={})
        mock_provider.build_batch_request = MagicMock(return_value={})
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


class TestExecuteParseOcrResult:
    def test_parses_mistral_response(self) -> None:
        raw_json = json.dumps({
            "choices": [
                {
                    "message": {
                        "content": "Extracted document text here.",
                    }
                }
            ],
            "model": "pixtral-large-latest",
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 100,
            },
        })

        # Mock the provider's parse_batch_result
        from forge.llm_providers.models import ProviderResponse

        mock_provider = MagicMock()
        mock_provider.parse_batch_result = MagicMock(
            return_value=ProviderResponse(
                text_content="Extracted document text here.",
                model_name="pixtral-large-latest",
                input_tokens=500,
                output_tokens=100,
                raw_response_json=raw_json,
            )
        )

        original_get_provider = None
        try:
            # We need to patch the import inside execute_parse_ocr_result
            import forge.llm_providers as llm_mod

            original_get_provider = llm_mod.get_provider
            llm_mod.get_provider = MagicMock(return_value=mock_provider)

            result = execute_parse_ocr_result(raw_json, provider_name="mistral")
            assert result.text == "Extracted document text here."
            assert result.model_name == "pixtral-large-latest"
            assert result.input_tokens == 500
            assert result.output_tokens == 100
        finally:
            if original_get_provider is not None:
                llm_mod.get_provider = original_get_provider

    def test_empty_text_returns_empty_string(self) -> None:
        from forge.llm_providers.models import ProviderResponse

        mock_provider = MagicMock()
        mock_provider.parse_batch_result = MagicMock(
            return_value=ProviderResponse(
                text_content=None,
                model_name="pixtral-large",
                input_tokens=10,
                output_tokens=0,
                raw_response_json="{}",
            )
        )

        import forge.llm_providers as llm_mod

        original = llm_mod.get_provider
        try:
            llm_mod.get_provider = MagicMock(return_value=mock_provider)
            result = execute_parse_ocr_result("{}", provider_name="mistral")
            assert result.text == ""
        finally:
            llm_mod.get_provider = original


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
