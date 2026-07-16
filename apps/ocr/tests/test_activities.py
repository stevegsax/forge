"""Tests for ocr.activities — pure helpers and engine-injected functions."""

from __future__ import annotations

import json

import pytest
import sqlalchemy as sa
from forge_contracts.models import dump_batch_result_payload

from ocr.activities import (
    CHUNK_SIZE_PAGES,
    MAX_FILE_SIZE_BYTES,
    MAX_PAGES,
    _derive_status,
    _mime_to_extension,
    _strip_image_prefix,
    build_ocr_batch_body,
    detect_mime_type,
    execute_build_request_blob,
    execute_list_ocr_jobs,
    execute_split_file_into_chunks,
    execute_store_ocr_result,
    parse_ocr_pages,
    rewrite_image_references,
    rewrite_ocr_uris_to_local,
    validate_file_size,
)
from ocr.models import OcrJobDerivedStatus
from ocr.store import (
    get_file_content,
    get_ocr_images,
    get_ocr_job_status,
    get_ocr_result,
    get_store_engine,
    save_file_content,
    upsert_ocr_job_status,
)

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestPure:
    def test_build_body_image_vs_document(self) -> None:
        img = build_ocr_batch_body("ZGF0YQ==", "image/png")
        assert img["document"]["type"] == "image_url"
        assert img["include_image_base64"] is True
        doc = build_ocr_batch_body("ZGF0YQ==", "application/pdf")
        assert doc["document"]["type"] == "document_url"

    def test_rewrite_image_references(self) -> None:
        md = "see ![alt](img-0.jpeg) here"
        out = rewrite_image_references(md, {"img-0.jpeg": "uuid-1"})
        assert out == "see ![alt](ocr-image://uuid-1) here"

    def test_rewrite_no_mapping_is_noop(self) -> None:
        assert rewrite_image_references("![a](img-0.jpeg)", {}) == "![a](img-0.jpeg)"

    def test_parse_ocr_pages(self) -> None:
        body = json.dumps(
            {
                "model": "mistral-ocr",
                "pages": [{"markdown": "page one ![a](img-0.jpeg)"}, {"markdown": "page two"}],
                "usage_info": {"pages_processed": 2, "doc_size_bytes": 99},
            }
        )
        result = parse_ocr_pages(body, {"img-0.jpeg": "uuid-1"})
        assert "ocr-image://uuid-1" in result.text
        assert "page two" in result.text
        assert result.page_count == 2
        assert result.model_name == "mistral-ocr"
        assert result.image_ids == ["uuid-1"]

    def test_strip_image_prefix(self) -> None:
        clean = b"\xff\xd8\xffreal"
        assert _strip_image_prefix(b"junk" + clean) == clean
        assert _strip_image_prefix(clean) == clean

    def test_derive_status(self) -> None:
        assert _derive_status("stored", "processing") == OcrJobDerivedStatus.SUCCEEDED.value
        assert _derive_status("failed", None) == OcrJobDerivedStatus.ERRORED.value
        assert _derive_status("submitted", "failed") == OcrJobDerivedStatus.ERRORED.value
        assert _derive_status("submitted", "submitted") == OcrJobDerivedStatus.PROCESSING.value


class TestDetectMimeType:
    def test_pdf(self) -> None:
        assert detect_mime_type("/tmp/test.pdf") == "application/pdf"

    def test_png(self) -> None:
        assert detect_mime_type("/tmp/test.png") == "image/png"

    def test_jpeg(self) -> None:
        assert detect_mime_type("/tmp/test.jpg") == "image/jpeg"

    def test_unknown(self) -> None:
        assert detect_mime_type("/tmp/test.xyz123") == "application/octet-stream"


class TestMimeToExtension:
    def test_jpeg(self) -> None:
        assert _mime_to_extension("image/jpeg") == ".jpeg"

    def test_png(self) -> None:
        assert _mime_to_extension("image/png") == ".png"

    def test_unknown(self) -> None:
        assert _mime_to_extension("application/x-unknown-thing") == ".bin"


class TestValidateFileSize:
    def test_pdf_not_rejected_for_size(self) -> None:
        """PDFs are never rejected by validate_file_size (they get split instead)."""
        validate_file_size(100 * 1024 * 1024, "application/pdf")

    def test_non_pdf_under_limit_passes(self) -> None:
        validate_file_size(MAX_FILE_SIZE_BYTES, "image/png")

    def test_non_pdf_over_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="Non-PDF file"):
            validate_file_size(MAX_FILE_SIZE_BYTES + 1, "image/png")


class TestRewriteOcrUrisToLocal:
    def test_rewrites_matching_uris(self) -> None:
        md = "![chart](ocr-image://abc-123) and ![logo](ocr-image://def-456)"
        mapping = {"abc-123": "abc-123.jpeg", "def-456": "def-456.png"}
        result = rewrite_ocr_uris_to_local(md, mapping)
        assert result == "![chart](abc-123.jpeg) and ![logo](def-456.png)"

    def test_leaves_unknown_uris_unchanged(self) -> None:
        md = "![x](ocr-image://unknown-id)"
        result = rewrite_ocr_uris_to_local(md, {"other": "other.jpeg"})
        assert result == md

    def test_empty_mapping(self) -> None:
        md = "![x](ocr-image://abc)"
        assert rewrite_ocr_uris_to_local(md, {}) == md

    def test_no_ocr_uris(self) -> None:
        md = "![x](https://example.com/img.png)"
        assert rewrite_ocr_uris_to_local(md, {"abc": "abc.jpeg"}) == md


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
    def test_non_pdf_single_chunk(self, migrated: str) -> None:
        """Non-PDF files produce a single chunk reusing the original blob."""
        engine = get_store_engine()
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

    def test_non_pdf_over_size_raises(self, migrated: str) -> None:
        """Non-PDF files exceeding the size limit are rejected."""
        engine = get_store_engine()
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

    def test_small_pdf_single_chunk(self, migrated: str) -> None:
        """PDFs under cutoffs produce a single chunk reusing the original blob."""
        engine = get_store_engine()
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
        assert get_file_content(engine, content_id) is not None

    def test_boundary_30_pages_single_chunk(self, migrated: str) -> None:
        """Exactly MAX_PAGES pages stays a single chunk."""
        engine = get_store_engine()
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

    def test_boundary_31_pages_splits(self, migrated: str) -> None:
        """MAX_PAGES + 1 pages triggers a split: 25 + 6."""
        engine = get_store_engine()
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

        assert result.chunks[0].page_start == 1
        assert result.chunks[0].page_end == CHUNK_SIZE_PAGES
        assert result.chunks[0].chunk_index == 0

        assert result.chunks[1].page_start == CHUNK_SIZE_PAGES + 1
        assert result.chunks[1].page_end == MAX_PAGES + 1
        assert result.chunks[1].chunk_index == 1

    def test_large_pdf_60_pages_3_chunks(self, migrated: str) -> None:
        """60-page PDF splits into 3 chunks: 25 + 25 + 10."""
        engine = get_store_engine()
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
        assert (result.chunks[0].page_start, result.chunks[0].page_end) == (1, 25)
        assert (result.chunks[1].page_start, result.chunks[1].page_end) == (26, 50)
        assert (result.chunks[2].page_start, result.chunks[2].page_end) == (51, 60)

    def test_original_blob_deleted_after_split(self, migrated: str) -> None:
        """After a multi-chunk split, the original blob is deleted; chunk blobs exist."""
        engine = get_store_engine()
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

        assert get_file_content(engine, content_id) is None
        for chunk in result.chunks:
            assert get_file_content(engine, chunk.content_id) is not None

    def test_original_blob_kept_for_single_chunk(self, migrated: str) -> None:
        """The single-chunk case preserves the original blob."""
        engine = get_store_engine()
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

    def test_chunk_pdfs_are_valid(self, migrated: str) -> None:
        """Each chunk blob is a valid PDF with the expected page count."""
        import fitz

        engine = get_store_engine()
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

        for chunk, expected in zip(result.chunks, [25, 25, 10], strict=True):
            blob = get_file_content(engine, chunk.content_id)
            assert blob is not None
            doc = fitz.open(stream=blob["data"], filetype="pdf")
            assert len(doc) == expected
            doc.close()

    def test_missing_content_raises(self, migrated: str) -> None:
        """Raises RuntimeError if the content_id is not found in the store."""
        engine = get_store_engine()
        with pytest.raises(RuntimeError, match="File content not found"):
            execute_split_file_into_chunks("nonexistent", "application/pdf", 1000, engine)


# ---------------------------------------------------------------------------
# execute_store_ocr_result
# ---------------------------------------------------------------------------


class TestStoreOcrResult:
    def test_inline_body_no_images(self, migrated: str) -> None:
        engine = get_store_engine()
        body = json.dumps(
            {"model": "m", "pages": [{"markdown": "hello"}], "usage_info": {"pages_processed": 1}}
        )
        result = execute_store_ocr_result(
            request_id="r1",
            document_id="doc-1",
            file_path="/a.pdf",
            batch_id="b1",
            workflow_id="wf",
            raw_response_json=body,
            s3_key=None,
            engine=engine,
        )
        assert result.text_length > 0
        assert get_ocr_result(engine, "doc-1")["text"] == "hello"
        assert get_ocr_job_status(engine, "r1")["status"] == "stored"

    def test_s3_envelope_with_images(self, migrated: str) -> None:
        from forge_contracts import s3_blobs

        body = json.dumps(
            {"model": "m", "pages": [{"markdown": "![x](img-0.jpeg)"}], "usage_info": {}}
        )
        images = [
            {
                "original_image_id": "img-0.jpeg",
                "page_index": 0,
                "image_base64": "ZmFrZQ==",
                "mime_type": "image/jpeg",
            }
        ]
        key = s3_blobs.build_key("batch-result-r2")
        envelope = dump_batch_result_payload(body, images).encode("utf-8")
        s3_blobs.put(key, envelope, "application/json")

        engine = get_store_engine()
        result = execute_store_ocr_result(
            request_id="r2",
            document_id="doc-2",
            file_path="",
            batch_id="b2",
            workflow_id="wf",
            raw_response_json=None,
            s3_key=key,
            engine=engine,
        )
        assert result.page_count == 1
        stored = get_ocr_result(engine, "doc-2")
        assert "ocr-image://" in stored["text"]
        imgs = get_ocr_images(engine, "doc-2")
        assert len(imgs) == 1
        assert imgs[0]["document_id"] == "doc-2"

    def test_idempotent_on_retry(self, migrated: str) -> None:
        engine = get_store_engine()
        body = json.dumps({"model": "m", "pages": [{"markdown": "hi"}], "usage_info": {}})
        kw = dict(
            request_id="r3",
            document_id="doc-3",
            file_path="",
            batch_id="b3",
            workflow_id="wf",
            raw_response_json=body,
            s3_key=None,
            engine=engine,
        )
        execute_store_ocr_result(**kw)
        execute_store_ocr_result(**kw)  # retry — must not raise or duplicate
        assert get_ocr_result(engine, "doc-3") is not None


# ---------------------------------------------------------------------------
# execute_build_request_blob
# ---------------------------------------------------------------------------


class TestBuildRequestBlob:
    def test_mints_id_and_stashes_blob(self, migrated: str) -> None:
        from forge_contracts import s3_blobs

        engine = get_store_engine()
        save_file_content(
            engine,
            content_id="c1",
            data=b"%PDF-1.4",
            mime_type="application/pdf",
            file_size_bytes=8,
        )
        ref = execute_build_request_blob(
            content_id="c1",
            mime_type="application/pdf",
            model_name="mistral:mistral-ocr-latest",
            engine=engine,
        )
        assert ref.request_id
        assert ref.model == "mistral-ocr-latest"
        requests = json.loads(s3_blobs.get(ref.s3_key).decode("utf-8"))
        assert requests[0]["custom_id"] == ref.request_id
        assert requests[0]["body"]["include_image_base64"] is True


# ---------------------------------------------------------------------------
# execute_list_ocr_jobs (status join)
# ---------------------------------------------------------------------------


class TestListOcrJobs:
    def test_status_join(self, migrated: str) -> None:
        from forge_contracts.batch_jobs import batch_jobs
        from forge_contracts.batch_jobs import metadata as bj_metadata

        engine = get_store_engine()
        # Platform owns batch_jobs — create it for the join (OCR migration doesn't).
        bj_metadata.create_all(engine)

        # A stored job, a still-processing job, and a provider-failed job.
        upsert_ocr_job_status(
            engine, request_id="r-ok", document_id="d-ok", file_path="/ok.pdf", status="stored"
        )
        upsert_ocr_job_status(
            engine, request_id="r-go", document_id="d-go", file_path="/go.pdf", status="submitted"
        )
        upsert_ocr_job_status(
            engine,
            request_id="r-bad",
            document_id="d-bad",
            file_path="/bad.pdf",
            status="submitted",
        )
        with engine.begin() as conn:
            conn.execute(
                sa.insert(batch_jobs).values(
                    id="r-bad", batch_id="b", workflow_id="wf", status="failed", provider="mistral"
                )
            )

        result = execute_list_ocr_jobs(engine)
        by_doc = {j.document_id: j.status for j in result.jobs}
        assert by_doc["d-ok"] == OcrJobDerivedStatus.SUCCEEDED.value
        assert by_doc["d-go"] == OcrJobDerivedStatus.PROCESSING.value
        assert by_doc["d-bad"] == OcrJobDerivedStatus.ERRORED.value
