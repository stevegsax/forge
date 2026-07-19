"""Tests for ocr.activities — pure helpers and engine-injected functions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from sax_platform.contracts.models import dump_batch_result_payload

from ocr.activities import (
    CHUNK_SIZE_PAGES,
    MAX_FILE_SIZE_BYTES,
    MAX_PAGES,
    _derive_status,
    _get_export_dir,
    _mime_to_extension,
    _strip_image_prefix,
    build_ocr_batch_body,
    compute_file_hash,
    detect_mime_type,
    execute_build_request_blob,
    execute_check_ocr_duplicate,
    execute_export_ocr_document,
    execute_list_ocr_jobs,
    execute_read_and_store_file,
    execute_reassemble_ocr_chunks,
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
    save_file_content,
    save_ocr_result,
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

    def test_rewrite_image_references_partial_mapping_leaves_unmapped(self) -> None:
        """A ref not present in the mapping is left untouched (not just dropped)."""
        md = "![a](img-0.jpeg) and ![b](img-1.jpeg)"
        out = rewrite_image_references(md, {"img-0.jpeg": "uuid-1"})
        assert out == "![a](ocr-image://uuid-1) and ![b](img-1.jpeg)"

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

    def test_strip_image_prefix_no_marker_returns_unchanged(self) -> None:
        """When no JPEG/PNG signature is found anywhere, the data passes through."""
        data = b"not an image at all"
        assert _strip_image_prefix(data) == data

    def test_derive_status(self) -> None:
        assert _derive_status("stored", "processing") == OcrJobDerivedStatus.SUCCEEDED.value
        assert _derive_status("failed", None) == OcrJobDerivedStatus.ERRORED.value
        assert _derive_status("submitted", "failed") == OcrJobDerivedStatus.ERRORED.value
        assert _derive_status("submitted", "submitted") == OcrJobDerivedStatus.PROCESSING.value

    def test_derive_status_unknown_combination_falls_through(self) -> None:
        """A status pair matching none of the known rules maps to UNKNOWN."""
        assert _derive_status("queued", "running") == OcrJobDerivedStatus.UNKNOWN.value

    def test_compute_file_hash_matches_sha256(self, tmp_path) -> None:
        path = tmp_path / "content.bin"
        data = b"some file bytes to hash" * 100
        path.write_bytes(data)
        assert compute_file_hash(str(path)) == hashlib.sha256(data).hexdigest()


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
# execute_read_and_store_file
# ---------------------------------------------------------------------------


class TestExecuteReadAndStoreFile:
    def test_reads_and_stores_bytes(self, store_engine, blobs, tmp_path) -> None:
        engine = store_engine
        path = tmp_path / "doc.pdf"
        data = b"%PDF-1.4 fake pdf bytes"
        path.write_bytes(data)

        ref = execute_read_and_store_file(str(path), engine, blobs)

        assert ref.mime_type == "application/pdf"
        assert ref.file_size_bytes == len(data)
        stored = get_file_content(engine, ref.content_id, blobs)
        assert stored is not None
        assert stored["data"] == data


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
    def test_non_pdf_single_chunk(self, store_engine, blobs) -> None:
        """Non-PDF files produce a single chunk reusing the original blob."""
        engine = store_engine
        content_id = "img-content"
        data = b"fake image bytes"
        save_file_content(
            engine,
            content_id=content_id,
            data=data,
            mime_type="image/png",
            file_size_bytes=len(data),
            blobs=blobs,
        )

        result = execute_split_file_into_chunks(content_id, "image/png", len(data), engine, blobs)

        assert len(result.chunks) == 1
        assert result.chunks[0].content_id == content_id  # reuses original
        assert result.chunks[0].chunk_index == 0
        assert result.chunks[0].page_start == 1
        assert result.chunks[0].page_end == 1
        assert result.total_pages == 1
        assert result.original_content_id == content_id

    def test_non_pdf_over_size_raises(self, store_engine, blobs) -> None:
        """Non-PDF files exceeding the size limit are rejected."""
        engine = store_engine
        content_id = "big-img"
        data = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        save_file_content(
            engine,
            content_id=content_id,
            data=data,
            mime_type="image/png",
            file_size_bytes=len(data),
            blobs=blobs,
        )

        with pytest.raises(ValueError, match="Non-PDF file"):
            execute_split_file_into_chunks(content_id, "image/png", len(data), engine, blobs)

    def test_small_pdf_single_chunk(self, store_engine, blobs) -> None:
        """PDFs under cutoffs produce a single chunk reusing the original blob."""
        engine = store_engine
        pdf_data = _create_test_pdf(10)
        content_id = "small-pdf"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
            blobs=blobs,
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine, blobs
        )

        assert len(result.chunks) == 1
        assert result.chunks[0].content_id == content_id  # reuses original
        assert result.chunks[0].page_start == 1
        assert result.chunks[0].page_end == 10
        assert result.total_pages == 10
        assert get_file_content(engine, content_id, blobs) is not None

    def test_boundary_30_pages_single_chunk(self, store_engine, blobs) -> None:
        """Exactly MAX_PAGES pages stays a single chunk."""
        engine = store_engine
        pdf_data = _create_test_pdf(MAX_PAGES)
        content_id = "boundary-30"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
            blobs=blobs,
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine, blobs
        )

        assert len(result.chunks) == 1
        assert result.total_pages == MAX_PAGES

    def test_boundary_31_pages_splits(self, store_engine, blobs) -> None:
        """MAX_PAGES + 1 pages triggers a split: 25 + 6."""
        engine = store_engine
        pdf_data = _create_test_pdf(MAX_PAGES + 1)
        content_id = "boundary-31"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
            blobs=blobs,
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine, blobs
        )

        assert len(result.chunks) == 2
        assert result.total_pages == MAX_PAGES + 1

        assert result.chunks[0].page_start == 1
        assert result.chunks[0].page_end == CHUNK_SIZE_PAGES
        assert result.chunks[0].chunk_index == 0

        assert result.chunks[1].page_start == CHUNK_SIZE_PAGES + 1
        assert result.chunks[1].page_end == MAX_PAGES + 1
        assert result.chunks[1].chunk_index == 1

    def test_large_pdf_60_pages_3_chunks(self, store_engine, blobs) -> None:
        """60-page PDF splits into 3 chunks: 25 + 25 + 10."""
        engine = store_engine
        pdf_data = _create_test_pdf(60)
        content_id = "large-60"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
            blobs=blobs,
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine, blobs
        )

        assert len(result.chunks) == 3
        assert result.total_pages == 60
        assert (result.chunks[0].page_start, result.chunks[0].page_end) == (1, 25)
        assert (result.chunks[1].page_start, result.chunks[1].page_end) == (26, 50)
        assert (result.chunks[2].page_start, result.chunks[2].page_end) == (51, 60)

    def test_original_blob_deleted_after_split(self, store_engine, blobs) -> None:
        """After a multi-chunk split, the original blob is deleted; chunk blobs exist."""
        engine = store_engine
        pdf_data = _create_test_pdf(MAX_PAGES + 1)
        content_id = "delete-test"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
            blobs=blobs,
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine, blobs
        )

        assert get_file_content(engine, content_id, blobs) is None
        for chunk in result.chunks:
            assert get_file_content(engine, chunk.content_id, blobs) is not None

    def test_original_blob_kept_for_single_chunk(self, store_engine, blobs) -> None:
        """The single-chunk case preserves the original blob."""
        engine = store_engine
        pdf_data = _create_test_pdf(5)
        content_id = "keep-test"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
            blobs=blobs,
        )

        execute_split_file_into_chunks(content_id, "application/pdf", len(pdf_data), engine, blobs)

        assert get_file_content(engine, content_id, blobs) is not None

    def test_chunk_pdfs_are_valid(self, store_engine, blobs) -> None:
        """Each chunk blob is a valid PDF with the expected page count."""
        import fitz

        engine = store_engine
        pdf_data = _create_test_pdf(60)
        content_id = "valid-chunks"
        save_file_content(
            engine,
            content_id=content_id,
            data=pdf_data,
            mime_type="application/pdf",
            file_size_bytes=len(pdf_data),
            blobs=blobs,
        )

        result = execute_split_file_into_chunks(
            content_id, "application/pdf", len(pdf_data), engine, blobs
        )

        for chunk, expected in zip(result.chunks, [25, 25, 10], strict=True):
            blob = get_file_content(engine, chunk.content_id, blobs)
            assert blob is not None
            doc = fitz.open(stream=blob["data"], filetype="pdf")
            assert len(doc) == expected
            doc.close()

    def test_missing_content_raises(self, store_engine, blobs) -> None:
        """Raises RuntimeError if the content_id is not found in the store."""
        engine = store_engine
        with pytest.raises(RuntimeError, match="File content not found"):
            execute_split_file_into_chunks("nonexistent", "application/pdf", 1000, engine, blobs)


# ---------------------------------------------------------------------------
# execute_store_ocr_result
# ---------------------------------------------------------------------------


class TestStoreOcrResult:
    def test_inline_body_no_images(self, store_engine, blobs) -> None:
        engine = store_engine
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
            blobs=blobs,
        )
        assert result.text_length > 0
        assert get_ocr_result(engine, "doc-1")["text"] == "hello"
        assert get_ocr_job_status(engine, "r1")["status"] == "stored"

    def test_s3_envelope_with_images(self, store_engine, blobs) -> None:
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
        key = blobs.build_key("batch-result-r2")
        envelope = dump_batch_result_payload(body, images).encode("utf-8")
        blobs.put(key, envelope, "application/json")

        engine = store_engine
        result = execute_store_ocr_result(
            request_id="r2",
            document_id="doc-2",
            file_path="",
            batch_id="b2",
            workflow_id="wf",
            raw_response_json=None,
            s3_key=key,
            engine=engine,
            blobs=blobs,
        )
        assert result.page_count == 1
        stored = get_ocr_result(engine, "doc-2")
        assert "ocr-image://" in stored["text"]
        imgs = get_ocr_images(engine, "doc-2")
        assert len(imgs) == 1
        assert imgs[0]["document_id"] == "doc-2"

    def test_idempotent_on_retry(self, store_engine, blobs) -> None:
        engine = store_engine
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
            blobs=blobs,
        )
        execute_store_ocr_result(**kw)
        execute_store_ocr_result(**kw)  # retry — must not raise or duplicate
        assert get_ocr_result(engine, "doc-3") is not None

    def test_no_body_raises(self, store_engine, blobs) -> None:
        """Neither an inline body nor an s3 envelope is a hard failure."""
        engine = store_engine
        with pytest.raises(RuntimeError, match="neither inline body nor"):
            execute_store_ocr_result(
                request_id="r4",
                document_id="doc-4",
                file_path="",
                batch_id="b4",
                workflow_id="wf",
                raw_response_json=None,
                s3_key=None,
                engine=engine,
                blobs=blobs,
            )

    def test_image_with_data_uri_prefix_is_decoded(self, store_engine, blobs) -> None:
        """image_base64 delivered as a data: URI has its header stripped before decode."""
        engine = store_engine
        body = json.dumps(
            {"model": "m", "pages": [{"markdown": "![x](img-0.jpeg)"}], "usage_info": {}}
        )
        images = [
            {
                "original_image_id": "img-0.jpeg",
                "page_index": 0,
                "image_base64": "data:image/png;base64,ZmFrZQ==",
            }
        ]
        key = blobs.build_key("batch-result-r5")
        envelope = dump_batch_result_payload(body, images).encode("utf-8")
        blobs.put(key, envelope, "application/json")

        execute_store_ocr_result(
            request_id="r5",
            document_id="doc-5",
            file_path="",
            batch_id="b5",
            workflow_id="wf",
            raw_response_json=None,
            s3_key=key,
            engine=engine,
            blobs=blobs,
        )
        imgs = get_ocr_images(engine, "doc-5")
        assert len(imgs) == 1
        assert imgs[0]["mime_type"] == "image/png"

    def test_file_hash_computed_when_file_exists(self, store_engine, blobs, tmp_path) -> None:
        engine = store_engine
        path = tmp_path / "source.pdf"
        data = b"%PDF-1.4 content"
        path.write_bytes(data)
        body = json.dumps({"model": "m", "pages": [{"markdown": "hi"}], "usage_info": {}})

        execute_store_ocr_result(
            request_id="r6",
            document_id="doc-6",
            file_path=str(path),
            batch_id="b6",
            workflow_id="wf",
            raw_response_json=body,
            s3_key=None,
            engine=engine,
            blobs=blobs,
        )
        stored = get_ocr_result(engine, "doc-6")
        assert stored["file_hash"] == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# execute_build_request_blob
# ---------------------------------------------------------------------------


class TestBuildRequestBlob:
    def test_mints_id_and_stashes_blob(self, store_engine, blobs) -> None:
        engine = store_engine
        save_file_content(
            engine,
            content_id="c1",
            data=b"%PDF-1.4",
            mime_type="application/pdf",
            file_size_bytes=8,
            blobs=blobs,
        )
        ref = execute_build_request_blob(
            content_id="c1",
            mime_type="application/pdf",
            model_name="mistral:mistral-ocr-latest",
            engine=engine,
            blobs=blobs,
        )
        assert ref.request_id
        assert ref.model == "mistral-ocr-latest"
        requests = json.loads(blobs.get(ref.s3_key).decode("utf-8"))
        assert requests[0]["custom_id"] == ref.request_id
        assert requests[0]["body"]["include_image_base64"] is True

    def test_missing_content_raises(self, store_engine, blobs) -> None:
        engine = store_engine
        with pytest.raises(RuntimeError, match="File content not found"):
            execute_build_request_blob(
                content_id="nonexistent",
                mime_type="application/pdf",
                model_name="mistral:mistral-ocr-latest",
                engine=engine,
                blobs=blobs,
            )


# ---------------------------------------------------------------------------
# execute_reassemble_ocr_chunks
# ---------------------------------------------------------------------------


class TestExecuteReassembleOcrChunks:
    def test_combines_chunk_results_and_cleans_up(self, store_engine) -> None:
        engine = store_engine
        for doc_id, text, tokens_in, tokens_out in (
            ("chunk-1", "page one", 5, 10),
            ("chunk-2", "page two", 7, 14),
        ):
            save_ocr_result(
                engine,
                document_id=doc_id,
                file_path="",
                text=text,
                model_name="mistral-ocr",
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                batch_id="b-chunk",
                workflow_id="wf-chunk",
            )

        result = execute_reassemble_ocr_chunks(
            document_id="combined-doc",
            chunk_document_ids=["chunk-1", "chunk-2"],
            file_path="",
            total_pages=2,
            engine=engine,
        )

        assert result.document_id == "combined-doc"
        assert result.page_count == 2
        combined = get_ocr_result(engine, "combined-doc")
        assert combined["text"] == "page one\n\npage two"
        assert combined["input_tokens"] == 12
        assert combined["output_tokens"] == 24
        assert combined["model_name"] == "mistral-ocr"
        # Chunk rows are removed after reassembly.
        assert get_ocr_result(engine, "chunk-1") is None
        assert get_ocr_result(engine, "chunk-2") is None

    def test_missing_chunk_raises(self, store_engine) -> None:
        engine = store_engine
        with pytest.raises(RuntimeError, match="OCR result not found"):
            execute_reassemble_ocr_chunks(
                document_id="combined-doc",
                chunk_document_ids=["no-such-chunk"],
                file_path="",
                total_pages=1,
                engine=engine,
            )


# ---------------------------------------------------------------------------
# _get_export_dir
# ---------------------------------------------------------------------------


class TestGetExportDir:
    def test_explicit_output_dir_wins(self) -> None:
        assert _get_export_dir("doc-1", "/explicit/dir") == Path("/explicit/dir")

    def test_falls_back_to_xdg_data_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", "/xdg/data")
        assert _get_export_dir("doc-1", "") == Path("/xdg/data") / "ocr" / "export" / "doc-1"

    def test_falls_back_to_home_when_no_xdg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        expected = Path.home() / ".local" / "share" / "ocr" / "export" / "doc-1"
        assert _get_export_dir("doc-1", "") == expected


# ---------------------------------------------------------------------------
# execute_export_ocr_document
# ---------------------------------------------------------------------------


class TestExecuteExportOcrDocument:
    def test_not_found_document(self, store_engine, blobs, tmp_path) -> None:
        engine = store_engine
        result = execute_export_ocr_document(
            document_id="no-such-doc", output_dir=str(tmp_path), engine=engine, blobs=blobs
        )
        assert result.status == "not_found"
        assert result.image_count == 0
        assert result.export_dir == ""

    def test_exports_text_and_images(self, store_engine, blobs, tmp_path) -> None:
        from ocr.store import ocr_image_id, save_ocr_image

        engine = store_engine
        image_id = ocr_image_id("req-export", "img-0.jpeg", 0)
        save_ocr_image(
            engine,
            image_id=image_id,
            document_id="doc-export",
            page_index=0,
            original_image_id="img-0.jpeg",
            data=b"\xff\xd8\xffimgdata",
            mime_type="image/jpeg",
            file_size_bytes=10,
            blobs=blobs,
        )
        save_ocr_result(
            engine,
            document_id="doc-export",
            file_path="/orig/report.pdf",
            text=f"see ![x](ocr-image://{image_id})",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="wf",
        )

        result = execute_export_ocr_document(
            document_id="doc-export", output_dir=str(tmp_path), engine=engine, blobs=blobs
        )

        assert result.status == "exported"
        assert result.image_count == 1
        md_path = Path(result.markdown_path)
        assert md_path.exists()
        content = md_path.read_text()
        assert f"]({image_id}.jpeg)" in content
        assert (tmp_path / f"{image_id}.jpeg").exists()


# ---------------------------------------------------------------------------
# execute_check_ocr_duplicate
# ---------------------------------------------------------------------------


class TestExecuteCheckOcrDuplicate:
    def test_no_prior_result_is_not_duplicate(self, store_engine, tmp_path) -> None:
        engine = store_engine
        path = tmp_path / "fresh.pdf"
        path.write_bytes(b"unique content")
        result = execute_check_ocr_duplicate(str(path), engine)
        assert result.is_duplicate is False
        assert result.existing_document_id == ""

    def test_matching_hash_is_duplicate(self, store_engine, tmp_path) -> None:
        from ocr.store import save_ocr_result as _save

        engine = store_engine
        data = b"same bytes both times"
        file_hash = hashlib.sha256(data).hexdigest()
        _save(
            engine,
            document_id="doc-existing",
            file_path="/other/path.pdf",
            text="t",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="wf",
            file_hash=file_hash,
        )
        path = tmp_path / "dup.pdf"
        path.write_bytes(data)

        result = execute_check_ocr_duplicate(str(path), engine)
        assert result.is_duplicate is True
        assert result.existing_document_id == "doc-existing"


# ---------------------------------------------------------------------------
# execute_list_ocr_jobs (status join)
# ---------------------------------------------------------------------------


class TestListOcrJobs:
    def test_status_join(self, store_engine) -> None:
        from sax_platform.contracts.batch_jobs import batch_jobs
        from sax_platform.contracts.batch_jobs import metadata as bj_metadata

        engine = store_engine
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

    def test_status_filter_narrows_results(self, store_engine) -> None:
        from sax_platform.contracts.batch_jobs import metadata as bj_metadata

        engine = store_engine
        bj_metadata.create_all(engine)
        upsert_ocr_job_status(
            engine, request_id="r-a", document_id="d-a", file_path="/a.pdf", status="stored"
        )
        upsert_ocr_job_status(
            engine, request_id="r-b", document_id="d-b", file_path="/b.pdf", status="submitted"
        )

        result = execute_list_ocr_jobs(engine, status_filter="stored")

        assert result.total == 1
        assert result.jobs[0].document_id == "d-a"


# ---------------------------------------------------------------------------
# Temporal activity methods — OcrStoreActivities bound-method shells
# ---------------------------------------------------------------------------


class TestActivityWrappers:
    """Exercises the ``@activity.defn`` bound methods directly as plain async calls.

    Per project convention these are called without a Temporal worker. The
    dependencies (store engine + ``S3Blobs``) are injected via the
    ``store_activities`` fixture: an ``OcrStoreActivities`` bound to the test's
    migrated sqlite store and a moto-backed bucket — no real service is reachable.
    """

    async def test_read_and_store_file_content(
        self, store_activities, store_engine, blobs, tmp_path
    ) -> None:
        path = tmp_path / "in.pdf"
        path.write_bytes(b"%PDF-1.4 bytes")
        ref = await store_activities.read_and_store_file_content(str(path))
        assert ref.mime_type == "application/pdf"
        assert get_file_content(store_engine, ref.content_id, blobs) is not None

    async def test_split_file_into_chunks(self, store_activities, store_engine, blobs) -> None:
        save_file_content(
            store_engine,
            content_id="split-me",
            data=b"img",
            mime_type="image/png",
            file_size_bytes=3,
            blobs=blobs,
        )
        input_json = json.dumps(
            {"content_id": "split-me", "mime_type": "image/png", "file_size_bytes": 3}
        )
        result = await store_activities.split_file_into_chunks(input_json)
        assert len(result.chunks) == 1
        assert result.chunks[0].content_id == "split-me"

    async def test_build_ocr_request_blob(self, store_activities, store_engine, blobs) -> None:
        save_file_content(
            store_engine,
            content_id="blob-me",
            data=b"%PDF-1.4",
            mime_type="application/pdf",
            file_size_bytes=8,
            blobs=blobs,
        )
        input_json = json.dumps(
            {
                "content_id": "blob-me",
                "mime_type": "application/pdf",
                "model_name": "mistral:mistral-ocr-latest",
            }
        )
        ref = await store_activities.build_ocr_request_blob(input_json)
        assert ref.model == "mistral-ocr-latest"

    async def test_delete_file_content_blob(self, store_activities, store_engine, blobs) -> None:
        save_file_content(
            store_engine,
            content_id="del-me",
            data=b"x",
            mime_type="image/png",
            file_size_bytes=1,
            blobs=blobs,
        )
        await store_activities.delete_file_content_blob("del-me")
        assert get_file_content(store_engine, "del-me", blobs) is None

    async def test_store_ocr_result(self, store_activities, store_engine) -> None:
        body = json.dumps({"model": "m", "pages": [{"markdown": "hi"}], "usage_info": {}})
        input_json = json.dumps(
            {
                "request_id": "wr-1",
                "document_id": "wdoc-1",
                "file_path": "",
                "batch_id": "wb-1",
                "workflow_id": "wf",
                "raw_response_json": body,
            }
        )
        result = await store_activities.store_ocr_result(input_json)
        assert result.document_id == "wdoc-1"
        assert get_ocr_result(store_engine, "wdoc-1") is not None

    async def test_upsert_ocr_status(self, store_activities, store_engine) -> None:
        input_json = json.dumps(
            {
                "request_id": "wr-2",
                "document_id": "wdoc-2",
                "file_path": "/a.pdf",
                "status": "submitted",
            }
        )
        await store_activities.upsert_ocr_status(input_json)
        assert get_ocr_job_status(store_engine, "wr-2")["status"] == "submitted"

    async def test_reassemble_ocr_chunks(self, store_activities, store_engine) -> None:
        for doc_id in ("wchunk-1", "wchunk-2"):
            save_ocr_result(
                store_engine,
                document_id=doc_id,
                file_path="",
                text="t",
                model_name="m",
                input_tokens=1,
                output_tokens=1,
                batch_id="b",
                workflow_id="wf",
            )
        input_json = json.dumps(
            {
                "document_id": "wcombined",
                "chunk_document_ids": ["wchunk-1", "wchunk-2"],
                "file_path": "",
                "total_pages": 2,
            }
        )
        result = await store_activities.reassemble_ocr_chunks(input_json)
        assert result.document_id == "wcombined"

    async def test_export_ocr_document(self, store_activities, store_engine, tmp_path) -> None:
        save_ocr_result(
            store_engine,
            document_id="wexport",
            file_path="/orig/f.pdf",
            text="hello",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="wf",
        )
        input_json = json.dumps({"document_id": "wexport", "output_dir": str(tmp_path)})
        result = await store_activities.export_ocr_document(input_json)
        assert result.status == "exported"

    async def test_check_ocr_duplicate(self, store_activities, tmp_path) -> None:
        path = tmp_path / "check.pdf"
        path.write_bytes(b"content")
        result = await store_activities.check_ocr_duplicate(str(path))
        assert result.is_duplicate is False

    async def test_mark_and_clear_ocr_removal(self, store_activities, store_engine) -> None:
        save_ocr_result(
            store_engine,
            document_id="wmark",
            file_path="",
            text="t",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="wf",
        )
        marked = await store_activities.mark_ocr_for_removal("wmark")
        assert marked.found is True
        assert get_ocr_result(store_engine, "wmark")["marked_for_removal"] is True

        cleared = await store_activities.clear_ocr_removal_mark("wmark")
        assert cleared.found is True
        assert get_ocr_result(store_engine, "wmark")["marked_for_removal"] is False

    async def test_list_ocr_jobs(self, store_activities, store_engine) -> None:
        from sax_platform.contracts.batch_jobs import metadata as bj_metadata

        bj_metadata.create_all(store_engine)
        upsert_ocr_job_status(
            store_engine,
            request_id="wr-list",
            document_id="wdoc-list",
            file_path="/x.pdf",
            status="stored",
        )
        input_json = json.dumps({"limit": 10, "status_filter": ""})
        result = await store_activities.list_ocr_jobs(input_json)
        assert result.total == 1
