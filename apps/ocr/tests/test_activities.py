"""Tests for ocr.activities — pure helpers and engine-injected functions."""

from __future__ import annotations

import json

import sqlalchemy as sa
from forge_contracts.models import dump_batch_result_payload

from ocr.activities import (
    _derive_status,
    _strip_image_prefix,
    build_ocr_batch_body,
    execute_build_request_blob,
    execute_list_ocr_jobs,
    execute_store_ocr_result,
    parse_ocr_pages,
    rewrite_image_references,
)
from ocr.models import OcrJobDerivedStatus
from ocr.store import (
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
