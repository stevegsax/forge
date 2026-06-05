"""Tests for ocr.store — OCR-owned tables and data access."""

from __future__ import annotations

import sqlalchemy as sa

from ocr.store import (
    find_ocr_result_by_hash,
    get_file_content,
    get_ocr_image,
    get_ocr_job_status,
    get_ocr_result,
    get_store_engine,
    ocr_image_id,
    save_file_content,
    save_ocr_image,
    save_ocr_result,
    upsert_ocr_job_status,
)


class TestOcrResult:
    def test_save_idempotent(self, migrated: str) -> None:
        engine = get_store_engine()
        kw = dict(
            document_id="doc-1",
            file_path="/a.pdf",
            text="hello",
            model_name="m",
            input_tokens=1,
            output_tokens=2,
            batch_id="b",
            workflow_id="wf",
        )
        assert save_ocr_result(engine, **kw) is True
        assert save_ocr_result(engine, **kw) is False
        assert get_ocr_result(engine, "doc-1")["text"] == "hello"

    def test_find_by_hash_excludes_removed(self, migrated: str) -> None:
        engine = get_store_engine()
        save_ocr_result(
            engine,
            document_id="doc-h",
            file_path="/a.pdf",
            text="t",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="wf",
            file_hash="abc123",
        )
        assert find_ocr_result_by_hash(engine, "abc123")["document_id"] == "doc-h"
        assert find_ocr_result_by_hash(engine, "nope") is None


class TestFileContentBlob:
    def test_round_trip_via_s3(self, migrated: str) -> None:
        engine = get_store_engine()
        save_file_content(
            engine, content_id="c1", data=b"bytes", mime_type="application/pdf", file_size_bytes=5
        )
        result = get_file_content(engine, "c1")
        assert result["data"] == b"bytes"
        assert result["mime_type"] == "application/pdf"


class TestOcrImage:
    def test_save_and_fetch_via_s3(self, migrated: str) -> None:
        engine = get_store_engine()
        image_id = ocr_image_id("req-1", "img-0.jpeg", 0)
        save_ocr_image(
            engine,
            image_id=image_id,
            document_id="doc-1",
            page_index=0,
            original_image_id="img-0.jpeg",
            data=b"\xff\xd8\xffimg",
            mime_type="image/jpeg",
            file_size_bytes=6,
        )
        fetched = get_ocr_image(engine, image_id)
        assert fetched["data"] == b"\xff\xd8\xffimg"
        assert fetched["document_id"] == "doc-1"


class TestOcrJobStatus:
    def test_upsert_inserts_then_updates(self, migrated: str) -> None:
        engine = get_store_engine()
        upsert_ocr_job_status(
            engine, request_id="r1", document_id="d1", file_path="/a", status="submitted"
        )
        assert get_ocr_job_status(engine, "r1")["status"] == "submitted"
        upsert_ocr_job_status(
            engine, request_id="r1", document_id="d1", status="stored"
        )
        row = get_ocr_job_status(engine, "r1")
        assert row["status"] == "stored"
        # Single row (upsert, not duplicate insert).
        with engine.connect() as conn:
            from ocr.store import OcrJobStatus

            count = conn.execute(
                sa.select(sa.func.count()).select_from(OcrJobStatus.__table__)
            ).scalar()
        assert count == 1
