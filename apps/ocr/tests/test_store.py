"""Tests for ocr.store — OCR-owned tables and data access."""

from __future__ import annotations

import sqlalchemy as sa

from ocr.store import (
    clear_ocr_removal_mark,
    delete_ocr_images_by_document,
    delete_ocr_results,
    find_ocr_result_by_file_path,
    find_ocr_result_by_hash,
    get_file_content,
    get_ocr_image,
    get_ocr_images,
    get_ocr_job_status,
    get_ocr_result,
    get_ocr_results_missing_hash,
    mark_ocr_for_removal,
    ocr_image_id,
    reassign_ocr_images_document_id,
    save_file_content,
    save_ocr_image,
    save_ocr_result,
    update_ocr_file_hash,
    update_ocr_images_document_id,
    upsert_ocr_job_status,
)


class TestOcrResult:
    def test_save_idempotent(self, store_engine) -> None:
        engine = store_engine
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

    def test_find_by_hash_excludes_removed(self, store_engine) -> None:
        engine = store_engine
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

    def test_get_missing_returns_none(self, store_engine) -> None:
        engine = store_engine
        assert get_ocr_result(engine, "no-such-doc") is None

    def test_delete_ocr_results(self, store_engine) -> None:
        engine = store_engine
        for doc_id in ("del-1", "del-2", "keep-1"):
            save_ocr_result(
                engine,
                document_id=doc_id,
                file_path="/a.pdf",
                text="t",
                model_name="m",
                input_tokens=0,
                output_tokens=0,
                batch_id="b",
                workflow_id="wf",
            )
        delete_ocr_results(engine, ["del-1", "del-2"])
        assert get_ocr_result(engine, "del-1") is None
        assert get_ocr_result(engine, "del-2") is None
        assert get_ocr_result(engine, "keep-1") is not None

    def test_find_by_file_path_excludes_removed(self, store_engine) -> None:
        engine = store_engine
        save_ocr_result(
            engine,
            document_id="doc-fp",
            file_path="/path/to/file.pdf",
            text="t",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="wf",
        )
        found = find_ocr_result_by_file_path(engine, "/path/to/file.pdf")
        assert found is not None
        assert found["document_id"] == "doc-fp"
        assert find_ocr_result_by_file_path(engine, "/no/such/file.pdf") is None

        mark_ocr_for_removal(engine, "doc-fp")
        assert find_ocr_result_by_file_path(engine, "/path/to/file.pdf") is None

    def test_missing_hash_lists_only_unhashed(self, store_engine) -> None:
        engine = store_engine
        save_ocr_result(
            engine,
            document_id="doc-no-hash",
            file_path="/a.pdf",
            text="t",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="wf",
        )
        save_ocr_result(
            engine,
            document_id="doc-has-hash",
            file_path="/b.pdf",
            text="t",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="wf",
            file_hash="somehash",
        )
        missing = get_ocr_results_missing_hash(engine)
        doc_ids = {row["document_id"] for row in missing}
        assert "doc-no-hash" in doc_ids
        assert "doc-has-hash" not in doc_ids

    def test_update_file_hash(self, store_engine) -> None:
        engine = store_engine
        save_ocr_result(
            engine,
            document_id="doc-update-hash",
            file_path="/a.pdf",
            text="t",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="wf",
        )
        assert update_ocr_file_hash(engine, "doc-update-hash", "newhash") is True
        assert get_ocr_result(engine, "doc-update-hash")["file_hash"] == "newhash"
        assert update_ocr_file_hash(engine, "no-such-doc", "x") is False


class TestRemovalMark:
    def test_mark_and_clear(self, store_engine) -> None:
        engine = store_engine
        save_ocr_result(
            engine,
            document_id="doc-mark",
            file_path="/a.pdf",
            text="t",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="wf",
        )
        assert mark_ocr_for_removal(engine, "doc-mark") is True
        assert get_ocr_result(engine, "doc-mark")["marked_for_removal"] is True
        assert mark_ocr_for_removal(engine, "no-such-doc") is False

        assert clear_ocr_removal_mark(engine, "doc-mark") is True
        assert get_ocr_result(engine, "doc-mark")["marked_for_removal"] is False
        assert clear_ocr_removal_mark(engine, "no-such-doc") is False


class TestFileContentBlob:
    def test_round_trip_via_s3(self, store_engine, blobs) -> None:
        engine = store_engine
        save_file_content(
            engine,
            content_id="c1",
            data=b"bytes",
            mime_type="application/pdf",
            file_size_bytes=5,
            blobs=blobs,
        )
        result = get_file_content(engine, "c1", blobs)
        assert result["data"] == b"bytes"
        assert result["mime_type"] == "application/pdf"


class TestOcrImage:
    def test_save_and_fetch_via_s3(self, store_engine, blobs) -> None:
        engine = store_engine
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
            blobs=blobs,
        )
        fetched = get_ocr_image(engine, image_id, blobs)
        assert fetched["data"] == b"\xff\xd8\xffimg"
        assert fetched["document_id"] == "doc-1"

    def test_get_missing_returns_none(self, store_engine, blobs) -> None:
        engine = store_engine
        assert get_ocr_image(engine, "no-such-image", blobs) is None

    def test_update_images_document_id(self, store_engine, blobs) -> None:
        engine = store_engine
        image_id = ocr_image_id("req-2", "img-0.jpeg", 0)
        save_ocr_image(
            engine,
            image_id=image_id,
            page_index=0,
            original_image_id="img-0.jpeg",
            data=b"\xff\xd8\xffimg",
            mime_type="image/jpeg",
            file_size_bytes=6,
            blobs=blobs,
        )
        update_ocr_images_document_id(engine, [image_id], "doc-assigned")
        assert get_ocr_image(engine, image_id, blobs)["document_id"] == "doc-assigned"
        # Empty id list is a no-op — must not raise.
        update_ocr_images_document_id(engine, [], "doc-noop")

    def test_reassign_images_document_id(self, store_engine, blobs) -> None:
        engine = store_engine
        image_id_1 = ocr_image_id("req-3", "img-0.jpeg", 0)
        image_id_2 = ocr_image_id("req-3", "img-1.jpeg", 1)
        for image_id, original in ((image_id_1, "img-0.jpeg"), (image_id_2, "img-1.jpeg")):
            save_ocr_image(
                engine,
                image_id=image_id,
                document_id="chunk-doc",
                page_index=0,
                original_image_id=original,
                data=b"\xff\xd8\xffimg",
                mime_type="image/jpeg",
                file_size_bytes=6,
                blobs=blobs,
            )
        reassign_ocr_images_document_id(engine, ["chunk-doc"], "combined-doc")
        images = get_ocr_images(engine, "combined-doc")
        assert {img["id"] for img in images} == {image_id_1, image_id_2}
        # Empty old-ids list is a no-op — must not raise.
        reassign_ocr_images_document_id(engine, [], "combined-doc")

    def test_delete_images_by_document(self, store_engine, blobs) -> None:
        engine = store_engine
        image_id = ocr_image_id("req-4", "img-0.jpeg", 0)
        save_ocr_image(
            engine,
            image_id=image_id,
            document_id="doc-to-delete",
            page_index=0,
            original_image_id="img-0.jpeg",
            data=b"\xff\xd8\xffimg",
            mime_type="image/jpeg",
            file_size_bytes=6,
            blobs=blobs,
        )
        delete_ocr_images_by_document(engine, ["doc-to-delete"], blobs)
        assert get_ocr_image(engine, image_id, blobs) is None
        assert get_ocr_images(engine, "doc-to-delete") == []
        # Empty document-ids list is a no-op — must not raise.
        delete_ocr_images_by_document(engine, [], blobs)


class TestOcrJobStatus:
    def test_upsert_inserts_then_updates(self, store_engine) -> None:
        engine = store_engine
        upsert_ocr_job_status(
            engine, request_id="r1", document_id="d1", file_path="/a", status="submitted"
        )
        assert get_ocr_job_status(engine, "r1")["status"] == "submitted"
        upsert_ocr_job_status(engine, request_id="r1", document_id="d1", status="stored")
        row = get_ocr_job_status(engine, "r1")
        assert row["status"] == "stored"
        # Single row (upsert, not duplicate insert).
        with engine.connect() as conn:
            from ocr.store import OcrJobStatus

            count = conn.execute(
                sa.select(sa.func.count()).select_from(OcrJobStatus.__table__)
            ).scalar()
        assert count == 1
