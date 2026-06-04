"""Tests for S3-backed OCR blob storage (s3_blobs wrapper + store blob functions).

S3 is mocked by the session-scoped ``_s3_backend`` moto fixture (conftest), and the
``forge_ocr_s3`` autouse fixture points ``FORGE_OCR_S3_BUCKET`` at the mocked bucket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import boto3
import pytest
from botocore.exceptions import ClientError
from forge_contracts import s3_blobs
from forge_contracts.s3_blobs import S3ConfigError

from forge.store import (
    delete_file_content,
    delete_ocr_images_by_document,
    get_file_content,
    get_ocr_image,
    get_store_engine,
    run_migrations,
    save_file_content,
    save_ocr_image,
)

if TYPE_CHECKING:
    from sqlalchemy import Engine


@pytest.fixture
def engine(forge_db_url: str) -> Engine:
    run_migrations(forge_db_url)
    return get_store_engine()


# ---------------------------------------------------------------------------
# s3_blobs wrapper
# ---------------------------------------------------------------------------


class TestS3BlobsWrapper:
    def test_build_key_no_prefix(self) -> None:
        assert s3_blobs.build_key("abc") == "abc"

    def test_build_key_with_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_OCR_S3_PREFIX", "ocr/")
        assert s3_blobs.build_key("abc") == "ocr/abc"

    def test_get_bucket_unset_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FORGE_OCR_S3_BUCKET", raising=False)
        with pytest.raises(S3ConfigError, match="FORGE_OCR_S3_BUCKET"):
            s3_blobs.get_bucket()

    def test_put_get_delete_roundtrip(self) -> None:
        s3_blobs.put("k1", b"hello", "text/plain")
        assert s3_blobs.get("k1") == b"hello"
        s3_blobs.delete("k1")
        with pytest.raises(ClientError):
            s3_blobs.get("k1")


# ---------------------------------------------------------------------------
# file_content_blobs → S3
# ---------------------------------------------------------------------------


class TestFileContentS3:
    def test_save_uploads_to_s3_and_get_returns_bytes(
        self, engine: Engine, forge_ocr_s3: str
    ) -> None:
        save_file_content(
            engine,
            content_id="c1",
            data=b"PDFBYTES",
            mime_type="application/pdf",
            file_size_bytes=8,
        )

        # The object lives in S3 under the built key...
        key = s3_blobs.build_key("c1")
        obj = boto3.client("s3").get_object(Bucket=forge_ocr_s3, Key=key)
        assert obj["Body"].read() == b"PDFBYTES"

        # ...and get_file_content fetches it back under the historical data key.
        blob = get_file_content(engine, "c1")
        assert blob is not None
        assert blob["data"] == b"PDFBYTES"
        assert blob["s3_key"] == key
        assert blob["mime_type"] == "application/pdf"

    def test_prefix_is_applied_end_to_end(
        self, engine: Engine, forge_ocr_s3: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FORGE_OCR_S3_PREFIX", "blobs/")
        save_file_content(
            engine, content_id="c-pref", data=b"X", mime_type="application/pdf", file_size_bytes=1
        )
        boto3.client("s3").get_object(Bucket=forge_ocr_s3, Key="blobs/c-pref")
        assert get_file_content(engine, "c-pref")["data"] == b"X"

    def test_save_raises_when_bucket_unset_and_writes_no_row(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FORGE_OCR_S3_BUCKET", raising=False)
        with pytest.raises(S3ConfigError, match="FORGE_OCR_S3_BUCKET"):
            save_file_content(
                engine, content_id="c2", data=b"x", mime_type="application/pdf", file_size_bytes=1
            )
        # No inline-in-DB fallback: nothing was persisted.
        monkeypatch.setenv("FORGE_OCR_S3_BUCKET", "forge-test-ocr")
        assert get_file_content(engine, "c2") is None

    def test_delete_removes_db_row_and_s3_object(
        self, engine: Engine, forge_ocr_s3: str
    ) -> None:
        save_file_content(
            engine, content_id="c3", data=b"bytes", mime_type="application/pdf", file_size_bytes=5
        )
        key = s3_blobs.build_key("c3")

        delete_file_content(engine, "c3")

        assert get_file_content(engine, "c3") is None
        with pytest.raises(ClientError):
            boto3.client("s3").get_object(Bucket=forge_ocr_s3, Key=key)


# ---------------------------------------------------------------------------
# ocr_images → S3
# ---------------------------------------------------------------------------


class TestOcrImageS3:
    def test_save_and_get_roundtrip(self, engine: Engine) -> None:
        save_ocr_image(
            engine,
            image_id="i1",
            document_id="d1",
            page_index=0,
            original_image_id="orig-1",
            data=b"IMG",
            mime_type="image/png",
            file_size_bytes=3,
        )
        img = get_ocr_image(engine, "i1")
        assert img is not None
        assert img["data"] == b"IMG"
        assert img["s3_key"] == s3_blobs.build_key("i1")

    def test_save_raises_when_bucket_unset(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FORGE_OCR_S3_BUCKET", raising=False)
        with pytest.raises(S3ConfigError, match="FORGE_OCR_S3_BUCKET"):
            save_ocr_image(
                engine,
                image_id="i2",
                page_index=0,
                original_image_id="o",
                data=b"x",
                mime_type="image/png",
                file_size_bytes=1,
            )

    def test_delete_by_document_removes_objects(
        self, engine: Engine, forge_ocr_s3: str
    ) -> None:
        save_ocr_image(
            engine,
            image_id="i3",
            document_id="d3",
            page_index=0,
            original_image_id="o3",
            data=b"IMG3",
            mime_type="image/png",
            file_size_bytes=4,
        )
        key = s3_blobs.build_key("i3")

        delete_ocr_images_by_document(engine, ["d3"])

        assert get_ocr_image(engine, "i3") is None
        with pytest.raises(ClientError):
            boto3.client("s3").get_object(Bucket=forge_ocr_s3, Key=key)
