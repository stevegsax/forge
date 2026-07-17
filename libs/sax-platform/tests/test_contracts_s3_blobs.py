"""Unit tests for the s3_blobs module.

The key/config helpers are pure. ``put``/``get``/``delete`` do real I/O
against boto3's S3 client, which every test here mocks via
``monkeypatch.setattr("boto3.client", ...)`` — no test in this module may
touch real AWS, and none reads the ambient ``AWS_*`` credential chain because
the client construction itself is replaced before any call.
"""

from __future__ import annotations

from unittest import mock

import pytest

from sax_platform.contracts import s3_blobs
from sax_platform.contracts.s3_blobs import S3ConfigError


def test_build_key_no_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_OCR_S3_PREFIX", raising=False)
    assert s3_blobs.build_key("abc") == "abc"


def test_build_key_with_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_OCR_S3_PREFIX", "ocr/")
    assert s3_blobs.build_key("abc") == "ocr/abc"


def test_get_bucket_unset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_OCR_S3_BUCKET", raising=False)
    with pytest.raises(S3ConfigError, match="FORGE_OCR_S3_BUCKET"):
        s3_blobs.get_bucket()


def test_get_bucket_returns_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_OCR_S3_BUCKET", "forge-test-ocr")
    assert s3_blobs.get_bucket() == "forge-test-ocr"


class TestClientOperations:
    """put/get/delete against a mocked boto3 client — never real S3."""

    def _mock_boto_client(self, monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
        fake_client = mock.MagicMock(name="fake_s3_client")
        monkeypatch.setattr("boto3.client", lambda service: fake_client)
        monkeypatch.setenv("FORGE_OCR_S3_BUCKET", "forge-test-ocr")
        return fake_client

    def test_put_uploads_bytes_with_content_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_client = self._mock_boto_client(monkeypatch)

        s3_blobs.put("some/key", b"hello", "text/plain")

        fake_client.put_object.assert_called_once_with(
            Bucket="forge-test-ocr", Key="some/key", Body=b"hello", ContentType="text/plain"
        )

    def test_get_returns_bytes_read_from_response_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = self._mock_boto_client(monkeypatch)
        fake_body = mock.MagicMock()
        fake_body.read.return_value = b"payload"
        fake_client.get_object.return_value = {"Body": fake_body}

        result = s3_blobs.get("some/key")

        assert result == b"payload"
        fake_client.get_object.assert_called_once_with(Bucket="forge-test-ocr", Key="some/key")

    def test_delete_calls_delete_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_client = self._mock_boto_client(monkeypatch)

        s3_blobs.delete("some/key")

        fake_client.delete_object.assert_called_once_with(Bucket="forge-test-ocr", Key="some/key")

    def test_put_raises_before_touching_client_when_bucket_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = mock.MagicMock(name="fake_s3_client")
        monkeypatch.setattr("boto3.client", lambda service: fake_client)
        monkeypatch.delenv("FORGE_OCR_S3_BUCKET", raising=False)

        with pytest.raises(S3ConfigError, match="FORGE_OCR_S3_BUCKET"):
            s3_blobs.put("some/key", b"x", "text/plain")

        fake_client.put_object.assert_not_called()
