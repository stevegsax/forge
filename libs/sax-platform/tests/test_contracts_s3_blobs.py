"""Unit tests for the S3Blobs client.

``put``/``get``/``delete`` do real I/O against boto3's S3 client, which every
test here mocks via ``monkeypatch.setattr("boto3.client", ...)`` — no test in
this module may touch real AWS, and none reads the ambient ``AWS_*`` credential
chain because the client construction itself is replaced before any call.
"""

from __future__ import annotations

from unittest import mock

import pytest

from sax_platform.contracts.s3_blobs import S3Blobs, S3ConfigError


class TestS3BlobsClass:
    """The explicit-config client: bucket + prefix at construction. boto3 is
    mocked throughout — never real AWS, never the ambient env."""

    def _mock_boto_client(self, monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
        fake_client = mock.MagicMock(name="fake_s3_client")
        monkeypatch.setattr("boto3.client", lambda service: fake_client)
        return fake_client

    def test_empty_bucket_raises_at_construction(self) -> None:
        with pytest.raises(S3ConfigError, match="bucket"):
            S3Blobs("")

    def test_build_key_without_prefix(self) -> None:
        assert S3Blobs("forge-test-ocr").build_key("abc") == "abc"

    def test_build_key_with_prefix(self) -> None:
        assert S3Blobs("forge-test-ocr", prefix="ocr/").build_key("abc") == "ocr/abc"

    def test_does_not_read_env_bucket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No FORGE_OCR_S3_BUCKET in the env: the class still works from its
        # constructor argument (unlike the module functions).
        fake_client = self._mock_boto_client(monkeypatch)
        monkeypatch.delenv("FORGE_OCR_S3_BUCKET", raising=False)

        S3Blobs("explicit-bucket").put("some/key", b"hello", "text/plain")

        fake_client.put_object.assert_called_once_with(
            Bucket="explicit-bucket", Key="some/key", Body=b"hello", ContentType="text/plain"
        )

    def test_put_uploads_bytes_with_content_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_client = self._mock_boto_client(monkeypatch)

        S3Blobs("forge-test-ocr").put("some/key", b"hello", "text/plain")

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

        result = S3Blobs("forge-test-ocr").get("some/key")

        assert result == b"payload"
        fake_client.get_object.assert_called_once_with(Bucket="forge-test-ocr", Key="some/key")

    def test_delete_calls_delete_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_client = self._mock_boto_client(monkeypatch)

        S3Blobs("forge-test-ocr").delete("some/key")

        fake_client.delete_object.assert_called_once_with(Bucket="forge-test-ocr", Key="some/key")

    def test_is_frozen(self) -> None:
        blobs = S3Blobs("forge-test-ocr")
        with pytest.raises(AttributeError, match="frozen"):
            blobs._bucket = "other"  # reassigning an existing slot is blocked too
        with pytest.raises(AttributeError):
            blobs.bucket = "other"  # type: ignore[attr-defined]  # new attrs blocked as well
