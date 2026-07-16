"""Pure unit tests for the s3_blobs key/config helpers.

The S3 I/O roundtrip (put/get/delete) is exercised against a mocked backend in
forge's integration suite (tests/test_s3_blobs.py) alongside the store blob
functions; here we cover only the pure, dependency-free helpers.
"""

from __future__ import annotations

import pytest

from forge_contracts import s3_blobs
from forge_contracts.s3_blobs import S3ConfigError


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
