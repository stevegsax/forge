"""Tests for ocr.settings — the frozen, fail-fast OcrSettings composite."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ocr.settings import OcrSettings


class TestOcrSettings:
    def test_composes_groups_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DB_URL", "sqlite:///settings-test.db")
        monkeypatch.setenv("FORGE_OCR_S3_BUCKET", "settings-bucket")
        monkeypatch.setenv("FORGE_OCR_S3_PREFIX", "pfx/")
        monkeypatch.setenv("MISTRAL_API_KEY", "mk")
        monkeypatch.setenv("FORGE_TEMPORAL_ADDRESS", "temporal-host:7233")

        settings = OcrSettings()

        assert settings.db.url == "sqlite:///settings-test.db"
        assert settings.blob.bucket == "settings-bucket"
        assert settings.blob.prefix == "pfx/"
        assert settings.llm.mistral_api_key == "mk"
        assert settings.temporal.address == "temporal-host:7233"

    def test_missing_db_url_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FORGE_DB_URL", raising=False)
        with pytest.raises(ValidationError):
            OcrSettings()

    def test_optional_groups_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DB_URL", "sqlite:///x.db")
        monkeypatch.delenv("FORGE_OCR_S3_BUCKET", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        # The conftest exports an address for FORGE_ENV=test; this test is about
        # what the group does with nothing set.
        monkeypatch.delenv("FORGE_TEMPORAL_ADDRESS", raising=False)

        settings = OcrSettings()

        assert settings.blob.bucket is None
        assert settings.blob.prefix == ""
        assert settings.llm.mistral_api_key is None
        # None = "no override"; the endpoint comes from resolve_temporal_target.
        assert settings.temporal.address is None

    def test_is_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DB_URL", "sqlite:///x.db")
        settings = OcrSettings()
        with pytest.raises(ValidationError):
            settings.temporal = settings.temporal  # type: ignore[misc]
