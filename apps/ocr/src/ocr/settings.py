"""Frozen per-process settings for the OCR app (T3.6, D89).

``OcrSettings`` is the OCR worker's composition-root config: constructed once,
at ``run_worker`` startup, from the environment, and fail-fast — a missing
``FORGE_DB_URL`` (the one hard requirement) raises at construction rather than
hours later on the first store access.

It composes the reusable ``sax_platform.config`` groups rather than declaring
its own env-reading fields, so the env-var names stay owned in exactly one place
(the platform groups) and every consumer reads them identically:

- ``temporal`` — Temporal frontend address + TLS/mTLS material.
- ``db`` — the shared database URL (``FORGE_DB_URL``; required).
- ``blob`` — the S3 blob bucket + key prefix (optional; ``None`` bucket ⇒ the
  worker builds no ``S3Blobs`` and blob-backed activities are unavailable).
- ``llm`` — provider API keys outside the Anthropic tier registry; OCR uses
  only ``mistral_api_key`` (optional; unset ⇒ the worker builds no
  ``MistralOcr`` capability — Phase 4 makes it required for self-polling).

Each group is itself a frozen ``pydantic-settings`` group that reads its own
flat env vars; composing them via ``default_factory`` means ``OcrSettings()``
triggers each group's env read in turn.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sax_platform.config import BlobSettings, DbSettings, LlmSettings, TemporalSettings

__all__ = ["OcrSettings"]


class OcrSettings(BaseSettings):
    """The OCR worker's frozen, construct-once, fail-fast settings composite."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    temporal: TemporalSettings = Field(default_factory=TemporalSettings)
    db: DbSettings = Field(default_factory=DbSettings)  # type: ignore[arg-type]
    blob: BlobSettings = Field(default_factory=BlobSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
