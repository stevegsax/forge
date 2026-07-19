"""Frozen pydantic-settings for pbook (T3.6 composition root).

The single keyed reader of pbook's environment. Constructed once at the
worker/CLI composition root and passed inward; no module below reads
``os.environ`` for these values anymore.

Mirrors the platform's settings-group convention (``sax_platform.config``):
each group is frozen — a config value that changed after construction would
be a bug, not a feature — and maps its fields to their env vars via
``validation_alias`` so the historical env var names are preserved
byte-for-byte while ``populate_by_name=True`` still lets tests (and callers
holding values in hand) construct a group directly from its field names.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["PbookDbSettings", "PbookSettings"]


class PbookDbSettings(BaseSettings):
    """The pbook store's PostgreSQL connection.

    ``url`` unset or empty disables the store entirely (``build_engine``
    returns ``None`` and every store-touching activity no-ops or errors as
    it did under the old ``get_database_url() is None`` path).
    ``pooler`` forces Supabase transaction-pooler mode (prepared statements
    off); host/port autodetection still applies on top of it downstream.
    """

    model_config = SettingsConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    url: str | None = Field(default=None, validation_alias="PBOOK_DATABASE_URL")
    pooler: bool = Field(default=False, validation_alias="PBOOK_DB_POOLER")


class PbookSettings(BaseSettings):
    """Whole-process pbook configuration, built once at worker/CLI start."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    db: PbookDbSettings = Field(default_factory=PbookDbSettings)
    log_path: str | None = Field(default=None, validation_alias="PBOOK_LOG_PATH")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
