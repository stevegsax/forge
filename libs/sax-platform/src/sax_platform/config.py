"""Reusable env-reading settings groups (T3.4, D89).

This module defines the settings *groups* only — frozen ``pydantic-settings``
classes that read the platform's existing environment variables under their
exact, unchanged names. It does not wire anything into forge, pbook, or ocr,
and it does not replace any existing point-of-use ``os.environ`` read (e.g.
``forge_contracts.temporal.build_tls_config``, ``forge_contracts.s3_blobs``,
``forge.logging_config.get_log_dir``). Those call sites keep reading the
environment directly until T3.6 migrates them onto these groups.

Each group is frozen (construct-once, no runtime mutation — a config value
that changed after construction would be a bug, not a feature) and maps its
fields to their env vars via ``validation_alias`` so the Python field name is
free to differ from the historical env var name while the env var name itself
is preserved byte-for-byte — other code and deployment env files
(``~/.config/forge/forge.env``, ``deploy/local-stack/.env``, launchd plists)
depend on those exact names.

``extra="ignore"``: a settings group only declares the env vars it cares
about; the process environment carries many others (``PATH``, unrelated
``FORGE_*`` vars owned by sibling groups, etc.) and construction must not
choke on them. ``populate_by_name=True``: in addition to the env alias, a
group can be constructed directly from its Python field names (useful for
tests and for callers that already have values in hand rather than in the
environment).

Boolean parsing (``TemporalSettings.tls``) uses pydantic's built-in str-to-bool
coercion: ``"1"``/``"true"``/``"yes"``/``"on"`` (case-insensitive) parse to
``True``; ``"0"``/``"false"``/``"no"``/``"off"`` parse to ``False``. This is a
stricter subset of ``forge_contracts.temporal``'s hand-rolled ``_truthy``
(which also treats unset/empty as falsy rather than raising) — reconciling
the two is T3.6's concern, once this module is wired in as the actual
point-of-use.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "BlobSettings",
    "DbSettings",
    "LlmSettings",
    "LogSettings",
    "TemporalSettings",
]


class TemporalSettings(BaseSettings):
    """Temporal frontend connection and TLS/mTLS configuration.

    Mirrors the env vars read by ``forge_contracts.temporal.build_tls_config``
    and ``forge.worker``'s ``FORGE_TEMPORAL_ADDRESS`` lookup.
    """

    model_config = SettingsConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    address: str = Field(default="localhost:7233", validation_alias="FORGE_TEMPORAL_ADDRESS")
    tls: bool = Field(default=False, validation_alias="FORGE_TEMPORAL_TLS")
    tls_server_ca: str | None = Field(default=None, validation_alias="FORGE_TEMPORAL_TLS_SERVER_CA")
    tls_client_cert: str | None = Field(
        default=None, validation_alias="FORGE_TEMPORAL_TLS_CLIENT_CERT"
    )
    tls_client_key: str | None = Field(
        default=None, validation_alias="FORGE_TEMPORAL_TLS_CLIENT_KEY"
    )
    tls_server_name: str | None = Field(
        default=None, validation_alias="FORGE_TEMPORAL_TLS_SERVER_NAME"
    )


class DbSettings(BaseSettings):
    """The shared database URL. Required — there is no usable default."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    url: str = Field(validation_alias="FORGE_DB_URL")


class BlobSettings(BaseSettings):
    """S3 blob storage configuration, mirroring ``forge_contracts.s3_blobs``."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    bucket: str | None = Field(default=None, validation_alias="FORGE_OCR_S3_BUCKET")
    prefix: str = Field(default="", validation_alias="FORGE_OCR_S3_PREFIX")


class LlmSettings(BaseSettings):
    """API keys for the LLM/OCR providers outside the Anthropic tier registry."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    mistral_api_key: str | None = Field(default=None, validation_alias="MISTRAL_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")


class LogSettings(BaseSettings):
    """Log directory resolution, mirroring ``forge.logging_config.get_log_dir``."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    log_dir: str | None = Field(default=None, validation_alias="FORGE_LOG_DIR")
    pbook_log_path: str | None = Field(default=None, validation_alias="PBOOK_LOG_PATH")
    xdg_state_home: str | None = Field(default=None, validation_alias="XDG_STATE_HOME")
