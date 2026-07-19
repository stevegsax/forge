"""Frozen, fail-fast process configuration for Forge (T3.6, D93).

``ForgeSettings`` is the single composition-root config object: constructed
ONCE at worker startup, it reads the environment exactly once (via its member
groups) and is frozen thereafter — a config value that changed after
construction would be a bug, not a feature. It composes the shared platform
settings groups (``sax_platform.config``) plus a forge-local
``TracingSettings`` group.

Fail-fast: because each group is built by ``default_factory`` at
``ForgeSettings()`` construction, an unset required variable raises
immediately. In particular ``DbSettings`` has no usable default for
``FORGE_DB_URL``, so ``ForgeSettings()`` raises if it is unset — the worker
refuses to start rather than discovering the missing URL mid-run.

The settings groups are the *only* keyed environment readers on the worker
side after T3.6; activity classes and the worker main receive their config
through this object rather than reading ``os.environ`` at point of use.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sax_platform.config import (
    BlobSettings,
    DbSettings,
    LlmSettings,
    LogSettings,
    TemporalSettings,
)

__all__ = ["ForgeSettings", "TracingSettings"]


class TracingSettings(BaseSettings):
    """OpenTelemetry exporter selection for forge tracing.

    Forge-local (no platform sibling): the exporter name is read from
    ``FORGE_OTEL_EXPORTER`` and handed to ``forge.tracing.init_tracing``.
    ``None`` means "unset" — tracing falls back to its console default.
    """

    model_config = SettingsConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    exporter: str | None = Field(default=None, validation_alias="FORGE_OTEL_EXPORTER")


def _load_db_settings() -> DbSettings:
    """Build :class:`~sax_platform.config.DbSettings`, reading ``FORGE_DB_URL``.

    ``DbSettings.url`` is required with no default; pydantic-settings populates it
    from the environment at construction. mypy's ``dataclass_transform`` view of
    the synthesized ``__init__`` treats ``url`` as a required argument and cannot
    see the env source, hence the narrow ignore. An unset ``FORGE_DB_URL`` raises
    here — the fail-fast the composition root depends on.
    """
    return DbSettings()  # type: ignore[call-arg]


class ForgeSettings(BaseModel):
    """The composed, frozen worker configuration built once at startup.

    Each field is a settings *group* built by ``default_factory`` when
    ``ForgeSettings()`` is constructed, so all environment reads happen at that
    single point and any missing required variable (notably ``FORGE_DB_URL``
    via :class:`~sax_platform.config.DbSettings`) raises there.
    """

    model_config = ConfigDict(frozen=True)

    temporal: TemporalSettings = Field(default_factory=TemporalSettings)
    db: DbSettings = Field(default_factory=_load_db_settings)
    blob: BlobSettings = Field(default_factory=BlobSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    log: LogSettings = Field(default_factory=LogSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
