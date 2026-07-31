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
(``~/.config/forge/envs/<env>.env``, launchd plists) depend on those exact
names.

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

Alongside the settings groups this module owns the *environment guard*
(``ForgeEnv``, ``ForgeEnvError``, ``resolve_forge_env``): a pure function that
turns the ``FORGE_ENV`` family of environment variables into a validated
target environment, refusing to fall back to any default so that reaching the
production store is always an explicit act, never the result of an unset
variable. It reads nothing itself — callers pass an explicit mapping (the
shell hands it ``os.environ``).

It also owns the Temporal *target* derivation (``TemporalTarget``,
``temporal_namespace_for``, ``resolve_temporal_target``): the address and
namespace a process connects to are computed from that validated environment
rather than read from the environment as separate variables, so a mismatched
pair is unconstructible. This replaced ``require_namespace_coherence``, which
validated a pairing an operator assembled by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, assert_never

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from sax_platform.contracts.constants import PRODUCT_SLUG

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "BlobSettings",
    "DbSettings",
    "ForgeEnv",
    "ForgeEnvError",
    "LlmSettings",
    "LogSettings",
    "TemporalSettings",
    "TemporalTarget",
    "parse_env_profile",
    "resolve_env_profile_path",
    "resolve_forge_env",
    "resolve_temporal_target",
    "temporal_namespace_for",
]


class TemporalSettings(BaseSettings):
    """Temporal frontend connection and TLS/mTLS configuration.

    Mirrors the env vars read by ``forge_contracts.temporal.build_tls_config``
    and ``forge.worker``'s ``FORGE_TEMPORAL_ADDRESS`` lookup.
    """

    model_config = SettingsConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    # An *override* for the frontend address, not the address itself. ``None``
    # means "no override": :func:`resolve_temporal_target` then supplies the
    # canonical endpoint for the declared environment. Only ``test`` — whose
    # server is an ephemeral per-job container on an arbitrary port — is
    # required to set it; for dev and prod an override that disagrees with the
    # environment's endpoint is refused rather than honoured.
    address: str | None = Field(default=None, validation_alias="FORGE_TEMPORAL_ADDRESS")
    # There is deliberately no ``namespace`` field, and no
    # ``FORGE_TEMPORAL_NAMESPACE``. The namespace is ``<slug>-<env>``, derived
    # from the declared environment by :func:`resolve_temporal_target` — a value
    # an operator cannot set, and therefore cannot set wrong.
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


class ForgeEnv(StrEnum):
    """The target environment a forge/ocr/pbook process runs against.

    There is deliberately no ``StrEnum`` member for "unset" or "default": a
    process must name its environment, and :func:`resolve_forge_env` refuses to
    invent one. The values are exact lowercase tokens — they travel through env
    files, launchd plists, and CLI ``--env`` flags, and matching is
    case-sensitive so a typo (``"PROD"``) fails loudly rather than routing to
    production.
    """

    PROD = "prod"
    DEV = "dev"
    TEST = "test"


class ForgeEnvError(Exception):
    """The target environment could not be resolved unambiguously.

    The message is written to be complete and actionable because it surfaces
    verbatim to an operator at process start — it names exactly which variable
    is wrong and what to do about it.
    """


# Declaration order (prod, dev, test) is the order shown in error messages.
_FORGE_ENV_VALUES: Final = tuple(member.value for member in ForgeEnv)


def resolve_forge_env(environ: Mapping[str, str]) -> ForgeEnv:
    """Resolve the target environment from an explicit environment mapping.

    Pure over ``environ`` — it performs no ``os.environ`` read of its own; the
    shell passes the process environment in. Rules are checked in order:

    1. ``FORGE_ENV`` missing or empty raises — there is no default environment.
    2. ``FORGE_ENV`` not one of prod/dev/test raises, naming the valid set.
    3. ``FORGE_ENV_TAG`` present and unequal to ``FORGE_ENV`` raises — the
       sourced env-profile file disagrees with the declared environment.
    4. ``FORGE_ENV == "prod"`` additionally requires ``FORGE_ENV_TAG == "prod"``
       (prod may only come from the tagged profile, never hand-assembled) and
       ``FORGE_PROD_ACK == "yes"`` — production access is an explicit act.
    5. Otherwise the matching :class:`ForgeEnv` member is returned. An absent
       ``FORGE_ENV_TAG`` is allowed off prod: hand-exported vars are fine for
       dev/test.
    """
    declared = environ.get("FORGE_ENV", "")
    if not declared:
        raise ForgeEnvError(
            "FORGE_ENV is unset: there is no default environment. "
            "Set FORGE_ENV to one of prod/dev/test. Workers receive it from "
            "their launchd plist; an interactive shell must export it (or pass "
            "the CLI's --env flag)."
        )
    if declared not in _FORGE_ENV_VALUES:
        raise ForgeEnvError(
            f"FORGE_ENV={declared!r} is not a valid environment. "
            "Valid values are prod, dev, test (exact lowercase)."
        )
    env = ForgeEnv(declared)

    tag = environ.get("FORGE_ENV_TAG", "")
    if tag and tag != declared:
        raise ForgeEnvError(
            f"FORGE_ENV_TAG={tag!r} does not match FORGE_ENV={declared!r}. "
            "The sourced env-profile file declares a different environment than "
            "FORGE_ENV claims — you likely sourced one profile then overrode "
            f"FORGE_ENV. Source the {declared} profile, or correct FORGE_ENV."
        )

    if env is ForgeEnv.PROD:
        ack = environ.get("FORGE_PROD_ACK", "")
        if tag != "prod" or ack != "yes":
            raise ForgeEnvError(
                "Targeting production is an explicit act. To run against prod "
                "you must BOTH source the tagged prod profile (which sets "
                "FORGE_ENV_TAG=prod) AND set FORGE_PROD_ACK=yes. Got "
                f"FORGE_ENV_TAG={tag or '<unset>'!r}, "
                f"FORGE_PROD_ACK={ack or '<unset>'!r}."
            )

    return env


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalTarget:
    """Where a process connects: one server address, one namespace.

    Both fields are derived together from the declared environment, so they can
    never disagree — the pairing that the old coherence check existed to police
    is no longer constructible.
    """

    address: str
    namespace: str


# The canonical frontend per environment. ``test`` is absent on purpose: its
# server is an ephemeral per-job container on an arbitrary port, so there is no
# canonical endpoint to name and the caller must supply one.
#
# These are org endpoints owned by sax-temporal (docs/namespaces.md). Dev is on
# the interim :7236 until forge's legacy server retires and frees :7233;
# changing that is a one-line edit here rather than an edit to every deployed
# profile.
_TEMPORAL_ADDRESSES: Final[Mapping[ForgeEnv, str]] = MappingProxyType(
    {
        ForgeEnv.PROD: "127.0.0.1:7243",
        ForgeEnv.DEV: "127.0.0.1:7236",
    }
)


def temporal_namespace_for(env: ForgeEnv, *, slug: str = PRODUCT_SLUG) -> str:
    """Derive the Temporal namespace for an environment: ``<slug>-<env>``.

    The org convention (sax-temporal/docs/namespaces.md). The bare slug and
    ``"default"`` are namespaces on no server, so a name that loses its suffix
    fails with "namespace not found" everywhere instead of reaching production.
    """
    return f"{slug}-{env.value}"


def resolve_temporal_target(
    env: ForgeEnv,
    *,
    address_override: str | None = None,
    slug: str = PRODUCT_SLUG,
) -> TemporalTarget:
    """Derive the address and namespace a process must connect to.

    Replaces the old ``require_namespace_coherence``. That function validated a
    pairing an operator had assembled by hand; this one constructs the pairing,
    so the incoherent combinations it used to reject can no longer be expressed.
    Pure over its inputs — the shell resolves the environment
    (:func:`resolve_forge_env`) and reads any override from
    :class:`TemporalSettings` before calling this, immediately before connecting.

    Environments are separated by *server*, and the per-environment namespace
    name is the backstop: point a dev-configured process at the prod server and
    ``forge-dev`` does not exist there, so it fails loudly instead of quietly
    polling production's queues.

    Args:
        env: the declared target environment.
        address_override: ``FORGE_TEMPORAL_ADDRESS``, or ``None`` when unset.
            Required for ``test``; for ``dev``/``prod`` it may only restate the
            environment's canonical endpoint.
        slug: the product slug owning the namespace. Defaults to
            :data:`~sax_platform.contracts.constants.PRODUCT_SLUG`; pbook passes
            its own once T6.4 removes forge's cross-queue dispatch.

    Raises:
        ForgeEnvError: when ``test`` supplies no address, or when ``dev``/``prod``
            supply one that is not that environment's server. The message names
            the fix because it surfaces verbatim to an operator at process start.
    """
    namespace = temporal_namespace_for(env, slug=slug)
    match env:
        case ForgeEnv.PROD | ForgeEnv.DEV:
            canonical = _TEMPORAL_ADDRESSES[env]
            if address_override is not None and address_override != canonical:
                raise ForgeEnvError(
                    f"FORGE_ENV={env.value} connects to {canonical} (the {env.value} "
                    f"Temporal server), but FORGE_TEMPORAL_ADDRESS={address_override!r}. "
                    "The address is derived from the environment, not configured — "
                    "unset FORGE_TEMPORAL_ADDRESS in the profile. If the server itself "
                    "moved, change it once in sax_platform.config._TEMPORAL_ADDRESSES "
                    "rather than per profile."
                )
            return TemporalTarget(address=canonical, namespace=namespace)
        case ForgeEnv.TEST:
            if address_override is None:
                raise ForgeEnvError(
                    "FORGE_ENV=test requires FORGE_TEMPORAL_ADDRESS: the test server is "
                    "an ephemeral per-job container on an arbitrary port, so there is no "
                    "canonical address to fall back to. Point it at the container the "
                    "suite started (the temporal_env fixture supplies this)."
                )
            return TemporalTarget(address=address_override, namespace=namespace)
        case _ as unreachable:  # pragma: no cover - exhaustiveness guard
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Env-profile parsing (pure) — the CLI ``--env`` flag's functional core
# ---------------------------------------------------------------------------

# Matches ``${NAME}`` (braced form only). A bare ``$`` or ``$NAME`` is left
# untouched so a secret value containing ``$`` is never corrupted; only the
# braced, name-shaped reference is a candidate for expansion. The name pattern
# is the POSIX env-var shape, so ``${1BAD}`` or ``${}`` never matches and is
# left literal wholesale.
_ENV_REF: Final = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_refs(value: str, expand_from: Mapping[str, str]) -> str:
    """Expand ``${NAME}`` references against *expand_from* (pure, braced-only).

    An unknown ``${NAME}`` (name absent from *expand_from*) is left literal, as
    is any unbraced ``$`` or ``$NAME``. No shell evaluation ever occurs.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in expand_from:
            return expand_from[name]
        return match.group(0)  # unknown ${NAME}: leave the reference literal

    return _ENV_REF.sub(_replace, value)


def parse_env_profile(text: str, *, expand_from: Mapping[str, str]) -> dict[str, str]:
    """Parse an env-profile file's *text* into a ``KEY -> VALUE`` mapping (pure).

    Tolerates the two authoring styles in use: strict ``KEY=VALUE`` lines and
    shell-style ``export KEY="value"`` lines. It performs no shell evaluation
    and never executes anything — only these mechanical steps, per line:

    1. Blank lines and ``#`` comment lines are skipped.
    2. One optional leading ``export `` is stripped.
    3. The line is split on its first ``=`` (values may contain ``=``).
    4. One matching pair of surrounding single or double quotes is stripped
       from the value.
    5. ``${NAME}`` references are expanded against *expand_from* (braced form
       only; unknown or unbraced references are left literal — see
       :func:`_expand_refs`).

    A non-comment line with no ``=`` (or an empty key) is malformed and raises
    :class:`ForgeEnvError` naming the 1-based line number — the shell surfaces
    that message and exits. Expansion is against an explicit mapping (the shell
    passes ``os.environ``) so the function stays pure and testable.
    """
    result: dict[str, str] = {}
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[len("export") :].lstrip()

        key, sep, value = line.partition("=")
        if not sep:
            raise ForgeEnvError(
                f"Malformed env-profile line {lineno}: {raw_line!r} has no '='. "
                "Each non-comment line must be KEY=VALUE (an optional leading "
                "'export ' is allowed)."
            )
        key = key.strip()
        if not key:
            raise ForgeEnvError(
                f"Malformed env-profile line {lineno}: {raw_line!r} has an empty key."
            )

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        result[key] = _expand_refs(value, expand_from)
    return result


def resolve_env_profile_path(value: str, *, xdg_config_home: str | None) -> Path:
    """Resolve a ``--env`` value to a profile file path (pure).

    A *value* containing a path separator (``/``) or ending in ``.env`` is taken
    verbatim as a filesystem path. Otherwise it is a profile *name* resolved to
    ``<xdg_config_home or ~/.config>/forge/envs/<value>.env`` — the XDG
    convention the deploy tree and workers use. ``xdg_config_home`` is passed in
    explicitly (the shell reads ``XDG_CONFIG_HOME``); the ``~/.config`` fallback
    is used only when it is ``None``.
    """
    if "/" in value or value.endswith(".env"):
        return Path(value)
    base = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return base / "forge" / "envs" / f"{value}.env"
