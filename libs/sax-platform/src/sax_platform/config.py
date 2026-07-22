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

Alongside the settings groups this module owns the *environment guard*
(``ForgeEnv``, ``ForgeEnvError``, ``resolve_forge_env``): a pure function that
turns the ``FORGE_ENV`` family of environment variables into a validated
target environment, refusing to fall back to any default so that reaching the
production store is always an explicit act, never the result of an unset
variable. It reads nothing itself — callers pass an explicit mapping (the
shell hands it ``os.environ``).
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, assert_never

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from sax_platform.contracts.constants import TEMPORAL_NAMESPACE

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
    "parse_env_profile",
    "require_namespace_coherence",
    "resolve_env_profile_path",
    "resolve_forge_env",
]


class TemporalSettings(BaseSettings):
    """Temporal frontend connection and TLS/mTLS configuration.

    Mirrors the env vars read by ``forge_contracts.temporal.build_tls_config``
    and ``forge.worker``'s ``FORGE_TEMPORAL_ADDRESS`` lookup.
    """

    model_config = SettingsConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    address: str = Field(default="localhost:7233", validation_alias="FORGE_TEMPORAL_ADDRESS")
    # The Temporal namespace this process connects to. Defaults to the shared
    # ``TEMPORAL_NAMESPACE`` ("default") so production — which sets nothing here —
    # keeps its historical behavior with zero config. A dev/test process declares
    # its own (e.g. ``forge-dev``) to isolate its queues/schedules/workflow ids
    # inside the shared local Temporal server; ``require_namespace_coherence``
    # enforces the pairing against ``FORGE_ENV``.
    namespace: str = Field(default=TEMPORAL_NAMESPACE, validation_alias="FORGE_TEMPORAL_NAMESPACE")
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


def require_namespace_coherence(env: ForgeEnv, namespace: str) -> None:
    """Refuse an env/namespace pairing that crosses the prod/staging line.

    A single Temporal server is shared across environments; a namespace isolates
    everything queue-scoped inside it — task queues, schedules, and workflow ids.
    A process's namespace must therefore match its declared environment, or it
    would poll the wrong environment's work. Pure over its two inputs: the shell
    resolves the env (:func:`resolve_forge_env`) and reads the namespace from
    :class:`TemporalSettings` before calling this, immediately before connecting.

    Rules:

    - :attr:`ForgeEnv.PROD` MUST use the shared ``"default"`` namespace
      (:data:`TEMPORAL_NAMESPACE`) — production *is* the default namespace, so
      any other value means a mis-assembled prod environment.
    - :attr:`ForgeEnv.DEV` / :attr:`ForgeEnv.TEST` MUST NOT use ``"default"``
      (that is production's namespace); a staging process must declare its own.

    Raises:
        ForgeEnvError: when the pairing is incoherent. The message names the fix
            because it surfaces verbatim to an operator at process start.
    """
    match env:
        case ForgeEnv.PROD:
            if namespace != TEMPORAL_NAMESPACE:
                raise ForgeEnvError(
                    f"FORGE_ENV=prod requires the {TEMPORAL_NAMESPACE!r} Temporal "
                    f"namespace, but FORGE_TEMPORAL_NAMESPACE={namespace!r}. Production "
                    "runs in the default namespace — unset FORGE_TEMPORAL_NAMESPACE "
                    "(or set it to 'default') in the prod profile."
                )
        case ForgeEnv.DEV | ForgeEnv.TEST:
            if namespace == TEMPORAL_NAMESPACE:
                raise ForgeEnvError(
                    f"FORGE_ENV={env.value} must not use the {TEMPORAL_NAMESPACE!r} "
                    "Temporal namespace — that is production's, and a dev/test process "
                    "there would poll production's queues and schedules. Set "
                    "FORGE_TEMPORAL_NAMESPACE=forge-dev in the dev profile (create the "
                    "namespace once with "
                    "`temporal operator namespace create --retention 72h -n forge-dev`)."
                )
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
