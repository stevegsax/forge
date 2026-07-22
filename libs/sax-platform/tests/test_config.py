"""Tests for sax_platform.config — the frozen env-reading settings groups.

These groups are not wired into any app yet (T3.6); this file only exercises
the mechanism: env values land in the right fields, defaults apply when a var
is unset, instances are frozen, and ``DbSettings`` requires ``FORGE_DB_URL``.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sax_platform.config import (
    BlobSettings,
    DbSettings,
    ForgeEnv,
    ForgeEnvError,
    LlmSettings,
    LogSettings,
    TemporalSettings,
    parse_env_profile,
    require_namespace_coherence,
    resolve_env_profile_path,
    resolve_forge_env,
)
from sax_platform.contracts.constants import TEMPORAL_NAMESPACE

_ALL_ENV_VARS = (
    "FORGE_TEMPORAL_ADDRESS",
    "FORGE_TEMPORAL_NAMESPACE",
    "FORGE_TEMPORAL_TLS",
    "FORGE_TEMPORAL_TLS_SERVER_CA",
    "FORGE_TEMPORAL_TLS_CLIENT_CERT",
    "FORGE_TEMPORAL_TLS_CLIENT_KEY",
    "FORGE_TEMPORAL_TLS_SERVER_NAME",
    "FORGE_DB_URL",
    "FORGE_OCR_S3_BUCKET",
    "FORGE_OCR_S3_PREFIX",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "FORGE_LOG_DIR",
    "PBOOK_LOG_PATH",
    "XDG_STATE_HOME",
)


@pytest.fixture(autouse=True)
def _clear_ambient_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from whatever the ambient shell env happens to hold.

    The ambient environment on this machine points at production (see
    CLAUDE.md), so tests must never inherit it — each test sets exactly the
    vars it asserts on.
    """
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestTemporalSettings:
    def test_env_values_read_into_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_TEMPORAL_ADDRESS", "temporal.example.com:7233")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", "true")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_SERVER_CA", "/etc/temporal/ca.pem")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_CLIENT_CERT", "/etc/temporal/cert.pem")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_CLIENT_KEY", "/etc/temporal/key.pem")
        monkeypatch.setenv("FORGE_TEMPORAL_TLS_SERVER_NAME", "temporal.internal")

        settings = TemporalSettings()

        assert settings.address == "temporal.example.com:7233"
        assert settings.tls is True
        assert settings.tls_server_ca == "/etc/temporal/ca.pem"
        assert settings.tls_client_cert == "/etc/temporal/cert.pem"
        assert settings.tls_client_key == "/etc/temporal/key.pem"
        assert settings.tls_server_name == "temporal.internal"

    @pytest.mark.parametrize("value", ["1", "yes", "on", "TRUE"])
    def test_tls_truthy_values_parse_true(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", value)
        assert TemporalSettings().tls is True

    @pytest.mark.parametrize("value", ["0", "no", "off", "false"])
    def test_tls_falsey_values_parse_false(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("FORGE_TEMPORAL_TLS", value)
        assert TemporalSettings().tls is False

    def test_defaults_apply_when_unset(self) -> None:
        settings = TemporalSettings()

        assert settings.address == "localhost:7233"
        assert settings.tls is False
        assert settings.tls_server_ca is None
        assert settings.tls_client_cert is None
        assert settings.tls_client_key is None
        assert settings.tls_server_name is None

    def test_namespace_defaults_to_shared_namespace_when_unset(self) -> None:
        # Prod sets nothing here, so the default must equal the shared namespace —
        # production's behavior stays identical with zero config.
        assert TemporalSettings().namespace == TEMPORAL_NAMESPACE == "default"

    def test_namespace_read_from_env_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_TEMPORAL_NAMESPACE", "forge-dev")
        assert TemporalSettings().namespace == "forge-dev"

    def test_frozen_instance_rejects_mutation(self) -> None:
        settings = TemporalSettings()

        with pytest.raises(ValidationError, match="frozen"):
            settings.address = "mutated:7233"  # type: ignore[misc]


class TestRequireNamespaceCoherence:
    """The env/namespace coherence check is pure over its two inputs.

    Prod must run in the shared ``default`` namespace; dev/test must run in any
    other. This keeps a dev/test process off production's queues and schedules in
    the shared Temporal server (and vice versa) — enforced at every connect.
    """

    @pytest.mark.parametrize(
        ("env", "namespace"),
        [
            # Prod IS the default namespace.
            (ForgeEnv.PROD, "default"),
            (ForgeEnv.PROD, TEMPORAL_NAMESPACE),
            # Dev/test in any non-default namespace are coherent.
            (ForgeEnv.DEV, "forge-dev"),
            (ForgeEnv.TEST, "forge-test"),
            (ForgeEnv.TEST, "anything-but-default"),
        ],
    )
    def test_coherent_pairings_pass(self, env: ForgeEnv, namespace: str) -> None:
        # A coherent pairing returns None and raises nothing.
        assert require_namespace_coherence(env, namespace) is None

    @pytest.mark.parametrize(
        ("env", "namespace", "match"),
        [
            # Prod in any non-default namespace is a mis-assembled prod env.
            (ForgeEnv.PROD, "forge-dev", "requires the 'default'"),
            (ForgeEnv.PROD, "forge-prod", "requires the 'default'"),
            # Dev/test in the default namespace would poll production's work.
            (ForgeEnv.DEV, "default", "must not use the 'default'"),
            (ForgeEnv.TEST, "default", "must not use the 'default'"),
            (ForgeEnv.DEV, TEMPORAL_NAMESPACE, "forge-dev"),
        ],
    )
    def test_incoherent_pairings_raise_with_actionable_message(
        self, env: ForgeEnv, namespace: str, match: str
    ) -> None:
        with pytest.raises(ForgeEnvError, match=match):
            require_namespace_coherence(env, namespace)


class TestDbSettings:
    def test_env_value_read_into_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DB_URL", "postgresql://user@host/db")

        assert DbSettings().url == "postgresql://user@host/db"

    def test_missing_env_var_raises(self) -> None:
        with pytest.raises(ValidationError, match="FORGE_DB_URL"):
            DbSettings()

    def test_frozen_instance_rejects_mutation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DB_URL", "postgresql://user@host/db")
        settings = DbSettings()

        with pytest.raises(ValidationError, match="frozen"):
            settings.url = "sqlite:///mutated.db"  # type: ignore[misc]


class TestBlobSettings:
    def test_env_values_read_into_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_OCR_S3_BUCKET", "forge-ocr-bucket")
        monkeypatch.setenv("FORGE_OCR_S3_PREFIX", "ocr/")

        settings = BlobSettings()

        assert settings.bucket == "forge-ocr-bucket"
        assert settings.prefix == "ocr/"

    def test_defaults_apply_when_unset(self) -> None:
        settings = BlobSettings()

        assert settings.bucket is None
        assert settings.prefix == ""

    def test_frozen_instance_rejects_mutation(self) -> None:
        settings = BlobSettings()

        with pytest.raises(ValidationError, match="frozen"):
            settings.prefix = "mutated/"  # type: ignore[misc]


class TestLlmSettings:
    def test_env_values_read_into_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

        settings = LlmSettings()

        assert settings.mistral_api_key == "mistral-secret"
        assert settings.openai_api_key == "openai-secret"

    def test_defaults_apply_when_unset(self) -> None:
        settings = LlmSettings()

        assert settings.mistral_api_key is None
        assert settings.openai_api_key is None

    def test_frozen_instance_rejects_mutation(self) -> None:
        settings = LlmSettings()

        with pytest.raises(ValidationError, match="frozen"):
            settings.mistral_api_key = "mutated"  # type: ignore[misc]


class TestLogSettings:
    def test_env_values_read_into_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_DIR", "/var/log/forge")
        monkeypatch.setenv("PBOOK_LOG_PATH", "/var/log/pbook.log")
        monkeypatch.setenv("XDG_STATE_HOME", "/home/user/.local/state")

        settings = LogSettings()

        assert settings.log_dir == "/var/log/forge"
        assert settings.pbook_log_path == "/var/log/pbook.log"
        assert settings.xdg_state_home == "/home/user/.local/state"

    def test_defaults_apply_when_unset(self) -> None:
        settings = LogSettings()

        assert settings.log_dir is None
        assert settings.pbook_log_path is None
        assert settings.xdg_state_home is None

    def test_frozen_instance_rejects_mutation(self) -> None:
        settings = LogSettings()

        with pytest.raises(ValidationError, match="frozen"):
            settings.log_dir = "/mutated"  # type: ignore[misc]


class TestResolveForgeEnv:
    """The environment guard is pure over an explicit mapping.

    Every case passes a hand-built ``dict`` — no monkeypatching — so the tests
    exercise the resolution rules directly and never touch the ambient shell
    env (which points at production on this machine).
    """

    @pytest.mark.parametrize(
        ("environ", "expected"),
        [
            # dev/test resolve with a matching tag...
            ({"FORGE_ENV": "dev", "FORGE_ENV_TAG": "dev"}, ForgeEnv.DEV),
            ({"FORGE_ENV": "test", "FORGE_ENV_TAG": "test"}, ForgeEnv.TEST),
            # ...and with the tag absent (hand-exported vars are fine off prod)...
            ({"FORGE_ENV": "dev"}, ForgeEnv.DEV),
            ({"FORGE_ENV": "test"}, ForgeEnv.TEST),
            # ...and an empty tag counts as absent, not as a mismatch.
            ({"FORGE_ENV": "dev", "FORGE_ENV_TAG": ""}, ForgeEnv.DEV),
            # prod resolves only with BOTH the tagged profile and the ack.
            (
                {
                    "FORGE_ENV": "prod",
                    "FORGE_ENV_TAG": "prod",
                    "FORGE_PROD_ACK": "yes",
                },
                ForgeEnv.PROD,
            ),
        ],
    )
    def test_resolves(self, environ: dict[str, str], expected: ForgeEnv) -> None:
        assert resolve_forge_env(environ) == expected

    @pytest.mark.parametrize(
        ("environ", "match"),
        [
            # Rule 1: missing or empty FORGE_ENV — no default is invented.
            ({}, "no default environment"),
            ({"FORGE_ENV": ""}, "no default environment"),
            # Rule 2: unknown value names the valid set.
            ({"FORGE_ENV": "staging"}, "not a valid environment"),
            # Rule 3: tag/env disagreement, in both directions.
            (
                {"FORGE_ENV": "dev", "FORGE_ENV_TAG": "prod"},
                "does not match",
            ),
            (
                {"FORGE_ENV": "test", "FORGE_ENV_TAG": "dev"},
                "does not match",
            ),
            # A prod tag under a dev claim is caught by the mismatch rule
            # before the prod-ack rule (rule 3 precedes rule 4).
            (
                {"FORGE_ENV": "prod", "FORGE_ENV_TAG": "dev", "FORGE_PROD_ACK": "yes"},
                "does not match",
            ),
            # Rule 4: prod without the tagged profile.
            ({"FORGE_ENV": "prod", "FORGE_PROD_ACK": "yes"}, "explicit act"),
            ({"FORGE_ENV": "prod", "FORGE_ENV_TAG": ""}, "explicit act"),
            # Rule 4: prod with the tag but no acknowledgement.
            ({"FORGE_ENV": "prod", "FORGE_ENV_TAG": "prod"}, "explicit act"),
            # Rule 4: the ack must be exactly "yes".
            (
                {
                    "FORGE_ENV": "prod",
                    "FORGE_ENV_TAG": "prod",
                    "FORGE_PROD_ACK": "true",
                },
                "explicit act",
            ),
        ],
    )
    def test_raises(self, environ: dict[str, str], match: str) -> None:
        with pytest.raises(ForgeEnvError, match=match):
            resolve_forge_env(environ)

    @pytest.mark.parametrize("value", ["PROD", "Prod", "DEV", "Test"])
    def test_case_sensitive_rejects_uppercase(self, value: str) -> None:
        """Matching is exact-lowercase only.

        ``FORGE_ENV`` values travel through env files and CLI flags; accepting
        case variants would let a shouted or title-cased typo silently route a
        process to a real environment (worst case, ``"PROD"`` -> production).
        Rejecting anything but the exact ``StrEnum`` value keeps that from ever
        happening — the token must match byte-for-byte.
        """
        with pytest.raises(ForgeEnvError, match="not a valid environment"):
            resolve_forge_env({"FORGE_ENV": value})


class TestParseEnvProfile:
    """``parse_env_profile`` is pure: text + explicit expansion mapping -> dict.

    No case reads ``os.environ`` — every ``${VAR}`` is expanded against a
    hand-built ``expand_from`` mapping, so the parser stays deterministic and
    never touches the ambient shell env.
    """

    def test_plain_key_value(self) -> None:
        assert parse_env_profile("FORGE_DB_URL=sqlite:///x.db", expand_from={}) == {
            "FORGE_DB_URL": "sqlite:///x.db"
        }

    def test_export_prefix_stripped(self) -> None:
        assert parse_env_profile("export FORGE_ENV_TAG=dev", expand_from={}) == {
            "FORGE_ENV_TAG": "dev"
        }

    def test_double_quotes_stripped(self) -> None:
        assert parse_env_profile('FORGE_DB_URL="sqlite:///x.db"', expand_from={}) == {
            "FORGE_DB_URL": "sqlite:///x.db"
        }

    def test_single_quotes_stripped(self) -> None:
        assert parse_env_profile("GREETING='hello world'", expand_from={}) == {
            "GREETING": "hello world"
        }

    def test_only_one_quote_pair_stripped(self) -> None:
        assert parse_env_profile("""A="'quoted'\"""", expand_from={}) == {"A": "'quoted'"}

    def test_braced_var_expanded(self) -> None:
        out = parse_env_profile(
            "FORGE_LOG_DIR=${XDG_STATE_HOME}/forge",
            expand_from={"XDG_STATE_HOME": "/home/u/.local/state"},
        )
        assert out == {"FORGE_LOG_DIR": "/home/u/.local/state/forge"}

    def test_expansion_runs_after_quote_strip(self) -> None:
        out = parse_env_profile('D="${H}/log"', expand_from={"H": "/state"})
        assert out == {"D": "/state/log"}

    def test_unknown_braced_var_left_literal(self) -> None:
        out = parse_env_profile("A=${NOPE}/x", expand_from={})
        assert out == {"A": "${NOPE}/x"}

    def test_bare_dollar_and_unbraced_name_left_literal(self) -> None:
        # A ``$`` or ``$NAME`` (unbraced) is never expanded — a secret value
        # containing ``$`` survives verbatim even when the name is in the map.
        out = parse_env_profile("PW=p$ss$WORD", expand_from={"WORD": "zzz", "ss": "!"})
        assert out == {"PW": "p$ss$WORD"}

    def test_comments_and_blank_lines_skipped(self) -> None:
        text = "# a comment\n\n   \nA=1\n#another\nB=2\n"
        assert parse_env_profile(text, expand_from={}) == {"A": "1", "B": "2"}

    def test_value_containing_equals_splits_on_first(self) -> None:
        out = parse_env_profile("URL=postgresql://u@h/db?a=b&c=d", expand_from={})
        assert out == {"URL": "postgresql://u@h/db?a=b&c=d"}

    def test_export_quotes_and_expansion_combined(self) -> None:
        out = parse_env_profile(
            'export FORGE_LOG_DIR="${H}/forge/log"', expand_from={"H": "/state"}
        )
        assert out == {"FORGE_LOG_DIR": "/state/forge/log"}

    def test_malformed_line_without_equals_raises_naming_lineno(self) -> None:
        text = "A=1\nGARBAGE_NO_EQUALS\nB=2\n"
        with pytest.raises(ForgeEnvError, match="line 2"):
            parse_env_profile(text, expand_from={})

    def test_empty_key_is_malformed(self) -> None:
        with pytest.raises(ForgeEnvError, match="line 1"):
            parse_env_profile("=orphan", expand_from={})


class TestResolveEnvProfilePath:
    """``resolve_env_profile_path`` classifies a name-vs-path value (pure)."""

    def test_bare_name_resolves_under_explicit_xdg(self) -> None:
        assert resolve_env_profile_path("dev", xdg_config_home="/cfg") == Path(
            "/cfg/forge/envs/dev.env"
        )

    def test_absolute_path_used_verbatim(self) -> None:
        assert resolve_env_profile_path("/abs/prod.env", xdg_config_home="/cfg") == Path(
            "/abs/prod.env"
        )

    def test_relative_path_with_separator_used_verbatim(self) -> None:
        assert resolve_env_profile_path("sub/dev.env", xdg_config_home="/cfg") == Path(
            "sub/dev.env"
        )

    def test_dotenv_suffix_without_separator_is_a_path(self) -> None:
        assert resolve_env_profile_path("local.env", xdg_config_home="/cfg") == Path("local.env")

    def test_name_falls_back_to_home_config_when_xdg_none(self) -> None:
        assert resolve_env_profile_path("dev", xdg_config_home=None) == (
            Path.home() / ".config" / "forge" / "envs" / "dev.env"
        )
