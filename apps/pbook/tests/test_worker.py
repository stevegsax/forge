"""Tests for pbook.worker (the composition root)."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pbook.worker as worker_mod


def _fake_stamp(base: str | None = None) -> str:
    """Deterministic stand-in for the git-version identity stamp.

    ``sax_platform.temporal.identity`` owns (and tests) version discovery; here
    only the wiring matters — that run_worker stamps the identity it was handed
    before connecting — so the real ``git`` call is replaced with a fixed suffix.
    """
    return f"{base or 'pid@host'}@testver"


def _base_patches():
    """Patch the I/O boundary of run_worker (logging, migrations, engine,
    Temporal connect, and the platform worker scaffold) so composition can be
    exercised without a DB, a network, or a running worker."""
    return (
        patch.object(worker_mod, "setup_logging", MagicMock()),
        patch.object(worker_mod, "_migrate_if_configured", MagicMock()),
        patch.object(worker_mod, "build_engine", MagicMock(return_value=None)),
        patch.object(worker_mod, "connect_temporal", AsyncMock(return_value=MagicMock())),
        patch.object(worker_mod, "run_platform_worker", AsyncMock()),
    )


class TestMigrateIfConfigured:
    def test_runs_migrations_when_url_set(self) -> None:
        db = MagicMock(url="postgresql://u@h/db")
        with patch.object(worker_mod, "run_migrations") as mock_run:
            worker_mod._migrate_if_configured(db)
        mock_run.assert_called_once_with("postgresql://u@h/db")

    def test_skips_when_url_unset(self) -> None:
        db = MagicMock(url=None)
        with patch.object(worker_mod, "run_migrations") as mock_run:
            worker_mod._migrate_if_configured(db)
        mock_run.assert_not_called()


class TestRunWorkerComposition:
    def test_no_provider_global_seam_remains(self) -> None:
        """The set_provider / _register_llm_provider seam is gone — the worker
        injects the provider via LlmActivities, not a module global."""
        assert not hasattr(worker_mod, "_register_llm_provider")
        assert not hasattr(worker_mod, "set_provider")

    @pytest.mark.asyncio
    async def test_adopts_platform_scaffold_and_registers_activities(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        p_log, p_mig, p_eng, p_conn, p_run = _base_patches()
        with p_log, p_mig, p_eng, p_conn, p_run as run_scaffold:
            await worker_mod.run_worker(address="localhost:7233")

        run_scaffold.assert_awaited_once()
        kwargs = run_scaffold.call_args.kwargs
        assert kwargs["task_queue"] == worker_mod.PBOOK_TASK_QUEUE
        assert kwargs["graceful_shutdown_timeout"] == timedelta(minutes=5)

        names = {getattr(a, "__name__", None) for a in kwargs["activities"]}
        # Generic steps, the two no-dep free functions, and a sample of the
        # engine-bound store + cli-op activities are all registered by name.
        assert {
            "llm_chat",
            "llm_embed",
            "validate_entry",
            "get_session_text_activity",
            "fetch_candidates",
            "add_entry_activity",
        } <= names

    @pytest.mark.asyncio
    async def test_builds_provider_and_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        p_log, p_mig, p_eng, p_conn, p_run = _base_patches()
        with (
            p_log,
            p_mig,
            p_eng,
            p_conn,
            p_run,
            patch.object(worker_mod, "make_client", MagicMock(return_value="CLIENT")) as mk,
            patch.object(worker_mod, "AnthropicLLM", MagicMock()) as allm,
            patch.object(worker_mod, "AsyncOpenAI", MagicMock(return_value="OAI")) as aoai,
            patch.object(worker_mod, "OpenAIEmbeddings", MagicMock()) as oemb,
        ):
            await worker_mod.run_worker()

        # Provider: AnthropicLLM(make_client()) — no hardcoded key.
        mk.assert_called_once_with()
        allm.assert_called_once_with("CLIENT")
        # Embedder: OpenAIEmbeddings(AsyncOpenAI(api_key=<settings.openai_api_key>)).
        aoai.assert_called_once_with(api_key="sk-test")
        oemb.assert_called_once_with("OAI")

    @pytest.mark.asyncio
    async def test_embedder_is_none_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        p_log, p_mig, p_eng, p_conn, p_run = _base_patches()
        with (
            p_log,
            p_mig,
            p_eng,
            p_conn,
            p_run,
            patch.object(worker_mod, "AsyncOpenAI", MagicMock()) as aoai,
            patch.object(worker_mod, "OpenAIEmbeddings", MagicMock()) as oemb,
        ):
            await worker_mod.run_worker()

        # No key ⇒ no client and no embedder constructed.
        aoai.assert_not_called()
        oemb.assert_not_called()

    @pytest.mark.asyncio
    async def test_connects_via_platform_chokepoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        p_log, p_mig, p_eng, p_conn, p_run = _base_patches()
        with (
            p_log,
            p_mig,
            p_eng,
            p_conn as conn,
            p_run,
            patch.object(worker_mod, "stamped_worker_identity", _fake_stamp),
        ):
            await worker_mod.run_worker(address="host:1234")
        conn.assert_awaited_once()
        # address is passed positionally; TLS settings + the resolved namespace
        # threaded via keyword (forge-test in the suite, from the autouse fixture).
        assert conn.call_args.args[0] == "host:1234"
        assert "settings" in conn.call_args.kwargs
        assert conn.call_args.kwargs["namespace"] == "forge-test"
        # The identity carries the launch-time code version; pbook's CLI supplies
        # no base, so the stamp falls back to the SDK-style {pid}@{hostname}.
        assert conn.call_args.kwargs["identity"] == "pid@host@testver"


class TestEnvGuard:
    """The worker resolves FORGE_ENV FIRST and fails fast without it (T0.9 ST-G2)."""

    @pytest.mark.asyncio
    async def test_requires_forge_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset FORGE_ENV raises before any settings/store setup.

        No I/O patches are applied: resolution is the worker's first act, so it
        raises without ever building settings or touching a database.
        """
        from sax_platform.config import ForgeEnvError

        monkeypatch.delenv("FORGE_ENV", raising=False)
        with pytest.raises(ForgeEnvError, match="no default environment"):
            await worker_mod.run_worker()

    @pytest.mark.asyncio
    async def test_logs_resolved_env(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        p_log, p_mig, p_eng, p_conn, p_run = _base_patches()
        with (
            p_log,
            p_mig,
            p_eng,
            p_conn,
            p_run,
            caplog.at_level(logging.INFO, logger="pbook.worker"),
        ):
            await worker_mod.run_worker()

        assert "pbook worker starting: env=test" in caplog.text


class TestNamespaceCoherence:
    """The worker refuses an env/namespace pairing that crosses the prod/staging line."""

    @pytest.mark.asyncio
    async def test_incoherent_namespace_fails_before_store_setup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FORGE_ENV=test + the ``default`` namespace fails fast, before migrations.

        The coherence check runs after settings are built but before ``build_engine``,
        migrations, or the client, so a mis-namespaced worker never touches a DB or
        the Temporal frontend.
        """
        from sax_platform.config import ForgeEnvError

        # env=test comes from the autouse _forge_env fixture; the default namespace
        # is incoherent with it.
        monkeypatch.setenv("FORGE_TEMPORAL_NAMESPACE", "default")
        p_log, p_mig, p_eng, p_conn, p_run = _base_patches()
        with (
            p_log,
            p_mig as mig,
            p_eng,
            p_conn as conn,
            p_run,
            pytest.raises(ForgeEnvError, match="must not use the 'default'"),
        ):
            await worker_mod.run_worker()

        mig.assert_not_called()
        conn.assert_not_awaited()
