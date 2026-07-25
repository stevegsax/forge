"""Tests for forge.worker — the composition root (T3.6)."""

from __future__ import annotations

import contextlib
from datetime import timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import forge.worker as worker_mod

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class TestInitStore:
    def test_runs_migrations_against_given_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        url = f"sqlite:///{tmp_path / 'forge.db'}"
        mock_run_migrations = MagicMock()
        monkeypatch.setattr("forge.store.run_migrations", mock_run_migrations)

        worker_mod._init_store(url)

        mock_run_migrations.assert_called_once_with(url)


class TestForgeSettingsFailFast:
    def test_missing_db_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ForgeSettings() (built first in run_worker) fails fast on an unset
        FORGE_DB_URL — the store-mandatory invariant the old _init_store held."""
        from pydantic import ValidationError

        from forge.settings import ForgeSettings

        monkeypatch.delenv("FORGE_DB_URL", raising=False)
        with pytest.raises(ValidationError):
            ForgeSettings()


# ---------------------------------------------------------------------------
# Composition root: run_worker builds each dependency once and injects it into
# the four activity classes. run_worker's SDK/settings imports are function-local,
# so the patches target the source modules the `from X import Y` lines resolve.
# ---------------------------------------------------------------------------


def _fake_stamp(base: str | None = None) -> str:
    """Deterministic stand-in for the git-version identity stamp.

    ``sax_platform.temporal.identity`` owns (and tests) version discovery; here
    only the wiring matters — that run_worker stamps the identity it was handed
    before connecting — so the real ``git`` call is replaced with a fixed suffix.
    """
    return f"{base or 'pid@host'}@testver"


def _fake_settings(
    *,
    bucket: str | None = "b",
    mistral_key: str | None = "mk",
    namespace: str = "forge-test",
) -> SimpleNamespace:
    return SimpleNamespace(
        db=SimpleNamespace(url="sqlite:///settings.db"),
        tracing=SimpleNamespace(exporter="console"),
        temporal=SimpleNamespace(address="settings-host:7233", namespace=namespace),
        blob=SimpleNamespace(bucket=bucket, prefix="pre/"),
        llm=SimpleNamespace(mistral_api_key=mistral_key),
        log=SimpleNamespace(),
    )


class _Composition:
    """Sentinels for every dependency the composition root builds, plus the
    patch set that injects them, so tests can assert the exact wiring."""

    def __init__(
        self,
        *,
        bucket: str | None = "b",
        mistral_key: str | None = "mk",
        namespace: str = "forge-test",
    ) -> None:
        self.settings = _fake_settings(bucket=bucket, mistral_key=mistral_key, namespace=namespace)
        self.client = MagicMock(name="temporal_client")
        self.sdk_client = MagicMock(name="sdk_client")
        self.llm = MagicMock(name="anthropic_llm")
        self.engine = MagicMock(name="store_engine")
        self.blobs = MagicMock(name="s3_blobs")
        self.mistral = MagicMock(name="mistral_ocr")

        self.init_store = MagicMock(name="_init_store")
        self.run_platform_worker = AsyncMock(name="run_platform_worker")
        self.make_client = MagicMock(return_value=self.sdk_client)
        self.get_store_engine = MagicMock(return_value=self.engine)
        self.s3blobs_cls = MagicMock(return_value=self.blobs)
        self.anthropic_llm_cls = MagicMock(return_value=self.llm)
        self.mistral_cls = MagicMock(return_value=self.mistral)
        self.connect = AsyncMock(return_value=self.client)
        self.init_tracing = MagicMock()
        self.shutdown_tracing = MagicMock()
        self.clean_prod_guard = MagicMock()

    @contextlib.contextmanager
    def apply(self) -> Iterator[None]:
        patches = [
            patch("forge.settings.ForgeSettings", return_value=self.settings),
            patch.object(worker_mod, "_init_store", self.init_store),
            patch.object(worker_mod, "connect_temporal", self.connect),
            patch.object(worker_mod, "stamped_worker_identity", _fake_stamp),
            patch.object(worker_mod, "require_clean_prod_code", self.clean_prod_guard),
            patch.object(worker_mod, "_run_platform_worker", self.run_platform_worker),
            patch("forge.tracing.init_tracing", self.init_tracing),
            patch("forge.tracing.shutdown_tracing", self.shutdown_tracing),
            patch("forge.logging_config.silence_noisy_loggers", MagicMock()),
            patch("sax_platform.llm.make_client", self.make_client),
            patch("sax_platform.llm.AnthropicLLM", self.anthropic_llm_cls),
            patch("sax_platform.db.get_store_engine", self.get_store_engine),
            patch("sax_platform.contracts.s3_blobs.S3Blobs", self.s3blobs_cls),
            patch("sax_platform.ocr.MistralOcr", self.mistral_cls),
            patch("sax_platform.ocr.make_mistral_client", MagicMock(return_value="mistral-sdk")),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield

    def registered(self) -> list:
        return self.run_platform_worker.await_args.kwargs["activities"]

    def bound(self, name: str) -> object:
        for act in self.registered():
            if getattr(act, "__name__", None) == name:
                return act
        raise AssertionError(f"activity {name!r} not registered")


class TestRunWorkerComposition:
    @pytest.mark.asyncio
    async def test_builds_once_and_wires_settings(self) -> None:
        comp = _Composition()
        with comp.apply():
            await worker_mod.run_worker(identity="worker-1")

        comp.init_store.assert_called_once_with(comp.settings.db.url)
        comp.init_tracing.assert_called_once_with(comp.settings.tracing.exporter)
        # The identity reaching Temporal is the caller's, stamped with the
        # launch-time code version (real discovery lives in sax_platform).
        comp.connect.assert_awaited_once_with(
            "settings-host:7233",
            identity="worker-1@testver",
            namespace="forge-test",
            settings=comp.settings.temporal,
        )

        # Exactly one SDK client and one store engine built.
        assert comp.make_client.call_count == 1
        comp.get_store_engine.assert_called_once_with(comp.settings.db.url)
        comp.anthropic_llm_cls.assert_called_once_with(comp.sdk_client)

        assert comp.run_platform_worker.await_args.kwargs["graceful_shutdown_timeout"] == timedelta(
            minutes=5
        )
        comp.shutdown_tracing.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_classes_share_the_one_engine_and_client(self) -> None:
        comp = _Composition()
        with comp.apply():
            await worker_mod.run_worker()

        assert comp.bound("persist_to_store").__self__._engine is comp.engine
        assert comp.bound("assemble_context").__self__._engine is comp.engine
        assert comp.bound("call_llm").__self__._llm is comp.llm

        batch = comp.bound("submit_batch_request").__self__
        assert batch._client is comp.sdk_client
        assert batch._engine is comp.engine
        assert batch._blob_store is comp.blobs

    @pytest.mark.asyncio
    async def test_registers_all_activity_names_unchanged(self) -> None:
        comp = _Composition()
        with comp.apply():
            await worker_mod.run_worker()

        names = {getattr(a, "__name__", None) for a in comp.registered()}
        expected = {
            "assemble_conflict_resolution_context",
            "assemble_exploration_context",
            "assemble_planner_context",
            "assemble_sanity_check_context",
            "commit_changes_activity",
            "create_worktree_activity",
            "detect_file_conflicts_activity",
            "remove_worktree_activity",
            "reset_worktree_activity",
            "validate_output",
            "validate_playbook_entry",
            "write_files",
            "write_output",
            "fetch_extraction_input",
            "save_extraction_results",
            "persist_to_store",
            "fetch_existing_playbooks",
            "fetch_playbook_ids",
            "export_single_playbook",
            "assemble_context",
            "assemble_step_context",
            "assemble_sub_task_context",
            "fulfill_context_requests",
            "call_llm",
            "call_planner",
            "call_exploration_llm",
            "call_sanity_check",
            "call_conflict_resolution",
            "call_extraction_llm",
            "review_manual_playbook",
            "submit_batch_request",
            "parse_llm_response",
            "batch_status",
            "fetch_batch_result",
        }
        assert expected <= names

    @pytest.mark.asyncio
    async def test_explicit_address_overrides_settings(self) -> None:
        comp = _Composition()
        with comp.apply():
            await worker_mod.run_worker(address="override:7233")

        # No caller identity: the stamp falls back to the SDK-style {pid}@{hostname}.
        comp.connect.assert_awaited_once_with(
            "override:7233",
            identity="pid@host@testver",
            namespace="forge-test",
            settings=comp.settings.temporal,
        )

    @pytest.mark.asyncio
    async def test_forge_never_builds_mistral_ocr(self) -> None:
        """Forge submits anthropic only (T4.2 ST3): the worker never constructs a
        MistralOcr client even when MISTRAL_API_KEY is set, and BatchActivities
        carries no ocr attribute."""
        comp = _Composition(mistral_key="mk")
        with comp.apply():
            await worker_mod.run_worker()

        comp.mistral_cls.assert_not_called()
        assert not hasattr(comp.bound("submit_batch_request").__self__, "_mistral_ocr")

    @pytest.mark.asyncio
    async def test_no_bucket_leaves_blob_store_none(self) -> None:
        comp = _Composition(bucket=None)
        with comp.apply():
            await worker_mod.run_worker()

        comp.s3blobs_cls.assert_not_called()
        assert comp.bound("submit_batch_request").__self__._blob_store is None

    @pytest.mark.asyncio
    async def test_shutdown_tracing_runs_when_worker_fails(self) -> None:
        comp = _Composition()
        comp.run_platform_worker = AsyncMock(side_effect=RuntimeError("worker boom"))
        with comp.apply(), pytest.raises(RuntimeError, match="worker boom"):
            await worker_mod.run_worker()

        comp.shutdown_tracing.assert_called_once_with()


class TestEnvGuard:
    """The worker resolves FORGE_ENV FIRST and fails fast without it (T0.9 ST-G2)."""

    @pytest.mark.asyncio
    async def test_requires_forge_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset FORGE_ENV raises before any settings/store/client setup.

        No composition patches are applied: resolution is the worker's first act,
        so it raises without ever building settings or touching a database.
        """
        from sax_platform.config import ForgeEnvError

        monkeypatch.delenv("FORGE_ENV", raising=False)
        with pytest.raises(ForgeEnvError, match="no default environment"):
            await worker_mod.run_worker()

    @pytest.mark.asyncio
    async def test_logs_resolved_env(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        comp = _Composition()
        with comp.apply(), caplog.at_level(logging.INFO, logger="forge.worker"):
            await worker_mod.run_worker()

        assert "forge worker starting: env=test" in caplog.text

    @pytest.mark.asyncio
    async def test_clean_prod_guard_runs_with_the_resolved_env(self) -> None:
        """The D103 guard is called with the resolved env, before store setup.

        It is a no-op off prod (the suite runs as ``test``); what this pins is the
        wiring — prod can only ever start on a committed checkout because every
        worker asks before it does anything else.
        """
        from sax_platform.config import ForgeEnv

        comp = _Composition()
        with comp.apply():
            await worker_mod.run_worker()

        comp.clean_prod_guard.assert_called_once_with(ForgeEnv.TEST)


class TestNamespaceCoherence:
    """The worker refuses an env/namespace pairing that crosses the prod/staging line."""

    @pytest.mark.asyncio
    async def test_incoherent_namespace_fails_before_store_setup(self) -> None:
        """FORGE_ENV=test + the ``default`` namespace fails fast, before migrations.

        The coherence check runs after settings are built but before ``_init_store``
        or the client is constructed, so a mis-namespaced worker never touches a
        database or the Temporal frontend.
        """
        from sax_platform.config import ForgeEnvError

        # env=test comes from the autouse forge_env fixture; the default namespace
        # is incoherent with it.
        comp = _Composition(namespace="default")
        with comp.apply(), pytest.raises(ForgeEnvError, match="must not use the 'default'"):
            await worker_mod.run_worker()

        comp.init_store.assert_not_called()
        comp.connect.assert_not_awaited()
