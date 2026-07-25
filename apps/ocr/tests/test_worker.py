"""Tests for ocr.worker — the T3.6 composition root.

The ``Worker`` construction plus the signal-handled graceful-drain loop live in
``sax_platform.temporal.worker.run_worker`` (T3.4) — shared and out of scope
here. What remains ocr's to test: the composition order and wiring — settings
read once (fail-fast), migrations, logging configured, the store engine built
ONCE and injected into ``OcrStoreActivities`` alongside the blob client, and the
right task queue / workflows / activities / shutdown timeout forwarded to the
shared runner.
"""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ocr.worker as worker_mod


def _fake_stamp(base: str | None = None) -> str:
    """Deterministic stand-in for the git-version identity stamp.

    ``sax_platform.temporal.identity`` owns (and tests) version discovery; here
    only the wiring matters — that run_worker stamps the identity it was handed
    before connecting — so the real ``git`` call is replaced with a fixed suffix.
    """
    return f"{base or 'pid@host'}@testver"


def _fake_settings(
    *,
    bucket: str = "bkt",
    prefix: str = "pre/",
    mistral_key: str | None = "k",
    namespace: str = "forge-test",
):
    settings = MagicMock(name="OcrSettings")
    settings.db.url = "sqlite:///x.db"
    settings.blob.bucket = bucket
    settings.blob.prefix = prefix
    settings.llm.mistral_api_key = mistral_key
    settings.temporal.address = "settings-temporal:7233"
    settings.temporal.namespace = namespace
    return settings


class TestBuildMistralOcr:
    def test_returns_none_when_key_unset(self) -> None:
        assert worker_mod._build_mistral_ocr(None) is None
        assert worker_mod._build_mistral_ocr("") is None

    def test_constructs_capability_when_key_set(self) -> None:
        mock_client = MagicMock(name="mistral-client")
        mock_capability = MagicMock(name="mistral-ocr-capability")
        mock_make_client = MagicMock(return_value=mock_client)
        mock_mistral_ocr_cls = MagicMock(return_value=mock_capability)

        with (
            patch("sax_platform.ocr.make_mistral_client", mock_make_client),
            patch("sax_platform.ocr.MistralOcr", mock_mistral_ocr_cls),
        ):
            result = worker_mod._build_mistral_ocr("real-key")

        mock_make_client.assert_called_once_with("real-key")
        mock_mistral_ocr_cls.assert_called_once_with(mock_client)
        assert result is mock_capability


class TestRunWorker:
    @pytest.mark.asyncio
    async def test_composition_root_wires_and_forwards(self) -> None:
        settings = _fake_settings()
        engine_sentinel = object()
        blobs_sentinel = object()
        mistral_sentinel = object()
        store_sentinel = MagicMock(name="store-activities")
        activities_sentinel = [MagicMock(name="bound-method")]
        client_sentinel = MagicMock(name="client")

        with (
            patch.object(worker_mod, "OcrSettings", return_value=settings),
            patch.object(worker_mod, "_init_store") as mock_init_store,
            patch.object(worker_mod, "setup_logging") as mock_setup_logging,
            patch.object(worker_mod, "silence_noisy_loggers") as mock_silence,
            patch.object(
                worker_mod, "get_store_engine", return_value=engine_sentinel
            ) as mock_get_engine,
            patch.object(worker_mod, "S3Blobs", return_value=blobs_sentinel) as mock_s3blobs,
            patch.object(worker_mod, "_build_mistral_ocr", return_value=mistral_sentinel),
            patch.object(worker_mod, "OcrStoreActivities", return_value=store_sentinel) as mock_cls,
            patch.object(
                worker_mod, "activity_methods", return_value=activities_sentinel
            ) as mock_activity_methods,
            patch.object(
                worker_mod, "connect_temporal", AsyncMock(return_value=client_sentinel)
            ) as mock_connect,
            patch.object(worker_mod, "stamped_worker_identity", _fake_stamp),
            patch.object(worker_mod, "_ensure_schedule", AsyncMock()) as mock_ensure_schedule,
            patch.object(worker_mod, "_run_worker", AsyncMock()) as mock_run_worker,
        ):
            await worker_mod.run_worker(identity="worker-123")

        # Settings drive migrations, logging, and the store engine.
        mock_init_store.assert_called_once_with("sqlite:///x.db")
        mock_setup_logging.assert_called_once_with("ocr", console=True)
        mock_silence.assert_called_once_with()

        # Engine built ONCE from settings.db.url — the load-bearing fix.
        mock_get_engine.assert_called_once_with("sqlite:///x.db")
        # Blob client bound to settings.blob bucket + prefix.
        mock_s3blobs.assert_called_once_with("bkt", "pre/")
        # Activities constructed with the injected engine + blobs + Mistral capability.
        mock_cls.assert_called_once_with(engine_sentinel, blobs_sentinel, mistral_sentinel)
        mock_activity_methods.assert_called_once_with(store_sentinel)

        # Connect via settings (address falls back to settings.temporal.address);
        # the resolved namespace is threaded so the worker joins the right lane, and
        # the identity carries the launch-time code version.
        mock_connect.assert_awaited_once_with(
            "settings-temporal:7233",
            identity="worker-123@testver",
            namespace="forge-test",
            settings=settings.temporal,
        )

        # The tracker Schedule is installed on the connected client before serving
        # work — the store children depend on its status hints.
        mock_ensure_schedule.assert_awaited_once_with(
            client_sentinel,
            worker_mod._TRACKER_SCHEDULE_ID,
            "OcrBatchTrackerWorkflow",
            worker_mod._TRACKER_INTERVAL,
        )

        # The shared runner received the right task queue / workflows / activities.
        mock_run_worker.assert_awaited_once()
        call = mock_run_worker.await_args
        assert call.args == (client_sentinel,)
        assert call.kwargs["task_queue"] == worker_mod.OCR_TASK_QUEUE
        assert call.kwargs["workflows"] == worker_mod.workflows()
        assert call.kwargs["activities"] is activities_sentinel
        assert call.kwargs["graceful_shutdown_timeout"] == timedelta(minutes=5)

    @pytest.mark.asyncio
    async def test_address_argument_overrides_settings(self) -> None:
        settings = _fake_settings()

        with (
            patch.object(worker_mod, "OcrSettings", return_value=settings),
            patch.object(worker_mod, "_init_store"),
            patch.object(worker_mod, "setup_logging"),
            patch.object(worker_mod, "silence_noisy_loggers"),
            patch.object(worker_mod, "get_store_engine", return_value=object()),
            patch.object(worker_mod, "S3Blobs", return_value=object()),
            patch.object(worker_mod, "_build_mistral_ocr", return_value=object()),
            patch.object(worker_mod, "OcrStoreActivities", return_value=MagicMock()),
            patch.object(worker_mod, "activity_methods", return_value=[]),
            patch.object(
                worker_mod, "connect_temporal", AsyncMock(return_value=MagicMock())
            ) as mock_connect,
            patch.object(worker_mod, "stamped_worker_identity", _fake_stamp),
            patch.object(worker_mod, "_ensure_schedule", AsyncMock()),
            patch.object(worker_mod, "_run_worker", AsyncMock()),
        ):
            await worker_mod.run_worker("override:7233")

        # No caller identity: the stamp falls back to the SDK-style {pid}@{hostname}.
        mock_connect.assert_awaited_once_with(
            "override:7233",
            identity="pid@host@testver",
            namespace="forge-test",
            settings=settings.temporal,
        )

    @pytest.mark.asyncio
    async def test_unset_mistral_key_fails_fast(self) -> None:
        """OCR now polls its own Mistral batches: a missing key fails at startup."""
        settings = _fake_settings(mistral_key=None)

        with (  # noqa: SIM117
            patch.object(worker_mod, "OcrSettings", return_value=settings),
            patch.object(worker_mod, "_init_store"),
            patch.object(worker_mod, "setup_logging"),
            patch.object(worker_mod, "silence_noisy_loggers"),
            patch.object(worker_mod, "get_store_engine", return_value="eng"),
            patch.object(worker_mod, "S3Blobs", return_value=object()),
            patch.object(worker_mod, "connect_temporal", AsyncMock(return_value=MagicMock())),
            patch.object(worker_mod, "_run_worker", AsyncMock()),
        ):
            # _build_mistral_ocr is NOT patched: the real builder returns None for
            # an empty key, and the composition root turns that into the error.
            with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
                await worker_mod.run_worker()

    @pytest.mark.asyncio
    async def test_unset_bucket_fails_fast(self) -> None:
        """OCR requires S3: the composition root builds the blob client
        unconditionally, so an unset bucket raises S3ConfigError at startup
        (from the real ``S3Blobs`` construction guard) rather than deferring the
        error to the first blob-touching activity."""
        from sax_platform.contracts.s3_blobs import S3ConfigError

        settings = _fake_settings(bucket="")

        with (  # noqa: SIM117
            patch.object(worker_mod, "OcrSettings", return_value=settings),
            patch.object(worker_mod, "_init_store"),
            patch.object(worker_mod, "setup_logging"),
            patch.object(worker_mod, "silence_noisy_loggers"),
            patch.object(worker_mod, "get_store_engine", return_value="eng"),
            patch.object(worker_mod, "_build_mistral_ocr", return_value=None),
            patch.object(worker_mod, "connect_temporal", AsyncMock(return_value=MagicMock())),
            patch.object(worker_mod, "_run_worker", AsyncMock()),
        ):
            # S3Blobs is NOT patched here: the real construction guard fires.
            with pytest.raises(S3ConfigError, match="bucket"):
                await worker_mod.run_worker()


class TestEnvGuard:
    """The worker resolves FORGE_ENV FIRST and fails fast without it (T0.9 ST-G2)."""

    @pytest.mark.asyncio
    async def test_requires_forge_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset FORGE_ENV raises before settings/store/logging setup.

        No patches are applied: resolution is the worker's first act, so it raises
        without ever building settings or touching a database.
        """
        from sax_platform.config import ForgeEnvError

        monkeypatch.delenv("FORGE_ENV", raising=False)
        with pytest.raises(ForgeEnvError, match="no default environment"):
            await worker_mod.run_worker()

    @pytest.mark.asyncio
    async def test_logs_resolved_env(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        settings = _fake_settings()
        with (
            patch.object(worker_mod, "OcrSettings", return_value=settings),
            patch.object(worker_mod, "_init_store"),
            patch.object(worker_mod, "setup_logging"),
            patch.object(worker_mod, "silence_noisy_loggers"),
            patch.object(worker_mod, "get_store_engine", return_value=object()),
            patch.object(worker_mod, "S3Blobs", return_value=object()),
            patch.object(worker_mod, "_build_mistral_ocr", return_value=object()),
            patch.object(worker_mod, "OcrStoreActivities", return_value=MagicMock()),
            patch.object(worker_mod, "activity_methods", return_value=[]),
            patch.object(worker_mod, "connect_temporal", AsyncMock(return_value=MagicMock())),
            patch.object(worker_mod, "_ensure_schedule", AsyncMock()),
            patch.object(worker_mod, "_run_worker", AsyncMock()),
            caplog.at_level(logging.INFO, logger="ocr.worker"),
        ):
            await worker_mod.run_worker()

        assert "ocr worker starting: env=test" in caplog.text


class TestNamespaceCoherence:
    """The worker refuses an env/namespace pairing that crosses the prod/staging line."""

    @pytest.mark.asyncio
    async def test_incoherent_namespace_fails_before_store_setup(self) -> None:
        """FORGE_ENV=test + the ``default`` namespace fails fast, before migrations.

        The coherence check runs after settings are built but before ``_init_store``,
        the Mistral capability, or the client, so a mis-namespaced worker never
        touches a database or the Temporal frontend.
        """
        from sax_platform.config import ForgeEnvError

        settings = _fake_settings(namespace="default")
        mock_init_store = MagicMock()
        mock_connect = AsyncMock()
        with (  # noqa: SIM117
            patch.object(worker_mod, "OcrSettings", return_value=settings),
            patch.object(worker_mod, "_init_store", mock_init_store),
            patch.object(worker_mod, "setup_logging"),
            patch.object(worker_mod, "silence_noisy_loggers"),
            patch.object(worker_mod, "connect_temporal", mock_connect),
            patch.object(worker_mod, "_run_worker", AsyncMock()),
        ):
            with pytest.raises(ForgeEnvError, match="must not use the 'default'"):
                await worker_mod.run_worker()

        mock_init_store.assert_not_called()
        mock_connect.assert_not_awaited()
