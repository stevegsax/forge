"""Tests for ocr.worker.

The ``Worker`` construction plus the signal-handled graceful-drain loop
(formerly ``_run_until_shutdown``/``_request_shutdown`` here) now live in
``sax_platform.temporal.worker.run_worker`` (T3.4, ST8) — shared across the
platform and its consumer apps, and out of scope for this module's own
tests. What remains ocr's to test: app-specific setup order (store
migrations, then the Mistral OCR DI seam, before connecting) and that
``run_worker`` forwards the right task queue / workflows / activities /
shutdown timeout to the shared runner.
"""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ocr.worker as worker_mod
from ocr.deps import get_mistral_ocr, reset_mistral_ocr


@pytest.fixture(autouse=True)
def _isolate_mistral_ocr_state() -> None:
    """Ensure a clean ocr.deps registry between tests, regardless of outcome."""
    reset_mistral_ocr()
    yield
    reset_mistral_ocr()


class TestInitMistralOcr:
    def test_constructs_client_and_registers_capability(self) -> None:
        mock_client = MagicMock(name="mistral-client")
        mock_capability = MagicMock(name="mistral-ocr-capability")
        mock_make_client = MagicMock(return_value=mock_client)
        mock_mistral_ocr_cls = MagicMock(return_value=mock_capability)

        with (
            patch("sax_platform.ocr.make_mistral_client", mock_make_client),
            patch("sax_platform.ocr.MistralOcr", mock_mistral_ocr_cls),
        ):
            worker_mod._init_mistral_ocr()

        mock_make_client.assert_called_once_with()
        mock_mistral_ocr_cls.assert_called_once_with(mock_client)
        assert get_mistral_ocr() is mock_capability


class TestRunWorker:
    @pytest.mark.asyncio
    async def test_bootstraps_worker_and_installs_mistral_ocr_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_client = MagicMock()
        mock_init_store = MagicMock()
        mock_init_mistral_ocr = MagicMock()
        mock_run_worker = AsyncMock()

        # A shared parent so call *order* (store, then the Mistral OCR seam,
        # before the worker connects) is observable, not just call counts.
        manager = MagicMock()
        manager.attach_mock(mock_init_store, "init_store")
        manager.attach_mock(mock_init_mistral_ocr, "init_mistral_ocr")

        monkeypatch.setenv("FORGE_TEMPORAL_ADDRESS", "temporal.example:7233")

        with (
            patch.object(worker_mod, "_init_store", mock_init_store),
            patch.object(worker_mod, "_init_mistral_ocr", mock_init_mistral_ocr),
            patch.object(
                worker_mod, "connect_temporal", AsyncMock(return_value=mock_client)
            ) as mock_connect,
            patch.object(worker_mod, "_run_worker", mock_run_worker),
        ):
            await worker_mod.run_worker(identity="worker-123")

        mock_init_store.assert_called_once_with()
        mock_init_mistral_ocr.assert_called_once_with()
        assert [call[0] for call in manager.mock_calls] == ["init_store", "init_mistral_ocr"]

        mock_connect.assert_awaited_once_with("temporal.example:7233", identity="worker-123")

        mock_run_worker.assert_awaited_once()
        call = mock_run_worker.await_args
        assert call.args == (mock_client,)
        assert call.kwargs["task_queue"] == worker_mod.OCR_TASK_QUEUE
        assert call.kwargs["workflows"] == worker_mod.workflows()
        assert call.kwargs["activities"] == worker_mod.activities()
        assert call.kwargs["graceful_shutdown_timeout"] == timedelta(minutes=5)

    @pytest.mark.asyncio
    async def test_uses_default_address_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_client = MagicMock()

        monkeypatch.delenv("FORGE_TEMPORAL_ADDRESS", raising=False)

        with (
            patch.object(worker_mod, "_init_store", MagicMock()),
            patch.object(worker_mod, "_init_mistral_ocr", MagicMock()),
            patch.object(
                worker_mod, "connect_temporal", AsyncMock(return_value=mock_client)
            ) as mock_connect,
            patch.object(worker_mod, "_run_worker", AsyncMock()),
        ):
            await worker_mod.run_worker()

        mock_connect.assert_awaited_once_with(worker_mod.DEFAULT_TEMPORAL_ADDRESS, identity=None)
