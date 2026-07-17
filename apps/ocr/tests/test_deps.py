"""Tests for ocr.deps — the Mistral OCR capability DI seam (T3.3)."""

from unittest.mock import MagicMock

import pytest

from ocr.deps import get_mistral_ocr, reset_mistral_ocr, set_mistral_ocr


@pytest.fixture(autouse=True)
def _isolate_state() -> None:
    """Ensure a clean registry between tests, regardless of test outcome."""
    reset_mistral_ocr()
    yield
    reset_mistral_ocr()


class TestMistralOcrRegistry:
    def test_roundtrips_registered_instance(self) -> None:
        instance = MagicMock(name="MistralOcr instance")

        set_mistral_ocr(instance)

        assert get_mistral_ocr() is instance

    def test_get_raises_runtime_error_when_unset(self) -> None:
        with pytest.raises(RuntimeError, match="No Mistral OCR capability registered"):
            get_mistral_ocr()

    def test_reset_clears_registered_instance(self) -> None:
        set_mistral_ocr(MagicMock(name="MistralOcr instance"))

        reset_mistral_ocr()

        with pytest.raises(RuntimeError, match="No Mistral OCR capability registered"):
            get_mistral_ocr()
