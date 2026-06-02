"""Tests for the generic llm_embed activity."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock

import pytest

from pbook.workflow_steps import llm_embed


class TestLLMEmbed:
    @pytest.mark.asyncio
    async def test_returns_base64_encoded_vector(self, monkeypatch):
        import numpy as np

        fake_vector = [1.0, 2.0, 3.0, 4.0]
        monkeypatch.setattr(
            "pbook.workflow_steps.embeddings.get_embedding",
            AsyncMock(return_value=fake_vector),
        )
        result = await llm_embed("hello world")
        assert isinstance(result, str)
        decoded = np.frombuffer(base64.b64decode(result), dtype=np.float32)
        np.testing.assert_allclose(decoded, fake_vector, rtol=1e-6)

    @pytest.mark.asyncio
    async def test_passes_text_through(self, monkeypatch):
        captured: dict[str, str] = {}

        async def _capture(text: str) -> list[float]:
            captured["text"] = text
            return [0.0, 0.0]

        monkeypatch.setattr(
            "pbook.workflow_steps.embeddings.get_embedding",
            _capture,
        )
        await llm_embed("the quick brown fox")
        assert captured["text"] == "the quick brown fox"
