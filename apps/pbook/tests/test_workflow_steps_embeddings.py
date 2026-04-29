"""Tests for the generic llm_embed activity."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock

import pytest

from pbook.workflow_steps import llm_embed


class TestLLMEmbed:
    @pytest.mark.asyncio
    async def test_returns_base64_encoded_bytes(self, monkeypatch):
        fake_bytes = b"\x01\x02\x03\x04"
        monkeypatch.setattr(
            "pbook.workflow_steps.embeddings.get_embedding",
            AsyncMock(return_value=fake_bytes),
        )
        result = await llm_embed("hello world")
        assert isinstance(result, str)
        assert base64.b64decode(result) == fake_bytes

    @pytest.mark.asyncio
    async def test_passes_text_through(self, monkeypatch):
        captured: dict[str, str] = {}

        async def _capture(text: str) -> bytes:
            captured["text"] = text
            return b"abcd"

        monkeypatch.setattr(
            "pbook.workflow_steps.embeddings.get_embedding", _capture,
        )
        await llm_embed("the quick brown fox")
        assert captured["text"] == "the quick brown fox"
