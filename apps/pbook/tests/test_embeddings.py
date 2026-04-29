"""Tests for pbook.embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from pbook.embeddings import cosine_similarity, get_client

# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def _vec(self, values: list[float]) -> bytes:
        return np.array(values, dtype=np.float32).tobytes()

    def test_identical_vectors(self):
        v = self._vec([1.0, 0.0, 0.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = self._vec([1.0, 0.0, 0.0])
        b = self._vec([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = self._vec([1.0, 0.0])
        b = self._vec([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        a = self._vec([1.0, 1.0, 0.0])
        b = self._vec([1.0, 0.0, 0.0])
        sim = cosine_similarity(a, b)
        assert 0.5 < sim < 1.0

    def test_zero_vector_returns_zero(self):
        a = self._vec([0.0, 0.0, 0.0])
        b = self._vec([1.0, 1.0, 1.0])
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors(self):
        z = self._vec([0.0, 0.0])
        assert cosine_similarity(z, z) == 0.0


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------


class TestGetClient:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Reset cached client
        import pbook.embeddings as mod
        monkeypatch.setattr(mod, "_client", None)

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            get_client()


# ---------------------------------------------------------------------------
# get_embedding
# ---------------------------------------------------------------------------


class TestGetEmbedding:
    @pytest.mark.asyncio
    async def test_returns_float32_bytes(self, monkeypatch):
        """get_embedding calls the OpenAI API and returns float32 bytes."""
        from typing import ClassVar

        import pbook.embeddings as mod

        fake_vector = [0.1, 0.2, 0.3]

        class FakeEmbeddingData:
            embedding: ClassVar = fake_vector

        class FakeResponse:
            data: ClassVar = [FakeEmbeddingData()]

        class FakeEmbeddings:
            async def create(self, *, input, model):
                return FakeResponse()

        class FakeClient:
            embeddings = FakeEmbeddings()

        monkeypatch.setattr(mod, "_client", FakeClient())

        result = await mod.get_embedding("test text")

        assert isinstance(result, bytes)
        decoded = np.frombuffer(result, dtype=np.float32)
        np.testing.assert_allclose(decoded, fake_vector, rtol=1e-6)

    @pytest.mark.asyncio
    async def test_strips_newlines(self, monkeypatch):
        """get_embedding replaces newlines with spaces in input text."""
        from typing import ClassVar

        import pbook.embeddings as mod

        captured_input = {}

        class FakeEmbeddingData:
            embedding: ClassVar = [0.0]

        class FakeResponse:
            data: ClassVar = [FakeEmbeddingData()]

        class FakeEmbeddings:
            async def create(self, *, input, model):
                captured_input["text"] = input[0]
                return FakeResponse()

        class FakeClient:
            embeddings = FakeEmbeddings()

        monkeypatch.setattr(mod, "_client", FakeClient())

        await mod.get_embedding("line one\nline two")
        assert captured_input["text"] == "line one line two"
