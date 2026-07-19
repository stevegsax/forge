"""Tests for the generic llm_embed activity."""

from __future__ import annotations

import base64

import pytest
from sax_platform.embeddings import EmbeddingResult
from temporalio.exceptions import ApplicationError

from pbook.roots import EmbeddingActivities


class _FakeEmbedder:
    """Structural stand-in for ``sax_platform.embeddings.OpenAIEmbeddings``.

    Records each ``embed`` call's text and returns a canned vector (or raises
    a canned error), so the step can be exercised without an OpenAI client.
    """

    def __init__(
        self, *, vector: list[float] | None = None, error: Exception | None = None
    ) -> None:
        self._vector = vector if vector is not None else [0.0]
        self._error = error
        self.calls: list[str] = []

    async def embed(self, text: str) -> EmbeddingResult:
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return EmbeddingResult(vector=self._vector, model="test", dimension=len(self._vector))


class TestLLMEmbed:
    @pytest.mark.asyncio
    async def test_returns_base64_encoded_vector(self):
        import numpy as np

        fake_vector = [1.0, 2.0, 3.0, 4.0]
        acts = EmbeddingActivities(_FakeEmbedder(vector=fake_vector))  # type: ignore[arg-type]
        result = await acts.llm_embed("hello world")
        assert isinstance(result, str)
        decoded = np.frombuffer(base64.b64decode(result), dtype=np.float32)
        np.testing.assert_allclose(decoded, fake_vector, rtol=1e-6)

    @pytest.mark.asyncio
    async def test_passes_text_through(self):
        embedder = _FakeEmbedder(vector=[0.0, 0.0])
        acts = EmbeddingActivities(embedder)  # type: ignore[arg-type]
        await acts.llm_embed("the quick brown fox")
        assert embedder.calls == ["the quick brown fox"]

    @pytest.mark.asyncio
    async def test_missing_embedder_raises_non_retryable(self):
        """A ``None`` embedder (no OPENAI_API_KEY at the composition root)
        surfaces as a clear, non-retryable ApplicationError so the bounded
        policy fails the session instead of hanging it."""
        acts = EmbeddingActivities(None)
        with pytest.raises(ApplicationError) as excinfo:
            await acts.llm_embed("hello")
        assert excinfo.value.non_retryable is True
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_transient_error_propagates_unwrapped(self):
        acts = EmbeddingActivities(
            _FakeEmbedder(error=ConnectionError("connection reset")),  # type: ignore[arg-type]
        )
        with pytest.raises(ConnectionError, match="connection reset"):
            await acts.llm_embed("hello")
