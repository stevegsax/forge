"""Tests for pbook.embeddings (the platform re-export shim).

Since T3.6, ``pbook.embeddings`` re-exports the codec, similarity, and the
``OpenAIEmbeddings`` client from ``sax_platform.embeddings`` — the module
holds no state and reads no environment (the old ``get_client``/
``get_embedding`` globals are gone; the client is injected at the worker
composition root). These tests confirm the re-exported surface works.
"""

from __future__ import annotations

import pytest

from pbook.embeddings import cosine_similarity, decode_embedding, encode_embedding

# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def _vec(self, values: list[float]) -> list[float]:
        return list(values)

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
# encode/decode round-trip
# ---------------------------------------------------------------------------


class TestCodec:
    def test_round_trip(self):
        import numpy as np

        vector = [0.1, 0.2, 0.3, -0.4]
        decoded = decode_embedding(encode_embedding(vector))
        np.testing.assert_allclose(decoded, vector, rtol=1e-6)
