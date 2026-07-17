"""Tests for sax_platform.embeddings.

The OpenAI SDK client is a constructor seam (`OpenAIEmbeddings.__init__(self,
client: AsyncOpenAI)`), not a module-level HTTP client — so these tests mock
the injected `client` object directly with `AsyncMock`/`MagicMock`, the same
pattern `test_ocr.py` uses for `MistralOcr`'s injected Mistral client. There
is no wire-format contract to verify beyond what the SDK itself already
tests; what's under test here is this module's own request shaping, response
parsing, the pure codec, and the `Embedder` protocol conformance.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from sax_platform.embeddings import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    Embedder,
    EmbeddingResult,
    OpenAIEmbeddings,
    cosine_similarity,
    decode_embedding,
    encode_embedding,
)

# ---------------------------------------------------------------------------
# Codec round-trip (pure)
# ---------------------------------------------------------------------------


def test_encode_decode_round_trip() -> None:
    vector = [0.1, -0.2, 0.3, 1.0, -1.0, 0.0]
    encoded = encode_embedding(vector)
    assert isinstance(encoded, str)

    decoded = decode_embedding(encoded)
    assert len(decoded) == len(vector)
    # float32 round-trip is not bit-exact against the float64 input.
    for original, restored in zip(vector, decoded, strict=True):
        assert restored == pytest.approx(original, abs=1e-6)


def test_encode_decode_empty_vector() -> None:
    encoded = encode_embedding([])
    assert decode_embedding(encoded) == []


def test_decode_matches_independent_numpy_encoding() -> None:
    """Cross-check against a hand-rolled encode to pin the exact wire format:
    little-endian float32 bytes, base64-encoded — not just self-consistency
    between encode_embedding and decode_embedding."""
    import base64

    vector = [1.5, -2.25, 3.75]
    raw = np.asarray(vector, dtype=np.float32).tobytes()
    hand_encoded = base64.b64encode(raw).decode("ascii")

    assert encode_embedding(vector) == hand_encoded
    assert decode_embedding(hand_encoded) == pytest.approx(vector, abs=1e-6)


# ---------------------------------------------------------------------------
# cosine_similarity (pure)
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors() -> None:
    vector = [1.0, 2.0, 3.0]
    assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_opposite_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_similarity_zero_vector_short_circuits() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# OpenAIEmbeddings (injected fake client)
# ---------------------------------------------------------------------------


def _fake_openai_client(vector: list[float]) -> MagicMock:
    """Build a fake AsyncOpenAI client whose `.embeddings.create` is an
    AsyncMock returning a canned response shaped like the real SDK's
    `CreateEmbeddingResponse` (only the attributes this module reads)."""
    canned_response = SimpleNamespace(data=[SimpleNamespace(embedding=vector)])
    client = MagicMock()
    client.embeddings.create = AsyncMock(return_value=canned_response)
    return client


async def test_openai_embeddings_embed_returns_embedding_result() -> None:
    vector = [0.1] * DEFAULT_EMBEDDING_DIM
    client = _fake_openai_client(vector)
    embedder = OpenAIEmbeddings(client)

    result = await embedder.embed("hello world")

    assert isinstance(result, EmbeddingResult)
    assert result.vector == vector
    assert result.model == DEFAULT_EMBEDDING_MODEL
    assert result.dimension == DEFAULT_EMBEDDING_DIM


async def test_openai_embeddings_normalizes_newlines() -> None:
    client = _fake_openai_client([1.0, 2.0, 3.0])
    embedder = OpenAIEmbeddings(client)

    await embedder.embed("line one\nline two")

    client.embeddings.create.assert_awaited_once_with(
        input=["line one line two"],
        model=DEFAULT_EMBEDDING_MODEL,
    )


async def test_openai_embeddings_uses_injected_model() -> None:
    client = _fake_openai_client([1.0, 2.0])
    embedder = OpenAIEmbeddings(client, model="text-embedding-3-large")

    result = await embedder.embed("text")

    assert result.model == "text-embedding-3-large"
    client.embeddings.create.assert_awaited_once_with(
        input=["text"],
        model="text-embedding-3-large",
    )


async def test_openai_embeddings_satisfies_embedder_protocol() -> None:
    client = _fake_openai_client([1.0])
    embedder = OpenAIEmbeddings(client)

    assert isinstance(embedder, Embedder)


def test_embedding_result_is_frozen() -> None:
    result = EmbeddingResult(vector=[1.0, 2.0], model=DEFAULT_EMBEDDING_MODEL, dimension=2)
    with pytest.raises(ValidationError):
        result.vector = [3.0, 4.0]  # type: ignore[misc]
