"""Embedding capability: an injected-client OpenAI embedder plus the
base64/float32 codec used at the Temporal payload boundary (T3.4).

Ported from `pbook.embeddings` (source of truth for behavior). The module-
level `get_client()`/global-singleton pattern and the free function
`get_embedding()` are gone — this module follows the T3.6 composition-root
convention already established by `sax_platform.ocr.MistralOcr`: the SDK
client (`openai.AsyncOpenAI`) is a constructor argument, never built or
looked up from the environment inside this module. Callers own the client's
lifecycle and are responsible for reading `OPENAI_API_KEY` (or whatever
credential source applies) at their own composition root.

`encode_embedding`/`decode_embedding`/`cosine_similarity` are ported
verbatim in behavior (byte-for-byte identical float32 codec, identical
zero-norm short circuit in `cosine_similarity`) — only the module they live
in has changed.

`OpenAIEmbeddings` never instantiates `openai.AsyncOpenAI` itself — the
client only appears in a type annotation — so with `from __future__ import
annotations` the SDK import lives under `TYPE_CHECKING` and this module
carries no runtime dependency on `openai` at import time (unlike
`sax_platform/ocr.py`, which does instantiate the Mistral SDK client and so
imports it eagerly). Either way, this module is a standalone sibling, not
re-exported through `sax_platform/__init__.py` — a consumer imports it
explicitly (`from sax_platform.embeddings import OpenAIEmbeddings`).
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai import AsyncOpenAI

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_EMBEDDING_MODEL",
    "Embedder",
    "EmbeddingResult",
    "OpenAIEmbeddings",
    "cosine_similarity",
    "decode_embedding",
    "encode_embedding",
]

# Default model for embeddings, and its output dimension — matches
# pbook.store.EMBEDDING_DIM (a pgvector `vector(1536)` column sized for
# text-embedding-3-small).
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class EmbeddingResult(BaseModel):
    """A vector embedding plus the model and dimension that produced it."""

    model_config = ConfigDict(frozen=True)

    vector: list[float]
    model: str
    dimension: int


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding a string of text into an `EmbeddingResult`."""

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed `text` and return the resulting vector plus provenance."""
        ...


# ---------------------------------------------------------------------------
# OpenAIEmbeddings — the network-touching shell
# ---------------------------------------------------------------------------


class OpenAIEmbeddings:
    """`Embedder` backed by the OpenAI embeddings API.

    The `AsyncOpenAI` client is a constructor argument — this class holds no
    other state and reads no environment variables itself (dependency
    injection, per T3.6's composition-root convention: see
    `sax_platform.ocr.MistralOcr` for the same shape). Callers own the
    client's lifecycle.
    """

    def __init__(self, client: AsyncOpenAI, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._client = client
        self._model = model

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate a vector embedding for `text` via the injected client.

        Ported from `pbook.embeddings.get_embedding`: the same
        newline-to-space normalization (the OpenAI API recommends collapsing
        newlines before embedding) and the same `input=[...]` single-item
        batch shape.
        """
        response = await self._client.embeddings.create(
            input=[text.replace("\n", " ")],
            model=self._model,
        )
        vector = list(response.data[0].embedding)
        return EmbeddingResult(vector=vector, model=self._model, dimension=len(vector))


# ---------------------------------------------------------------------------
# Codec + similarity — pure functions (Function Core)
# ---------------------------------------------------------------------------


def encode_embedding(vector: Sequence[float]) -> str:
    """Encode a vector as base64 float32 bytes for the Temporal boundary."""
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def decode_embedding(encoded: str) -> list[float]:
    """Decode a base64 float32 byte string back into a `list[float]`."""
    raw = base64.b64decode(encoded)
    values: list[float] = np.frombuffer(raw, dtype=np.float32).tolist()
    return values


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Accepts any float sequence (`list` or `numpy.ndarray`).
    """
    vec_a = np.asarray(a, dtype=np.float32)
    vec_b = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
