"""Embedding utilities for the playbook service.

Thin adapter over :mod:`sax_platform.embeddings` (T3.4). The base64/float32
codec (`encode_embedding`/`decode_embedding`), `cosine_similarity`, and
`DEFAULT_EMBEDDING_MODEL` are re-exported unchanged — `sax_platform` owns
their one implementation now; pbook's former duplicates are gone. What
stays here is the `OPENAI_API_KEY` composition seam: `get_client()` builds
the `openai.AsyncOpenAI` client from the environment (env wiring stays
pbook's own until T3.6), and `get_embedding()` delegates through
`sax_platform.embeddings.OpenAIEmbeddings` using that client, preserving
the original ``list[float]`` return type existing callers expect.
"""

from __future__ import annotations

import logging
import os

from openai import AsyncOpenAI
from sax_platform.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    OpenAIEmbeddings,
    cosine_similarity,
    decode_embedding,
    encode_embedding,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "cosine_similarity",
    "decode_embedding",
    "encode_embedding",
    "get_client",
    "get_embedding",
]

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Get or create the AsyncOpenAI client.

    Raises ``RuntimeError`` if OPENAI_API_KEY is not set.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            msg = (
                "OPENAI_API_KEY not set. Embedding operations require an OpenAI API key. "
                "Please set the OPENAI_API_KEY environment variable."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def get_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    """Generate a vector embedding for the given text.

    Returns the embedding as a ``list[float]`` ready to store in a
    pgvector column. Delegates to
    `sax_platform.embeddings.OpenAIEmbeddings`, constructed with the
    client from `get_client()`.
    """
    logger.debug("Generating embedding for text (len=%d) using %s", len(text), model)
    embedder = OpenAIEmbeddings(get_client(), model)
    result = await embedder.embed(text)
    return result.vector
