"""Embedding utilities for the playbook service.

Uses the OpenAI API to generate vector embeddings for semantic search
and de-duplication. Vectors are plain ``list[float]`` throughout — that
is what pgvector columns accept and return. For the Temporal payload
boundary they are base64-encoded as float32 bytes (compact and
deterministic); see :func:`encode_embedding` / :func:`decode_embedding`.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Default model for embeddings (1536-dim — see pbook.store.EMBEDDING_DIM).
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

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
    pgvector column.
    """
    client = get_client()
    logger.debug("Generating embedding for text (len=%d) using %s", len(text), model)

    response = await client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=model,
    )
    return list(response.data[0].embedding)


def encode_embedding(vector: Sequence[float]) -> str:
    """Encode a vector as base64 float32 bytes for the Temporal boundary."""
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def decode_embedding(encoded: str) -> list[float]:
    """Decode a base64 float32 byte string back into a ``list[float]``."""
    raw = base64.b64decode(encoded)
    values: list[float] = np.frombuffer(raw, dtype=np.float32).tolist()
    return values


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Accepts any float sequence (``list`` or ``numpy.ndarray``).
    """
    vec_a = np.asarray(a, dtype=np.float32)
    vec_b = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
