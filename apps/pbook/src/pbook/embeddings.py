"""Embedding utilities for the playbook service.

Uses OpenAI API to generate vector embeddings for semantic search
and de-duplication, as prescribed by the ACE framework.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Default model for embeddings
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


async def get_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> bytes:
    """Generate a vector embedding for the given text.

    Returns the embedding as a float32 byte array (BLOB-compatible).
    """
    client = get_client()
    logger.debug("Generating embedding for text (len=%d) using %s", len(text), model)

    response = await client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=model,
    )

    vector = response.data[0].embedding
    return np.array(vector, dtype=np.float32).tobytes()


def cosine_similarity(a: bytes, b: bytes) -> float:
    """Compute cosine similarity between two float32-encoded vector blobs."""
    vec_a = np.frombuffer(a, dtype=np.float32)
    vec_b = np.frombuffer(b, dtype=np.float32)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
