"""Embedding utilities for the playbook service.

Thin re-export shim over :mod:`sax_platform.embeddings` (T3.4/T3.6).
``sax_platform`` owns the one implementation of the base64/float32 codec
(`encode_embedding`/`decode_embedding`), `cosine_similarity`, the
`OpenAIEmbeddings` client, and `DEFAULT_EMBEDDING_MODEL`.

As of T3.6 the module-global `OPENAI_API_KEY` composition seam is gone —
the old client-cache singleton and free embedding function are deleted. The
worker's composition root (`pbook.worker`) builds the `openai.AsyncOpenAI`
client from settings and threads an `OpenAIEmbeddings` into
:class:`~pbook.roots.EmbeddingActivities`; no module here reads the
environment for a credential.
"""

from __future__ import annotations

from sax_platform.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    OpenAIEmbeddings,
    cosine_similarity,
    decode_embedding,
    encode_embedding,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "OpenAIEmbeddings",
    "cosine_similarity",
    "decode_embedding",
    "encode_embedding",
]
