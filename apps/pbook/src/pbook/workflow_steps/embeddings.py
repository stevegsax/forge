"""Generic embedding activity.

Thin wrapper around :func:`pbook.embeddings.get_embedding` that returns
the vector as a base64-encoded float32 string so it can be passed
through Temporal's JSON payload boundary.
"""

from __future__ import annotations

from temporalio import activity

from pbook.embeddings import encode_embedding, get_embedding


@activity.defn
async def llm_embed(text: str) -> str:
    """Compute an embedding for ``text`` and return base64-encoded bytes."""
    vector = await get_embedding(text)
    return encode_embedding(vector)
