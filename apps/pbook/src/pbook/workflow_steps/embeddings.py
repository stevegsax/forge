"""Generic embedding activity.

Thin wrapper around :func:`pbook.embeddings.get_embedding` that returns
the float32 vector as a base64-encoded string so it can be passed
through Temporal's JSON payload boundary.
"""

from __future__ import annotations

import base64

from temporalio import activity

from pbook.embeddings import get_embedding


@activity.defn
async def llm_embed(text: str) -> str:
    """Compute an embedding for ``text`` and return base64-encoded bytes."""
    raw = await get_embedding(text)
    return base64.b64encode(raw).decode("ascii")
