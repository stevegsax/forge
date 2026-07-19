"""Generic embedding step.

Provides :func:`execute_llm_embed`, the logic behind the ``llm_embed``
activity (the ``@activity.defn`` bound method lives on
:class:`pbook.roots.EmbeddingActivities`, which threads in the injected
embedder). Returns the vector as a base64-encoded float32 string so it can
be passed through Temporal's JSON payload boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from temporalio.exceptions import ApplicationError

from pbook.embeddings import encode_embedding
from pbook.workflow_steps._errors import is_nonretryable_auth_error

if TYPE_CHECKING:
    from sax_platform.embeddings import OpenAIEmbeddings


async def execute_llm_embed(embedder: OpenAIEmbeddings | None, text: str) -> str:
    """Compute an embedding for ``text`` and return base64-encoded bytes.

    ``embedder`` is ``None`` when no ``OPENAI_API_KEY`` was configured at the
    composition root; that surfaces here as a clear, non-retryable error
    rather than a hang, matching the old missing-key RuntimeError.
    """
    try:
        if embedder is None:
            msg = "OPENAI_API_KEY not set. Embedding operations require an OpenAI API key."
            raise RuntimeError(msg)
        result = await embedder.embed(text)
        vector = result.vector
    except Exception as exc:
        # A missing OPENAI_API_KEY (RuntimeError) or an invalid one
        # (AuthenticationError) can never succeed on retry — fail fast and
        # non-retryably so the workflow surfaces the error instead of the
        # session hanging at "running". Transient errors stay retryable.
        if is_nonretryable_auth_error(exc):
            raise ApplicationError(
                f"llm_embed: provider authentication/configuration error "
                f"({type(exc).__name__}): {exc}",
                type=type(exc).__name__,
                non_retryable=True,
            ) from exc
        raise
    return encode_embedding(vector)
