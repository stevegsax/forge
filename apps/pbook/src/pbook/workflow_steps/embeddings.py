"""Generic embedding activity.

Thin wrapper around :func:`pbook.embeddings.get_embedding` that returns
the vector as a base64-encoded float32 string so it can be passed
through Temporal's JSON payload boundary.
"""

from __future__ import annotations

from temporalio import activity
from temporalio.exceptions import ApplicationError

from pbook.embeddings import encode_embedding, get_embedding
from pbook.workflow_steps._errors import is_nonretryable_auth_error


@activity.defn
async def llm_embed(text: str) -> str:
    """Compute an embedding for ``text`` and return base64-encoded bytes."""
    try:
        vector = await get_embedding(text)
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
