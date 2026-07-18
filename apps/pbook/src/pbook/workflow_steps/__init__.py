"""Generic, reusable Temporal workflow steps for LLM and embedding work.

Activities (``llm_chat``, ``llm_embed``) live here so any workflow can
make structured-output chat calls or compute embeddings without
re-implementing the provider/parsing/heartbeat ritual. Output types are
resolved by name via the frozen mapping in
:mod:`pbook.workflow_steps.output_types`.

Designed to be reusable beyond pbook — forge could adopt these in a
future round, at which point the contract may grow an ``include_raw``
flag on :class:`LLMChatResult` for forge's message-log path.
"""

from pbook.workflow_steps.embeddings import llm_embed
from pbook.workflow_steps.llm import LLMChatInput, LLMChatResult, llm_chat
from pbook.workflow_steps.output_types import OUTPUT_TYPES, resolve_output_type

__all__ = [
    "OUTPUT_TYPES",
    "LLMChatInput",
    "LLMChatResult",
    "llm_chat",
    "llm_embed",
    "resolve_output_type",
]
