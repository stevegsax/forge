"""Generic, reusable Temporal workflow steps for LLM and embedding work.

The ``llm_chat`` and ``llm_embed`` activities are bound methods on
:class:`pbook.roots.LlmActivities` / :class:`pbook.roots.EmbeddingActivities`
(the composition root threads in the provider/embedder). This package holds
the dependency-injected step logic (:func:`execute_llm_chat`,
:func:`execute_llm_embed`) plus the shared input/output models and the frozen
output-type mapping any workflow needs.

Designed to be reusable beyond pbook — forge could adopt these in a
future round, at which point the contract may grow an ``include_raw``
flag on :class:`LLMChatResult` for forge's message-log path.
"""

from pbook.workflow_steps.embeddings import execute_llm_embed
from pbook.workflow_steps.llm import LLMChatInput, LLMChatResult, execute_llm_chat
from pbook.workflow_steps.output_types import OUTPUT_TYPES, resolve_output_type

__all__ = [
    "OUTPUT_TYPES",
    "LLMChatInput",
    "LLMChatResult",
    "execute_llm_chat",
    "execute_llm_embed",
    "resolve_output_type",
]
