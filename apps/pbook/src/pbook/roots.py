"""Class-based Temporal activities — pbook's dependency-injection seam (T3.6).

Temporal's sanctioned dependency injection for activities is a class whose
``@activity.defn`` methods close over constructor-injected dependencies. The
worker's composition root (:mod:`pbook.worker`) builds the engine, the LLM
provider, and the embedder ONCE, constructs these classes, and registers the
bound methods.

Each method's ``__name__`` equals the historical activity name — all pbook
workflows invoke activities by string (``workflow.execute_activity("llm_chat",
...)``), and a bound method keeps its ``__name__`` — so this class conversion
is invisible to every workflow and to the by-name workflow-mock tests.

The methods are thin shells: they delegate to the per-module activity
functions (in ``pbook.activities`` and ``pbook.workflow_steps``), passing the
injected dependency. The store-touching functions preserve the engine-is-None
disabled behavior they had under the old ``get_store_engine()`` seam (return
empty / no-op for the workflow activities; raise for the CLI-op activities via
``_require_engine``).

The two no-dependency activities — ``validate_entry`` (pure JSON validation)
and ``get_session_text_activity`` (transcript file rendering) — stay bare
``@activity.defn`` free functions in their modules; classes exist only to
carry dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from temporalio import activity

from pbook.activities import cli_ops, export, extraction, maintenance, retrieval, review
from pbook.workflow_steps.embeddings import execute_llm_embed
from pbook.workflow_steps.llm import LLMChatInput, LLMChatResult, execute_llm_chat

if TYPE_CHECKING:
    from sax_platform.embeddings import OpenAIEmbeddings
    from sqlalchemy import Engine

    from pbook.llm import SupportsComplete


class StoreActivities:
    """Every store-touching activity, bound to one engine per process.

    ``engine`` is ``None`` when ``PBOOK_DATABASE_URL`` is unset (store
    disabled). One engine per process is load-bearing: the platform engine
    factory caps the Postgres pool, so building a fresh pool per activity
    call would exhaust the managed-database connection cap.
    """

    def __init__(self, engine: Engine | None) -> None:
        self._engine = engine

    # --- retrieval -------------------------------------------------------

    @activity.defn
    async def fetch_candidates(self, input_json: str) -> list[dict[str, Any]]:
        return await retrieval.fetch_candidates(self._engine, input_json)

    @activity.defn
    async def compute_similarities_by_id(self, input_json: str) -> dict[str, float]:
        return await retrieval.compute_similarities_by_id(self._engine, input_json)

    @activity.defn
    async def score_and_pack(self, input_json: str) -> dict[str, Any]:
        return await retrieval.score_and_pack(self._engine, input_json)

    @activity.defn
    async def record_retrieval_event(self, entry_ids_json: str) -> None:
        await retrieval.record_retrieval_event(self._engine, entry_ids_json)

    # --- export ----------------------------------------------------------

    @activity.defn
    async def fetch_entry_ids(self, input_json: str) -> list[int]:
        return await export.fetch_entry_ids(self._engine, input_json)

    @activity.defn
    async def export_single_entry(self, entry_id: int) -> dict[str, Any]:
        return await export.export_single_entry(self._engine, entry_id)

    # --- extraction (persistence side) -----------------------------------

    @activity.defn
    async def save_extracted_entries(self, input_json: str) -> int:
        return await extraction.save_extracted_entries(self._engine, input_json)

    @activity.defn
    async def record_ingested_session(self, input_json: str) -> None:
        await extraction.record_ingested_session(self._engine, input_json)

    @activity.defn
    async def record_ingested_session_error(self, input_json: str) -> None:
        await extraction.record_ingested_session_error(self._engine, input_json)

    # --- review (persistence side) ---------------------------------------

    @activity.defn
    async def fetch_existing_entries(self, limit: int = 50) -> list[dict[str, Any]]:
        return await review.fetch_existing_entries(self._engine, limit)

    @activity.defn
    async def find_duplicates(self, input_json: str) -> list[dict[str, Any]]:
        return await review.find_duplicates(self._engine, input_json)

    # --- maintenance -----------------------------------------------------

    @activity.defn
    async def fetch_all_entries_for_maintenance(self) -> list[dict[str, Any]]:
        return await maintenance.fetch_all_entries_for_maintenance(self._engine)

    @activity.defn
    async def cluster_similar_entries(self, input_json: str) -> list[list[int]]:
        return await maintenance.cluster_similar_entries(self._engine, input_json)

    @activity.defn
    async def prune_entries(self, entry_ids: list[int]) -> int:
        return await maintenance.prune_entries(self._engine, entry_ids)

    @activity.defn
    async def save_consolidated_entry(self, input_json: str) -> int:
        return await maintenance.save_consolidated_entry(self._engine, input_json)

    # --- CLI-op activities ----------------------------------------------

    @activity.defn
    async def get_entry_activity(self, input: dict[str, Any]) -> dict[str, Any] | None:
        return await cli_ops.get_entry_activity(self._engine, input)

    @activity.defn
    async def list_entries_activity(self, input: dict[str, Any]) -> list[dict[str, Any]]:
        return await cli_ops.list_entries_activity(self._engine, input)

    @activity.defn
    async def list_sources_activity(self, input: dict[str, Any]) -> dict[str, Any]:
        return await cli_ops.list_sources_activity(self._engine, input)

    @activity.defn
    async def list_tags_activity(self, input: dict[str, Any]) -> dict[str, Any]:
        return await cli_ops.list_tags_activity(self._engine, input)

    @activity.defn
    async def review_queue_activity(self, input: dict[str, Any]) -> dict[str, Any]:
        return await cli_ops.review_queue_activity(self._engine, input)

    @activity.defn
    async def list_sessions_activity(self, input: dict[str, Any]) -> list[dict[str, Any]]:
        return await cli_ops.list_sessions_activity(self._engine, input)

    @activity.defn
    async def check_duplicate_activity(self, input: dict[str, Any]) -> list[dict[str, Any]]:
        return await cli_ops.check_duplicate_activity(self._engine, input)

    @activity.defn
    async def add_entry_activity(self, input: dict[str, Any]) -> dict[str, Any]:
        return await cli_ops.add_entry_activity(self._engine, input)

    @activity.defn
    async def approve_entry_activity(self, input: dict[str, Any]) -> dict[str, Any]:
        return await cli_ops.approve_entry_activity(self._engine, input)

    @activity.defn
    async def reject_entry_activity(self, input: dict[str, Any]) -> dict[str, Any]:
        return await cli_ops.reject_entry_activity(self._engine, input)

    @activity.defn
    async def update_entry_activity(self, input: dict[str, Any]) -> dict[str, Any]:
        return await cli_ops.update_entry_activity(self._engine, input)

    @activity.defn
    async def record_feedback_activity(self, input: dict[str, Any]) -> dict[str, Any]:
        return await cli_ops.record_feedback_activity(self._engine, input)

    @activity.defn
    async def filter_already_ingested_activity(self, input: dict[str, Any]) -> dict[str, Any]:
        return await cli_ops.filter_already_ingested_activity(self._engine, input)

    @activity.defn
    async def record_started_sessions_activity(self, input: dict[str, Any]) -> dict[str, Any]:
        return await cli_ops.record_started_sessions_activity(self._engine, input)

    @activity.defn
    async def prune_activity(self, input: dict[str, Any]) -> dict[str, Any]:
        return await cli_ops.prune_activity(self._engine, input)

    def all_activities(self) -> list[Any]:
        """The bound methods to register on the worker (one engine per process)."""
        return [
            self.fetch_candidates,
            self.compute_similarities_by_id,
            self.score_and_pack,
            self.record_retrieval_event,
            self.fetch_entry_ids,
            self.export_single_entry,
            self.save_extracted_entries,
            self.record_ingested_session,
            self.record_ingested_session_error,
            self.fetch_existing_entries,
            self.find_duplicates,
            self.fetch_all_entries_for_maintenance,
            self.cluster_similar_entries,
            self.prune_entries,
            self.save_consolidated_entry,
            self.get_entry_activity,
            self.list_entries_activity,
            self.list_sources_activity,
            self.list_tags_activity,
            self.review_queue_activity,
            self.list_sessions_activity,
            self.check_duplicate_activity,
            self.add_entry_activity,
            self.approve_entry_activity,
            self.reject_entry_activity,
            self.update_entry_activity,
            self.record_feedback_activity,
            self.filter_already_ingested_activity,
            self.record_started_sessions_activity,
            self.prune_activity,
        ]


class LlmActivities:
    """The generic structured-output chat activity, bound to one provider."""

    def __init__(self, provider: SupportsComplete) -> None:
        self._provider = provider

    @activity.defn
    async def llm_chat(self, input: LLMChatInput) -> LLMChatResult:
        return await execute_llm_chat(self._provider, input)


class EmbeddingActivities:
    """The generic embedding activity, bound to one embedder.

    ``embedder`` is ``None`` when no ``OPENAI_API_KEY`` was configured; that
    surfaces at call time as a clear, non-retryable error (see
    :func:`pbook.workflow_steps.embeddings.execute_llm_embed`).
    """

    def __init__(self, embedder: OpenAIEmbeddings | None) -> None:
        self._embedder = embedder

    @activity.defn
    async def llm_embed(self, text: str) -> str:
        return await execute_llm_embed(self._embedder, text)
