"""Pydantic data models for the playbook service.

Design follows Function Core / Imperative Shell:
- All models are pure data classes with validation.
- CapabilityTier / ModelConfig / resolve_model are duplicated from Forge
  (30 lines, no shared state) to avoid a dependency on the Forge package.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Entry types
# ---------------------------------------------------------------------------


class EntryType(StrEnum):
    """Content type for playbook entries."""

    PITFALL = "pitfall"  # Extracted from experience — unexpected + actionable
    CURATED = "curated"  # Human-submitted general advice


# ---------------------------------------------------------------------------
# Playbook entry
# ---------------------------------------------------------------------------


class PlaybookEntry(BaseModel):
    """A single playbook entry — the universal write model.

    Covers both content types.
    """

    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    entry_type: EntryType = EntryType.CURATED
    source_project: str = ""
    source_task_id: str = ""
    needs_review: bool = False
    helpful_count: int = 0
    harmful_count: int = 0
    retrieval_count: int = 0
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievalMode(StrEnum):
    """Intent mode for retrieval ranking."""

    CREATE = "create"  # Boosts general knowledge
    FIX = "fix"  # Boosts project-specific pitfalls


class RetrievalInput(BaseModel):
    """Input for the retrieval workflow.

    When ``query`` is non-empty, the workflow computes the query embedding
    and ranks candidates by cosine similarity (semantic-primary). When
    empty, ranking falls back to tag overlap + mode boost.
    """

    tags: list[str] = Field(default_factory=list)
    mode: RetrievalMode = RetrievalMode.CREATE
    token_budget: int = 5000
    project: str = ""
    approved_only: bool = False
    query: str = ""
    threshold: float = 0.0
    include_rejected: bool = False


class RetrievalResult(BaseModel):
    """Output from the retrieval workflow."""

    entries: list[dict[str, Any]] = Field(default_factory=list)
    token_count: int = 0
    total_candidates: int = 0


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class FeedbackInput(BaseModel):
    """Input for recording feedback on a retrieved playbook entry."""

    entry_id: int
    helpful: bool
    source_project: str = ""
    context: str = Field(default="", description="Why the entry was helpful or harmful")


# ---------------------------------------------------------------------------
# Push API — experience data for extraction
# ---------------------------------------------------------------------------


class PushExperienceInput(BaseModel):
    """Structured input for pushing raw experience to the extraction path.

    No ``outcome`` field — the extraction LLM determines what is noteworthy.
    """

    project: str = Field(description="Project that generated this experience")
    problem: str = Field(description="What unexpected situation occurred")
    resolution: str = Field(description="How it was resolved")
    context: str = Field(default="", description="Relevant context (code, errors, etc.)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary key-value pairs")


# ---------------------------------------------------------------------------
# Model routing (duplicated from Forge — 30 lines, no shared state)
# ---------------------------------------------------------------------------


class CapabilityTier(StrEnum):
    """Capability tier for model routing."""

    REASONING = "reasoning"
    GENERATION = "generation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"


_DEFAULT_TIER_MODELS: dict[CapabilityTier, str] = {
    CapabilityTier.REASONING: "anthropic:claude-opus-4-6",
    CapabilityTier.GENERATION: "anthropic:claude-sonnet-4-6",
    CapabilityTier.SUMMARIZATION: "anthropic:claude-sonnet-4-6",
    CapabilityTier.CLASSIFICATION: "anthropic:claude-haiku-4-5-20251001",
}


class ModelConfig(BaseModel):
    """Maps capability tiers to concrete model names."""

    reasoning: str = Field(default=_DEFAULT_TIER_MODELS[CapabilityTier.REASONING])
    generation: str = Field(default=_DEFAULT_TIER_MODELS[CapabilityTier.GENERATION])
    summarization: str = Field(default=_DEFAULT_TIER_MODELS[CapabilityTier.SUMMARIZATION])
    classification: str = Field(default=_DEFAULT_TIER_MODELS[CapabilityTier.CLASSIFICATION])


def resolve_model(tier: CapabilityTier, config: ModelConfig) -> str:
    """Resolve a capability tier to a concrete model name."""
    return {
        CapabilityTier.REASONING: config.reasoning,
        CapabilityTier.GENERATION: config.generation,
        CapabilityTier.SUMMARIZATION: config.summarization,
        CapabilityTier.CLASSIFICATION: config.classification,
    }[tier]


# ---------------------------------------------------------------------------
# CLI-op workflow inputs/outputs
#
# Every CLI command that touches the DB submits a workflow; the worker
# is the only process that opens the database. The worker's
# PBOOK_DATABASE_URL is the single source of truth for which DB is in play.
# `pbook migrate` is the lone exception (schema setup must precede the
# worker's connection).
# ---------------------------------------------------------------------------


class GetEntryInput(BaseModel):
    """Input to ``GetEntryWorkflow``."""

    entry_id: int


class ListEntriesInput(BaseModel):
    """Input to ``ListEntriesWorkflow``.

    All filters are optional; ``needs_review_only`` keeps the queue-style
    filter from the original CLI ``list --needs-review`` flag.
    """

    tags: list[str] = Field(default_factory=list)
    entry_type: str | None = None
    project: str | None = None
    needs_review_only: bool = False
    include_rejected: bool = False
    limit: int = 20


class ListSourcesInput(BaseModel):
    """Input to ``ListSourcesWorkflow``."""

    entry_id: int


class ListTagsInput(BaseModel):
    """Input to ``ListTagsWorkflow`` — no fields, present for symmetry."""


class ReviewQueueInput(BaseModel):
    """Input to ``ReviewQueueWorkflow``."""

    limit: int = 20
    by_experience: bool = False


class ReviewQueueResult(BaseModel):
    """Output from ``ReviewQueueWorkflow``.

    ``by_experience=False``: entries populated, clusters/singletons empty.
    ``by_experience=True``: clusters/singletons populated, entries empty.
    """

    entries: list[dict[str, Any]] = Field(default_factory=list)
    clusters: list[dict[str, Any]] = Field(default_factory=list)
    singletons: list[dict[str, Any]] = Field(default_factory=list)


class ListSessionsInput(BaseModel):
    """Input to ``ListSessionsWorkflow``."""

    project: str | None = None
    limit: int = 20


class GetSessionTextInput(BaseModel):
    """Input to ``GetSessionTextWorkflow``.

    The activity reads the JSONL transcript from disk inside the worker
    process. ``path`` overrides default resolution (which scans
    ``~/.claude/projects/``); ``raw=True`` returns the JSONL bytes
    verbatim instead of the rendered USER:/ASSISTANT: format.
    """

    session_id: str
    path: str | None = None
    raw: bool = False


class GetSessionTextResult(BaseModel):
    """Output from ``GetSessionTextWorkflow``."""

    text: str
    project_name: str = ""


class AddEntryInput(BaseModel):
    """Input to ``AddEntryWorkflow``."""

    entry: PlaybookEntry
    needs_review: bool = False


class AddEntryResult(BaseModel):
    """Output from ``AddEntryWorkflow``."""

    id: int
    title: str
    needs_review: bool
    rejected: bool = False


class ApproveEntryInput(BaseModel):
    """Input to ``ApproveEntryWorkflow``."""

    entry_id: int


class RejectEntryInput(BaseModel):
    """Input to ``RejectEntryWorkflow``."""

    entry_id: int
    reason: str | None = None


class UpdateEntryInput(BaseModel):
    """Input to ``UpdateEntryWorkflow``.

    ``updates`` is a column-name → value dict; the activity validates
    against the entries table schema.
    """

    entry_id: int
    updates: dict[str, Any] = Field(default_factory=dict)


class RecordFeedbackInput(BaseModel):
    """Input to ``RecordFeedbackWorkflow``."""

    entry_id: int
    helpful: bool
    context: str = ""


class CheckDuplicateInput(BaseModel):
    """Input to ``CheckDuplicateWorkflow``."""

    title: str
    tags: list[str] | None = None


class PruneInput(BaseModel):
    """Input to ``PruneWorkflow``.

    ``apply=False`` (dry-run) returns candidates without changing state.
    ``apply=True`` marks each candidate for review with the
    ``pattern:prune-candidate`` tag and ``needs_review=True``.
    """

    min_retrievals: int = 5
    max_harmful_ratio: float = 0.5
    max_stale_days: int = 90
    apply: bool = False


class PruneResult(BaseModel):
    """Output from ``PruneWorkflow``."""

    candidates: list[dict[str, Any]] = Field(default_factory=list)
    applied_count: int = 0


class EntryStatusResult(BaseModel):
    """Common output for the simple write workflows (approve/reject/update).

    Mirrors what the CLI used to emit before the refactor so existing
    JSON consumers keep working.
    """

    id: int
    title: str
    approved: bool
    needs_review: bool = False
    rejected: bool = False
    rejection_reason: str | None = None


class FilterAlreadyIngestedInput(BaseModel):
    """Input to ``FilterAlreadyIngestedWorkflow``."""

    session_ids: list[str] = Field(default_factory=list)


class IngestedSessionStub(BaseModel):
    """One session payload for ``RecordStartedSessionsWorkflow``."""

    session_id: str
    project_name: str


class RecordStartedSessionsInput(BaseModel):
    """Input to ``RecordStartedSessionsWorkflow``."""

    sessions: list[IngestedSessionStub] = Field(default_factory=list)
    workflow_id: str
    run_id: str
