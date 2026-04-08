"""Pydantic data models for the playbook service.

Design follows Function Core / Imperative Shell:
- All models are pure data classes with validation.
- CapabilityTier / ModelConfig / resolve_model are duplicated from Forge
  (30 lines, no shared state) to avoid a dependency on the Forge package.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Entry types
# ---------------------------------------------------------------------------


class EntryType(StrEnum):
    """Content type for playbook entries."""

    PITFALL = "pitfall"       # Extracted from experience — unexpected + actionable
    CURATED = "curated"       # Human-submitted general advice
    API_DOC = "api_doc"       # Library documentation record


# ---------------------------------------------------------------------------
# Playbook entry
# ---------------------------------------------------------------------------


class PlaybookEntry(BaseModel):
    """A single playbook entry — the universal write model.

    Covers all three content types.  For API_DOC entries the ``content``
    field holds the structured ApiDocRecord serialized as JSON.
    """

    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    entry_type: EntryType = EntryType.CURATED
    source_project: str = ""
    source_task_id: str = ""
    needs_review: bool = False


# ---------------------------------------------------------------------------
# API documentation record
# ---------------------------------------------------------------------------


class ApiDocRecord(BaseModel):
    """Structured API documentation for a single library method.

    Stored as the ``content`` of a PlaybookEntry with entry_type=API_DOC.
    """

    library: str               # e.g. "sqlalchemy"
    method: str                # Fully qualified, e.g. "sqlalchemy.create_engine"
    summary: str               # 1-2 sentences
    signature: str             # Method signature with type hints
    examples: list[str] = Field(default_factory=list)
    doc_url: str = ""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievalMode(StrEnum):
    """Intent mode for retrieval ranking."""

    CREATE = "create"   # Boosts general knowledge + API docs
    FIX = "fix"         # Boosts project-specific pitfalls


class RetrievalInput(BaseModel):
    """Input for the retrieval workflow."""

    tags: list[str] = Field(default_factory=list)
    mode: RetrievalMode = RetrievalMode.CREATE
    token_budget: int = 5000
    project: str = ""
    approved_only: bool = False


class RetrievalResult(BaseModel):
    """Output from the retrieval workflow."""

    entries: list[dict] = Field(default_factory=list)
    token_count: int = 0
    total_candidates: int = 0


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
    metadata: dict = Field(default_factory=dict, description="Arbitrary key-value pairs")


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
    CapabilityTier.GENERATION: "anthropic:claude-sonnet-4-5-20250929",
    CapabilityTier.SUMMARIZATION: "anthropic:claude-sonnet-4-5-20250929",
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
