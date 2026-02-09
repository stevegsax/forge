"""Token estimation and priority-based context packing.

Pure functions only. Implements bin-packing with priority ordering.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from forge.code_intel.repo_map import RepoMap, estimate_tokens

if TYPE_CHECKING:
    from forge.code_intel.graph import RankedFile


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Representation(StrEnum):
    """How a context item is represented in the prompt."""

    FULL = "full"
    SIGNATURES = "signatures"
    REPO_MAP = "repo_map"
    PLAYBOOK = "playbook"


class ContextItem(BaseModel):
    """A single item to include in context."""

    file_path: str
    content: str = Field(description="The text to include in the prompt.")
    representation: Representation
    priority: int = Field(description="Lower number = higher priority.")
    importance: float = Field(default=0.0, description="PageRank score for ranking within tiers.")
    estimated_tokens: int


class ContextBudget(BaseModel):
    """Token budget breakdown."""

    model_max_tokens: int = Field(description="Total model context window.")
    reserved_for_output: int = Field(description="Tokens reserved for LLM response.")
    reserved_for_task: int = Field(description="Tokens used by task description, instructions.")
    available_for_context: int = Field(description="What remains for file context.")


class PackedContext(BaseModel):
    """Result of packing context items into the budget."""

    items: list[ContextItem] = Field(default_factory=list)
    repo_map: RepoMap | None = None
    total_estimated_tokens: int = 0
    budget_utilization: float = Field(default=0.0, description="0.0 to 1.0.")
    items_included: int = 0
    items_reduced: int = Field(default=0, description="Items downgraded from full to signatures.")
    items_truncated: int = Field(default=0, description="Items that didn't fit at all.")


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def compute_budget(
    model_max_tokens: int,
    reserved_for_output: int,
    task_description_tokens: int,
) -> ContextBudget:
    """Compute the available token budget for file context."""
    available = model_max_tokens - reserved_for_output - task_description_tokens
    return ContextBudget(
        model_max_tokens=model_max_tokens,
        reserved_for_output=reserved_for_output,
        reserved_for_task=task_description_tokens,
        available_for_context=max(0, available),
    )


def pack_context(
    items: list[ContextItem],
    budget_tokens: int,
    signature_fallbacks: dict[str, str] | None = None,
) -> PackedContext:
    """Pack context items into the token budget.

    Priority-then-importance ordering. When an item would exceed the budget,
    attempts graceful degradation (full -> signatures) before skipping.

    Args:
        items: Context items to pack, in any order.
        budget_tokens: Maximum tokens available for context.
        signature_fallbacks: Optional dict of file_path -> signature text
            for items that can be reduced from full to signatures.
    """
    if not items:
        return PackedContext()

    signature_fallbacks = signature_fallbacks or {}

    # Sort by priority (ascending), then importance (descending) within tier
    sorted_items = sorted(items, key=lambda i: (i.priority, -i.importance))

    packed: list[ContextItem] = []
    total_tokens = 0
    reduced_count = 0
    truncated_count = 0

    for item in sorted_items:
        if total_tokens + item.estimated_tokens <= budget_tokens:
            packed.append(item)
            total_tokens += item.estimated_tokens
        elif item.representation == Representation.FULL and item.file_path in signature_fallbacks:
            # Try graceful degradation: full -> signatures
            sig_text = signature_fallbacks[item.file_path]
            sig_tokens = estimate_tokens(sig_text)
            if total_tokens + sig_tokens <= budget_tokens:
                reduced_item = item.model_copy(
                    update={
                        "content": sig_text,
                        "representation": Representation.SIGNATURES,
                        "estimated_tokens": sig_tokens,
                    }
                )
                packed.append(reduced_item)
                total_tokens += sig_tokens
                reduced_count += 1
            else:
                truncated_count += 1
        else:
            truncated_count += 1

    utilization = total_tokens / budget_tokens if budget_tokens > 0 else 0.0

    return PackedContext(
        items=packed,
        total_estimated_tokens=total_tokens,
        budget_utilization=utilization,
        items_included=len(packed),
        items_reduced=reduced_count,
        items_truncated=truncated_count,
    )


def build_context_items(
    target_file_contents: dict[str, str],
    direct_import_contents: dict[str, str],
    transitive_summaries: dict[str, str],
    ranked_files: list[RankedFile],
    manual_context_contents: dict[str, str],
    repo_map_text: str | None = None,
) -> list[ContextItem]:
    """Build context items with assigned priority tiers.

    Priority tiers:
        2 = target files (current content)
        3 = direct imports (full content)
        4 = transitive imports (signatures only)
        5 = repo map
        6 = manually specified context_files
    """
    items: list[ContextItem] = []

    # Build importance lookup
    importance_map = {f.file_path: f.importance for f in ranked_files}

    # Priority 2: target files
    for path, content in target_file_contents.items():
        items.append(
            ContextItem(
                file_path=path,
                content=content,
                representation=Representation.FULL,
                priority=2,
                importance=importance_map.get(path, 1.0),
                estimated_tokens=estimate_tokens(content),
            )
        )

    # Priority 3: direct imports
    for path, content in direct_import_contents.items():
        if path not in target_file_contents:
            items.append(
                ContextItem(
                    file_path=path,
                    content=content,
                    representation=Representation.FULL,
                    priority=3,
                    importance=importance_map.get(path, 0.5),
                    estimated_tokens=estimate_tokens(content),
                )
            )

    # Priority 4: transitive imports (signatures)
    for path, sig_text in transitive_summaries.items():
        if path not in target_file_contents and path not in direct_import_contents:
            items.append(
                ContextItem(
                    file_path=path,
                    content=sig_text,
                    representation=Representation.SIGNATURES,
                    priority=4,
                    importance=importance_map.get(path, 0.1),
                    estimated_tokens=estimate_tokens(sig_text),
                )
            )

    # Priority 5: repo map
    if repo_map_text:
        items.append(
            ContextItem(
                file_path="__repo_map__",
                content=repo_map_text,
                representation=Representation.REPO_MAP,
                priority=5,
                importance=0.0,
                estimated_tokens=estimate_tokens(repo_map_text),
            )
        )

    # Priority 6: manual context files
    for path, content in manual_context_contents.items():
        if path not in target_file_contents and path not in direct_import_contents:
            items.append(
                ContextItem(
                    file_path=path,
                    content=content,
                    representation=Representation.FULL,
                    priority=6,
                    importance=0.0,
                    estimated_tokens=estimate_tokens(content),
                )
            )

    return items
