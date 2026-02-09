"""Tests for forge.code_intel.budget — token estimation + priority packing."""

from __future__ import annotations

from forge.code_intel.budget import (
    ContextBudget,
    ContextItem,
    PackedContext,
    Representation,
    build_context_items,
    compute_budget,
    pack_context,
)
from forge.code_intel.graph import RankedFile, Relationship
from forge.code_intel.repo_map import estimate_tokens

# ---------------------------------------------------------------------------
# compute_budget
# ---------------------------------------------------------------------------


class TestComputeBudget:
    def test_basic(self) -> None:
        budget = compute_budget(
            model_max_tokens=200_000,
            reserved_for_output=16_000,
            task_description_tokens=4_000,
        )
        assert isinstance(budget, ContextBudget)
        assert budget.available_for_context == 180_000

    def test_zero_available(self) -> None:
        budget = compute_budget(
            model_max_tokens=1000,
            reserved_for_output=800,
            task_description_tokens=300,
        )
        assert budget.available_for_context == 0


# ---------------------------------------------------------------------------
# pack_context
# ---------------------------------------------------------------------------


def _make_item(
    path: str,
    content: str,
    priority: int,
    importance: float = 0.5,
    representation: Representation = Representation.FULL,
) -> ContextItem:
    return ContextItem(
        file_path=path,
        content=content,
        representation=representation,
        priority=priority,
        importance=importance,
        estimated_tokens=estimate_tokens(content),
    )


class TestPackContext:
    def test_empty_items(self) -> None:
        result = pack_context([], budget_tokens=1000)
        assert isinstance(result, PackedContext)
        assert result.items_included == 0

    def test_all_fit(self) -> None:
        items = [
            _make_item("a.py", "short", priority=2),
            _make_item("b.py", "also short", priority=3),
        ]
        result = pack_context(items, budget_tokens=10000)
        assert result.items_included == 2
        assert result.items_truncated == 0

    def test_priority_ordering(self) -> None:
        items = [
            _make_item("low.py", "x" * 40, priority=6),
            _make_item("high.py", "y" * 40, priority=2),
        ]
        result = pack_context(items, budget_tokens=10000)
        assert result.items[0].file_path == "high.py"
        assert result.items[1].file_path == "low.py"

    def test_importance_ordering_within_tier(self) -> None:
        items = [
            _make_item("less.py", "x" * 40, priority=3, importance=0.1),
            _make_item("more.py", "y" * 40, priority=3, importance=0.9),
        ]
        result = pack_context(items, budget_tokens=10000)
        assert result.items[0].file_path == "more.py"
        assert result.items[1].file_path == "less.py"

    def test_truncation_when_over_budget(self) -> None:
        items = [
            _make_item("fits.py", "a" * 20, priority=2),
            _make_item("big.py", "b" * 40000, priority=3),
        ]
        result = pack_context(items, budget_tokens=100)
        assert result.items_included == 1
        assert result.items_truncated == 1
        assert result.items[0].file_path == "fits.py"

    def test_graceful_degradation(self) -> None:
        items = [
            _make_item("large.py", "x" * 400, priority=3),
        ]
        sigs = {"large.py": "def f():"}  # Much smaller
        result = pack_context(items, budget_tokens=10, signature_fallbacks=sigs)
        assert result.items_included == 1
        assert result.items_reduced == 1
        assert result.items[0].representation == Representation.SIGNATURES

    def test_degradation_also_fails(self) -> None:
        items = [
            _make_item("large.py", "x" * 400, priority=3),
        ]
        sigs = {"large.py": "y" * 400}  # Also too big
        result = pack_context(items, budget_tokens=10, signature_fallbacks=sigs)
        assert result.items_included == 0
        assert result.items_truncated == 1

    def test_budget_utilization(self) -> None:
        items = [_make_item("a.py", "a" * 200, priority=2)]
        result = pack_context(items, budget_tokens=100)
        assert result.budget_utilization == 50 / 100  # 200 chars / 4 = 50 tokens


# ---------------------------------------------------------------------------
# build_context_items
# ---------------------------------------------------------------------------


class TestBuildContextItems:
    def test_target_files_priority_2(self) -> None:
        items = build_context_items(
            target_file_contents={"target.py": "code"},
            direct_import_contents={},
            transitive_summaries={},
            ranked_files=[],
            manual_context_contents={},
        )
        assert len(items) == 1
        assert items[0].priority == 2
        assert items[0].file_path == "target.py"

    def test_direct_imports_priority_3(self) -> None:
        items = build_context_items(
            target_file_contents={},
            direct_import_contents={"dep.py": "dep code"},
            transitive_summaries={},
            ranked_files=[],
            manual_context_contents={},
        )
        assert len(items) == 1
        assert items[0].priority == 3

    def test_transitive_priority_4(self) -> None:
        items = build_context_items(
            target_file_contents={},
            direct_import_contents={},
            transitive_summaries={"trans.py": "def f():"},
            ranked_files=[],
            manual_context_contents={},
        )
        assert len(items) == 1
        assert items[0].priority == 4
        assert items[0].representation == Representation.SIGNATURES

    def test_repo_map_priority_5(self) -> None:
        items = build_context_items(
            target_file_contents={},
            direct_import_contents={},
            transitive_summaries={},
            ranked_files=[],
            manual_context_contents={},
            repo_map_text="repo map content",
        )
        assert len(items) == 1
        assert items[0].priority == 5
        assert items[0].representation == Representation.REPO_MAP

    def test_manual_context_priority_6(self) -> None:
        items = build_context_items(
            target_file_contents={},
            direct_import_contents={},
            transitive_summaries={},
            ranked_files=[],
            manual_context_contents={"manual.txt": "info"},
        )
        assert len(items) == 1
        assert items[0].priority == 6

    def test_no_duplicates(self) -> None:
        """Files in target_files should not be duplicated in direct imports."""
        items = build_context_items(
            target_file_contents={"shared.py": "code"},
            direct_import_contents={"shared.py": "code"},
            transitive_summaries={"shared.py": "def f():"},
            ranked_files=[],
            manual_context_contents={},
        )
        assert len(items) == 1
        assert items[0].priority == 2

    def test_importance_from_ranked_files(self) -> None:
        ranked = [
            RankedFile(
                file_path="dep.py",
                module_name="pkg.dep",
                importance=0.75,
                distance=1,
                relationship=Relationship.DIRECT_IMPORT,
            ),
        ]
        items = build_context_items(
            target_file_contents={},
            direct_import_contents={"dep.py": "code"},
            transitive_summaries={},
            ranked_files=ranked,
            manual_context_contents={},
        )
        assert items[0].importance == 0.75

    def test_all_tiers_combined(self) -> None:
        items = build_context_items(
            target_file_contents={"target.py": "target"},
            direct_import_contents={"direct.py": "direct"},
            transitive_summaries={"transitive.py": "def f():"},
            ranked_files=[],
            manual_context_contents={"manual.txt": "info"},
            repo_map_text="repo map",
        )
        priorities = {item.priority for item in items}
        assert priorities == {2, 3, 4, 5, 6}
