"""Tests for pbook.activities.maintenance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pbook.activities.maintenance import identify_prune_candidates

# ---------------------------------------------------------------------------
# identify_prune_candidates
# ---------------------------------------------------------------------------


class TestIdentifyPruneCandidates:
    def _entry(self, **overrides) -> dict:
        base = {
            "id": 1,
            "title": "Test",
            "content": "Content",
            "tags": ["lang:python"],
            "entry_type": "curated",
            "helpful_count": 0,
            "harmful_count": 0,
            "retrieval_count": 0,
            "created_at": datetime.now(UTC),
        }
        base.update(overrides)
        return base

    def test_harmful_entry_flagged(self):
        entry = self._entry(
            retrieval_count=10,
            harmful_count=6,
            helpful_count=2,
        )
        candidates = identify_prune_candidates([entry])
        assert len(candidates) == 1
        assert "harmful ratio" in candidates[0]["prune_reason"]

    def test_harmful_ratio_at_threshold_not_flagged(self):
        entry = self._entry(
            retrieval_count=10,
            harmful_count=5,  # exactly 50%, not exceeding
            helpful_count=0,
        )
        candidates = identify_prune_candidates([entry])
        assert len(candidates) == 0

    def test_harmful_below_min_retrievals_not_flagged(self):
        entry = self._entry(
            retrieval_count=3,
            harmful_count=3,
        )
        candidates = identify_prune_candidates([entry], min_retrievals=5)
        assert len(candidates) == 0

    def test_stale_unretrieved_entry_flagged(self):
        old_date = datetime.now(UTC) - timedelta(days=200)
        entry = self._entry(
            retrieval_count=0,
            created_at=old_date,
        )
        candidates = identify_prune_candidates([entry])
        assert len(candidates) == 1
        assert "never retrieved" in candidates[0]["prune_reason"]

    def test_recent_unretrieved_entry_not_flagged(self):
        recent_date = datetime.now(UTC) - timedelta(days=30)
        entry = self._entry(
            retrieval_count=0,
            created_at=recent_date,
        )
        candidates = identify_prune_candidates([entry])
        assert len(candidates) == 0

    def test_healthy_entry_not_flagged(self):
        entry = self._entry(
            retrieval_count=10,
            helpful_count=8,
            harmful_count=1,
        )
        candidates = identify_prune_candidates([entry])
        assert len(candidates) == 0

    def test_retrieved_entry_not_stale(self):
        """An old entry that has been retrieved should not be flagged as stale."""
        old_date = datetime.now(UTC) - timedelta(days=300)
        entry = self._entry(
            retrieval_count=2,  # retrieved but below harmful threshold
            harmful_count=0,
            created_at=old_date,
        )
        candidates = identify_prune_candidates([entry])
        assert len(candidates) == 0

    def test_empty_list(self):
        assert identify_prune_candidates([]) == []

    def test_multiple_entries_mixed(self):
        old_date = datetime.now(UTC) - timedelta(days=200)
        entries = [
            self._entry(id=1, retrieval_count=10, harmful_count=8),  # harmful
            self._entry(id=2, retrieval_count=10, helpful_count=9),  # healthy
            self._entry(id=3, retrieval_count=0, created_at=old_date),  # stale
        ]
        candidates = identify_prune_candidates(entries)
        assert len(candidates) == 2
        flagged_ids = {c["id"] for c in candidates}
        assert flagged_ids == {1, 3}

    def test_string_created_at(self):
        """Handles ISO format strings from JSON serialization."""
        old_date = (datetime.now(UTC) - timedelta(days=200)).isoformat()
        entry = self._entry(
            retrieval_count=0,
            created_at=old_date,
        )
        candidates = identify_prune_candidates([entry])
        assert len(candidates) == 1

    def test_custom_thresholds(self):
        entry = self._entry(
            retrieval_count=3,
            harmful_count=2,
        )
        # Default min_retrievals=5 would skip this, but custom=3 catches it
        candidates = identify_prune_candidates(
            [entry],
            min_retrievals=3,
            max_harmful_ratio=0.5,
        )
        assert len(candidates) == 1
