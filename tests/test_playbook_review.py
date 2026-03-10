"""Tests for forge.activities.playbook_review."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from forge.activities.playbook_review import (
    apply_suggestions,
    build_review_system_prompt,
    build_review_user_prompt,
    review_playbook_entry,
)
from forge.models import PlaybookEntry, PlaybookReviewResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_entry() -> PlaybookEntry:
    return PlaybookEntry(
        title="Query OCR documents",
        content="Use get_db_path() to connect and query ocr_results table.",
        tags=["domain:database"],
        source_task_id="manual-ocr",
    )


@pytest.fixture
def existing_playbooks() -> list[dict]:
    return [
        {
            "title": "Retry on timeout",
            "tags_json": '["pattern:retry", "domain:network"]',
        },
        {
            "title": "Use WAL mode for SQLite",
            "tags_json": '["domain:database"]',
        },
    ]


# ---------------------------------------------------------------------------
# build_review_system_prompt
# ---------------------------------------------------------------------------


class TestBuildReviewSystemPrompt:
    def test_includes_criteria(self) -> None:
        prompt = build_review_system_prompt([])
        assert "Clarity" in prompt
        assert "Duplication" in prompt

    def test_includes_existing_playbook_titles(
        self, existing_playbooks: list[dict]
    ) -> None:
        prompt = build_review_system_prompt(existing_playbooks)
        assert "Retry on timeout" in prompt
        assert "Use WAL mode for SQLite" in prompt

    def test_empty_existing(self) -> None:
        prompt = build_review_system_prompt([])
        assert "Existing playbook entries" not in prompt

    def test_handles_list_tags(self) -> None:
        """Tags already parsed as list instead of JSON string."""
        entries = [{"title": "Entry", "tags_json": ["tag1", "tag2"]}]
        prompt = build_review_system_prompt(entries)
        assert "tag1, tag2" in prompt


# ---------------------------------------------------------------------------
# build_review_user_prompt
# ---------------------------------------------------------------------------


class TestBuildReviewUserPrompt:
    def test_includes_all_fields(self, sample_entry: PlaybookEntry) -> None:
        prompt = build_review_user_prompt(sample_entry)
        assert "Query OCR documents" in prompt
        assert "get_db_path()" in prompt
        assert "domain:database" in prompt
        assert "manual-ocr" in prompt


# ---------------------------------------------------------------------------
# apply_suggestions
# ---------------------------------------------------------------------------


class TestApplySuggestions:
    def test_no_suggestions(self, sample_entry: PlaybookEntry) -> None:
        review = PlaybookReviewResult(approved=True)
        result = apply_suggestions(sample_entry, review)
        assert result.title == sample_entry.title
        assert result.content == sample_entry.content
        assert result.tags == sample_entry.tags

    def test_title_suggestion(self, sample_entry: PlaybookEntry) -> None:
        review = PlaybookReviewResult(approved=True, suggested_title="Better title")
        result = apply_suggestions(sample_entry, review)
        assert result.title == "Better title"
        assert result.content == sample_entry.content

    def test_content_suggestion(self, sample_entry: PlaybookEntry) -> None:
        review = PlaybookReviewResult(approved=True, suggested_content="Improved content")
        result = apply_suggestions(sample_entry, review)
        assert result.content == "Improved content"
        assert result.title == sample_entry.title

    def test_tags_merged(self, sample_entry: PlaybookEntry) -> None:
        review = PlaybookReviewResult(
            approved=True,
            suggested_tags=["domain:database", "pattern:query"],
        )
        result = apply_suggestions(sample_entry, review)
        assert "domain:database" in result.tags
        assert "pattern:query" in result.tags
        # No duplicates
        assert result.tags.count("domain:database") == 1

    def test_all_suggestions(self, sample_entry: PlaybookEntry) -> None:
        review = PlaybookReviewResult(
            approved=True,
            suggested_title="New title",
            suggested_content="New content",
            suggested_tags=["new-tag"],
        )
        result = apply_suggestions(sample_entry, review)
        assert result.title == "New title"
        assert result.content == "New content"
        assert "new-tag" in result.tags
        assert "domain:database" in result.tags

    def test_preserves_other_fields(self, sample_entry: PlaybookEntry) -> None:
        review = PlaybookReviewResult(approved=True, suggested_title="X")
        result = apply_suggestions(sample_entry, review)
        assert result.source_task_id == sample_entry.source_task_id
        assert result.source_workflow_id == sample_entry.source_workflow_id


# ---------------------------------------------------------------------------
# review_playbook_entry (integration with mocked provider)
# ---------------------------------------------------------------------------


class TestReviewPlaybookEntry:
    @pytest.mark.asyncio
    async def test_calls_provider_and_parses_result(
        self, sample_entry: PlaybookEntry
    ) -> None:
        mock_response = AsyncMock()
        mock_response.tool_input = {
            "approved": True,
            "rejection_reason": "",
            "suggested_tags": ["extra-tag"],
            "suggested_title": "",
            "suggested_content": "",
        }

        mock_provider = AsyncMock()
        mock_provider.build_request_params.return_value = {"mock": True}
        mock_provider.call.return_value = mock_response

        with patch(
            "forge.llm_providers.registry.get_provider",
            return_value=mock_provider,
        ):
            result = await review_playbook_entry(sample_entry, [])

        assert isinstance(result, PlaybookReviewResult)
        assert result.approved is True
        assert result.suggested_tags == ["extra-tag"]
        mock_provider.build_request_params.assert_called_once()
        mock_provider.call.assert_called_once()
