"""Tests for review activities and workflow."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from temporalio import activity
from temporalio.worker import Worker

from pbook.activities.review import validate_entry
from pbook.llm import ReviewResult
from pbook.models import PlaybookEntry
from pbook.prompts.review import (
    apply_suggestions,
    build_review_system_prompt,
    build_review_user_prompt,
)
from pbook.roots import StoreActivities
from pbook.store import build_entry_dict, save_entries
from pbook.worker import PBOOK_TASK_QUEUE
from pbook.workflow_steps import LLMChatResult
from pbook.workflows.manual_entry import ManualEntryWorkflow
from tests.conftest import encode_test_embedding, setup_db

if TYPE_CHECKING:
    from pathlib import Path

    from temporalio.testing import WorkflowEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _setup_db(_tmp_path: Path | None = None):
    """Return the session test engine (migrations already applied)."""
    return setup_db()[0]


def _manual_entry_activities(engine, mock_chat, mock_embed, find_duplicates):
    """ManualEntryWorkflow's activities: two engine-bound store methods, the
    no-dep validate_entry free function, plus the LLM/embed/dup stubs."""
    store = StoreActivities(engine)
    return [
        validate_entry,
        store.fetch_existing_entries,
        mock_chat,
        store.save_extracted_entries,
        mock_embed,
        find_duplicates,
    ]


# ---------------------------------------------------------------------------
# build_review_system_prompt
# ---------------------------------------------------------------------------


class TestBuildReviewSystemPrompt:
    def test_includes_quality_bar(self):
        prompt = build_review_system_prompt([])
        assert "BETTER TO REJECT" in prompt
        assert "MINIMAL" in prompt
        assert "ACCURATE" in prompt

    def test_includes_existing_entries(self):
        existing = [
            {"title": "Existing tip", "tags": ["lang:python"]},
        ]
        prompt = build_review_system_prompt(existing)
        assert "Existing tip" in prompt
        assert "duplication" in prompt.lower()


# ---------------------------------------------------------------------------
# build_review_user_prompt
# ---------------------------------------------------------------------------


class TestBuildReviewUserPrompt:
    def test_formats_entry(self):
        entry = PlaybookEntry(
            title="Test",
            content="Content",
            tags=["lang:python"],
            source_project="forge",
        )
        prompt = build_review_user_prompt(entry)
        assert "Test" in prompt
        assert "Content" in prompt
        assert "lang:python" in prompt
        assert "forge" in prompt


# ---------------------------------------------------------------------------
# apply_suggestions
# ---------------------------------------------------------------------------


class TestApplySuggestions:
    def test_applies_title(self):
        entry = PlaybookEntry(title="Old", content="c", tags=["lang:python"])
        review = ReviewResult(
            approved=True,
            suggested_title="New Title",
        )
        result = apply_suggestions(entry, review)
        assert result.title == "New Title"
        assert result.content == "c"  # Unchanged

    def test_applies_content(self):
        entry = PlaybookEntry(title="T", content="old", tags=["lang:python"])
        review = ReviewResult(
            approved=True,
            suggested_content="new content",
        )
        result = apply_suggestions(entry, review)
        assert result.content == "new content"

    def test_merges_tags(self):
        entry = PlaybookEntry(title="T", content="c", tags=["lang:python"])
        review = ReviewResult(
            approved=True,
            suggested_tags=["lib:sqlalchemy"],
        )
        result = apply_suggestions(entry, review)
        assert "lang:python" in result.tags
        assert "lib:sqlalchemy" in result.tags

    def test_no_suggestions_keeps_original(self):
        entry = PlaybookEntry(title="T", content="c", tags=["lang:python"])
        review = ReviewResult(approved=True)
        result = apply_suggestions(entry, review)
        assert result == entry


# ---------------------------------------------------------------------------
# validate_entry activity
# ---------------------------------------------------------------------------


class TestValidateEntry:
    @pytest.mark.asyncio
    async def test_valid_entry(self):
        raw = json.dumps({"title": "T", "content": "C", "tags": ["lang:python"]})
        result = json.loads(await validate_entry(raw))
        assert result["valid"] is True
        assert result["entry"]["title"] == "T"

    @pytest.mark.asyncio
    async def test_invalid_entry(self):
        result = json.loads(await validate_entry("not json"))
        assert result["valid"] is False
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# fetch_existing_entries activity
# ---------------------------------------------------------------------------


class TestFetchExistingEntries:
    @pytest.mark.asyncio
    async def test_returns_recent_entries(self, tmp_path):
        engine = _setup_db(tmp_path)

        entry = PlaybookEntry(title="Existing", content="Content", tags=["lang:python"])
        save_entries(engine, [build_entry_dict(entry)])

        result = await StoreActivities(engine).fetch_existing_entries(50)
        assert len(result) == 1
        assert result[0]["title"] == "Existing"
        # Embeddings are stripped before crossing the wire.
        assert "embedding" not in result[0]

    @pytest.mark.asyncio
    async def test_returns_empty_when_store_empty(self):
        result = await StoreActivities(_setup_db()).fetch_existing_entries(50)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self):
        result = await StoreActivities(None).fetch_existing_entries(50)
        assert result == []


# ---------------------------------------------------------------------------
# ManualEntryWorkflow
# ---------------------------------------------------------------------------


def _make_chat_stub(tool_input: dict):
    @activity.defn(name="llm_chat")
    async def _llm_chat(_input) -> LLMChatResult:
        return LLMChatResult(
            tool_input=tool_input,
            model_name="anthropic:test",
            input_tokens=0,
            output_tokens=0,
            latency_ms=1.0,
        )

    return _llm_chat


def _make_embed_stub(value: str):
    @activity.defn(name="llm_embed")
    async def _llm_embed(_text: str) -> str:
        return value

    return _llm_embed


def _make_find_duplicates_stub():
    @activity.defn(name="find_duplicates")
    async def _find_duplicates(_input_json: str) -> list:
        return []

    return _find_duplicates


class TestManualEntryWorkflow:
    @pytest.mark.asyncio
    async def test_approved_entry(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
    ) -> None:
        engine = _setup_db(tmp_path)

        mock_chat = _make_chat_stub(
            {
                "approved": True,
                "rejection_reason": "",
                "suggested_title": "",
                "suggested_content": "",
                "suggested_tags": [],
            }
        )
        mock_embed = _make_embed_stub(encode_test_embedding(0.1))

        raw_json = json.dumps(
            {
                "title": "Good entry",
                "content": "Useful advice",
                "tags": ["lang:python"],
            }
        )

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ManualEntryWorkflow],
            activities=_manual_entry_activities(
                engine, mock_chat, mock_embed, _make_find_duplicates_stub()
            ),
        ):
            result = await env.client.execute_workflow(
                ManualEntryWorkflow.run,
                raw_json,
                id="test-manual-approved",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result["approved"] is True
        assert result["entries_saved"] == 1

    @pytest.mark.asyncio
    async def test_rejected_entry(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
    ) -> None:
        engine = _setup_db(tmp_path)

        mock_chat = _make_chat_stub(
            {
                "approved": False,
                "rejection_reason": "Too generic",
                "suggested_title": "",
                "suggested_content": "",
                "suggested_tags": [],
            }
        )
        mock_embed = _make_embed_stub(encode_test_embedding(0.1))

        raw_json = json.dumps(
            {
                "title": "Bad entry",
                "content": "Write clean code",
                "tags": ["lang:python"],
            }
        )

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ManualEntryWorkflow],
            activities=_manual_entry_activities(
                engine, mock_chat, mock_embed, _make_find_duplicates_stub()
            ),
        ):
            result = await env.client.execute_workflow(
                ManualEntryWorkflow.run,
                raw_json,
                id="test-manual-rejected",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result["approved"] is False
        assert result["rejection_reason"] == "Too generic"

    @pytest.mark.asyncio
    async def test_invalid_json(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
        monkeypatch,
    ) -> None:

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ManualEntryWorkflow],
            activities=[validate_entry],
        ):
            result = await env.client.execute_workflow(
                ManualEntryWorkflow.run,
                "not valid json",
                id="test-manual-invalid",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result["approved"] is False
        assert "validation_error" in result
