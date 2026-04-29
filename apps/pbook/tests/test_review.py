"""Tests for review activities and workflow."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from sax_llm.models import ProviderResponse
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from pbook.activities.extraction import save_extracted_entries
from pbook.activities.review import (
    execute_review_call,
    fetch_existing_entries,
    review_entry,
    validate_entry,
)
from pbook.llm import ReviewResult, reset_provider
from pbook.models import PlaybookEntry
from pbook.prompts.review import (
    apply_suggestions,
    build_review_system_prompt,
    build_review_user_prompt,
)
from pbook.store import (
    build_entry_dict,
    get_engine,
    run_migrations,
    save_entries,
)
from pbook.worker import PBOOK_TASK_QUEUE
from pbook.workflows.manual_entry import ManualEntryWorkflow

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


@pytest.fixture(autouse=True)
def _cleanup_provider():
    yield
    reset_provider()


def _setup_db(tmp_path: Path):
    db_path = tmp_path / "test.db"
    run_migrations(db_path)
    return get_engine(db_path)


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
            {"title": "Existing tip", "tags_json": '["lang:python"]'},
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
# execute_review_call
# ---------------------------------------------------------------------------


class TestExecuteReviewCall:
    @pytest.mark.asyncio
    async def test_calls_provider(self):
        mock_response = ProviderResponse(
            tool_input={
                "approved": True,
                "rejection_reason": "",
                "suggested_title": "",
                "suggested_content": "",
                "suggested_tags": [],
            },
            model_name="test",
            input_tokens=0,
            output_tokens=0,
            raw_response_json="{}",
        )
        provider = MagicMock()
        provider.build_request_params.return_value = {}
        provider.call = AsyncMock(return_value=mock_response)

        result = await execute_review_call("system", "user", provider)
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_rejection(self):
        mock_response = ProviderResponse(
            tool_input={
                "approved": False,
                "rejection_reason": "Too generic",
                "suggested_title": "",
                "suggested_content": "",
                "suggested_tags": [],
            },
            model_name="test",
            input_tokens=0,
            output_tokens=0,
            raw_response_json="{}",
        )
        provider = MagicMock()
        provider.build_request_params.return_value = {}
        provider.call = AsyncMock(return_value=mock_response)

        result = await execute_review_call("system", "user", provider)
        assert result.approved is False
        assert result.rejection_reason == "Too generic"


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
    async def test_returns_recent_entries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)

        entry = PlaybookEntry(title="Existing", content="Content", tags=["lang:python"])
        save_entries(engine, [build_entry_dict(entry)])

        result = await fetch_existing_entries(50)
        assert len(result) == 1
        assert result[0]["title"] == "Existing"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "nonexistent.db"))
        result = await fetch_existing_entries(50)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", "")
        result = await fetch_existing_entries(50)
        assert result == []


# ---------------------------------------------------------------------------
# review_entry activity
# ---------------------------------------------------------------------------


class TestReviewEntryActivity:
    @pytest.mark.asyncio
    async def test_approved_entry(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        from pbook.llm import set_provider

        mock_response = ProviderResponse(
            tool_input={
                "approved": True,
                "rejection_reason": "",
                "suggested_title": "Better Title",
                "suggested_content": "",
                "suggested_tags": [],
            },
            model_name="test",
            input_tokens=0,
            output_tokens=0,
            raw_response_json="{}",
        )
        provider = MagicMock()
        provider.build_request_params.return_value = {}
        provider.call = AsyncMock(return_value=mock_response)
        set_provider(provider)

        input_json = json.dumps({
            "entry": {
                "title": "Original",
                "content": "Some advice",
                "tags": ["lang:python"],
            },
            "existing_entries": [],
        })

        result = json.loads(await review_entry(input_json))
        assert result["approved"] is True
        assert result["final_entry"]["title"] == "Better Title"

    @pytest.mark.asyncio
    async def test_rejected_entry(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        from pbook.llm import set_provider

        mock_response = ProviderResponse(
            tool_input={
                "approved": False,
                "rejection_reason": "Too vague",
                "suggested_title": "",
                "suggested_content": "",
                "suggested_tags": [],
            },
            model_name="test",
            input_tokens=0,
            output_tokens=0,
            raw_response_json="{}",
        )
        provider = MagicMock()
        provider.build_request_params.return_value = {}
        provider.call = AsyncMock(return_value=mock_response)
        set_provider(provider)

        input_json = json.dumps({
            "entry": {
                "title": "Bad",
                "content": "Write good code",
                "tags": ["lang:python"],
            },
            "existing_entries": [],
        })

        result = json.loads(await review_entry(input_json))
        assert result["approved"] is False
        assert result["rejection_reason"] == "Too vague"


# ---------------------------------------------------------------------------
# ManualEntryWorkflow
# ---------------------------------------------------------------------------


class TestManualEntryWorkflow:
    @pytest.mark.asyncio
    async def test_approved_entry(
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        @activity.defn(name="review_entry")
        async def mock_review(input_json: str) -> str:
            data = json.loads(input_json)
            return json.dumps({
                "approved": True,
                "rejection_reason": "",
                "final_entry": data["entry"],
            })

        @activity.defn(name="compute_embedding")
        async def mock_compute_embedding(text: str) -> str:
            return base64.b64encode(b"fake-embedding").decode("ascii")

        @activity.defn(name="find_duplicates")
        async def mock_find_duplicates(input_json: str) -> list:
            return []

        raw_json = json.dumps({
            "title": "Good entry",
            "content": "Useful advice",
            "tags": ["lang:python"],
        })

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ManualEntryWorkflow],
            activities=[
                validate_entry,
                fetch_existing_entries,
                mock_review,
                save_extracted_entries,
                mock_compute_embedding,
                mock_find_duplicates,
            ],
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
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        @activity.defn(name="review_entry")
        async def mock_reject(input_json: str) -> str:
            return json.dumps({
                "approved": False,
                "rejection_reason": "Too generic",
                "final_entry": {},
            })

        @activity.defn(name="compute_embedding")
        async def mock_compute_embedding(text: str) -> str:
            return base64.b64encode(b"fake-embedding").decode("ascii")

        @activity.defn(name="find_duplicates")
        async def mock_find_duplicates(input_json: str) -> list:
            return []

        raw_json = json.dumps({
            "title": "Bad entry",
            "content": "Write clean code",
            "tags": ["lang:python"],
        })

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ManualEntryWorkflow],
            activities=[
                validate_entry,
                fetch_existing_entries,
                mock_reject,
                save_extracted_entries,
                mock_compute_embedding,
                mock_find_duplicates,
            ],
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
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))

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
