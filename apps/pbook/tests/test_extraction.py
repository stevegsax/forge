"""Tests for extraction activities and workflow."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from temporalio import activity
from temporalio.worker import Worker

from pbook.models import PushExperienceInput
from pbook.prompts.extraction import (
    build_extraction_system_prompt,
    build_extraction_user_prompt,
)
from pbook.roots import StoreActivities
from pbook.store import list_recent_entries
from pbook.worker import PBOOK_TASK_QUEUE
from pbook.workflow_steps import LLMChatResult
from pbook.workflows.extraction import ExtractionWorkflow
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


# ---------------------------------------------------------------------------
# build_extraction_system_prompt
# ---------------------------------------------------------------------------


class TestBuildExtractionSystemPrompt:
    def test_includes_quality_bar(self):
        exp = PushExperienceInput(
            project="forge",
            problem="Base64 prefix in OCR response",
            resolution="Strip data URI prefix before decoding",
        )
        prompt = build_extraction_system_prompt([exp])

        assert "UNEXPECTED" in prompt
        assert "ACTIONABLE" in prompt
        assert "better to extract nothing" in prompt.lower()

    def test_includes_experience_data(self):
        exp = PushExperienceInput(
            project="forge",
            problem="Connection timeout",
            resolution="Increase pool size",
            context="SQLAlchemy with SQLite",
        )
        prompt = build_extraction_system_prompt([exp])

        assert "forge" in prompt
        assert "Connection timeout" in prompt
        assert "Increase pool size" in prompt
        assert "SQLAlchemy with SQLite" in prompt

    def test_multiple_experiences(self):
        exps = [
            PushExperienceInput(
                project="forge",
                problem="Problem A",
                resolution="Fix A",
            ),
            PushExperienceInput(
                project="forge",
                problem="Problem B",
                resolution="Fix B",
            ),
        ]
        prompt = build_extraction_system_prompt(exps)

        assert "Problem A" in prompt
        assert "Problem B" in prompt

    def test_includes_metadata(self):
        exp = PushExperienceInput(
            project="forge",
            problem="Connection error",
            resolution="Retry with backoff",
            metadata={"environment": "production", "retry_count": 3},
        )
        prompt = build_extraction_system_prompt([exp])

        assert "environment" in prompt
        assert "production" in prompt

    def test_warns_against_red_herring_lessons(self):
        """The prompt must instruct the LLM to verify candidates against
        the Resolution — symptoms and dismissed hypotheses in the Problem
        text are common over-extraction traps (see #258/259/260 cluster
        where MISTRAL_API_KEY appeared in the Problem but was explicitly
        NOT the cause)."""
        exp = PushExperienceInput(
            project="forge",
            problem="x",
            resolution="y",
        )
        prompt = build_extraction_system_prompt([exp])
        assert "Resolution" in prompt
        assert "red herring" in prompt.lower()

    def test_warns_against_multiple_entries_per_experience(self):
        """Multiple entries from one experience must be rare; if produced,
        they should reflect distinct root causes, not different framings
        of the same lesson."""
        exp = PushExperienceInput(
            project="forge",
            problem="x",
            resolution="y",
        )
        prompt = build_extraction_system_prompt([exp])
        assert "0 or 1 entries" in prompt
        assert "root cause" in prompt.lower()


# ---------------------------------------------------------------------------
# build_extraction_user_prompt
# ---------------------------------------------------------------------------


class TestBuildExtractionUserPrompt:
    def test_includes_quality_reminder(self):
        prompt = build_extraction_user_prompt()
        assert "unexpected" in prompt.lower()
        assert "empty list" in prompt.lower()


# ---------------------------------------------------------------------------
# save_extracted_entries activity
# ---------------------------------------------------------------------------


class TestSaveExtractedEntries:
    @pytest.mark.asyncio
    async def test_saves_with_needs_review(self, tmp_path):
        engine = _setup_db(tmp_path)

        input_data = json.dumps(
            {
                "entries": [
                    {"title": "Test", "content": "Content", "tags": ["lang:python"]},
                ],
                "project": "forge",
            }
        )

        count = await StoreActivities(engine).save_extracted_entries(input_data)
        assert count == 1

        entries = list_recent_entries(engine)
        assert len(entries) == 1
        assert entries[0]["needs_review"] is True
        assert entries[0]["entry_type"] == "pitfall"
        assert entries[0]["source_project"] == "forge"

    @pytest.mark.asyncio
    async def test_empty_entries(self):
        count = await StoreActivities(None).save_extracted_entries(
            json.dumps({"entries": [], "project": ""})
        )
        assert count == 0


# ---------------------------------------------------------------------------
# ExtractionWorkflow
# ---------------------------------------------------------------------------


def _make_chat_stub(tool_input: dict):
    """Build an @activity.defn(name='llm_chat') closure returning canned tool_input."""

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


class TestExtractionWorkflow:
    @pytest.mark.asyncio
    async def test_extraction_creates_entries(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
    ) -> None:
        engine = _setup_db(tmp_path)

        mock_chat = _make_chat_stub(
            {
                "entries": [
                    {
                        "title": "Mock lesson",
                        "content": "Mock content",
                        "tags": ["lang:python"],
                    },
                ],
            }
        )
        mock_embed = _make_embed_stub(encode_test_embedding(0.5))

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ExtractionWorkflow],
            activities=[mock_chat, mock_embed, StoreActivities(engine).save_extracted_entries],
        ):
            result = await env.client.execute_workflow(
                ExtractionWorkflow.run,
                json.dumps(
                    {
                        "experiences": [
                            {
                                "project": "forge",
                                "problem": "test problem",
                                "resolution": "test resolution",
                            },
                        ],
                        "project": "forge",
                    }
                ),
                id="test-extraction-1",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result["entries_created"] == 1

        # Verify embedding was saved (1536-dim, leading value preserved).
        entries = list_recent_entries(engine)
        assert len(entries[0]["embedding"]) == 1536
        assert entries[0]["embedding"][0] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_extraction_empty_experiences(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
        monkeypatch,
    ) -> None:

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ExtractionWorkflow],
            activities=[],
        ):
            result = await env.client.execute_workflow(
                ExtractionWorkflow.run,
                json.dumps({"experiences": [], "project": ""}),
                id="test-extraction-empty",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result["entries_created"] == 0

    @pytest.mark.asyncio
    async def test_extraction_no_entries_extracted(
        self,
        env: WorkflowEnvironment,
        tmp_path: Path,
        monkeypatch,
    ) -> None:

        mock_chat = _make_chat_stub({"entries": []})
        mock_embed = _make_embed_stub("")

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ExtractionWorkflow],
            activities=[mock_chat, mock_embed],
        ):
            result = await env.client.execute_workflow(
                ExtractionWorkflow.run,
                json.dumps(
                    {
                        "experiences": [
                            {
                                "project": "forge",
                                "problem": "normal thing",
                                "resolution": "normal fix",
                            }
                        ],
                        "project": "forge",
                    }
                ),
                id="test-extraction-nothing",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result["entries_created"] == 0


# ---------------------------------------------------------------------------
# Session lifecycle activities
# ---------------------------------------------------------------------------


class TestRecordIngestedSessionActivities:
    """Direct unit tests for the cross-queue lifecycle callbacks.

    The activities are thin wrappers around the store helpers. They run
    against the session's isolated test database (see conftest), never the
    developer's real store.
    """

    @pytest.mark.asyncio
    async def test_completion_activity_writes_completed_row(self):
        engine = setup_db()[0]
        await StoreActivities(engine).record_ingested_session(
            json.dumps(
                {
                    "session_id": "s-good",
                    "project_name": "alpha",
                    "experiences_found": 4,
                    "entries_created": 3,
                }
            )
        )

        from pbook.store import list_ingested_sessions

        rows = list_ingested_sessions(engine)
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s-good"
        assert rows[0]["status"] == "completed"
        assert rows[0]["experiences_found"] == 4
        assert rows[0]["entries_created"] == 3

    @pytest.mark.asyncio
    async def test_error_activity_writes_error_row(self):
        engine = setup_db()[0]
        await StoreActivities(engine).record_ingested_session_error(
            json.dumps(
                {
                    "session_id": "s-bad",
                    "project_name": "alpha",
                    "error_message": "malformed_llm_response",
                }
            )
        )

        from pbook.store import list_ingested_sessions

        rows = list_ingested_sessions(engine)
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s-bad"
        assert rows[0]["status"] == "error"
        assert rows[0]["error_message"] == "malformed_llm_response"

    @pytest.mark.asyncio
    async def test_disabled_db_is_a_noop(self):
        # With the store disabled (engine None), the callbacks no-op.
        store = StoreActivities(None)
        await store.record_ingested_session(json.dumps({"session_id": "s-x"}))
        await store.record_ingested_session_error(
            json.dumps({"session_id": "s-x", "error_message": "boom"})
        )
