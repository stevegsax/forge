"""Tests for extraction activities and workflow."""

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

from pbook.activities.extraction import (
    build_extraction_system_prompt,
    build_extraction_user_prompt,
    execute_extraction_call,
    save_extracted_entries,
)
from pbook.llm import reset_provider
from pbook.models import PushExperienceInput
from pbook.store import (
    get_engine,
    list_recent_entries,
    run_migrations,
)
from pbook.worker import PBOOK_TASK_QUEUE
from pbook.workflows.extraction import ExtractionWorkflow

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
                project="forge", problem="Problem A", resolution="Fix A",
            ),
            PushExperienceInput(
                project="forge", problem="Problem B", resolution="Fix B",
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


# ---------------------------------------------------------------------------
# build_extraction_user_prompt
# ---------------------------------------------------------------------------


class TestBuildExtractionUserPrompt:
    def test_includes_quality_reminder(self):
        prompt = build_extraction_user_prompt()
        assert "unexpected" in prompt.lower()
        assert "empty list" in prompt.lower()


# ---------------------------------------------------------------------------
# execute_extraction_call
# ---------------------------------------------------------------------------


class TestExecuteExtractionCall:
    @pytest.mark.asyncio
    async def test_calls_provider(self):
        mock_response = ProviderResponse(
            tool_input={
                "entries": [
                    {
                        "title": "Strip base64 prefix",
                        "content": "Mistral OCR returns data URI prefix.",
                        "tags": ["ocr", "mistral"],
                    }
                ]
            },
            model_name="test-model",
            input_tokens=100,
            output_tokens=50,
            raw_response_json="{}",
        )

        provider = MagicMock()
        provider.build_request_params.return_value = {"mock": True}
        provider.call = AsyncMock(return_value=mock_response)

        result, in_tok, out_tok, latency = await execute_extraction_call(
            "system", "user", provider,
        )

        assert len(result.entries) == 1
        assert result.entries[0].title == "Strip base64 prefix"
        assert in_tok == 100
        assert out_tok == 50
        assert latency > 0

    @pytest.mark.asyncio
    async def test_empty_extraction(self):
        mock_response = ProviderResponse(
            tool_input={"entries": []},
            model_name="test",
            input_tokens=0,
            output_tokens=0,
            raw_response_json="{}",
        )
        provider = MagicMock()
        provider.build_request_params.return_value = {}
        provider.call = AsyncMock(return_value=mock_response)

        result, _, _, _ = await execute_extraction_call(
            "system", "user", provider,
        )
        assert result.entries == []


# ---------------------------------------------------------------------------
# save_extracted_entries activity
# ---------------------------------------------------------------------------


class TestSaveExtractedEntries:
    @pytest.mark.asyncio
    async def test_saves_with_needs_review(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        input_data = json.dumps({
            "entries": [
                {"title": "Test", "content": "Content", "tags": ["lang:python"]},
            ],
            "project": "forge",
        })

        count = await save_extracted_entries(input_data)
        assert count == 1

        engine = get_engine(tmp_path / "test.db")
        entries = list_recent_entries(engine)
        assert len(entries) == 1
        assert entries[0]["needs_review"] is True
        assert entries[0]["entry_type"] == "pitfall"
        assert entries[0]["source_project"] == "forge"

    @pytest.mark.asyncio
    async def test_empty_entries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        count = await save_extracted_entries(json.dumps({"entries": [], "project": ""}))
        assert count == 0


# ---------------------------------------------------------------------------
# ExtractionWorkflow
# ---------------------------------------------------------------------------


class TestExtractionWorkflow:
    @pytest.mark.asyncio
    async def test_extraction_creates_entries(
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        @activity.defn(name="extract_from_experience")
        async def mock_extract(input_json: str) -> str:
            return json.dumps({
                "entries": [
                    {
                        "title": "Mock lesson",
                        "content": "Mock content",
                        "tags": ["lang:python"],
                    }
                ]
            })

        @activity.defn(name="compute_embedding")
        async def mock_compute_embedding(text: str) -> str:
            return base64.b64encode(b"fake-embedding").decode("ascii")

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ExtractionWorkflow],
            activities=[mock_extract, save_extracted_entries, mock_compute_embedding],
        ):
            result = await env.client.execute_workflow(
                ExtractionWorkflow.run,
                json.dumps({
                    "experiences": [
                        {
                            "project": "forge",
                            "problem": "test problem",
                            "resolution": "test resolution",
                        }
                    ],
                    "project": "forge",
                }),
                id="test-extraction-1",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result["entries_created"] == 1
        
        # Verify embedding was saved
        engine = get_engine(tmp_path / "test.db")
        entries = list_recent_entries(engine)
        assert entries[0]["embedding"] == b"fake-embedding"

    @pytest.mark.asyncio
    async def test_extraction_empty_experiences(
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))

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
        self, env: WorkflowEnvironment, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))

        @activity.defn(name="extract_from_experience")
        async def mock_extract_empty(input_json: str) -> str:
            return json.dumps({"entries": []})

        @activity.defn(name="compute_embedding")
        async def mock_compute_embedding(text: str) -> str:
            return ""

        async with Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=[ExtractionWorkflow],
            activities=[mock_extract_empty, mock_compute_embedding],
        ):
            result = await env.client.execute_workflow(
                ExtractionWorkflow.run,
                json.dumps({
                    "experiences": [
                        {
                            "project": "forge",
                            "problem": "normal thing",
                            "resolution": "normal fix",
                        }
                    ],
                    "project": "forge",
                }),
                id="test-extraction-nothing",
                task_queue=PBOOK_TASK_QUEUE,
            )

        assert result["entries_created"] == 0
