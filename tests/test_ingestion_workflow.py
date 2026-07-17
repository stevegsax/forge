"""Integration tests for TranscriptIngestionWorkflow and BatchIngestionWorkflow.

These tests use the Temporal time-skipping test server and mock activities
to exercise the full workflow orchestration, including:

- The prepare_transcript activity (mocked to return canned data)
- The batch submit/wait/parse cycle (driven by signals)
- Cross-queue activities and child workflows on PBOOK_TASK_QUEUE
  (registered on a second in-test worker with mock activities/workflows)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from temporalio import activity, workflow
from temporalio.worker import Worker

from forge.ingestion_workflow import (
    PBOOK_TASK_QUEUE,
    BatchIngestionWorkflow,
    TranscriptIngestionWorkflow,
)
from forge.models import (
    BatchResult,
    BatchSubmitInput,
    BatchSubmitResult,
    ParsedLLMResponse,
    ParseResponseInput,
    ThinkingPolicy,
)
from forge.persist_models import PersistRequest, PersistResult
from forge.workflows import FORGE_TASK_QUEUE

if TYPE_CHECKING:
    from temporalio.testing import WorkflowEnvironment


# ---------------------------------------------------------------------------
# Shared mock state
# ---------------------------------------------------------------------------

_CALL_LOG: list[str] = []

# Canned prepare_transcript output — tests mutate this before running.
_PREPARED_JSON: str = ""

# Canned analysis JSON returned from parse_llm_response.
_ANALYSIS_JSON: str = ""

# Count of sessions recorded as ingested (cross-queue).
_RECORDED_SESSIONS: list[dict] = []

# Canned extraction result returned from mock pbook ExtractionWorkflow.
_EXTRACTION_RESULT: dict = {"entries_created": 0}

# Captured submit_batch_request inputs — regression coverage for the shared
# thinking fallback: transcript analysis omits `thinking` entirely and must
# land disabled via workflow_blocks.batch_submit_and_wait's shared fallback.
_SUBMIT_BATCH_INPUTS: list[BatchSubmitInput] = []


def _reset_state(
    *,
    prepared: dict | None = None,
    analysis: dict | None = None,
    extraction_result: dict | None = None,
) -> None:
    """Reset module-level state between tests."""
    global _PREPARED_JSON, _ANALYSIS_JSON, _EXTRACTION_RESULT
    _CALL_LOG.clear()
    _RECORDED_SESSIONS.clear()
    _SUBMIT_BATCH_INPUTS.clear()
    # Empty string activates the echo-input fallback in mock_prepare_transcript.
    _PREPARED_JSON = json.dumps(prepared) if prepared is not None else ""
    _ANALYSIS_JSON = json.dumps(analysis or {"experiences": []})
    _EXTRACTION_RESULT = extraction_result or {"entries_created": 0}


# ---------------------------------------------------------------------------
# Forge-side mock activities (registered on FORGE_TASK_QUEUE)
# ---------------------------------------------------------------------------


@activity.defn(name="prepare_transcript")
async def mock_prepare_transcript(input_json: str) -> str:
    _CALL_LOG.append("prepare_transcript")
    if _PREPARED_JSON:
        return _PREPARED_JSON
    # Default: empty-transcript early-exit, but echo the caller's session_id
    # so per-session aggregation in fan-out tests can be verified.
    payload = json.loads(input_json)
    return json.dumps(
        {
            "transcript_text": "",
            "system_prompt": "",
            "user_prompt": "",
            "project": payload.get("project", ""),
            "session_id": payload.get("session_id", ""),
            "message_count": 0,
            "char_count": 0,
        }
    )


@activity.defn(name="submit_batch_request")
async def mock_submit_batch(input: BatchSubmitInput) -> BatchSubmitResult:
    _CALL_LOG.append(f"submit_batch:{input.output_type_name}")
    _SUBMIT_BATCH_INPUTS.append(input)
    return BatchSubmitResult(request_id="req-ingest-1", batch_id="msgbatch_ingest1")


@activity.defn(name="parse_llm_response")
async def mock_parse_response(input: ParseResponseInput) -> ParsedLLMResponse:
    _CALL_LOG.append(f"parse:{input.output_type_name}")
    return ParsedLLMResponse(
        parsed_json=_ANALYSIS_JSON,
        model_name="mock-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=1.0,
    )


@activity.defn(name="persist_to_store")
async def mock_persist_to_store(req: PersistRequest) -> PersistResult:
    """No-op survivable-write mock (batch path now persists the submission)."""
    return PersistResult(kind=req.kind, applied=True)


_FORGE_MOCK_ACTIVITIES = [
    mock_prepare_transcript,
    mock_submit_batch,
    mock_parse_response,
    mock_persist_to_store,
]


# ---------------------------------------------------------------------------
# pbook-side mock activities and workflow (registered on PBOOK_TASK_QUEUE)
# ---------------------------------------------------------------------------


@activity.defn(name="record_ingested_session")
async def mock_record_ingested_session(input_json: str) -> None:
    _CALL_LOG.append("record_ingested_session")
    _RECORDED_SESSIONS.append(json.loads(input_json))


@activity.defn(name="record_ingested_session_error")
async def mock_record_ingested_session_error(input_json: str) -> None:
    _CALL_LOG.append("record_ingested_session_error")
    _RECORDED_SESSIONS.append(json.loads(input_json))


@activity.defn(name="mock_run_extraction")
async def mock_run_extraction(input_json: str) -> dict:
    """Activity stand-in for pbook's ExtractionWorkflow body.

    Workflow sandbox restricts access to mutable module globals, so the
    canned result and call logging must happen inside an activity.
    """
    payload = json.loads(input_json)
    _CALL_LOG.append(f"ExtractionWorkflow:{len(payload.get('experiences', []))}")
    return _EXTRACTION_RESULT


@workflow.defn(name="ExtractionWorkflow")
class MockExtractionWorkflow:
    """Stand-in for pbook's ExtractionWorkflow.

    Delegates to ``mock_run_extraction`` activity so the canned result
    can be controlled from test code via module-level globals.
    """

    @workflow.run
    async def run(self, input_json: str) -> dict:
        from datetime import timedelta

        return await workflow.execute_activity(
            "mock_run_extraction",
            input_json,
            start_to_close_timeout=timedelta(seconds=10),
            result_type=dict,
        )


_PBOOK_MOCK_ACTIVITIES = [
    mock_record_ingested_session,
    mock_record_ingested_session_error,
    mock_run_extraction,
]
_PBOOK_MOCK_WORKFLOWS = [MockExtractionWorkflow]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


async def _drive_transcript_workflow(
    env: WorkflowEnvironment,
    *,
    input_json: str,
    workflow_id: str,
    expect_batch_call: bool,
) -> dict:
    """Start TranscriptIngestionWorkflow, deliver batch signal if expected, return result."""
    async with (
        Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[TranscriptIngestionWorkflow],
            activities=_FORGE_MOCK_ACTIVITIES,
        ),
        Worker(
            env.client,
            task_queue=PBOOK_TASK_QUEUE,
            workflows=_PBOOK_MOCK_WORKFLOWS,
            activities=_PBOOK_MOCK_ACTIVITIES,
        ),
    ):
        handle = await env.client.start_workflow(
            TranscriptIngestionWorkflow.run,
            input_json,
            id=workflow_id,
            task_queue=FORGE_TASK_QUEUE,
        )
        if expect_batch_call:
            await handle.signal(
                TranscriptIngestionWorkflow.batch_result_received,
                BatchResult(
                    request_id="req-ingest-1",
                    batch_id="msgbatch_ingest1",
                    raw_response_json=_ANALYSIS_JSON,
                    result_type="TranscriptAnalysisResult",
                ),
            )
        return await handle.result()


# ---------------------------------------------------------------------------
# TranscriptIngestionWorkflow tests
# ---------------------------------------------------------------------------


class TestTranscriptIngestionEarlyExit:
    @pytest.mark.asyncio
    async def test_empty_transcript_returns_zero_counts(self, env: WorkflowEnvironment) -> None:
        _reset_state(
            prepared={
                "transcript_text": "",
                "system_prompt": "",
                "user_prompt": "",
                "project": "p",
                "session_id": "s1",
                "message_count": 0,
                "char_count": 0,
            }
        )

        result = await _drive_transcript_workflow(
            env,
            input_json=json.dumps({"path": "/tmp/fake", "session_id": "s1"}),
            workflow_id="test-ingest-empty",
            expect_batch_call=False,
        )

        assert result == {
            "experiences_found": 0,
            "entries_created": 0,
            "session_id": "s1",
        }
        assert "prepare_transcript" in _CALL_LOG
        # No batch call for empty transcript
        assert not any(c.startswith("submit_batch") for c in _CALL_LOG)

    @pytest.mark.asyncio
    async def test_too_few_messages_returns_early(self, env: WorkflowEnvironment) -> None:
        _reset_state(
            prepared={
                "transcript_text": "USER: hi",
                "system_prompt": "sp",
                "user_prompt": "up",
                "project": "p",
                "session_id": "s-small",
                "message_count": 2,  # Below 3-message threshold
                "char_count": 8,
            }
        )

        result = await _drive_transcript_workflow(
            env,
            input_json=json.dumps({"path": "/tmp/fake", "session_id": "s-small"}),
            workflow_id="test-ingest-small",
            expect_batch_call=False,
        )

        assert result["experiences_found"] == 0
        assert result["entries_created"] == 0
        assert result["session_id"] == "s-small"
        assert not any(c.startswith("submit_batch") for c in _CALL_LOG)


class TestTranscriptIngestionNoExperiences:
    @pytest.mark.asyncio
    async def test_llm_returns_no_experiences_records_session(
        self, env: WorkflowEnvironment
    ) -> None:
        _reset_state(
            prepared={
                "transcript_text": "USER: x\nASSISTANT: y\nUSER: z",
                "system_prompt": "sp",
                "user_prompt": "up",
                "project": "demo",
                "session_id": "s-empty",
                "message_count": 3,
                "char_count": 30,
            },
            analysis={"experiences": []},
        )

        result = await _drive_transcript_workflow(
            env,
            input_json=json.dumps({"path": "/tmp/fake", "session_id": "s-empty"}),
            workflow_id="test-ingest-no-exp",
            expect_batch_call=True,
        )

        assert result["experiences_found"] == 0
        assert result["entries_created"] == 0
        assert "submit_batch:TranscriptAnalysisResult" in _CALL_LOG
        assert "parse:TranscriptAnalysisResult" in _CALL_LOG
        # Session should be recorded with 0 counts
        assert "record_ingested_session" in _CALL_LOG
        # Transcript analysis omits `thinking`; the shared fallback in
        # batch_submit_and_wait must resolve it to disabled.
        assert len(_SUBMIT_BATCH_INPUTS) == 1
        assert _SUBMIT_BATCH_INPUTS[0].thinking == ThinkingPolicy(enabled=False)
        assert len(_RECORDED_SESSIONS) == 1
        recorded = _RECORDED_SESSIONS[0]
        assert recorded["session_id"] == "s-empty"
        assert recorded["experiences_found"] == 0
        assert recorded["entries_created"] == 0


class TestTranscriptIngestionHappyPath:
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, env: WorkflowEnvironment) -> None:
        _reset_state(
            prepared={
                "transcript_text": "USER: how\nASSISTANT: like this\nUSER: thanks",
                "system_prompt": "sp",
                "user_prompt": "up",
                "project": "myproj",
                "session_id": "s-happy",
                "message_count": 3,
                "char_count": 40,
            },
            analysis={
                "experiences": [
                    {
                        "problem": "test hangs",
                        "resolution": "reset the loop",
                        "context": "pytest-asyncio",
                    },
                    {
                        "problem": "import error",
                        "resolution": "install pbook",
                    },
                ]
            },
            extraction_result={"entries_created": 2},
        )

        result = await _drive_transcript_workflow(
            env,
            input_json=json.dumps({"path": "/tmp/fake", "session_id": "s-happy"}),
            workflow_id="test-ingest-happy",
            expect_batch_call=True,
        )

        assert result["experiences_found"] == 2
        assert result["entries_created"] == 2
        assert result["session_id"] == "s-happy"

        # Verify full activity sequence
        assert "prepare_transcript" in _CALL_LOG
        assert "submit_batch:TranscriptAnalysisResult" in _CALL_LOG
        assert "parse:TranscriptAnalysisResult" in _CALL_LOG
        assert "ExtractionWorkflow:2" in _CALL_LOG
        assert "record_ingested_session" in _CALL_LOG

        # Session recorded with correct counts
        assert len(_RECORDED_SESSIONS) == 1
        recorded = _RECORDED_SESSIONS[0]
        assert recorded["experiences_found"] == 2
        assert recorded["entries_created"] == 2
        assert recorded["project_name"] == "myproj"


class TestTranscriptIngestionMalformedResponse:
    @pytest.mark.asyncio
    async def test_malformed_json_returns_error_result(self, env: WorkflowEnvironment) -> None:
        """If the LLM returns unparseable JSON, the workflow must not crash.

        It returns an error marker and a 0-count result without calling
        the extraction pipeline.
        """
        global _ANALYSIS_JSON
        _reset_state(
            prepared={
                "transcript_text": "USER: x\nASSISTANT: y\nUSER: z",
                "system_prompt": "sp",
                "user_prompt": "up",
                "project": "p",
                "session_id": "s-bad",
                "message_count": 3,
                "char_count": 30,
            },
        )
        # Override analysis JSON with malformed content AFTER reset
        _ANALYSIS_JSON = "this is not json {{{"

        result = await _drive_transcript_workflow(
            env,
            input_json=json.dumps({"path": "/tmp/fake", "session_id": "s-bad"}),
            workflow_id="test-ingest-bad-json",
            expect_batch_call=True,
        )

        assert result["experiences_found"] == 0
        assert result["entries_created"] == 0
        assert result.get("error") == "malformed_llm_response"
        # Extraction workflow should NOT have been called
        assert not any(c.startswith("ExtractionWorkflow") for c in _CALL_LOG)
        # The success-path callback must NOT fire on a malformed response.
        assert "record_ingested_session" not in _CALL_LOG
        # The error-path callback must fire so `pbook sessions` can show
        # status=error instead of leaving the row stuck on `running`.
        assert "record_ingested_session_error" in _CALL_LOG
        assert len(_RECORDED_SESSIONS) == 1
        recorded = _RECORDED_SESSIONS[0]
        assert recorded["session_id"] == "s-bad"
        assert recorded["error_message"] == "malformed_llm_response"


# ---------------------------------------------------------------------------
# BatchIngestionWorkflow tests
# ---------------------------------------------------------------------------


class TestBatchIngestionWorkflow:
    @pytest.mark.asyncio
    async def test_empty_sessions_returns_zero_counts(self, env: WorkflowEnvironment) -> None:
        _reset_state()

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[BatchIngestionWorkflow, TranscriptIngestionWorkflow],
            activities=_FORGE_MOCK_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                BatchIngestionWorkflow.run,
                json.dumps({"sessions": []}),
                id="test-batch-ingest-empty",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result == {
            "sessions_processed": 0,
            "total_experiences": 0,
            "total_entries_created": 0,
            "per_session": [],
        }

    @pytest.mark.asyncio
    async def test_fan_out_aggregates_child_results(self, env: WorkflowEnvironment) -> None:
        """BatchIngestionWorkflow fans out to TranscriptIngestionWorkflow children.

        We use the early-exit path (empty transcript) in the child so no
        batch signals are needed — every child returns zero counts but
        the parent should still correctly aggregate them.
        """
        _reset_state(
            prepared={
                "transcript_text": "",
                "system_prompt": "",
                "user_prompt": "",
                "project": "p",
                "session_id": "",
                "message_count": 0,
                "char_count": 0,
            }
        )

        sessions = [
            {"path": "/tmp/a", "project": "p", "session_id": "s-a"},
            {"path": "/tmp/b", "project": "p", "session_id": "s-b"},
            {"path": "/tmp/c", "project": "p", "session_id": "s-c"},
        ]

        async with (
            Worker(
                env.client,
                task_queue=FORGE_TASK_QUEUE,
                workflows=[BatchIngestionWorkflow, TranscriptIngestionWorkflow],
                activities=_FORGE_MOCK_ACTIVITIES,
            ),
            Worker(
                env.client,
                task_queue=PBOOK_TASK_QUEUE,
                workflows=_PBOOK_MOCK_WORKFLOWS,
                activities=_PBOOK_MOCK_ACTIVITIES,
            ),
        ):
            result = await env.client.execute_workflow(
                BatchIngestionWorkflow.run,
                json.dumps({"sessions": sessions}),
                id="test-batch-ingest-fanout",
                task_queue=FORGE_TASK_QUEUE,
            )

        assert result["sessions_processed"] == 3
        assert result["total_experiences"] == 0
        assert result["total_entries_created"] == 0
        assert len(result["per_session"]) == 3
        # All three children should have run prepare_transcript
        assert _CALL_LOG.count("prepare_transcript") == 3

    @pytest.mark.asyncio
    async def test_parent_awaits_children_before_returning(self, env: WorkflowEnvironment) -> None:
        """Regression: BatchIngestionWorkflow must await child completion.

        ChildWorkflowHandle subclasses asyncio.Task, so calling `.result()`
        on it invokes the synchronous Task.result() and raises
        InvalidStateError("Result is not set.") before the child finishes.
        The parent's except-handler would then swallow the exception,
        return zeros with `session_id=""`, complete, and Temporal would
        terminate the still-running children "by parent close policy".

        This test pins the parent down to actually waiting for each child
        to complete: per-session results must carry the input session_id
        and must NOT have an `error` key.
        """
        # Echo path: mock_prepare_transcript reflects each input session_id
        # back as the prepared session_id, taking the empty-transcript
        # early-exit so no batch signals are required.
        _reset_state()

        sessions = [
            {"path": "/tmp/a", "project": "p", "session_id": "s-alpha"},
            {"path": "/tmp/b", "project": "p", "session_id": "s-bravo"},
        ]

        async with Worker(
            env.client,
            task_queue=FORGE_TASK_QUEUE,
            workflows=[BatchIngestionWorkflow, TranscriptIngestionWorkflow],
            activities=_FORGE_MOCK_ACTIVITIES,
        ):
            result = await env.client.execute_workflow(
                BatchIngestionWorkflow.run,
                json.dumps({"sessions": sessions}),
                id="test-batch-awaits-children",
                task_queue=FORGE_TASK_QUEUE,
            )

        per_session = result["per_session"]
        assert len(per_session) == 2
        # Children completed normally — no swallowed exceptions.
        assert all("error" not in r for r in per_session), per_session
        # Each per_session entry carries the child's actual session_id,
        # which is only populated on the success path.
        returned_ids = {r["session_id"] for r in per_session}
        assert returned_ids == {"s-alpha", "s-bravo"}
