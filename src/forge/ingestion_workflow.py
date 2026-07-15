"""Temporal workflows for ingesting Claude Code transcripts.

Orchestrates transcript analysis via forge's batch API, then feeds
identified experiences into pbook's extraction pipeline cross-queue.

TranscriptIngestionWorkflow: processes a single session
BatchIngestionWorkflow: fans out to process multiple sessions
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from forge.models import (
        AssembledContext,
        BatchResult,
        CapabilityTier,
        ModelConfig,
        ParsedLLMResponse,
        ThinkingConfig,
        resolve_model,
    )
    from forge.workflow_blocks import batch_submit_and_wait

PBOOK_TASK_QUEUE = "pbook-task-queue"

_PREPARE_TIMEOUT = timedelta(seconds=120)
_SAVE_TIMEOUT = timedelta(seconds=30)

_RETRY = RetryPolicy(maximum_attempts=2)


def _as_int(value: object) -> int:
    """Narrow a JSON-derived count to int for summation.

    Child-workflow results are typed ``dict[str, object]``; the count
    fields are always ints at runtime, but the static type is ``object``.
    """
    return value if isinstance(value, int) else 0


@workflow.defn
class TranscriptIngestionWorkflow:
    """Ingest a single Claude Code transcript and extract playbook entries.

    Uses forge's batch API for the analysis LLM call, then calls
    pbook's ExtractionWorkflow cross-queue for the extraction pipeline.

    Input JSON: {"path": str, "project": str, "session_id": str}
    Output: {"experiences_found": int, "entries_created": int, "session_id": str}
    """

    def __init__(self) -> None:
        self._batch_results: dict[str, BatchResult] = {}

    @workflow.signal
    async def batch_result_received(self, result: BatchResult) -> None:
        # Correlate by request_id, first delivery wins: at-least-once signalling
        # can redeliver a result, and a stale/duplicate must not be mistaken for
        # another call's (INTERIM; this whole path moves to pbook in Phase 6).
        self._batch_results.setdefault(result.request_id, result)

    @workflow.run
    async def run(self, input_json: str) -> dict[str, object]:
        data = json.loads(input_json)
        session_id = data.get("session_id", "")

        # Step 1: Read, parse, and render transcript
        prepared_json = await workflow.execute_activity(
            "prepare_transcript",
            input_json,
            start_to_close_timeout=_PREPARE_TIMEOUT,
            retry_policy=_RETRY,
            result_type=str,
        )

        prepared = json.loads(prepared_json)
        if not prepared.get("transcript_text") or prepared.get("message_count", 0) < 3:
            return {
                "experiences_found": 0,
                "entries_created": 0,
                "session_id": session_id,
            }

        # Step 2: Submit analysis to batch API. Use the SUMMARIZATION tier
        # (Sonnet by default) — transcript analysis is a summarization-like
        # task, and this keeps ingestion off the more expensive generation
        # tier even if someone overrides it to Opus in ModelConfig.
        model_name = resolve_model(CapabilityTier.SUMMARIZATION, ModelConfig())

        context = AssembledContext(
            task_id=f"ingest-{session_id}",
            system_prompt=prepared["system_prompt"],
            user_prompt=prepared["user_prompt"],
            model_name=model_name,
        )

        parsed: ParsedLLMResponse = await batch_submit_and_wait(
            self._batch_results,
            context,
            "TranscriptAnalysisResult",
            thinking=ThinkingConfig(),
            max_tokens=4096,
        )

        # Step 3: Parse analysis result into experiences
        try:
            analysis = json.loads(parsed.parsed_json)
        except json.JSONDecodeError:
            workflow.logger.error(
                "Malformed LLM response for session %s: %s",
                session_id,
                parsed.parsed_json[:200],
            )
            await workflow.execute_activity(
                "record_ingested_session_error",
                json.dumps(
                    {
                        "session_id": session_id,
                        "project_name": prepared.get("project", ""),
                        "error_message": "malformed_llm_response",
                    }
                ),
                task_queue=PBOOK_TASK_QUEUE,
                start_to_close_timeout=_SAVE_TIMEOUT,
                result_type=type(None),
            )
            return {
                "experiences_found": 0,
                "entries_created": 0,
                "session_id": session_id,
                "error": "malformed_llm_response",
            }
        experiences = analysis.get("experiences", [])

        if not experiences:
            # Record session as ingested with 0 results via pbook cross-queue
            await workflow.execute_activity(
                "record_ingested_session",
                json.dumps(
                    {
                        "session_id": session_id,
                        "project_name": prepared.get("project", ""),
                        "experiences_found": 0,
                        "entries_created": 0,
                    }
                ),
                task_queue=PBOOK_TASK_QUEUE,
                start_to_close_timeout=_SAVE_TIMEOUT,
                result_type=type(None),
            )
            return {
                "experiences_found": 0,
                "entries_created": 0,
                "session_id": session_id,
            }

        # Step 4: Convert to PushExperienceInput format and call
        # pbook's ExtractionWorkflow cross-queue. The metadata carries
        # session_id, the rich situation text (for future "discuss this
        # play" flows), and a stable experience_hash so re-ingestion
        # is idempotent on the entry_sources side.
        project = prepared.get("project", "")
        push_experiences = []
        for exp in experiences:
            problem = exp["problem"]
            resolution = exp["resolution"]
            context = exp.get("context", "")
            digest = hashlib.sha256(
                f"{problem}\x00{resolution}\x00{context}".encode(),
            ).hexdigest()
            push_experiences.append(
                {
                    "project": project,
                    "problem": problem,
                    "resolution": resolution,
                    "context": context,
                    "metadata": {
                        "source": "claude-code-transcript",
                        "session_id": session_id,
                        "experience_hash": digest,
                        "situation": exp.get("situation", ""),
                    },
                }
            )

        run_suffix = workflow.uuid4().hex[:8]
        extraction_result = await workflow.execute_child_workflow(
            "ExtractionWorkflow",
            json.dumps({"experiences": push_experiences, "project": project}),
            task_queue=PBOOK_TASK_QUEUE,
            id=f"pbook-extract-ingest-{session_id}-{run_suffix}",
        )

        entries_created = extraction_result.get("entries_created", 0)

        # Step 5: Record session as ingested
        await workflow.execute_activity(
            "record_ingested_session",
            json.dumps(
                {
                    "session_id": session_id,
                    "project_name": project,
                    "experiences_found": len(experiences),
                    "entries_created": entries_created,
                }
            ),
            task_queue=PBOOK_TASK_QUEUE,
            start_to_close_timeout=_SAVE_TIMEOUT,
            result_type=type(None),
        )

        return {
            "experiences_found": len(experiences),
            "entries_created": entries_created,
            "session_id": session_id,
        }


@workflow.defn
class BatchIngestionWorkflow:
    """Fan out to process multiple Claude Code transcript sessions.

    Input JSON: {"sessions": [{"path": str, "project": str, "session_id": str}, ...]}
    Output: {
        "sessions_processed": int,
        "total_experiences": int,
        "total_entries_created": int,
        "per_session": [{"session_id": str, "experiences_found": int, "entries_created": int}, ...],
    }
    """

    @workflow.run
    async def run(self, input_json: str) -> dict[str, object]:
        data = json.loads(input_json)
        sessions = data.get("sessions", [])

        if not sessions:
            return {
                "sessions_processed": 0,
                "total_experiences": 0,
                "total_entries_created": 0,
                "per_session": [],
            }

        # Fan out: start a child workflow per session. Append a short
        # UUID suffix so re-ingesting the same session never collides
        # with a previous run's workflow ID.
        run_suffix = workflow.uuid4().hex[:8]
        pending = []
        for session in sessions:
            handle = await workflow.start_child_workflow(
                TranscriptIngestionWorkflow.run,
                json.dumps(session),
                id=f"ingest-session-{session['session_id']}-{run_suffix}",
            )
            pending.append((session, handle))

        # Gather results
        results = []
        for session, handle in pending:
            try:
                # ChildWorkflowHandle subclasses asyncio.Task; await the
                # handle directly. `handle.result()` is the synchronous
                # Task.result() and raises before the child finishes.
                result = await handle
                results.append(result)
            except Exception as exc:
                workflow.logger.warning(
                    "Session ingestion failed for %s: %s",
                    session.get("session_id", ""),
                    exc,
                )
                # Tell pbook the child failed so `pbook sessions` can show
                # the error instead of leaving the row stuck on `running`.
                await workflow.execute_activity(
                    "record_ingested_session_error",
                    json.dumps(
                        {
                            "session_id": session.get("session_id", ""),
                            "project_name": session.get("project", ""),
                            "error_message": str(exc),
                        }
                    ),
                    task_queue=PBOOK_TASK_QUEUE,
                    start_to_close_timeout=_SAVE_TIMEOUT,
                    result_type=type(None),
                )
                results.append(
                    {
                        "experiences_found": 0,
                        "entries_created": 0,
                        "session_id": session.get("session_id", ""),
                        "error": str(exc),
                    }
                )

        total_exp = sum(_as_int(r.get("experiences_found", 0)) for r in results)
        total_entries = sum(_as_int(r.get("entries_created", 0)) for r in results)

        return {
            "sessions_processed": len(results),
            "total_experiences": total_exp,
            "total_entries_created": total_entries,
            "per_session": results,
        }
