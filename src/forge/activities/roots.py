"""Composition-root activity classes for Forge (T3.6, D93).

Temporal's sanctioned dependency-injection pattern: each activity that carries a
process-wide dependency (the store engine, the LLM client, the batch SDK client,
the blob store) is a ``@activity.defn``
**method** on a small class constructed ONCE in ``forge.worker`` with the
dependencies the composition root built. The worker registers the *bound
methods*; because a bound method proxies ``__name__`` to the underlying
function, the registered activity name is unchanged from the former free-function
shells — workflows invoke by string, so the class conversion is invisible to
them.

The methods are thin: they open the OTel span / heartbeat exactly as the former
shells did and delegate to the ``execute_*`` core functions (and pure prompt
builders / store helpers) in the per-activity modules, passing the
constructor-held dependency. No module-global client, engine, or registry
survives.

Sandbox-light discipline (see ``tests/test_sandbox_light.py``): this module is
chain-imported into the Temporal workflow sandbox via ``forge.activities``, so it
must not eagerly import an HTTP stack. The SDK-loading imports
(``sax_platform.llm.batch`` etc.) stay function-local inside the ``execute_*``
cores; ``forge.store`` is imported inside the methods. The dependency *types*
(AsyncAnthropic, MistralOcr, S3Blobs, AnthropicLLM, Engine) are
TYPE_CHECKING-only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from sax_platform.temporal.heartbeat import heartbeat_during
from temporalio import activity

from forge.activities.batch_fetch import execute_batch_status, execute_fetch_batch_result
from forge.activities.batch_parse import execute_parse_llm_response
from forge.activities.batch_submit import (
    DEFAULT_MODEL,
    execute_batch_submit,
)
from forge.activities.conflict_resolution import execute_conflict_resolution_call
from forge.activities.context import (
    _assemble_context_inner,
    _assemble_step_context_inner,
    _assemble_sub_task_context_inner,
)
from forge.activities.exploration import execute_exploration_call, fulfill_requests
from forge.activities.extraction import (
    build_extraction_system_prompt,
    build_extraction_user_prompt,
    execute_extraction_call,
)
from forge.activities.llm import execute_llm_call
from forge.activities.persist import execute_persist
from forge.activities.planner import execute_planner_call
from forge.activities.playbook_export import db_row_to_playbook_entry
from forge.activities.playbook_review import apply_suggestions, review_playbook_entry
from forge.activities.sanity_check import execute_sanity_check_call

# Activity parameter/return types must be runtime-importable: Temporal resolves
# each registered activity's type hints (get_type_hints) against this module's
# globals to drive the pydantic data converter, so these cannot move into a
# TYPE_CHECKING block (noqa: TC001). forge.models / forge.persist_models are pure
# pydantic (sandbox-light).
from forge.models import (
    AssembleContextInput,
    AssembledContext,
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    BatchFetchResult,
    BatchStatusInput,
    BatchStatusResult,
    BatchSubmitInput,
    BatchSubmitResult,
    ConflictResolutionCallInput,
    ConflictResolutionCallResult,
    ContextResult,
    ExplorationInput,
    ExplorationResponse,
    ExportSinglePlaybookInput,
    ExtractionCallResult,
    ExtractionInput,
    FetchBatchResultInput,
    FetchExistingPlaybooksInput,
    FetchExtractionInput,
    FetchPlaybookIdsInput,
    FulfillContextInput,
    LLMCallResult,
    ParsedLLMResponse,
    ParseResponseInput,
    PlanCallResult,
    PlannerInput,
    PlaybookEntry,
    ReviewManualPlaybookInput,
    ReviewManualPlaybookResult,
    SanityCheckCallResult,
    SanityCheckInput,
    SaveExtractionInput,
)
from forge.persist_models import PersistRequest, PersistResult  # noqa: TC001
from forge.tracing import get_tracer, llm_call_attributes

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anthropic import AsyncAnthropic
    from pydantic import BaseModel
    from sax_platform.contracts.s3_blobs import S3Blobs
    from sax_platform.llm import AnthropicLLM
    from sqlalchemy import Engine

__all__ = [
    "BatchActivities",
    "ContextActivities",
    "LlmActivities",
    "StoreActivities",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StoreActivities — every activity that reads/writes the store, over one engine
# ---------------------------------------------------------------------------


class StoreActivities:
    """Store-backed activities bound to the single process-wide engine.

    Replaces the per-call ``get_store_engine()`` in each former shell with one
    engine built at worker startup (T3.6: "one engine per process" — each engine
    carries a bounded Postgres pool, so a fresh pool per activity call could
    exhaust the managed-database connection cap).
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @activity.defn
    async def fetch_extraction_input(self, input: FetchExtractionInput) -> ExtractionInput:
        """Read unextracted runs from the store and build the extraction prompt.

        Returns an ExtractionInput with empty source_workflow_ids if no runs found.
        """
        import json

        from forge.store import get_unextracted_runs

        runs = get_unextracted_runs(self._engine, limit=input.limit)

        if not runs:
            return ExtractionInput(system_prompt="", user_prompt="", source_workflow_ids=[])

        for run in runs:
            if "result_json" in run:
                try:
                    run["result"] = json.loads(run["result_json"])
                except (json.JSONDecodeError, TypeError):
                    run["result"] = {}

        return ExtractionInput(
            system_prompt=build_extraction_system_prompt(runs),
            user_prompt=build_extraction_user_prompt(),
            source_workflow_ids=[r["workflow_id"] for r in runs],
        )

    @activity.defn
    async def save_extraction_results(self, input: SaveExtractionInput) -> None:
        """Write extracted playbook entries to the store."""
        from forge.store import build_playbook_dict, save_playbooks

        dicts = [
            build_playbook_dict(entry, input.extraction_workflow_id) for entry in input.entries
        ]
        save_playbooks(self._engine, dicts)

    @activity.defn
    async def persist_to_store(self, req: PersistRequest) -> PersistResult:
        """Apply one idempotent, survivable store write, dispatched on ``req.kind``."""
        return await execute_persist(req, self._engine)

    @activity.defn
    async def fetch_existing_playbooks(
        self, input: FetchExistingPlaybooksInput
    ) -> list[dict[str, Any]]:
        """Query recent playbooks for duplication context."""
        from forge.store import list_recent_playbooks

        return list_recent_playbooks(self._engine, limit=input.limit)

    @activity.defn
    async def fetch_playbook_ids(self, input: FetchPlaybookIdsInput) -> list[int]:
        """Query store for matching playbook IDs."""
        from forge.store import get_playbook_ids

        return get_playbook_ids(
            self._engine,
            tags=input.tags if input.tags else None,
            source_task_id=input.source_task_id,
            limit=input.limit,
        )

    @activity.defn
    async def export_single_playbook(self, input: ExportSinglePlaybookInput) -> PlaybookEntry:
        """Fetch one playbook row by ID and convert to PlaybookEntry."""
        from forge.store import get_playbook_by_id

        row = get_playbook_by_id(self._engine, input.playbook_id)
        if row is None:
            msg = f"Playbook {input.playbook_id} not found"
            raise RuntimeError(msg)
        return db_row_to_playbook_entry(row)


# ---------------------------------------------------------------------------
# ContextActivities — context assembly + provider dispatch, over one engine
# ---------------------------------------------------------------------------


class ContextActivities:
    """Context-assembly activities bound to the single process-wide engine.

    The engine threads into playbook loading (single-step / planned / fan-out
    assembly) and the store-backed exploration providers (``past_runs`` /
    ``playbooks``).
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @activity.defn
    async def assemble_context(self, input: AssembleContextInput) -> AssembledContext:
        """Read context files and assemble the prompts for the LLM call."""
        tracer = get_tracer()
        with tracer.start_as_current_span("forge.assemble_context"):
            logger.info("Assemble context: task_id=%s", input.task_id)
            return await _assemble_context_inner(input, self._engine)

    @activity.defn
    async def assemble_step_context(self, input: AssembleStepContextInput) -> AssembledContext:
        """Read context files from the worktree and assemble step-level prompts."""
        return await _assemble_step_context_inner(input, self._engine)

    @activity.defn
    async def assemble_sub_task_context(
        self, input: AssembleSubTaskContextInput
    ) -> AssembledContext:
        """Read context files from the parent worktree and assemble sub-task prompts."""
        return await _assemble_sub_task_context_inner(input, self._engine)

    @activity.defn
    async def fulfill_context_requests(self, input: FulfillContextInput) -> list[ContextResult]:
        """Dispatch context requests to the provider registry (store engine threaded)."""
        tracer = get_tracer()
        with tracer.start_as_current_span("forge.fulfill_context_requests") as span:
            logger.info("Fulfilling %d context requests", len(input.requests))
            requests_as_dicts: list[dict[str, object]] = [
                {"provider": r.provider, "params": r.params} for r in input.requests
            ]

            # Provider handlers run subprocesses (git, ruff, rg) and
            # repo-proportional grimp/file scans; offload the whole dispatch off
            # the event loop.
            results = await asyncio.to_thread(
                fulfill_requests,
                requests_as_dicts,
                input.repo_root,
                input.worktree_path,
                self._engine,
            )

            span.set_attributes(
                {
                    "forge.exploration.providers_called": len(results),
                    "forge.exploration.total_tokens": sum(r.estimated_tokens for r in results),
                }
            )

            return results


# ---------------------------------------------------------------------------
# LlmActivities — sync-lane structured-output calls, over one AnthropicLLM
# ---------------------------------------------------------------------------


class LlmActivities:
    """LLM sync-lane activities bound to one ``AnthropicLLM`` client.

    The single client (sharing the composition root's SDK client with the batch
    lane) is injected once, rather than each activity resolving its own from a
    module-global cache.
    """

    def __init__(self, llm: AnthropicLLM) -> None:
        self._llm = llm

    @activity.defn
    async def call_llm(self, context: AssembledContext) -> LLMCallResult:
        """Call the LLM for a single-step generation and extract structured results."""
        tracer = get_tracer()
        with tracer.start_as_current_span("forge.call_llm") as span:
            logger.info("LLM call start: task_id=%s model=%s", context.task_id, context.model_name)
            async with heartbeat_during():
                result = await execute_llm_call(context, self._llm)
            logger.info(
                "LLM call done: task_id=%s tokens=%din/%dout latency=%.0fms",
                context.task_id,
                result.input_tokens,
                result.output_tokens,
                result.latency_ms,
            )
            span.set_attributes(
                llm_call_attributes(
                    model_name=result.model_name,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                    task_id=context.task_id,
                    cache_creation_input_tokens=result.cache_creation_input_tokens,
                    cache_read_input_tokens=result.cache_read_input_tokens,
                )
            )
            return result

    @activity.defn
    async def call_planner(self, input: PlannerInput) -> PlanCallResult:
        """Call the LLM for planning and extract the structured plan."""
        tracer = get_tracer()
        with tracer.start_as_current_span("forge.call_planner") as span:
            logger.info("Planner call: task_id=%s", input.task_id)
            async with heartbeat_during():
                result = await execute_planner_call(input, self._llm)
            logger.info("Plan produced: task_id=%s steps=%d", input.task_id, len(result.plan.steps))
            span.set_attributes(
                llm_call_attributes(
                    model_name=result.model_name,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                    task_id=input.task_id,
                    cache_creation_input_tokens=result.cache_creation_input_tokens,
                    cache_read_input_tokens=result.cache_read_input_tokens,
                )
            )
            return result

    @activity.defn
    async def call_exploration_llm(self, input: ExplorationInput) -> ExplorationResponse:
        """Call the exploration LLM to decide what context to request."""
        from pathlib import Path

        from forge.activities.context import (
            _read_project_instructions,
            build_project_instructions_section,
        )

        tracer = get_tracer()
        with tracer.start_as_current_span("forge.call_exploration_llm") as span:
            logger.info(
                "Exploration call: task_id=%s round=%d/%d",
                input.task_id,
                input.round_number,
                input.max_rounds,
            )
            project_instructions = ""
            if input.repo_root:
                project_instructions = build_project_instructions_section(
                    _read_project_instructions(Path(input.repo_root))
                )

            start = time.monotonic()
            async with heartbeat_during():
                response = await execute_exploration_call(input, self._llm, project_instructions)
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(
                "Exploration result: task_id=%s requests=%d", input.task_id, len(response.requests)
            )
            span.set_attributes(
                {
                    "forge.exploration.round": input.round_number,
                    "forge.exploration.requests_count": len(response.requests),
                    "forge.exploration.latency_ms": elapsed_ms,
                }
            )
            return response

    @activity.defn
    async def call_sanity_check(self, input: SanityCheckInput) -> SanityCheckCallResult:
        """Call the LLM for a plan sanity check and extract the verdict."""
        tracer = get_tracer()
        with tracer.start_as_current_span("forge.call_sanity_check") as span:
            logger.info("Sanity check call: task_id=%s", input.task_id)
            async with heartbeat_during():
                result = await execute_sanity_check_call(input, self._llm)
            logger.info(
                "Sanity verdict: task_id=%s verdict=%s",
                input.task_id,
                result.response.verdict.value,
            )
            span.set_attributes(
                llm_call_attributes(
                    model_name=result.model_name,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                    task_id=input.task_id,
                    cache_creation_input_tokens=result.cache_creation_input_tokens,
                    cache_read_input_tokens=result.cache_read_input_tokens,
                )
            )
            return result

    @activity.defn
    async def call_conflict_resolution(
        self, input: ConflictResolutionCallInput
    ) -> ConflictResolutionCallResult:
        """Call the LLM to merge conflicting sub-task file versions."""
        tracer = get_tracer()
        with tracer.start_as_current_span("forge.call_conflict_resolution") as span:
            logger.info("Conflict resolution call: task_id=%s", input.task_id)
            async with heartbeat_during():
                result = await execute_conflict_resolution_call(input, self._llm)
            span.set_attributes(
                llm_call_attributes(
                    model_name=result.model_name,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                    task_id=input.task_id,
                    cache_creation_input_tokens=result.cache_creation_input_tokens,
                    cache_read_input_tokens=result.cache_read_input_tokens,
                )
            )
            return result

    @activity.defn
    async def call_extraction_llm(self, input: ExtractionInput) -> ExtractionCallResult:
        """Call the LLM to extract playbook entries from completed runs."""
        tracer = get_tracer()
        with tracer.start_as_current_span("forge.call_extraction_llm") as span:
            async with heartbeat_during():
                result = await execute_extraction_call(input, self._llm)
            span.set_attributes(
                llm_call_attributes(
                    model_name=result.model_name,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                    task_id="__extraction__",
                    cache_creation_input_tokens=result.cache_creation_input_tokens,
                    cache_read_input_tokens=result.cache_read_input_tokens,
                )
            )
            return result

    @activity.defn
    async def review_manual_playbook(
        self, input: ReviewManualPlaybookInput
    ) -> ReviewManualPlaybookResult:
        """Review a proposed playbook entry via LLM and apply suggestions."""
        review = await review_playbook_entry(
            input.entry, input.existing_playbooks, self._llm, model_name=input.model_name
        )
        if not review.approved:
            return ReviewManualPlaybookResult(
                approved=False,
                rejection_reason=review.rejection_reason,
                final_entry=input.entry,
            )
        final_entry = apply_suggestions(input.entry, review)
        return ReviewManualPlaybookResult(approved=True, final_entry=final_entry)


# ---------------------------------------------------------------------------
# BatchActivities — batch-lane submit/poll/parse, over the shared SDK client
# ---------------------------------------------------------------------------


class BatchActivities:
    """Batch-lane activities bound to the composition root's shared dependencies.

    Replaces the two per-module batch-client caches and the module S3-blob
    functions. The retired signal-path poller also held the Temporal client to
    signal waiting workflows; the timer-loop transport (T4.1) has each workflow
    poll and fetch its own result, so no client is held here. Forge submits
    anthropic only (T4.2 ST3) — no MistralOcr client. ``blob_store`` is optional:
    when ``FORGE_OCR_S3_BUCKET`` is unset at startup it is ``None`` and a code
    path that needs it raises a clear ``RuntimeError`` at point of use.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        output_types: Mapping[str, type[BaseModel]],
        engine: Engine,
        blob_store: S3Blobs | None,
    ) -> None:
        self._client = client
        self._output_types = output_types
        self._engine = engine
        self._blob_store = blob_store

    @activity.defn
    async def submit_batch_request(self, input: BatchSubmitInput) -> BatchSubmitResult:
        """Build and submit a generic (in-platform) batch request."""
        from sax_platform.llm.tiers import split_provider

        tracer = get_tracer()
        with tracer.start_as_current_span("forge.submit_batch_request") as span:
            logger.info(
                "Batch submitted: task_id=%s output_type=%s",
                input.context.task_id,
                input.output_type_name,
            )
            result = await execute_batch_submit(input, self._client, self._output_types)

            span.set_attributes(
                {
                    "forge.batch.request_id": result.request_id,
                    "forge.batch.batch_id": result.batch_id,
                    "forge.batch.output_type": input.output_type_name,
                    "forge.batch.workflow_id": input.workflow_id,
                }
            )

            provider_name, _ = split_provider(input.context.model_name or DEFAULT_MODEL)
            # Thread the provider back to the workflow, which persists the
            # submission survivably (Phase C) — the activity writes no store row.
            return result.model_copy(update={"provider": provider_name})

    @activity.defn
    async def parse_llm_response(self, input: ParseResponseInput) -> ParsedLLMResponse:
        """Classify a delivered batch result line into a typed ParsedLLMResponse."""
        from forge.message_log import write_message_log

        tracer = get_tracer()
        with tracer.start_as_current_span("forge.parse_llm_response") as span:
            logger.info(
                "Parse response: task_id=%s output_type=%s", input.task_id, input.output_type_name
            )
            # The body arrives inline or via an S3 pointer (a result envelope);
            # fetch and unwrap when only s3_key is set. The generic path ignores
            # any images the envelope carries.
            if input.s3_key is not None:
                from sax_platform.contracts.models import parse_batch_result_payload

                if self._blob_store is None:
                    msg = (
                        "parse_llm_response: S3 blob store not configured "
                        "(FORGE_OCR_S3_BUCKET unset) but the result was delivered by s3_key."
                    )
                    raise RuntimeError(msg)
                envelope = self._blob_store.get(input.s3_key)
                raw_json, _images = parse_batch_result_payload(envelope.decode("utf-8"))
            else:
                raw_json = input.raw_response_json
            if raw_json is None:
                msg = (
                    "parse_llm_response: no body resolved (both raw_response_json and s3_key empty)"
                )
                raise ValueError(msg)

            if input.log_messages and input.worktree_path:
                write_message_log(input.worktree_path, "response", raw_json)

            result = execute_parse_llm_response(
                raw_json, input.output_type_name, self._output_types
            )

            span.set_attributes(
                {
                    "forge.batch.output_type": input.output_type_name or "",
                    "forge.batch.task_id": input.task_id,
                    "forge.batch.model_name": result.model_name,
                }
            )
            return result

    @activity.defn
    async def batch_status(self, input: BatchStatusInput) -> BatchStatusResult:
        """Poll one batch's normalized lifecycle state (timer-loop transport, T4.1)."""
        tracer = get_tracer()
        with tracer.start_as_current_span("forge.batch_status") as span:
            result = await execute_batch_status(input, client=self._client)
            logger.info("Batch status: batch_id=%s state=%s", input.batch_id, result.state)
            span.set_attributes(
                {
                    "forge.batch.batch_id": input.batch_id,
                    "forge.batch.provider": input.provider,
                    "forge.batch.state": result.state,
                }
            )
            return result

    @activity.defn
    async def fetch_batch_result(self, input: FetchBatchResultInput) -> BatchFetchResult:
        """Fetch this waiter's batch result line, claim-checked (timer-loop transport, T4.1)."""
        tracer = get_tracer()
        with tracer.start_as_current_span("forge.fetch_batch_result") as span:
            logger.info(
                "Fetch batch result: batch_id=%s request_id=%s",
                input.batch_id,
                input.request_id,
            )
            async with heartbeat_during():
                result = await execute_fetch_batch_result(
                    input,
                    client=self._client,
                    put_result_blob=self._put_result_blob,
                )
            delivery = "error" if result.error else ("pointer" if result.s3_key else "inline")
            span.set_attributes(
                {
                    "forge.batch.batch_id": input.batch_id,
                    "forge.batch.request_id": input.request_id,
                    "forge.batch.provider": input.provider,
                    "forge.batch.delivery": delivery,
                }
            )
            return result

    def _put_result_blob(self, custom_id: str, data: bytes) -> str:
        """Upload a batch-result envelope and return its S3 key (pointer delivery).

        Used by the ``fetch_batch_result`` activity: a RuntimeError is raised at
        point of use when the blob store is unconfigured (``FORGE_OCR_S3_BUCKET``
        unset) but a result requires pointer delivery.
        """
        if self._blob_store is None:
            msg = (
                "fetch_batch_result: S3 blob store not configured "
                "(FORGE_OCR_S3_BUCKET unset) but a result requires pointer delivery."
            )
            raise RuntimeError(msg)
        key = self._blob_store.build_key(f"batch-result-{custom_id}")
        self._blob_store.put(key, data, "application/json")
        return key
