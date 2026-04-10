"""Temporal workflow for playbook retrieval.

Fetches candidates, ranks by intent mode, packs within token budget,
and records which entries were served for helpfulness tracking.
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pbook.activities.retrieval import rank_and_pack
    from pbook.models import RetrievalInput, RetrievalResult

_FETCH_TIMEOUT = timedelta(seconds=30)
_RECORD_TIMEOUT = timedelta(seconds=10)


@workflow.defn
class RetrievalWorkflow:
    """Fetch, rank, and pack playbook entries within a token budget."""

    @workflow.run
    async def run(self, input: RetrievalInput) -> RetrievalResult:
        # Step 1: Fetch candidates from the store
        candidates = await workflow.execute_activity(
            "fetch_candidates",
            input.model_dump_json(),
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )

        if not candidates:
            return RetrievalResult(entries=[], token_count=0, total_candidates=0)

        # Step 2: Rank and pack (pure function, runs in workflow thread)
        packed, token_count = rank_and_pack(
            candidates,
            input.tags,
            input.mode,
            input.token_budget,
        )

        # Step 3: Record retrieval (fire-and-forget, non-blocking)
        entry_ids = [e["id"] for e in packed if "id" in e]
        if entry_ids:
            try:
                await workflow.execute_activity(
                    "record_retrieval_event",
                    json.dumps(entry_ids),
                    start_to_close_timeout=_RECORD_TIMEOUT,
                )
            except Exception:
                # Recording is best-effort; retrieval still succeeds
                workflow.logger.warning("Failed to record retrieval event")

        return RetrievalResult(
            entries=packed,
            token_count=token_count,
            total_candidates=len(candidates),
        )
