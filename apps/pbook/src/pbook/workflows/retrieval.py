"""Temporal workflow for playbook retrieval.

Fetches candidates, ranks by intent mode, and packs within token budget.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pbook.activities.retrieval import rank_and_pack
    from pbook.models import RetrievalInput, RetrievalResult

_FETCH_TIMEOUT = timedelta(seconds=30)


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

        return RetrievalResult(
            entries=packed,
            token_count=token_count,
            total_candidates=len(candidates),
        )
