"""Temporal workflow for playbook retrieval.

The workflow body ferries only ranking *metadata* between activities;
heavy entry data (title, content, embeddings) stays server-side. This
keeps activity payloads well under Temporal's 512 KB warning threshold
and avoids pydantic re-serializing megabyte-scale dicts on every replay.

Steps:
1. ``fetch_candidates``        — minimal dicts (id + ranking fields)
2. ``llm_embed`` + ``compute_similarities_by_id`` — only when query is set;
   embeddings are loaded from the DB inside the similarity activity
3. ``score_and_pack``          — server-side rank + top-N load + token-pack
4. ``record_retrieval_event``  — fire-and-forget bookkeeping
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pbook.models import RetrievalInput, RetrievalResult
    from pbook.workflow_steps.retry import LLM_RETRY_POLICY


_FETCH_TIMEOUT = timedelta(seconds=30)
_RECORD_TIMEOUT = timedelta(seconds=10)
_EMBED_TIMEOUT = timedelta(seconds=60)
_SIMILARITY_TIMEOUT = timedelta(seconds=30)
_PACK_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class RetrievalWorkflow:
    """Fetch, rank, and pack playbook entries within a token budget."""

    @workflow.run
    async def run(self, input: RetrievalInput) -> RetrievalResult:
        # 1. Fetch minimal candidate metadata. No content, no embeddings.
        meta = await workflow.execute_activity(
            "fetch_candidates",
            input.model_dump_json(),
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )
        if not meta:
            return RetrievalResult(entries=[], token_count=0, total_candidates=0)

        # 2. Free-text query path: embed the query, then score similarity
        # by ID. Embedding bytes are loaded server-side and never cross
        # the wire.
        similarities: dict | None = None
        if input.query:
            query_b64 = await workflow.execute_activity(
                "llm_embed",
                input.query,
                start_to_close_timeout=_EMBED_TIMEOUT,
                result_type=str,
                retry_policy=LLM_RETRY_POLICY,
            )
            similarities = await workflow.execute_activity(
                "compute_similarities_by_id",
                json.dumps({
                    "query_embedding_b64": query_b64,
                    "ids": [m["id"] for m in meta if "id" in m],
                }),
                start_to_close_timeout=_SIMILARITY_TIMEOUT,
                result_type=dict,
            )

        # 3. Score, rank, and pack — full content loaded server-side
        # for top-N only.
        result = await workflow.execute_activity(
            "score_and_pack",
            json.dumps({
                "meta": meta,
                "similarities": similarities,
                "tags": input.tags,
                "mode": input.mode.value,
                "token_budget": input.token_budget,
                "threshold": input.threshold,
            }),
            start_to_close_timeout=_PACK_TIMEOUT,
            result_type=dict,
        )
        packed = result["packed"]
        token_count = result["token_count"]

        # 4. Record retrieval (fire-and-forget; failure logs but doesn't fail
        # the workflow — the user gets their entries either way).
        entry_ids = [e["id"] for e in packed if "id" in e]
        if entry_ids:
            try:
                await workflow.execute_activity(
                    "record_retrieval_event",
                    json.dumps(entry_ids),
                    start_to_close_timeout=_RECORD_TIMEOUT,
                )
            except Exception:
                workflow.logger.warning("Failed to record retrieval event")

        return RetrievalResult(
            entries=packed,
            token_count=token_count,
            total_candidates=len(meta),
        )
