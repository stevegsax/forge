"""Temporal workflow for playbook retrieval.

Fetches candidates, optionally ranks by free-text semantic similarity
to a query embedding, packs within a token budget, and records which
entries were served for helpfulness tracking.
"""

from __future__ import annotations

import base64
import json
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pbook.activities.retrieval import rank_and_pack
    from pbook.models import RetrievalInput, RetrievalResult

def _encode_candidate_embedding(emb: object) -> str:
    """Normalize a candidate embedding to base64 for cross-activity transport.

    Temporal serializes activity results through Pydantic; raw bytes
    can come back as either ``bytes``, ``list[int]`` (per-byte), or the
    empty/None case. Handle each so the workflow body stays sandbox-safe.
    """
    if emb is None or emb == "":
        return ""
    if isinstance(emb, bytes):
        return base64.b64encode(emb).decode("ascii")
    if isinstance(emb, list):
        return base64.b64encode(bytes(emb)).decode("ascii")
    if isinstance(emb, str):
        # Assume already base64 (defensive — uncommon but cheap to allow).
        return emb
    return ""


_FETCH_TIMEOUT = timedelta(seconds=30)
_RECORD_TIMEOUT = timedelta(seconds=10)
_EMBED_TIMEOUT = timedelta(seconds=60)
_SIMILARITY_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class RetrievalWorkflow:
    """Fetch, rank, and pack playbook entries within a token budget."""

    @workflow.run
    async def run(self, input: RetrievalInput) -> RetrievalResult:
        # Step 1: Fetch candidates from the store.
        candidates = await workflow.execute_activity(
            "fetch_candidates",
            input.model_dump_json(),
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )

        if not candidates:
            return RetrievalResult(entries=[], token_count=0, total_candidates=0)

        # Step 2: Free-text query path. Embed the query, then compute
        # cosine similarity in an activity (numpy operations are
        # nondeterministic from Temporal's perspective and would trip
        # the workflow sandbox if done inline).
        similarities: dict[int, float] | None = None
        if input.query:
            query_embedding_b64 = await workflow.execute_activity(
                "llm_embed",
                input.query,
                start_to_close_timeout=_EMBED_TIMEOUT,
                result_type=str,
            )
            sim_input = json.dumps({
                "query_embedding_b64": query_embedding_b64,
                "candidates": [
                    {
                        "id": c["id"],
                        "embedding_b64": _encode_candidate_embedding(c.get("embedding")),
                    }
                    for c in candidates
                ],
            })
            sim_strkey = await workflow.execute_activity(
                "compute_similarities",
                sim_input,
                start_to_close_timeout=_SIMILARITY_TIMEOUT,
                result_type=dict,
            )
            similarities = {int(k): float(v) for k, v in sim_strkey.items()}

        # Step 3: Rank and pack (pure function, runs in workflow thread).
        packed, token_count = rank_and_pack(
            candidates,
            input.tags,
            input.mode,
            input.token_budget,
            similarities=similarities,
            threshold=input.threshold,
        )

        # Step 4: Record retrieval (fire-and-forget, non-blocking).
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
