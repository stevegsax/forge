"""Temporal workflow for manually submitted playbook entries.

Validates, reviews via LLM, and saves a manually submitted playbook entry.
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow

_VALIDATE_TIMEOUT = timedelta(seconds=30)
_FETCH_TIMEOUT = timedelta(seconds=30)
_REVIEW_TIMEOUT = timedelta(minutes=2)
_REVIEW_HEARTBEAT = timedelta(seconds=60)
_SAVE_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class ManualEntryWorkflow:
    """Validate, review, and save a manually submitted playbook entry."""

    @workflow.run
    async def run(self, raw_json: str) -> dict:
        # Step 1: Validate raw JSON
        validation_json = await workflow.execute_activity(
            "validate_entry",
            raw_json,
            start_to_close_timeout=_VALIDATE_TIMEOUT,
            result_type=str,
        )
        validation = json.loads(validation_json)

        if not validation["valid"]:
            return {
                "approved": False,
                "validation_error": validation["error"],
            }

        entry = validation["entry"]

        # Step 2: Compute embedding for the proposed entry
        text_to_embed = f"{entry['title']}\n{entry['content']}"
        entry_embedding = await workflow.execute_activity(
            "compute_embedding",
            text_to_embed,
            start_to_close_timeout=timedelta(seconds=60),
            result_type=str,
        )
        entry["embedding"] = entry_embedding

        # Step 3: Find semantic duplicates to prevent context collapse (ACE)
        duplicates = await workflow.execute_activity(
            "find_duplicates",
            json.dumps({"embedding": entry_embedding, "threshold": 0.85}),
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )

        # Step 4: Fetch recent entries for broader context
        existing = await workflow.execute_activity(
            "fetch_existing_entries",
            50,
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )

        # Combine duplicates and existing entries for the reviewer
        # Prioritize duplicates in the review context
        context_entries = duplicates + [
            e for e in existing 
            if e["id"] not in {d["id"] for d in duplicates}
        ]

        # Step 5: LLM review
        review_input = json.dumps({
            "entry": entry,
            "existing_entries": context_entries[:50],  # Keep within context limits
        })
        review_json = await workflow.execute_activity(
            "review_entry",
            review_input,
            start_to_close_timeout=_REVIEW_TIMEOUT,
            heartbeat_timeout=_REVIEW_HEARTBEAT,
            result_type=str,
        )
        review = json.loads(review_json)

        if not review["approved"]:
            return {
                "approved": False,
                "rejection_reason": review["rejection_reason"],
            }

        # Step 6: Save the reviewed entry
        final_entry = review["final_entry"]
        save_input = json.dumps({
            "entries": [final_entry],
            "project": final_entry.get("source_project", ""),
        })
        count = await workflow.execute_activity(
            "save_extracted_entries",
            save_input,
            start_to_close_timeout=_SAVE_TIMEOUT,
            result_type=int,
        )

        return {
            "approved": True,
            "entry": final_entry,
            "entries_saved": count,
        }
