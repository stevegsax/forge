"""Temporal workflow for knowledge extraction from pushed experience.

Receives experience data, calls extraction LLM, validates and saves entries.
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow

_LLM_TIMEOUT = timedelta(minutes=5)
_LLM_HEARTBEAT = timedelta(seconds=60)
_SAVE_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class ExtractionWorkflow:
    """Extract lessons from pushed experience data.

    1. Call extraction LLM with experience reports.
    2. Save valid entries with needs_review=True.
    """

    @workflow.run
    async def run(self, input_json: str) -> dict:
        data = json.loads(input_json)
        experiences_raw = data.get("experiences", [])
        project = data.get("project", "")

        if not experiences_raw:
            return {"entries_created": 0}

        # Step 1: Call extraction LLM
        extraction_json = await workflow.execute_activity(
            "extract_from_experience",
            json.dumps(experiences_raw),
            start_to_close_timeout=_LLM_TIMEOUT,
            heartbeat_timeout=_LLM_HEARTBEAT,
            result_type=str,
        )

        extraction = json.loads(extraction_json)
        entries = extraction.get("entries", [])

        if not entries:
            return {"entries_created": 0}

        # Step 2: Generate embeddings for each entry in parallel
        # ACE requires semantic de-duplication and vector search
        for entry in entries:
            text_to_embed = f"{entry['title']}\n{entry['content']}"
            entry["embedding"] = await workflow.execute_activity(
                "compute_embedding",
                text_to_embed,
                start_to_close_timeout=timedelta(seconds=60),
                result_type=str,
            )

        # Step 3: Save entries with needs_review=True
        save_input = json.dumps({"entries": entries, "project": project})
        count = await workflow.execute_activity(
            "save_extracted_entries",
            save_input,
            start_to_close_timeout=_SAVE_TIMEOUT,
            result_type=int,
        )

        return {"entries_created": count}
