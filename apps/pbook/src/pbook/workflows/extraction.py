"""Temporal workflow for knowledge extraction from pushed experience.

Receives experience data, calls the extraction LLM once per experience
(so each extracted entry is naturally attributed to its origin),
computes embeddings, and saves entries via match-or-attach.
"""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow

_LLM_TIMEOUT = timedelta(minutes=5)
_LLM_HEARTBEAT = timedelta(seconds=60)
_SAVE_TIMEOUT = timedelta(seconds=30)
_EMBED_TIMEOUT = timedelta(seconds=60)


@workflow.defn
class ExtractionWorkflow:
    """Extract lessons from pushed experience data.

    Per experience:

    1. Call extraction LLM (single experience in / 0..K entries out).
    2. Embed each candidate entry and the experience's situation text.
    3. Save via match-or-attach: an existing semantically-similar entry
       gains a new entry_sources row; a novel candidate becomes a new
       entry plus its first entry_sources row.

    Looping per-experience keeps attribution clean (every entry from a
    given LLM call comes from a single experience) without complicating
    the extraction prompt.
    """

    @workflow.run
    async def run(self, input_json: str) -> dict:
        data = json.loads(input_json)
        experiences_raw = data.get("experiences", [])
        project = data.get("project", "")

        if not experiences_raw:
            return {"entries_created": 0}

        total_created = 0

        for exp in experiences_raw:
            metadata = exp.get("metadata", {}) or {}
            situation_text = metadata.get("situation", "")

            # Step 1: extract entries from this single experience.
            extraction_json = await workflow.execute_activity(
                "extract_from_experience",
                json.dumps([exp]),
                start_to_close_timeout=_LLM_TIMEOUT,
                heartbeat_timeout=_LLM_HEARTBEAT,
                result_type=str,
            )
            extraction = json.loads(extraction_json)
            entries = extraction.get("entries", [])
            if not entries:
                continue

            # Step 2: embed each candidate entry.
            for entry in entries:
                text_to_embed = f"{entry['title']}\n{entry['content']}"
                entry["embedding"] = await workflow.execute_activity(
                    "compute_embedding",
                    text_to_embed,
                    start_to_close_timeout=_EMBED_TIMEOUT,
                    result_type=str,
                )

            # Step 3: embed the situation once per experience (shared
            # across every entry this experience produces).
            situation_embedding = ""
            if situation_text:
                situation_embedding = await workflow.execute_activity(
                    "compute_embedding",
                    situation_text,
                    start_to_close_timeout=_EMBED_TIMEOUT,
                    result_type=str,
                )

            # Step 4: match-or-attach each entry plus its source row.
            save_input = json.dumps({
                "entries": entries,
                "project": project,
                "source": {
                    "session_id": metadata.get("session_id", ""),
                    "project_name": exp.get("project", project),
                    "experience_hash": metadata.get("experience_hash"),
                    "source_context": situation_text,
                    "source_context_embedding": situation_embedding,
                },
            })
            count = await workflow.execute_activity(
                "save_extracted_entries",
                save_input,
                start_to_close_timeout=_SAVE_TIMEOUT,
                result_type=int,
            )
            total_created += count

        return {"entries_created": total_created}
