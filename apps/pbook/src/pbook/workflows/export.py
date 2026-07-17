"""Temporal workflow for exporting playbook entries.

Fans out one activity per entry for parallel conversion,
then gathers results.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow

_FETCH_TIMEOUT = timedelta(seconds=30)
_EXPORT_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class ExportWorkflow:
    """Fetch matching entry IDs, fan-out export per row, gather results."""

    @workflow.run
    async def run(self, input_json: str) -> dict[str, Any]:
        # Step 1: Fetch matching IDs
        ids: list[int] = await workflow.execute_activity(
            "fetch_entry_ids",
            input_json,
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )

        if not ids:
            return {"entries": [], "count": 0}

        # Step 2: Fan-out — start one activity per row
        handles = []
        for entry_id in ids:
            handle = workflow.start_activity(
                "export_single_entry",
                entry_id,
                start_to_close_timeout=_EXPORT_TIMEOUT,
                result_type=dict,
            )
            handles.append(handle)

        # Step 3: Gather
        entries: list[dict[str, Any]] = []
        for handle in handles:
            entry: dict[str, Any] = await handle
            entries.append(entry)

        return {"entries": entries, "count": len(entries)}
