"""Cross-repo Temporal identifiers shared by Forge and its consumer apps.

These are part of the wire contract: the platform and a consumer app must agree
on the namespace and task-queue names, so they live here rather than being
redefined (and risking drift) in each repo. No signal is part of the wire
contract anymore — the timer-loop batch transport (D88, T4.1/T4.2) has each
workflow poll and fetch its own result, so cross-workflow signaling is gone.
"""

from __future__ import annotations

from typing import Final

# Temporal namespace the platform and consumer apps connect to. No explicit
# namespace was set historically (implicit "default"); the split makes it
# explicit so cross-queue activity and child workflows share one namespace.
TEMPORAL_NAMESPACE: Final = "default"

# Task queues. Each worker owns one queue; an activity/workflow runs on the
# worker registered for the queue it is scheduled on.
FORGE_TASK_QUEUE: Final = "forge-task-queue"
OCR_TASK_QUEUE: Final = "ocr-task-queue"
