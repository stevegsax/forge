"""Cross-repo Temporal identifiers shared by Forge and its consumer apps.

These are part of the wire contract: the platform and a consumer app must agree
on the namespace, task-queue names, and signal name, so they live here rather
than being redefined (and risking drift) in each repo.
"""

from __future__ import annotations

from typing import Final

# Temporal namespace the platform and consumer apps connect to. No explicit
# namespace was set historically (implicit "default"); the split makes it
# explicit so cross-queue signaling and child workflows share one namespace.
TEMPORAL_NAMESPACE: Final = "default"

# Task queues. Each worker owns one queue; an activity/workflow runs on the
# worker registered for the queue it is scheduled on.
FORGE_TASK_QUEUE: Final = "forge-task-queue"
OCR_TASK_QUEUE: Final = "ocr-task-queue"

# Signal the platform poller sends to a waiting consumer workflow when a batch
# result is ready. Temporal binds signals by the handler METHOD NAME, so the
# receiving ``@workflow.signal`` method must be named exactly this (or bind it
# explicitly with ``@workflow.signal(name=BATCH_RESULT_SIGNAL)``).
BATCH_RESULT_SIGNAL: Final = "batch_result_received"
