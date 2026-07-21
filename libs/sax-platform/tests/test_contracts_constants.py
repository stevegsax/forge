"""Tests for the cross-repo Temporal identifier constants.

These are part of the wire contract between the platform and its consumer
apps, so the values themselves (not just their existence) are pinned here —
an accidental rename would silently misroute cross-queue activities.
"""

from __future__ import annotations

from sax_platform.contracts.constants import (
    FORGE_TASK_QUEUE,
    OCR_TASK_QUEUE,
    TEMPORAL_NAMESPACE,
)


def test_namespace_is_default() -> None:
    assert TEMPORAL_NAMESPACE == "default"


def test_task_queue_names() -> None:
    assert FORGE_TASK_QUEUE == "forge-task-queue"
    assert OCR_TASK_QUEUE == "ocr-task-queue"


def test_task_queues_are_distinct() -> None:
    """Each worker owns one queue; a collision would misroute activities."""
    assert FORGE_TASK_QUEUE != OCR_TASK_QUEUE
