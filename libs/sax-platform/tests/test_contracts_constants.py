"""Tests for the cross-repo Temporal identifier constants.

These are part of the wire contract between the platform and its consumer
apps, so the values themselves (not just their existence) are pinned here —
an accidental rename would silently misroute cross-queue activities.
"""

from __future__ import annotations

from sax_platform.contracts.constants import (
    FORGE_TASK_QUEUE,
    OCR_TASK_QUEUE,
    PRODUCT_SLUG,
)


def test_product_slug() -> None:
    # forge, ocr and pbook all share this slug's namespace today: ocr is not its
    # own slug (D102) and pbook's split waits on T6.4.
    assert PRODUCT_SLUG == "forge"


def test_no_namespace_constant() -> None:
    """The namespace is `<slug>-<env>`, so it cannot be a wire constant.

    A `Final` namespace here is what let every environment share one name; the
    value is derived per environment in sax_platform.config instead.
    """
    import sax_platform.contracts.constants as constants

    assert not hasattr(constants, "TEMPORAL_NAMESPACE")


def test_task_queue_names() -> None:
    assert FORGE_TASK_QUEUE == "forge-task-queue"
    assert OCR_TASK_QUEUE == "ocr-task-queue"


def test_task_queues_are_distinct() -> None:
    """Each worker owns one queue; a collision would misroute activities."""
    assert FORGE_TASK_QUEUE != OCR_TASK_QUEUE
