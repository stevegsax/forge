"""Cross-repo Temporal identifiers shared by Forge and its consumer apps.

These are part of the wire contract: the platform and a consumer app must agree
on the namespace and task-queue names, so they live here rather than being
redefined (and risking drift) in each repo. No signal is part of the wire
contract anymore — the timer-loop batch transport (D88, T4.1/T4.2) has each
workflow poll and fetch its own result, so cross-workflow signaling is gone.
"""

from __future__ import annotations

from typing import Final

# The registration slug that owns the Temporal namespace for every process in
# this monorepo. The namespace itself is ``<slug>-<env>`` — it varies by
# environment, so it is *derived*
# (:func:`sax_platform.config.temporal_namespace_for`) rather than pinned here.
# The bare slug and ``"default"`` are namespaces on no server, so a process that
# fails to derive one cannot land anywhere.
#
# forge, ocr and pbook all resolve to this single slug today: ocr is
# deliberately not its own slug (D102), and pbook shares forge's namespace until
# T6.4 deletes forge's cross-queue dispatch into ``pbook-task-queue`` (child
# workflows and cross-queue activities are namespace-bound). See
# sax-temporal/docs/namespaces.md.
PRODUCT_SLUG: Final = "forge"

# Task queues. Each worker owns one queue; an activity/workflow runs on the
# worker registered for the queue it is scheduled on.
FORGE_TASK_QUEUE: Final = "forge-task-queue"
OCR_TASK_QUEUE: Final = "ocr-task-queue"
