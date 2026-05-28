"""Pure request/result models for the survivable ``persist_to_store`` activity.

These models cross the workflow boundary, so this module imports **only** pure
pydantic models (``forge.models``) — never ``sqlalchemy`` or ``forge.store``. That
keeps it safe to import inside the Temporal workflow sandbox under
``workflow.unsafe.imports_passed_through()``.

Each store write is expressed as a discriminated ``PersistRequest`` variant (tagged
by ``kind``). The ``persist_to_store`` activity dispatches on ``kind`` and applies
an idempotent write, returning a ``PersistResult`` whose ``applied`` flag reports
whether a new row was written (vs. a duplicate absorbed on retry).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from forge.models import PlaybookEntry, TaskResult


class PersistInteraction(BaseModel):
    """An LLM interaction row (interactions table), keyed by idempotency_key."""

    kind: Literal["interaction"] = "interaction"
    idempotency_key: str
    task_id: str
    role: str
    system_prompt: str
    user_prompt: str
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    step_id: str | None = None
    sub_task_id: str | None = None
    explanation: str = ""
    context_stats_json: str | None = None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class PersistRun(BaseModel):
    """A workflow run result (runs table), keyed by workflow_id."""

    kind: Literal["run"] = "run"
    workflow_id: str
    task_result: TaskResult


class PersistBatchSubmission(BaseModel):
    """A submitted batch job (batch_jobs table), keyed by request_id."""

    kind: Literal["batch_submission"] = "batch_submission"
    request_id: str
    batch_id: str
    workflow_id: str
    provider: str = "anthropic"
    file_path: str | None = None
    document_id: str | None = None


class PersistBatchFailure(BaseModel):
    """A failed batch submission (batch_jobs table), keyed by request_id."""

    kind: Literal["batch_failure"] = "batch_failure"
    request_id: str
    workflow_id: str
    error_message: str
    provider: str = "anthropic"
    file_path: str | None = None
    document_id: str | None = None


class PersistBatchStatus(BaseModel):
    """A batch job status transition (plain UPDATE on batch_jobs.id)."""

    kind: Literal["batch_status"] = "batch_status"
    request_id: str
    status: str
    error_message: str | None = None


class PersistOcrResult(BaseModel):
    """An OCR result row (ocr_results table), keyed by document_id."""

    kind: Literal["ocr_result"] = "ocr_result"
    document_id: str
    file_path: str
    text: str
    model_name: str
    input_tokens: int
    output_tokens: int
    batch_id: str
    workflow_id: str
    page_count: int = 0
    file_hash: str | None = None


class PersistPlaybooks(BaseModel):
    """Extracted playbook entries (playbooks table), keyed per-entry by uuid5."""

    kind: Literal["playbooks"] = "playbooks"
    extraction_workflow_id: str
    entries: list[PlaybookEntry]


PersistRequest = Annotated[
    PersistInteraction
    | PersistRun
    | PersistBatchSubmission
    | PersistBatchFailure
    | PersistBatchStatus
    | PersistOcrResult
    | PersistPlaybooks,
    Field(discriminator="kind"),
]


class PersistResult(BaseModel):
    """Outcome of a persist: which kind ran and whether a new row was written."""

    kind: str
    applied: bool
