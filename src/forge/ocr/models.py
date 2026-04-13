"""Data models for OCR workflows."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class OcrJobDerivedStatus(StrEnum):
    """Aggregated OCR job status as shown by ``ocr list``.

    Derived from the underlying ``BatchJobStatus`` values of a submission's
    chunks. See ``execute_list_ocr_jobs`` for the derivation rules. This is
    a display-level label, distinct from ``BatchJobStatus`` which is the
    DB-level per-chunk state.
    """

    PROCESSING = "processing"
    """At least one chunk is still SUBMITTED or STORING — not done yet."""

    SUCCEEDED = "succeeded"
    """Every chunk reached BatchJobStatus.SUCCEEDED."""

    ERRORED = "errored"
    """At least one chunk is in ERRORED / FAILED / EXPIRED / CANCELED /
    MISSING."""

    UNKNOWN = "unknown"
    """Chunks are in a combination that doesn't match any rule above."""


class OcrSubmitInput(BaseModel):
    """Input to the OcrSubmitWorkflow."""

    file_path: str

    @field_validator("file_path")
    @classmethod
    def file_path_must_be_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            msg = "file_path must be a non-empty string"
            raise ValueError(msg)
        return v

    model_name: str = "mistral:mistral-ocr-latest"
    max_tokens: int = 16384
    document_id: str = Field(default="", description="Auto-generated if empty.")
    skip_duplicate_detection: bool = Field(
        default=False,
        description="Skip duplicate detection and re-submit even if already OCR'd.",
    )


class OcrBatchRef(BaseModel):
    """Lightweight reference returned by submit_ocr_batch activity.

    Contains only the batch tracking metadata needed by the workflow.
    """

    batch_id: str
    request_id: str


class OcrSubmitResult(BaseModel):
    """Result from OcrSubmitWorkflow — returned immediately after batch submission.

    The workflow does not wait for OCR to complete. Child workflows
    (OcrStoreWorkflow / OcrGatherWorkflow) continue running independently
    and will store the results when the batch finishes.
    """

    document_id: str
    batch_refs: list[OcrBatchRef] = Field(default_factory=list)
    chunk_count: int = 0
    skipped: bool = False
    skip_reason: str = ""


class OcrStoreInput(BaseModel):
    """Input to the OcrStoreWorkflow."""

    batch_id: str
    request_id: str
    document_id: str
    file_path: str  # original source file (metadata)
    gather_workflow_id: str = ""  # if set, signal this workflow on completion


class OcrStoreResult(BaseModel):
    """Result from the OcrStoreWorkflow."""

    document_id: str
    text_length: int
    page_count: int = 0
    stored: bool = True
    skipped: bool = False
    skip_reason: str = ""


class FileContentResult(BaseModel):
    """Base64-encoded file content used for LLM batch submission."""

    base64_data: str
    mime_type: str
    file_size_bytes: int


class FileContentRef(BaseModel):
    """Lightweight reference to file content stored in the database."""

    content_id: str
    mime_type: str
    file_size_bytes: int


class OcrParseResult(BaseModel):
    """Result from parse_ocr_result activity."""

    text: str
    model_name: str
    input_tokens: int
    output_tokens: int
    page_count: int = 0
    image_count: int = 0
    image_ids: list[str] = Field(default_factory=list)


class ChunkRef(BaseModel):
    """Reference to a single chunk of a split document."""

    content_id: str  # FK to file_content_blobs
    mime_type: str
    file_size_bytes: int
    chunk_index: int  # 0-based sequence number
    page_start: int  # 1-based first page in this chunk
    page_end: int  # 1-based last page in this chunk


class SplitResult(BaseModel):
    """Result from split_file_into_chunks activity."""

    chunks: list[ChunkRef]
    total_pages: int
    original_content_id: str


class OcrReassembleInput(BaseModel):
    """Input to the reassemble_ocr_chunks activity."""

    document_id: str
    chunk_document_ids: list[str]  # ordered
    file_path: str
    total_pages: int


class OcrGatherInput(BaseModel):
    """Input to the OcrGatherWorkflow."""

    document_id: str
    chunk_document_ids: list[str]  # ordered
    store_workflow_ids: list[str]  # OcrStoreWorkflow IDs to await
    file_path: str
    total_pages: int


class OcrExportInput(BaseModel):
    """Input to the OcrExportWorkflow."""

    document_id: str
    output_dir: str = Field(
        default="",
        description=(
            "Override export directory. Defaults to $XDG_DATA_HOME/forge/ocr-export/<document_id>."
        ),
    )


class OcrExportResult(BaseModel):
    """Result from the OcrExportWorkflow."""

    document_id: str
    export_dir: str
    markdown_path: str
    image_count: int
    status: str = "exported"


class OcrMarkInput(BaseModel):
    """Input to mark/clear removal workflows. One document per invocation."""

    document_id: str


class OcrDuplicateCheckResult(BaseModel):
    """Result from check_ocr_duplicate activity."""

    is_duplicate: bool
    existing_document_id: str = ""


class OcrMarkResult(BaseModel):
    """Result from mark/clear removal workflows."""

    document_id: str
    found: bool


class OcrJobEntry(BaseModel):
    """A single OCR job submission as seen by the user.

    ``status`` values are ``OcrJobDerivedStatus`` members, serialized as
    their string value for JSON compatibility.
    """

    file_path: str
    document_id: str = ""
    status: str = ""
    chunk_count: int = 1
    created_at: str = ""


class OcrListJobsInput(BaseModel):
    """Input to the OcrListJobsWorkflow."""

    limit: int = Field(default=50, description="Maximum number of jobs to return.")
    status_filter: str = Field(
        default="",
        description="Filter by aggregate status (processing, succeeded, errored).",
    )


class OcrListJobsResult(BaseModel):
    """Result from the OcrListJobsWorkflow."""

    jobs: list[OcrJobEntry] = Field(default_factory=list)
    total: int = 0


class OcrSyncInput(BaseModel):
    """Input to the OcrSyncWorkflow (synchronous OCR path)."""

    file_path: str

    @field_validator("file_path")
    @classmethod
    def file_path_must_be_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            msg = "file_path must be a non-empty string"
            raise ValueError(msg)
        return v

    model_name: str = "mistral:mistral-ocr-latest"
    document_id: str = Field(default="", description="Auto-generated if empty.")
    skip_duplicate_detection: bool = Field(
        default=False,
        description="Skip duplicate detection and re-submit even if already OCR'd.",
    )
