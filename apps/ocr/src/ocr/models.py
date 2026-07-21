"""Data models for OCR workflows."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class OcrProcessingStatus(StrEnum):
    """Coarse processing lifecycle owned by OCR (single-writer: the OCR workflow).

    Written to the ``ocr_job_status`` projection and validated at the store
    boundary (``upsert_ocr_job_status``), exactly as ``BatchJobStatus`` is in the
    platform contracts. Distinct from the platform's provider-batch
    ``BatchJobStatus`` — the two are joined on ``request_id`` for the user-facing
    status view (see ``_derive_status``).
    """

    SUBMITTED = "submitted"
    """The batch has been submitted to the provider; the store child is polling."""

    PROCESSING = "processing"
    """Reserved in-flight state (no writer today; kept so the derivation table and
    any future intermediate writer stay covered)."""

    STORED = "stored"
    """Terminal success: the fetched result's text + images have been stored."""

    FAILED = "failed"
    """Terminal failure: submit error, a provider-terminal batch, a give-up at the
    ceiling, or a fetch/store error."""


class OcrJobDerivedStatus(StrEnum):
    """Aggregated OCR job status as shown by ``ocr list``.

    A display-level label derived from a submission's ``OcrProcessingStatus``
    (authoritative) and the joined provider ``BatchJobStatus``. See
    ``_derive_status`` for the full derivation table.
    """

    PROCESSING = "processing"
    """OCR is still submitted/processing and the provider batch has not failed."""

    SUCCEEDED = "succeeded"
    """OCR reached the terminal STORED state."""

    ERRORED = "errored"
    """OCR reached FAILED, or the provider batch is FAILED / EXPIRED / MISSING."""

    UNKNOWN = "unknown"
    """The stored OCR status is an unrecognized (legacy) value."""


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
    """Lightweight reference to a submitted batch (batch_id + correlation id)."""

    batch_id: str
    request_id: str


class OcrBatchRequestRef(BaseModel):
    """Reference to a pre-built /v1/ocr request blob stashed in S3.

    ``request_id`` is the single correlation id (== provider custom_id ==
    platform ``batch_jobs`` PK), minted once when the blob is built. The submit
    workflow hands ``s3_key`` + ``model`` to ocr's own ``submit_ocr_batch``
    activity (via ``OcrSubmitBatchInput``).
    """

    request_id: str
    s3_key: str
    model: str


class OcrSubmitResult(BaseModel):
    """Result from OcrSubmitWorkflow — returned once every chunk has stored.

    The workflow submits each chunk's batch, then awaits its parent-awaited
    OcrStoreWorkflow children (each polls its own Mistral batch and stores the
    result). It returns after all children complete (and, for a split document,
    after inline reassembly). ``batch_refs`` records the submitted batches.
    """

    document_id: str
    batch_refs: list[OcrBatchRef] = Field(default_factory=list)
    chunk_count: int = 0
    skipped: bool = False
    skip_reason: str = ""


class OcrStoreInput(BaseModel):
    """Input to the OcrStoreWorkflow.

    Carries the real provider ``batch_id`` (the child polls it directly); the
    signal-era ``gather_workflow_id``/empty-``batch_id`` fields are gone.
    """

    batch_id: str
    request_id: str
    document_id: str
    file_path: str  # original source file (metadata)


class OcrSubmitBatchInput(BaseModel):
    """Input to the ``submit_ocr_batch`` activity.

    ``s3_key`` locates the pre-built /v1/ocr request blob in ocr's own S3;
    ``model`` is the provider model id (no provider prefix).
    """

    s3_key: str
    model: str


class OcrBatchStatusInput(BaseModel):
    """Input to the ``ocr_batch_status`` activity (a status-only poll)."""

    batch_id: str


class OcrFetchStoreInput(BaseModel):
    """Input to the ``fetch_and_store_ocr_result`` activity.

    The activity downloads the finished batch, selects this request's entry by
    ``request_id``, and stores text + images under ``document_id``. Result bytes
    never transit workflow history — only the small summary returns.
    """

    batch_id: str
    request_id: str
    document_id: str
    file_path: str
    workflow_id: str


class OcrSplitInput(BaseModel):
    """Input to the ``split_file_into_chunks`` activity."""

    content_id: str
    mime_type: str
    file_size_bytes: int


class OcrBuildRequestInput(BaseModel):
    """Input to the ``build_ocr_request_blob`` activity.

    ``model_name`` is the full ``provider:model`` id; the activity strips the
    prefix when recording the request blob's provider model.
    """

    content_id: str
    mime_type: str
    model_name: str


class OcrStatusUpsertInput(BaseModel):
    """Input to the ``upsert_ocr_status`` activity (single-writer status projection).

    ``status`` is a validated ``OcrProcessingStatus`` — every write site passes an
    enum member, so an out-of-vocabulary status can never reach the activity.
    """

    request_id: str
    document_id: str
    file_path: str = ""
    status: OcrProcessingStatus
    error_message: str | None = None


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


class OcrExportInput(BaseModel):
    """Input to the OcrExportWorkflow."""

    document_id: str
    output_dir: str = Field(
        default="",
        description=(
            "Override export directory. Defaults to $XDG_DATA_HOME/ocr/export/<document_id>."
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

    ``status`` is the derived display label (``OcrJobDerivedStatus``), serialized
    as its string value for JSON compatibility.
    """

    file_path: str
    document_id: str = ""
    status: OcrJobDerivedStatus = OcrJobDerivedStatus.UNKNOWN
    chunk_count: int = 1
    created_at: str = ""


class OcrListJobsInput(BaseModel):
    """Input to the OcrListJobsWorkflow."""

    limit: int = Field(default=50, description="Maximum number of jobs to return.")
    status_filter: str = Field(
        default="",
        description=(
            "Filter by derived aggregate status (processing, succeeded, errored, unknown)."
        ),
    )


class OcrListJobsResult(BaseModel):
    """Result from the OcrListJobsWorkflow."""

    jobs: list[OcrJobEntry] = Field(default_factory=list)
    total: int = 0


# (OcrSyncInput removed — synchronous OCR path deleted in the OCR-out cut.)
