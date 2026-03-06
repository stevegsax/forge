"""Data models for OCR workflows."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OcrSubmitInput(BaseModel):
    """Input to the OcrSubmitWorkflow."""

    file_path: str
    model_name: str = "mistral:mistral-ocr-latest"
    max_tokens: int = 16384
    document_id: str = Field(default="", description="Auto-generated if empty.")


class OcrBatchRef(BaseModel):
    """Lightweight reference returned by submit_ocr_batch activity.

    Contains only the batch tracking metadata needed by the workflow;
    callers receive OcrStoreResult once OCR completes.
    """

    batch_id: str
    request_id: str


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
            "Override export directory. Defaults to"
            " $XDG_DATA_HOME/forge/ocr-export/<document_id>."
        ),
    )


class OcrExportResult(BaseModel):
    """Result from the OcrExportWorkflow."""

    document_id: str
    export_dir: str
    markdown_path: str
    image_count: int


class OcrMarkInput(BaseModel):
    """Input to mark/clear removal workflows. One document per invocation."""

    document_id: str


class OcrMarkResult(BaseModel):
    """Result from mark/clear removal workflows."""

    document_id: str
    found: bool
