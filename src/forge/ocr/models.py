"""Data models for OCR workflows."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OcrSubmitInput(BaseModel):
    """Input to the OcrSubmitWorkflow."""

    file_path: str
    model_name: str = "mistral:mistral-ocr-latest"
    max_tokens: int = 16384
    document_id: str = Field(default="", description="Auto-generated if empty.")


class OcrSubmitResult(BaseModel):
    """Result from the OcrSubmitWorkflow."""

    batch_id: str
    request_id: str
    document_id: str
    workflow_id: str
    chunk_count: int = 1
    total_pages: int = 0


class OcrStoreInput(BaseModel):
    """Input to the OcrStoreWorkflow."""

    batch_id: str
    request_id: str
    document_id: str
    file_path: str  # original source file (metadata)


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
