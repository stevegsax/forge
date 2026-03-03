"""Data models for OCR workflows."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OcrSubmitInput(BaseModel):
    """Input to the OcrSubmitWorkflow."""

    file_path: str
    model_name: str = "mistral:pixtral-large-latest"
    max_tokens: int = 16384
    document_id: str = Field(default="", description="Auto-generated if empty.")


class OcrSubmitResult(BaseModel):
    """Result from the OcrSubmitWorkflow."""

    batch_id: str
    request_id: str
    document_id: str
    workflow_id: str


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
    """Result from read_file_as_base64 activity."""

    base64_data: str
    mime_type: str
    file_size_bytes: int


class OcrParseResult(BaseModel):
    """Result from parse_ocr_result activity."""

    text: str
    model_name: str
    input_tokens: int
    output_tokens: int
