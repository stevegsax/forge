"""OCR activities for Forge.

Activities for reading files, submitting OCR batch requests,
parsing results, and storing extracted text.

Design follows Function Core / Imperative Shell:
- Pure functions: build_ocr_messages, detect_mime_type
- Testable functions: execute_read_file, execute_submit_ocr_batch,
  execute_parse_ocr_result, execute_store_ocr_result
- Imperative shell: Temporal activities wrapping the testable functions
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from temporalio import activity

from forge.llm_providers.models import (
    DocumentContent,
    ImageContent,
    Message,
    TextContent,
)
from forge.ocr.models import (
    FileContentRef,
    FileContentResult,
    OcrParseResult,
    OcrStoreResult,
    OcrSubmitResult,
)

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from forge.llm_providers.protocol import LLMProvider
    from forge.ocr.models import OcrSubmitInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def detect_mime_type(file_path: str) -> str:
    """Detect MIME type from file extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def build_ocr_messages(
    base64_data: str,
    mime_type: str,
    instruction: str = "Extract all text from this document. Return the full text content.",
) -> list[Message]:
    """Build multimodal messages for OCR."""
    if mime_type.startswith("image/"):
        content_block: ImageContent | DocumentContent = ImageContent(
            media_type=mime_type,
            data=base64_data,
        )
    else:
        content_block = DocumentContent(
            media_type=mime_type,
            data=base64_data,
        )

    return [
        Message(role="system", content="You are a document OCR assistant."),
        Message(
            role="user",
            content=[content_block, TextContent(text=instruction)],
        ),
    ]


def build_ocr_batch_body(base64_data: str, mime_type: str) -> dict:
    """Build the request body for the /v1/ocr batch endpoint."""
    data_uri = f"data:{mime_type};base64,{base64_data}"
    if mime_type.startswith("image/"):
        doc: dict = {"type": "image_url", "image_url": data_uri}
    else:
        doc = {"type": "document_url", "document_url": data_uri}
    return {"document": doc}


# ---------------------------------------------------------------------------
# Testable functions
# ---------------------------------------------------------------------------


def execute_read_file(file_path: str) -> FileContentResult:
    """Read a file and return base64-encoded content with MIME type."""
    path = Path(file_path)
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    mime_type = detect_mime_type(file_path)
    return FileContentResult(
        base64_data=encoded,
        mime_type=mime_type,
        file_size_bytes=len(raw),
    )


def execute_read_and_store_file(file_path: str, engine: Engine) -> FileContentRef:
    """Read a file and store raw bytes in the database.

    Returns a lightweight reference suitable for Temporal payloads.
    """
    from forge.store import save_file_content

    path = Path(file_path)
    raw = path.read_bytes()
    mime_type = detect_mime_type(file_path)
    content_id = str(uuid.uuid4())

    save_file_content(
        engine,
        content_id=content_id,
        data=raw,
        mime_type=mime_type,
        file_size_bytes=len(raw),
    )

    return FileContentRef(
        content_id=content_id,
        mime_type=mime_type,
        file_size_bytes=len(raw),
    )


async def execute_submit_ocr_batch(
    input: OcrSubmitInput,
    file_content: FileContentResult,
    provider: LLMProvider,
    workflow_id: str,
) -> OcrSubmitResult:
    """Build OCR request body and submit to /v1/ocr batch endpoint."""
    from forge.llm_providers import parse_model_id

    document_id = input.document_id or str(uuid.uuid4())
    _, model = parse_model_id(input.model_name)

    body = build_ocr_batch_body(file_content.base64_data, file_content.mime_type)
    request_id = str(uuid.uuid4())
    batch_request = {"custom_id": request_id, "body": body}
    batch_id = await provider.submit_batch(
        [batch_request], model, endpoint="/v1/ocr"
    )

    return OcrSubmitResult(
        batch_id=batch_id,
        request_id=request_id,
        document_id=document_id,
        workflow_id=workflow_id,
    )


def execute_parse_ocr_result(
    raw_json: str,
    provider_name: str = "mistral",
) -> OcrParseResult:
    """Parse OCR batch result into extracted text.

    The OCR endpoint returns ``pages[].markdown`` and ``usage_info``
    instead of the chat-completion format.  ``provider_name`` is kept
    for signature compatibility but is no longer used.
    """
    data = json.loads(raw_json)
    pages = data.get("pages", [])
    text = "\n\n".join(page.get("markdown", "") for page in pages)
    model_name = data.get("model", "")
    usage = data.get("usage_info", {})

    return OcrParseResult(
        text=text,
        model_name=model_name,
        input_tokens=usage.get("pages_processed", 0),
        output_tokens=usage.get("doc_size_bytes", 0),
    )


def execute_store_ocr_result(
    *,
    document_id: str,
    file_path: str,
    text: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    batch_id: str,
    workflow_id: str,
) -> OcrStoreResult:
    """Store OCR result in the database."""
    from forge.store import get_db_path, get_engine, save_ocr_result

    db_path = get_db_path()
    if db_path is None:
        return OcrStoreResult(
            document_id=document_id,
            text_length=len(text),
            stored=False,
        )

    engine = get_engine(db_path)
    save_ocr_result(
        engine,
        document_id=document_id,
        file_path=file_path,
        text=text,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        batch_id=batch_id,
        workflow_id=workflow_id,
    )
    return OcrStoreResult(
        document_id=document_id,
        text_length=len(text),
    )


# ---------------------------------------------------------------------------
# Imperative shell (Temporal activities)
# ---------------------------------------------------------------------------


@activity.defn
async def read_and_store_file_content(file_path: str) -> FileContentRef:
    """Activity: read file and store raw bytes in the database.

    Returns a lightweight FileContentRef instead of the full file content,
    avoiding Temporal's 2MB payload limit for large files.
    """
    from forge.store import get_db_path, get_engine

    logger.info("Reading and storing file: %s", file_path)
    db_path = get_db_path()
    if db_path is None:
        msg = "Cannot store file content: database is disabled"
        raise RuntimeError(msg)

    engine = get_engine(db_path)
    return execute_read_and_store_file(file_path, engine)


@activity.defn
async def submit_ocr_batch(input_json: str) -> OcrSubmitResult:
    """Activity: submit OCR batch request.

    Takes JSON-serialized OcrSubmitInput + FileContentRef to avoid
    complex activity input. Loads file bytes from the database,
    base64-encodes in memory, and submits to the provider.
    """
    from forge.llm_providers import get_provider, parse_model_id
    from forge.ocr.models import OcrSubmitInput
    from forge.store import (
        delete_file_content,
        get_db_path,
        get_engine,
        get_file_content,
        record_batch_failure,
        record_batch_submission,
    )

    data = json.loads(input_json)
    ocr_input = OcrSubmitInput.model_validate(data["submit_input"])
    store_workflow_id = data["store_workflow_id"]

    # Load file content from database
    db_path = get_db_path()
    if db_path is None:
        msg = "Cannot load file content: database is disabled"
        raise RuntimeError(msg)

    engine = get_engine(db_path)

    file_content_ref = data["file_content_ref"]
    content_id = file_content_ref["content_id"]
    blob = get_file_content(engine, content_id)
    if blob is None:
        msg = f"File content not found for content_id={content_id}"
        raise RuntimeError(msg)

    # Build FileContentResult from stored bytes (base64-encode in memory)
    encoded = base64.b64encode(blob["data"]).decode("ascii")
    file_content = FileContentResult(
        base64_data=encoded,
        mime_type=blob["mime_type"],
        file_size_bytes=blob["file_size_bytes"],
    )

    provider = get_provider(ocr_input.model_name)
    provider_name, _ = parse_model_id(ocr_input.model_name)

    try:
        result = await execute_submit_ocr_batch(
            ocr_input, file_content, provider, store_workflow_id
        )
    except Exception as exc:
        request_id = str(uuid.uuid4())
        try:
            record_batch_failure(
                engine,
                request_id=request_id,
                workflow_id=store_workflow_id,
                error_message=str(exc),
                provider=provider_name,
            )
        except Exception:
            logger.error("Failed to record batch failure", exc_info=True)
        raise

    # Clean up the BLOB after successful submission
    try:
        delete_file_content(engine, content_id)
    except Exception:
        logger.error("Failed to delete file content blob %s", content_id, exc_info=True)

    # Record batch submission for the poller to find — if recording fails,
    # the batch is submitted but untracked. A duplicate on retry is better
    # than a lost batch.
    record_batch_submission(
        engine,
        request_id=result.request_id,
        batch_id=result.batch_id,
        workflow_id=store_workflow_id,
        provider=provider_name,
    )

    return result


@activity.defn
async def parse_ocr_result(raw_json: str) -> OcrParseResult:
    """Activity: parse raw OCR batch result."""
    logger.info("Parsing OCR result")
    return execute_parse_ocr_result(raw_json, provider_name="mistral")


@activity.defn
async def store_ocr_result(input_json: str) -> OcrStoreResult:
    """Activity: store OCR result in database."""
    data = json.loads(input_json)
    logger.info("Storing OCR result: document_id=%s", data.get("document_id", ""))
    return execute_store_ocr_result(**data)
