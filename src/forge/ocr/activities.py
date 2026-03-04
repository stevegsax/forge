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
import re
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
    ChunkRef,
    FileContentRef,
    FileContentResult,
    OcrParseResult,
    OcrStoreResult,
    OcrSubmitResult,
    SplitResult,
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
    return {"document": doc, "include_image_base64": True}


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PAGES = 30
CHUNK_SIZE_PAGES = 25


def validate_file_size(file_size_bytes: int, mime_type: str) -> None:
    """Raise ValueError if a non-PDF file exceeds the size cutoff."""
    if mime_type != "application/pdf" and file_size_bytes > MAX_FILE_SIZE_BYTES:
        msg = (
            f"Non-PDF file ({mime_type}) exceeds maximum size of "
            f"{MAX_FILE_SIZE_BYTES} bytes: {file_size_bytes} bytes"
        )
        raise ValueError(msg)


# Matches markdown image references like ![alt](img-0.jpeg) or ![alt](img-12.png)
_IMAGE_REF_PATTERN = re.compile(r"(!\[[^\]]*\])\(([^)]+)\)")


def rewrite_image_references(markdown: str, image_mapping: dict[str, str]) -> str:
    """Rewrite markdown image references to use ocr-image:// URIs.

    Replaces ``![alt](img-0.jpeg)`` with ``![alt](ocr-image://{uuid})``
    for each image ID found in *image_mapping*.
    """
    if not image_mapping:
        return markdown

    def _replace(match: re.Match) -> str:
        alt_bracket = match.group(1)
        image_ref = match.group(2)
        if image_ref in image_mapping:
            return f"{alt_bracket}(ocr-image://{image_mapping[image_ref]})"
        return match.group(0)

    return _IMAGE_REF_PATTERN.sub(_replace, markdown)


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

    If ``_image_mapping`` is present in the JSON (injected by the batch
    poller after storing images), image references in the markdown are
    rewritten from ``img-N.jpeg`` to ``ocr-image://{uuid}``.
    """
    data = json.loads(raw_json)

    # Pop image mapping if present (injected by batch poll activity)
    image_mapping: dict[str, str] = data.pop("_image_mapping", {})
    image_ids = list(image_mapping.values())

    pages = data.get("pages", [])
    page_texts: list[str] = []
    for page in pages:
        md = page.get("markdown", "")
        if image_mapping:
            md = rewrite_image_references(md, image_mapping)
        page_texts.append(md)

    text = "\n\n".join(page_texts)
    model_name = data.get("model", "")
    usage = data.get("usage_info", {})

    return OcrParseResult(
        text=text,
        model_name=model_name,
        input_tokens=usage.get("pages_processed", 0),
        output_tokens=usage.get("doc_size_bytes", 0),
        page_count=len(pages),
        image_count=len(image_ids),
        image_ids=image_ids,
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
    page_count: int = 0,
    image_ids: list[str] | None = None,
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
        page_count=page_count,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        batch_id=batch_id,
        workflow_id=workflow_id,
    )

    # Best-effort: update document_id on pre-stored OCR images
    if image_ids:
        try:
            from forge.store import update_ocr_images_document_id

            update_ocr_images_document_id(engine, image_ids, document_id)
        except Exception:
            logger.warning(
                "Failed to update document_id on OCR images for %s",
                document_id,
                exc_info=True,
            )

    return OcrStoreResult(
        document_id=document_id,
        text_length=len(text),
        page_count=page_count,
    )


def execute_split_file_into_chunks(
    content_id: str,
    mime_type: str,
    file_size_bytes: int,
    engine: Engine,
) -> SplitResult:
    """Split a stored file into chunks for parallel OCR processing.

    Non-PDF files are validated for size and returned as a single chunk.
    PDFs under the size/page cutoff are returned as a single chunk reusing
    the original blob. Large PDFs are split into CHUNK_SIZE_PAGES-page
    chunks, each saved as a new blob; the original blob is deleted.
    """
    import fitz

    from forge.store import delete_file_content, get_file_content, save_file_content

    # Non-PDF: validate and return single chunk
    if mime_type != "application/pdf":
        validate_file_size(file_size_bytes, mime_type)
        return SplitResult(
            chunks=[
                ChunkRef(
                    content_id=content_id,
                    mime_type=mime_type,
                    file_size_bytes=file_size_bytes,
                    chunk_index=0,
                    page_start=1,
                    page_end=1,
                )
            ],
            total_pages=1,
            original_content_id=content_id,
        )

    # Load PDF blob from DB
    blob = get_file_content(engine, content_id)
    if blob is None:
        msg = f"File content not found for content_id={content_id}"
        raise RuntimeError(msg)

    pdf_data = blob["data"]
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    total_pages = len(doc)

    # Small PDF: single chunk reusing original blob
    if total_pages <= MAX_PAGES and file_size_bytes <= MAX_FILE_SIZE_BYTES:
        doc.close()
        return SplitResult(
            chunks=[
                ChunkRef(
                    content_id=content_id,
                    mime_type=mime_type,
                    file_size_bytes=file_size_bytes,
                    chunk_index=0,
                    page_start=1,
                    page_end=total_pages,
                )
            ],
            total_pages=total_pages,
            original_content_id=content_id,
        )

    # Large PDF: split into chunks
    chunks: list[ChunkRef] = []
    for chunk_index, start_page in enumerate(range(0, total_pages, CHUNK_SIZE_PAGES)):
        end_page = min(start_page + CHUNK_SIZE_PAGES, total_pages)

        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
        chunk_bytes = chunk_doc.tobytes()
        chunk_doc.close()

        chunk_content_id = str(uuid.uuid4())
        save_file_content(
            engine,
            content_id=chunk_content_id,
            data=chunk_bytes,
            mime_type="application/pdf",
            file_size_bytes=len(chunk_bytes),
        )

        chunks.append(
            ChunkRef(
                content_id=chunk_content_id,
                mime_type="application/pdf",
                file_size_bytes=len(chunk_bytes),
                chunk_index=chunk_index,
                page_start=start_page + 1,  # 1-based
                page_end=end_page,  # 1-based
            )
        )

    doc.close()

    # Delete original blob now that chunks are saved
    delete_file_content(engine, content_id)

    return SplitResult(
        chunks=chunks,
        total_pages=total_pages,
        original_content_id=content_id,
    )


def execute_reassemble_ocr_chunks(
    *,
    document_id: str,
    chunk_document_ids: list[str],
    file_path: str,
    total_pages: int,
    engine: Engine,
) -> OcrStoreResult:
    """Combine OCR results from multiple chunks into a single result.

    Reads each chunk's ocr_results row, joins text, sums tokens,
    stores the combined result under the real document_id, and
    deletes the chunk rows.  Also reassigns OCR images from chunk
    document_ids to the final document_id.
    """
    from forge.store import delete_ocr_results, get_ocr_result, save_ocr_result

    texts: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    model_name = ""
    batch_id = ""

    for chunk_doc_id in chunk_document_ids:
        row = get_ocr_result(engine, chunk_doc_id)
        if row is None:
            msg = f"OCR result not found for chunk document_id={chunk_doc_id}"
            raise RuntimeError(msg)
        texts.append(row["text"])
        total_input_tokens += row["input_tokens"]
        total_output_tokens += row["output_tokens"]
        if not model_name:
            model_name = row["model_name"]
        if not batch_id:
            batch_id = row["batch_id"]

    combined_text = "\n\n".join(texts)

    save_ocr_result(
        engine,
        document_id=document_id,
        file_path=file_path,
        text=combined_text,
        page_count=total_pages,
        model_name=model_name,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        batch_id=batch_id,
        workflow_id=f"reassemble-{document_id}",
    )

    # Best-effort: reassign OCR images from chunk doc IDs to the final doc ID
    try:
        from forge.store import reassign_ocr_images_document_id

        reassign_ocr_images_document_id(engine, chunk_document_ids, document_id)
    except Exception:
        logger.warning(
            "Failed to reassign OCR images for document %s",
            document_id,
            exc_info=True,
        )

    # Clean up chunk rows
    delete_ocr_results(engine, chunk_document_ids)

    return OcrStoreResult(
        document_id=document_id,
        text_length=len(combined_text),
        page_count=total_pages,
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


@activity.defn
async def split_file_into_chunks(input_json: str) -> SplitResult:
    """Activity: split a stored file into chunks for parallel OCR.

    Takes JSON with content_id, mime_type, file_size_bytes.
    Returns SplitResult with ordered ChunkRef list.
    """
    from forge.store import get_db_path, get_engine

    data = json.loads(input_json)
    logger.info("Splitting file into chunks: content_id=%s", data.get("content_id", ""))

    db_path = get_db_path()
    if db_path is None:
        msg = "Cannot split file: database is disabled"
        raise RuntimeError(msg)

    engine = get_engine(db_path)
    return execute_split_file_into_chunks(
        content_id=data["content_id"],
        mime_type=data["mime_type"],
        file_size_bytes=data["file_size_bytes"],
        engine=engine,
    )


@activity.defn
async def reassemble_ocr_chunks(input_json: str) -> OcrStoreResult:
    """Activity: combine OCR results from multiple chunks into one.

    Takes JSON with document_id, chunk_document_ids, file_path, total_pages.
    Returns OcrStoreResult for the combined document.
    """
    from forge.store import get_db_path, get_engine

    data = json.loads(input_json)
    logger.info("Reassembling OCR chunks: document_id=%s", data.get("document_id", ""))

    db_path = get_db_path()
    if db_path is None:
        msg = "Cannot reassemble chunks: database is disabled"
        raise RuntimeError(msg)

    engine = get_engine(db_path)
    return execute_reassemble_ocr_chunks(
        document_id=data["document_id"],
        chunk_document_ids=data["chunk_document_ids"],
        file_path=data["file_path"],
        total_pages=data["total_pages"],
        engine=engine,
    )
