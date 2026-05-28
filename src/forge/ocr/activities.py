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

from sax_llm.models import (
    DocumentContent,
    ExtractedImage,
    ImageContent,
    Message,
    TextContent,
)
from temporalio import activity

from forge.ocr.models import (
    ChunkRef,
    FileContentRef,
    FileContentResult,
    OcrBatchRef,
    OcrDuplicateCheckResult,
    OcrExportResult,
    OcrJobDerivedStatus,
    OcrJobEntry,
    OcrListJobsResult,
    OcrMarkResult,
    OcrParseResult,
    OcrStoreResult,
    SplitResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sax_llm.protocol import LLMProvider
    from sqlalchemy import Engine

    from forge.ocr.models import OcrSubmitInput

    # Callable that stores extracted images and returns {original_id: uuid}
    StoreImagesFn = Callable[[list[ExtractedImage]], dict[str, str]]

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


# Matches ocr-image:// URIs in markdown: ![alt](ocr-image://{uuid})
_OCR_IMAGE_URI_PATTERN = re.compile(r"(!\[[^\]]*\])\(ocr-image://([^)]+)\)")


def _mime_to_extension(mime_type: str) -> str:
    """Convert a MIME type to a file extension (with leading dot)."""
    ext = mimetypes.guess_extension(mime_type)
    # mimetypes returns .jpe or .jpg on some platforms for image/jpeg
    if ext in (".jpe", ".jpg"):
        return ".jpeg"
    return ext or ".bin"


# JPEG Start-Of-Image marker
_JPEG_SOI = b"\xff\xd8\xff"
# PNG signature
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _strip_image_prefix(data: bytes) -> bytes:
    """Strip corrupt prefix bytes from image data.

    Early OCR image storage decoded data-URI-prefixed base64 without
    stripping the ``data:image/...;base64,`` header, producing garbage
    bytes before the real image start.  This function finds the true
    image signature and strips the prefix.
    """
    if data[:3] == _JPEG_SOI or data[:8] == _PNG_SIG:
        return data
    # Try to find the real start
    for marker in (_JPEG_SOI, _PNG_SIG):
        idx = data.find(marker)
        if idx > 0:
            return data[idx:]
    return data


def rewrite_ocr_uris_to_local(
    markdown: str,
    image_id_to_filename: dict[str, str],
) -> str:
    """Rewrite ``ocr-image://{uuid}`` references to local filenames.

    Used during export to make the markdown reference exported image files.
    """
    if not image_id_to_filename:
        return markdown

    def _replace(match: re.Match) -> str:
        alt_bracket = match.group(1)
        image_id = match.group(2)
        if image_id in image_id_to_filename:
            return f"{alt_bracket}({image_id_to_filename[image_id]})"
        return match.group(0)

    return _OCR_IMAGE_URI_PATTERN.sub(_replace, markdown)


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
) -> OcrBatchRef:
    """Build OCR request body and submit to /v1/ocr batch endpoint."""
    from sax_llm import parse_model_id

    _, model = parse_model_id(input.model_name)

    body = build_ocr_batch_body(file_content.base64_data, file_content.mime_type)
    request_id = str(uuid.uuid4())
    batch_request = {"custom_id": request_id, "body": body}
    batch_id = await provider.submit_batch([batch_request], model, endpoint="/v1/ocr")

    return OcrBatchRef(
        batch_id=batch_id,
        request_id=request_id,
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
    from forge.store import get_store_engine, save_ocr_result

    file_hash = None
    if file_path and Path(file_path).is_file():
        file_hash = compute_file_hash(file_path)

    engine = get_store_engine()
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
        file_hash=file_hash,
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

    file_hash = None
    if file_path and Path(file_path).is_file():
        file_hash = compute_file_hash(file_path)

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
        file_hash=file_hash,
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


def _get_export_dir(document_id: str, output_dir: str) -> Path:
    """Resolve the export directory for a document.

    If *output_dir* is provided, use it directly. Otherwise default to
    ``$XDG_DATA_HOME/forge/ocr-export/<document_id>``.
    """
    import os

    if output_dir:
        return Path(output_dir)

    xdg = os.environ.get("XDG_DATA_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "forge" / "ocr-export" / document_id


def execute_export_ocr_document(
    *,
    document_id: str,
    output_dir: str,
    engine: Engine,
) -> OcrExportResult:
    """Export OCR text and images to the filesystem.

    Creates a directory containing ``<document_id>.md`` and all associated
    images with ``ocr-image://`` URIs rewritten to local filenames.
    """
    from forge.store import get_ocr_image, get_ocr_images, get_ocr_result

    # Load text
    ocr_row = get_ocr_result(engine, document_id)
    if ocr_row is None:
        return OcrExportResult(
            document_id=document_id,
            export_dir="",
            markdown_path="",
            image_count=0,
            status="not_found",
        )

    text: str = ocr_row["text"]

    # Load image metadata
    image_rows = get_ocr_images(engine, document_id)

    # Build id→filename mapping and write images
    export_path = _get_export_dir(document_id, output_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    id_to_filename: dict[str, str] = {}
    for img_meta in image_rows:
        image_id: str = img_meta["id"]
        ext = _mime_to_extension(img_meta["mime_type"])
        filename = f"{image_id}{ext}"
        id_to_filename[image_id] = filename

        # Load full image blob
        img_full = get_ocr_image(engine, image_id)
        if img_full is not None:
            (export_path / filename).write_bytes(_strip_image_prefix(img_full["data"]))

    # Rewrite URIs and write markdown
    exported_text = rewrite_ocr_uris_to_local(text, id_to_filename)
    original_stem = Path(ocr_row["file_path"]).stem if ocr_row.get("file_path") else document_id
    md_path = export_path / f"{original_stem}.md"
    md_path.write_text(exported_text, encoding="utf-8")

    return OcrExportResult(
        document_id=document_id,
        export_dir=str(export_path),
        markdown_path=str(md_path),
        image_count=len(image_rows),
    )


async def execute_call_ocr_sync(
    *,
    base64_data: str,
    mime_type: str,
    model_name: str,
    document_id: str,
    file_path: str,
    workflow_id: str,
    provider: LLMProvider,
    store_images_fn: StoreImagesFn | None = None,
) -> OcrStoreResult:
    """Call OCR synchronously and store the result.

    This replaces the batch-submit → poll → signal → parse → store pipeline
    with a single direct API call.  Reuses ``execute_parse_ocr_result`` and
    ``execute_store_ocr_result`` for the parse/store steps.

    *store_images_fn* accepts a list of ``ExtractedImage`` and returns a
    mapping of ``{original_image_id: stored_uuid}``.  In production this
    is wired to the database; in tests it can be a stub.
    """
    from sax_llm import parse_model_id
    from sax_llm.mistral import _extract_images_from_response

    _, model = parse_model_id(model_name)

    data_uri = f"data:{mime_type};base64,{base64_data}"
    response_body = await provider.call_ocr(
        document_data_uri=data_uri,
        model=model,
        include_image_base64=True,
    )

    # Extract and store images (mirrors batch poller logic)
    extracted = _extract_images_from_response(response_body)
    image_mapping: dict[str, str] = {}
    if extracted and store_images_fn is not None:
        image_mapping = store_images_fn(extracted)

    # Inject mapping so execute_parse_ocr_result rewrites image refs
    response_body["_image_mapping"] = image_mapping
    raw_json = json.dumps(response_body)

    parse_result = execute_parse_ocr_result(raw_json)

    return execute_store_ocr_result(
        document_id=document_id,
        file_path=file_path,
        text=parse_result.text,
        model_name=parse_result.model_name,
        input_tokens=parse_result.input_tokens,
        output_tokens=parse_result.output_tokens,
        page_count=parse_result.page_count,
        batch_id="",
        workflow_id=workflow_id,
        image_ids=parse_result.image_ids,
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
    from forge.store import get_store_engine

    logger.info("Reading and storing file: %s", file_path)
    engine = get_store_engine()
    return execute_read_and_store_file(file_path, engine)


@activity.defn
async def submit_ocr_batch(input_json: str) -> OcrBatchRef:
    """Activity: submit OCR batch request.

    Takes JSON-serialized OcrSubmitInput + FileContentRef to avoid
    complex activity input. Loads file bytes from the database,
    base64-encodes in memory, and submits to the provider.
    """
    from sax_llm import get_provider

    from forge.ocr.models import OcrSubmitInput
    from forge.store import get_file_content, get_store_engine

    data = json.loads(input_json)
    ocr_input = OcrSubmitInput.model_validate(data["submit_input"])

    # Load file content from database
    engine = get_store_engine()

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

    # No store writes here: OcrSubmitWorkflow records the submission (and deletes
    # the blob) after this returns, and records a failure if this raises. Keeping
    # the store out of this activity means a DB blip never re-runs the expensive
    # submit (fixes the double-submit-on-DB-error bug).
    return await execute_submit_ocr_batch(ocr_input, file_content, provider)


@activity.defn
async def delete_file_content_blob(content_id: str) -> None:
    """Delete a stored file-content blob (DB row + S3 object) after submission."""
    from forge.store import delete_file_content, get_store_engine

    engine = get_store_engine()
    delete_file_content(engine, content_id)


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
async def update_batch_job_status(input_json: str) -> None:
    """Activity: update the status column of a batch_jobs row.

    Called by OcrStoreWorkflow (and any other consumer) to promote a row
    to ``SUCCEEDED`` after its downstream write completes, or to
    ``ERRORED`` when the parse/store step fails after signal delivery.

    Input JSON keys: ``request_id``, ``status``, optional ``error_message``.
    ``status`` must be a valid ``BatchJobStatus`` value — ``update_batch_status``
    raises ``ValueError`` on unknown strings.
    """
    from forge.models import BatchJobStatus
    from forge.store import get_store_engine, update_batch_status

    data = json.loads(input_json)
    engine = get_store_engine()
    update_batch_status(
        engine,
        request_id=data["request_id"],
        status=BatchJobStatus(data["status"]),
        error_message=data.get("error_message"),
    )
    logger.info(
        "Updated batch_jobs status: request_id=%s status=%s",
        data["request_id"],
        data["status"],
    )


@activity.defn
async def call_ocr_sync(input_json: str) -> OcrStoreResult:
    """Activity: call OCR synchronously and store the result.

    Replaces the batch-submit → poll → signal → parse → store pipeline
    with a single direct API call.  Takes JSON with keys:

    - ``file_path``: path to file on disk (used when no content_id)
    - ``content_id``: blob ID in file_content_blobs (used for chunks)
    - ``mime_type``: MIME type of the file
    - ``model_name``: provider:model identifier
    - ``document_id``: target document ID
    - ``workflow_id``: calling workflow's ID
    """
    from sax_llm import get_provider, parse_model_id

    from forge.store import (
        get_file_content,
        get_store_engine,
        save_ocr_image,
    )

    data = json.loads(input_json)
    document_id = data["document_id"]
    file_path = data["file_path"]
    model_name = data["model_name"]
    workflow_id = data["workflow_id"]
    content_id = data.get("content_id", "")
    mime_type = data.get("mime_type", "")

    logger.info("Sync OCR: document_id=%s file=%s", document_id, file_path)

    # Load file content — from blob store (chunks) or filesystem
    if content_id:
        engine = get_store_engine()
        blob = get_file_content(engine, content_id)
        if blob is None:
            msg = f"File content not found for content_id={content_id}"
            raise RuntimeError(msg)
        b64_data = base64.b64encode(blob["data"]).decode("ascii")
        mime_type = mime_type or blob["mime_type"]
    else:
        file_result = execute_read_file(file_path)
        b64_data = file_result.base64_data
        mime_type = mime_type or file_result.mime_type

    provider = get_provider(model_name)
    _provider_name, _ = parse_model_id(model_name)

    # Build image storage closure (mirrors batch poller pattern)
    engine = get_store_engine()

    def _store_images(images: list[ExtractedImage]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for img in images:
            image_id = str(uuid.uuid4())
            raw_b64 = img.image_base64
            img_mime = img.mime_type
            if raw_b64.startswith("data:"):
                header, raw_b64 = raw_b64.split(",", 1)
                img_mime = header.split(":")[1].split(";")[0]
            img_data = base64.b64decode(raw_b64)
            save_ocr_image(
                engine,
                image_id=image_id,
                page_index=img.page_index,
                original_image_id=img.original_image_id,
                data=img_data,
                mime_type=img_mime,
                file_size_bytes=len(img_data),
                top_left_x=img.top_left_x,
                top_left_y=img.top_left_y,
                bottom_right_x=img.bottom_right_x,
                bottom_right_y=img.bottom_right_y,
            )
            mapping[img.original_image_id] = image_id
        return mapping

    store_images_fn = _store_images

    return await execute_call_ocr_sync(
        base64_data=b64_data,
        mime_type=mime_type,
        model_name=model_name,
        document_id=document_id,
        file_path=file_path,
        workflow_id=workflow_id,
        provider=provider,
        store_images_fn=store_images_fn,
    )


@activity.defn
async def split_file_into_chunks(input_json: str) -> SplitResult:
    """Activity: split a stored file into chunks for parallel OCR.

    Takes JSON with content_id, mime_type, file_size_bytes.
    Returns SplitResult with ordered ChunkRef list.
    """
    from forge.store import get_store_engine

    data = json.loads(input_json)
    logger.info("Splitting file into chunks: content_id=%s", data.get("content_id", ""))

    engine = get_store_engine()
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
    from forge.store import get_store_engine

    data = json.loads(input_json)
    logger.info("Reassembling OCR chunks: document_id=%s", data.get("document_id", ""))

    engine = get_store_engine()
    return execute_reassemble_ocr_chunks(
        document_id=data["document_id"],
        chunk_document_ids=data["chunk_document_ids"],
        file_path=data["file_path"],
        total_pages=data["total_pages"],
        engine=engine,
    )


@activity.defn
async def export_ocr_document(input_json: str) -> OcrExportResult:
    """Activity: export OCR text and images to the filesystem.

    Takes JSON with document_id and optional output_dir.
    Returns OcrExportResult with export paths and image count.
    """
    from forge.store import get_store_engine

    data = json.loads(input_json)
    logger.info("Exporting OCR document: document_id=%s", data.get("document_id", ""))

    engine = get_store_engine()
    return execute_export_ocr_document(
        document_id=data["document_id"],
        output_dir=data.get("output_dir", ""),
        engine=engine,
    )


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    import hashlib

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def execute_check_ocr_duplicate(
    file_path: str,
    engine: Engine,
) -> OcrDuplicateCheckResult:
    """Check whether a file has already been successfully OCR'd.

    Computes the SHA-256 hash of the file and checks whether a matching
    ``ocr_results`` row exists that is **not** marked for removal.
    The hash is always returned so workflows can thread it through to
    the save step.
    """
    from forge.store import find_ocr_result_by_hash

    file_hash = compute_file_hash(file_path)

    row = find_ocr_result_by_hash(engine, file_hash)
    if row is not None:
        return OcrDuplicateCheckResult(
            is_duplicate=True,
            existing_document_id=row["document_id"],
        )
    return OcrDuplicateCheckResult(is_duplicate=False)


@activity.defn
async def check_ocr_duplicate(file_path: str) -> OcrDuplicateCheckResult:
    """Activity: check if a file has already been successfully OCR'd."""
    from forge.store import get_store_engine

    logger.info("Checking for duplicate OCR result: %s", file_path)

    engine = get_store_engine()
    return execute_check_ocr_duplicate(file_path, engine)


@activity.defn
async def mark_ocr_for_removal(document_id: str) -> OcrMarkResult:
    """Activity: set marked_for_removal=True on an OCR document."""
    from forge.store import get_store_engine
    from forge.store import mark_ocr_for_removal as _mark

    logger.info("Marking OCR document for removal: %s", document_id)

    engine = get_store_engine()
    found = _mark(engine, document_id)
    return OcrMarkResult(document_id=document_id, found=found)


@activity.defn
async def clear_ocr_removal_mark(document_id: str) -> OcrMarkResult:
    """Activity: set marked_for_removal=False on an OCR document."""
    from forge.store import clear_ocr_removal_mark as _clear
    from forge.store import get_store_engine

    logger.info("Clearing removal mark on OCR document: %s", document_id)

    engine = get_store_engine()
    found = _clear(engine, document_id)
    return OcrMarkResult(document_id=document_id, found=found)


# ---------------------------------------------------------------------------
# List OCR jobs
# ---------------------------------------------------------------------------


def execute_list_ocr_jobs(
    engine: Engine,
    *,
    limit: int = 50,
    status_filter: str = "",
) -> OcrListJobsResult:
    """Query OCR submissions grouped by document_id.

    Returns one entry per submission — distinct from chunking, and
    distinct from resubmissions of the same file_path. Aggregate
    status is derived in SQL from the underlying batch_jobs rows, and
    the ocr_results row is joined on document_id when available.

    Status logic (computed in SQL, matched to OcrJobDerivedStatus):
    - Any chunk errored or failed -> "errored"
    - Any chunk still submitted or storing -> "processing"
    - All chunks succeeded -> "succeeded"
    - Otherwise -> "unknown"

    Resubmissions each get a distinct row in the output — the old
    errored submission and the new one are both visible, so logs can
    be reviewed and the old row can be cleaned up later.

    When ``status_filter`` is provided the filter is applied in SQL
    before ``ORDER BY`` / ``LIMIT``, so the returned rows are the
    newest-first ``limit`` rows that match the filter (instead of
    silently returning fewer rows when the limit was consumed by
    other statuses).
    """
    import sqlalchemy as sa

    from forge.models import BatchJobStatus
    from forge.store import BatchJob, OcrResult

    bj = BatchJob.__table__
    ocr = OcrResult.__table__

    # Aggregate batch_jobs by document_id (the submission-level grouper)
    # plus file_path so we can surface both in the output without a
    # second join. All chunks of one submission share both values.
    sub = (
        sa.select(
            bj.c.document_id,
            bj.c.file_path,
            sa.func.max(bj.c.created_at).label("created_at"),
            sa.func.count().label("chunk_count"),
            sa.func.count(sa.case((bj.c.status == BatchJobStatus.ERRORED, 1))).label("n_errored"),
            sa.func.count(sa.case((bj.c.status == BatchJobStatus.FAILED, 1))).label("n_failed"),
            sa.func.count(sa.case((bj.c.status == BatchJobStatus.SUBMITTED, 1))).label(
                "n_submitted"
            ),
            sa.func.count(sa.case((bj.c.status == BatchJobStatus.STORING, 1))).label("n_storing"),
            sa.func.count(sa.case((bj.c.status == BatchJobStatus.SUCCEEDED, 1))).label(
                "n_succeeded"
            ),
        )
        .where(bj.c.provider == "mistral")
        .where(bj.c.file_path.isnot(None))
        .where(bj.c.document_id.isnot(None))
        .group_by(bj.c.document_id, bj.c.file_path)
    ).subquery("agg")

    # Derive list-level status in SQL so the optional status_filter can
    # be applied before LIMIT. The CASE order mirrors the precedence
    # in OcrJobDerivedStatus: errored beats processing beats succeeded.
    status_expr = sa.case(
        (
            sa.or_(sub.c.n_errored > 0, sub.c.n_failed > 0),
            sa.literal(OcrJobDerivedStatus.ERRORED.value),
        ),
        (
            sa.or_(sub.c.n_submitted > 0, sub.c.n_storing > 0),
            sa.literal(OcrJobDerivedStatus.PROCESSING.value),
        ),
        (
            sub.c.n_succeeded == sub.c.chunk_count,
            sa.literal(OcrJobDerivedStatus.SUCCEEDED.value),
        ),
        else_=sa.literal(OcrJobDerivedStatus.UNKNOWN.value),
    )

    # Left join ocr_results on document_id (unique), so resubmissions
    # pick up only their own completed row. Rows marked for removal
    # are excluded so soft-deleted submissions don't bleed through.
    stmt = sa.select(
        sub.c.document_id.label("submission_document_id"),
        sub.c.file_path,
        sub.c.created_at,
        sub.c.chunk_count,
        status_expr.label("status"),
        ocr.c.document_id.label("result_document_id"),
    ).select_from(
        sub.outerjoin(
            ocr,
            sa.and_(
                ocr.c.document_id == sub.c.document_id,
                ocr.c.marked_for_removal == sa.false(),
            ),
        )
    )

    if status_filter:
        stmt = stmt.where(status_expr == status_filter)

    stmt = stmt.order_by(sub.c.created_at.desc()).limit(limit)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()

    jobs: list[OcrJobEntry] = []
    for row in rows:
        # Prefer the ocr_results document_id (confirms a stored result),
        # but fall back to the batch_jobs document_id for in-flight or
        # errored submissions so callers can still reference the row.
        document_id = row["result_document_id"] or row["submission_document_id"] or ""

        # BatchJob.created_at uses the UTCDateTime type decorator, so
        # the value here is already a tz-aware UTC datetime.
        jobs.append(
            OcrJobEntry(
                file_path=row["file_path"],
                document_id=document_id,
                status=row["status"],
                chunk_count=row["chunk_count"],
                created_at=row["created_at"].isoformat() if row["created_at"] else "",
            )
        )

    return OcrListJobsResult(jobs=jobs, total=len(jobs))


@activity.defn
async def list_ocr_jobs(input_json: str) -> OcrListJobsResult:
    """Activity: list OCR job submissions grouped by file_path."""
    from forge.store import get_store_engine

    data = json.loads(input_json)
    logger.info(
        "Listing OCR jobs: limit=%s status_filter=%s",
        data.get("limit", 50),
        data.get("status_filter", ""),
    )

    engine = get_store_engine()
    return execute_list_ocr_jobs(
        engine,
        limit=data.get("limit", 50),
        status_filter=data.get("status_filter", ""),
    )
