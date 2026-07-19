"""OCR activities — file IO, request-blob build, result parse/store, status, listing.

Consumes the Forge platform purely through ``sax_platform.contracts`` + Temporal
string-name cross-queue calls. The provider submit itself is the platform's job (the
opaque-blob submit SPI); OCR builds the request body, hands the platform a pointer,
and owns all image storage / markdown rewriting / status projection.

Function Core / Imperative Shell: pure helpers (body build, markdown rewrite, page
parse) are separated from the Temporal activities that do IO.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sax_platform.contracts.models import resolve_batch_result
from temporalio import activity

from ocr.models import (
    ChunkRef,
    FileContentRef,
    OcrBatchRequestRef,
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
    from sax_platform.contracts.s3_blobs import S3Blobs
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def detect_mime_type(file_path: str) -> str:
    """Detect MIME type from file extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def build_ocr_batch_body(base64_data: str, mime_type: str) -> dict[str, Any]:
    """Build the request body for the /v1/ocr batch endpoint."""
    data_uri = f"data:{mime_type};base64,{base64_data}"
    doc: dict[str, Any]
    if mime_type.startswith("image/"):
        doc = {"type": "image_url", "image_url": data_uri}
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
    """Rewrite ``![alt](img-0.jpeg)`` references to ``ocr-image://{uuid}`` URIs."""
    if not image_mapping:
        return markdown

    def _replace(match: re.Match[str]) -> str:
        alt_bracket = match.group(1)
        image_ref = match.group(2)
        if image_ref in image_mapping:
            return f"{alt_bracket}(ocr-image://{image_mapping[image_ref]})"
        return match.group(0)

    return _IMAGE_REF_PATTERN.sub(_replace, markdown)


# Matches ocr-image:// URIs in markdown: ![alt](ocr-image://{uuid})
_OCR_IMAGE_URI_PATTERN = re.compile(r"(!\[[^\]]*\])\(ocr-image://([^)]+)\)")


def rewrite_ocr_uris_to_local(markdown: str, image_id_to_filename: dict[str, str]) -> str:
    """Rewrite ``ocr-image://{uuid}`` references to local filenames (for export)."""
    if not image_id_to_filename:
        return markdown

    def _replace(match: re.Match[str]) -> str:
        alt_bracket = match.group(1)
        image_id = match.group(2)
        if image_id in image_id_to_filename:
            return f"{alt_bracket}({image_id_to_filename[image_id]})"
        return match.group(0)

    return _OCR_IMAGE_URI_PATTERN.sub(_replace, markdown)


def _mime_to_extension(mime_type: str) -> str:
    """Convert a MIME type to a file extension (with leading dot)."""
    ext = mimetypes.guess_extension(mime_type)
    if ext in (".jpe", ".jpg"):
        return ".jpeg"
    return ext or ".bin"


_JPEG_SOI = b"\xff\xd8\xff"
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _strip_image_prefix(data: bytes) -> bytes:
    """Strip any corrupt prefix bytes before a real image signature."""
    if data[:3] == _JPEG_SOI or data[:8] == _PNG_SIG:
        return data
    for marker in (_JPEG_SOI, _PNG_SIG):
        idx = data.find(marker)
        if idx > 0:
            return data[idx:]
    return data


def parse_ocr_pages(body_json: str, image_mapping: dict[str, str]) -> OcrParseResult:
    """Parse the OCR response body into extracted text (pure).

    The body is the provider OCR format (``pages[].markdown`` + ``usage_info``);
    image references are rewritten using *image_mapping* (original id -> stored uuid).
    """
    data = json.loads(body_json)
    pages = data.get("pages", [])
    page_texts: list[str] = []
    for page in pages:
        md = page.get("markdown", "")
        if image_mapping:
            md = rewrite_image_references(md, image_mapping)
        page_texts.append(md)

    text = "\n\n".join(page_texts)
    usage = data.get("usage_info", {})
    return OcrParseResult(
        text=text,
        model_name=data.get("model", ""),
        input_tokens=usage.get("pages_processed", 0),
        output_tokens=usage.get("doc_size_bytes", 0),
        page_count=len(pages),
        image_count=len(image_mapping),
        image_ids=list(image_mapping.values()),
    )


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    import hashlib

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Testable functions (engine injected)
# ---------------------------------------------------------------------------


def execute_read_and_store_file(
    file_path: str,
    engine: Engine,
    blobs: S3Blobs,
) -> FileContentRef:
    """Read a file and store raw bytes (S3-backed) keyed by a new content id."""
    from ocr.store import save_file_content

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
        blobs=blobs,
    )
    return FileContentRef(content_id=content_id, mime_type=mime_type, file_size_bytes=len(raw))


def execute_split_file_into_chunks(
    content_id: str,
    mime_type: str,
    file_size_bytes: int,
    engine: Engine,
    blobs: S3Blobs,
) -> SplitResult:
    """Split a stored file into chunks for parallel OCR processing."""
    import fitz

    from ocr.store import delete_file_content, get_file_content, save_file_content

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

    blob = get_file_content(engine, content_id, blobs)
    if blob is None:
        msg = f"File content not found for content_id={content_id}"
        raise RuntimeError(msg)

    doc = fitz.open(stream=blob["data"], filetype="pdf")
    total_pages = len(doc)

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
            blobs=blobs,
        )
        chunks.append(
            ChunkRef(
                content_id=chunk_content_id,
                mime_type="application/pdf",
                file_size_bytes=len(chunk_bytes),
                chunk_index=chunk_index,
                page_start=start_page + 1,
                page_end=end_page,
            )
        )

    doc.close()
    delete_file_content(engine, content_id, blobs)
    return SplitResult(chunks=chunks, total_pages=total_pages, original_content_id=content_id)


def execute_build_request_blob(
    *,
    content_id: str,
    mime_type: str,
    model_name: str,
    engine: Engine,
    blobs: S3Blobs,
) -> OcrBatchRequestRef:
    """Build the /v1/ocr batch request and stash it to S3 as an opaque blob.

    Mints the single correlation id (request_id == provider custom_id ==
    batch_jobs PK) and returns it with the blob key so the submit workflow can
    hand the platform a pointer.
    """
    from ocr.store import get_file_content

    blob = get_file_content(engine, content_id, blobs)
    if blob is None:
        msg = f"File content not found for content_id={content_id}"
        raise RuntimeError(msg)

    encoded = base64.b64encode(blob["data"]).decode("ascii")
    body = build_ocr_batch_body(encoded, blob["mime_type"] or mime_type)
    request_id = str(uuid.uuid4())
    requests = [{"custom_id": request_id, "body": body}]

    s3_key = blobs.build_key(f"ocr-request-{request_id}")
    blobs.put(s3_key, json.dumps(requests).encode("utf-8"), "application/json")

    _provider, model = model_name.split(":", 1) if ":" in model_name else ("mistral", model_name)
    return OcrBatchRequestRef(request_id=request_id, s3_key=s3_key, model=model)


def execute_store_ocr_result(
    *,
    request_id: str,
    document_id: str,
    file_path: str,
    batch_id: str,
    workflow_id: str,
    raw_response_json: str | None,
    s3_key: str | None,
    engine: Engine,
    blobs: S3Blobs,
) -> OcrStoreResult:
    """Resolve the delivered result, store images, save text + status (idempotent)."""
    from sax_platform.contracts.models import BatchResult

    from ocr.store import ocr_image_id, save_ocr_image, save_ocr_result, upsert_ocr_job_status

    body, images = resolve_batch_result(
        BatchResult(
            request_id=request_id,
            batch_id=batch_id,
            raw_response_json=raw_response_json,
            s3_key=s3_key,
            result_type="succeeded",
        ),
        blobs,
    )
    if body is None:
        msg = "OCR result has neither inline body nor s3 envelope"
        raise RuntimeError(msg)

    # Store images first (deterministic ids → idempotent on retry), building the
    # original-id -> stored-uuid mapping the markdown rewrite needs.
    image_mapping: dict[str, str] = {}
    for img in images:
        original_image_id = img["original_image_id"]
        page_index = img["page_index"]
        image_id = ocr_image_id(request_id, original_image_id, page_index)
        raw_b64 = img["image_base64"]
        mime_type = img.get("mime_type", "image/jpeg")
        if isinstance(raw_b64, str) and raw_b64.startswith("data:"):
            header, raw_b64 = raw_b64.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
        data = base64.b64decode(raw_b64)
        save_ocr_image(
            engine,
            image_id=image_id,
            document_id=document_id,
            page_index=page_index,
            original_image_id=original_image_id,
            data=data,
            mime_type=mime_type,
            file_size_bytes=len(data),
            top_left_x=img.get("top_left_x"),
            top_left_y=img.get("top_left_y"),
            bottom_right_x=img.get("bottom_right_x"),
            bottom_right_y=img.get("bottom_right_y"),
            blobs=blobs,
        )
        image_mapping[original_image_id] = image_id

    parse_result = parse_ocr_pages(body, image_mapping)

    file_hash = None
    if file_path and Path(file_path).is_file():
        file_hash = compute_file_hash(file_path)

    save_ocr_result(
        engine,
        document_id=document_id,
        file_path=file_path,
        text=parse_result.text,
        page_count=parse_result.page_count,
        model_name=parse_result.model_name,
        input_tokens=parse_result.input_tokens,
        output_tokens=parse_result.output_tokens,
        batch_id=batch_id,
        workflow_id=workflow_id,
        file_hash=file_hash,
    )
    upsert_ocr_job_status(
        engine,
        request_id=request_id,
        document_id=document_id,
        file_path=file_path,
        status="stored",
    )
    return OcrStoreResult(
        document_id=document_id,
        text_length=len(parse_result.text),
        page_count=parse_result.page_count,
    )


def execute_reassemble_ocr_chunks(
    *,
    document_id: str,
    chunk_document_ids: list[str],
    file_path: str,
    total_pages: int,
    engine: Engine,
) -> OcrStoreResult:
    """Combine OCR results from multiple chunks into a single result."""
    from ocr.store import (
        delete_ocr_results,
        get_ocr_result,
        reassign_ocr_images_document_id,
        save_ocr_result,
    )

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
        model_name = model_name or row["model_name"]
        batch_id = batch_id or row["batch_id"]

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
    reassign_ocr_images_document_id(engine, chunk_document_ids, document_id)
    delete_ocr_results(engine, chunk_document_ids)
    return OcrStoreResult(
        document_id=document_id, text_length=len(combined_text), page_count=total_pages
    )


def _get_export_dir(document_id: str, output_dir: str) -> Path:
    import os

    if output_dir:
        return Path(output_dir)
    xdg = os.environ.get("XDG_DATA_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "ocr" / "export" / document_id


def execute_export_ocr_document(
    *,
    document_id: str,
    output_dir: str,
    engine: Engine,
    blobs: S3Blobs,
) -> OcrExportResult:
    """Export OCR text + images to the filesystem."""
    from ocr.store import get_ocr_image, get_ocr_images, get_ocr_result

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
    image_rows = get_ocr_images(engine, document_id)
    export_path = _get_export_dir(document_id, output_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    id_to_filename: dict[str, str] = {}
    for img_meta in image_rows:
        image_id: str = img_meta["id"]
        filename = f"{image_id}{_mime_to_extension(img_meta['mime_type'])}"
        id_to_filename[image_id] = filename
        img_full = get_ocr_image(engine, image_id, blobs)
        if img_full is not None:
            (export_path / filename).write_bytes(_strip_image_prefix(img_full["data"]))

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


def execute_check_ocr_duplicate(file_path: str, engine: Engine) -> OcrDuplicateCheckResult:
    """Check whether a file has already been successfully OCR'd (by content hash)."""
    from ocr.store import find_ocr_result_by_hash

    file_hash = compute_file_hash(file_path)
    row = find_ocr_result_by_hash(engine, file_hash)
    if row is not None:
        return OcrDuplicateCheckResult(is_duplicate=True, existing_document_id=row["document_id"])
    return OcrDuplicateCheckResult(is_duplicate=False)


def execute_list_ocr_jobs(
    engine: Engine, *, limit: int = 50, status_filter: str = ""
) -> OcrListJobsResult:
    """List OCR submissions: OCR's own status table joined to the platform batch_jobs.

    ``ocr_job_status`` (OCR single-writer) is the source of truth for the user-facing
    status; the platform ``batch_jobs`` read model is LEFT-joined on ``request_id``
    for the provider-batch detail. No ``forge`` import — only the contracts read model.
    """
    import sqlalchemy as sa
    from sax_platform.contracts.batch_jobs import batch_jobs as bj

    from ocr.store import OcrJobStatus

    js = OcrJobStatus.__table__
    stmt = (
        sa.select(
            js.c.request_id,
            js.c.document_id,
            js.c.file_path,
            js.c.status.label("ocr_status"),
            js.c.created_at,
            bj.c.status.label("provider_status"),
        )
        .select_from(js.outerjoin(bj, bj.c.id == js.c.request_id))
        .order_by(js.c.created_at.desc())
    )
    if status_filter:
        stmt = stmt.where(js.c.status == status_filter)
    stmt = stmt.limit(limit)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()

    jobs = [
        OcrJobEntry(
            file_path=row["file_path"],
            document_id=row["document_id"],
            status=_derive_status(row["ocr_status"], row["provider_status"]),
            created_at=row["created_at"].isoformat() if row["created_at"] else "",
        )
        for row in rows
    ]
    return OcrListJobsResult(jobs=jobs, total=len(jobs))


def _derive_status(ocr_status: str, provider_status: str | None) -> str:
    """Map the (ocr, provider) status pair to a coarse display status."""
    if ocr_status == "stored":
        return OcrJobDerivedStatus.SUCCEEDED.value
    if ocr_status == "failed" or provider_status in {"failed", "expired", "missing"}:
        return OcrJobDerivedStatus.ERRORED.value
    if ocr_status in {"submitted", "processing"}:
        return OcrJobDerivedStatus.PROCESSING.value
    return OcrJobDerivedStatus.UNKNOWN.value


# ---------------------------------------------------------------------------
# Imperative shell — class-based Temporal activities (T3.6 composition root)
# ---------------------------------------------------------------------------


class OcrStoreActivities:
    """Dependency-carrying OCR activities: one store engine + one blob client.

    Temporal's sanctioned dependency injection (T3.6): the process-wide store
    engine and :class:`S3Blobs` are built once at worker startup and injected
    here, replacing the per-call ``get_store_engine()`` each shell used to build
    (a fresh pooled Postgres engine per activity invocation). OCR requires S3, so
    ``blobs`` is required — the worker fails fast at startup on an unset bucket
    rather than deferring the error to the first blob-touching activity.

    Each method is a bare ``@activity.defn`` so its *registered* name equals the
    method ``__name__`` — the exact names the OCR workflows invoke by string.
    Bound methods preserve ``__name__``, so converting the former module-level
    activity functions into methods is invisible to the workflows and to the
    by-name activity mocks in the workflow tests.
    """

    def __init__(self, engine: Engine, blobs: S3Blobs) -> None:
        self._engine = engine
        self._blobs = blobs

    @activity.defn
    async def read_and_store_file_content(self, file_path: str) -> FileContentRef:
        """Activity: read a file and store its bytes (S3-backed)."""
        logger.info("Reading and storing file: %s", file_path)
        return execute_read_and_store_file(file_path, self._engine, self._blobs)

    @activity.defn
    async def split_file_into_chunks(self, input_json: str) -> SplitResult:
        """Activity: split a stored file into OCR chunks."""
        data = json.loads(input_json)
        return execute_split_file_into_chunks(
            content_id=data["content_id"],
            mime_type=data["mime_type"],
            file_size_bytes=data["file_size_bytes"],
            engine=self._engine,
            blobs=self._blobs,
        )

    @activity.defn
    async def build_ocr_request_blob(self, input_json: str) -> OcrBatchRequestRef:
        """Activity: build the /v1/ocr request and stash it to S3; mint request_id."""
        data = json.loads(input_json)
        return execute_build_request_blob(
            content_id=data["content_id"],
            mime_type=data["mime_type"],
            model_name=data["model_name"],
            engine=self._engine,
            blobs=self._blobs,
        )

    @activity.defn
    async def delete_file_content_blob(self, content_id: str) -> None:
        """Activity: delete a stored file-content blob (DB row + S3 object)."""
        from ocr.store import delete_file_content

        delete_file_content(self._engine, content_id, self._blobs)

    @activity.defn
    async def store_ocr_result(self, input_json: str) -> OcrStoreResult:
        """Activity: resolve the delivered result and store text + images + status."""
        data = json.loads(input_json)
        logger.info("Storing OCR result: document_id=%s", data.get("document_id", ""))
        return execute_store_ocr_result(
            request_id=data["request_id"],
            document_id=data["document_id"],
            file_path=data["file_path"],
            batch_id=data["batch_id"],
            workflow_id=data["workflow_id"],
            raw_response_json=data.get("raw_response_json"),
            s3_key=data.get("s3_key"),
            engine=self._engine,
            blobs=self._blobs,
        )

    @activity.defn
    async def upsert_ocr_status(self, input_json: str) -> None:
        """Activity: upsert the OCR processing-status row (single-writer)."""
        from ocr.store import upsert_ocr_job_status

        data = json.loads(input_json)
        upsert_ocr_job_status(
            self._engine,
            request_id=data["request_id"],
            document_id=data["document_id"],
            file_path=data.get("file_path", ""),
            status=data["status"],
            error_message=data.get("error_message"),
        )

    @activity.defn
    async def reassemble_ocr_chunks(self, input_json: str) -> OcrStoreResult:
        """Activity: combine OCR results from multiple chunks into one."""
        data = json.loads(input_json)
        return execute_reassemble_ocr_chunks(
            document_id=data["document_id"],
            chunk_document_ids=data["chunk_document_ids"],
            file_path=data["file_path"],
            total_pages=data["total_pages"],
            engine=self._engine,
        )

    @activity.defn
    async def export_ocr_document(self, input_json: str) -> OcrExportResult:
        """Activity: export OCR text and images to the filesystem."""
        data = json.loads(input_json)
        return execute_export_ocr_document(
            document_id=data["document_id"],
            output_dir=data.get("output_dir", ""),
            engine=self._engine,
            blobs=self._blobs,
        )

    @activity.defn
    async def check_ocr_duplicate(self, file_path: str) -> OcrDuplicateCheckResult:
        """Activity: check if a file has already been successfully OCR'd."""
        return execute_check_ocr_duplicate(file_path, self._engine)

    @activity.defn
    async def mark_ocr_for_removal(self, document_id: str) -> OcrMarkResult:
        """Activity: set marked_for_removal=True on an OCR document."""
        from ocr.store import mark_ocr_for_removal as _mark

        found = _mark(self._engine, document_id)
        return OcrMarkResult(document_id=document_id, found=found)

    @activity.defn
    async def clear_ocr_removal_mark(self, document_id: str) -> OcrMarkResult:
        """Activity: set marked_for_removal=False on an OCR document."""
        from ocr.store import clear_ocr_removal_mark as _clear

        found = _clear(self._engine, document_id)
        return OcrMarkResult(document_id=document_id, found=found)

    @activity.defn
    async def list_ocr_jobs(self, input_json: str) -> OcrListJobsResult:
        """Activity: list OCR submissions (status join)."""
        data = json.loads(input_json)
        return execute_list_ocr_jobs(
            self._engine,
            limit=data.get("limit", 50),
            status_filter=data.get("status_filter", ""),
        )
