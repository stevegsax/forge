"""OCR activities — file IO, request-blob build, batch submit/poll/fetch, status, listing.

OCR owns its Mistral batches end-to-end (T4.2): it builds the /v1/ocr request,
submits the batch, polls its status, and fetches + stores the result through the
injected ``MistralOcr`` capability. It consumes the platform only for the
cross-queue ``batch_jobs`` ledger (via ``sax_platform.contracts`` + Temporal
string-name calls) and owns all image storage / markdown rewriting / status
projection.

Function Core / Imperative Shell: pure helpers (body build, markdown rewrite, page
parse, status derivation) are separated from the Temporal activities that do IO.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never

from sax_platform.contracts.models import BatchJobStatus
from sax_platform.ocr import BatchPollStatus, ExtractedImage
from temporalio import activity
from temporalio.exceptions import ApplicationError

from ocr.models import (
    ChunkRef,
    FileContentRef,
    OcrBatchRequestRef,
    OcrBatchStatusInput,
    OcrBuildRequestInput,
    OcrDuplicateCheckResult,
    OcrExportInput,
    OcrExportResult,
    OcrFetchStoreInput,
    OcrJobDerivedStatus,
    OcrJobEntry,
    OcrListJobsInput,
    OcrListJobsResult,
    OcrMarkResult,
    OcrParseResult,
    OcrProcessingStatus,
    OcrReassembleInput,
    OcrSplitInput,
    OcrStatusUpsertInput,
    OcrStoreResult,
    OcrSubmitBatchInput,
    SplitResult,
)

if TYPE_CHECKING:
    from sax_platform.contracts.s3_blobs import S3Blobs
    from sax_platform.ocr import MistralOcr
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

# Mistral's normalized poll statuses mapped onto the ``wait_batch_ended`` state
# vocabulary. PENDING and IN_PROGRESS both mean "keep waiting" ("in_progress");
# ENDED signals the fetch; the three provider-terminal states pass through. An
# unrecognized status falls back to "in_progress" (the safe non-terminal default).
_POLL_STATUS_TO_STATE: dict[BatchPollStatus, str] = {
    BatchPollStatus.PENDING: "in_progress",
    BatchPollStatus.IN_PROGRESS: "in_progress",
    BatchPollStatus.ENDED: "ended",
    BatchPollStatus.FAILED: "failed",
    BatchPollStatus.EXPIRED: "expired",
    BatchPollStatus.CANCELED: "canceled",
}


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
    raw_response_json: str,
    extracted_images: list[ExtractedImage],
    engine: Engine,
    blobs: S3Blobs,
) -> OcrStoreResult:
    """Store the fetched batch entry's images + text + status (idempotent).

    Takes the entry's raw OCR body and its already-extracted images directly (the
    fetch happens in ``fetch_and_store_ocr_result``); the retired result-envelope
    resolve indirection is gone.
    """
    from ocr.store import ocr_image_id, save_ocr_image, save_ocr_result, upsert_ocr_job_status

    # Store images first (deterministic ids → idempotent on retry), building the
    # original-id -> stored-uuid mapping the markdown rewrite needs.
    image_mapping: dict[str, str] = {}
    for img in extracted_images:
        image_id = ocr_image_id(request_id, img.original_image_id, img.page_index)
        raw_b64 = img.image_base64
        mime_type = img.mime_type
        if raw_b64.startswith("data:"):
            header, raw_b64 = raw_b64.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
        data = base64.b64decode(raw_b64)
        save_ocr_image(
            engine,
            image_id=image_id,
            document_id=document_id,
            page_index=img.page_index,
            original_image_id=img.original_image_id,
            data=data,
            mime_type=mime_type,
            file_size_bytes=len(data),
            top_left_x=img.top_left_x,
            top_left_y=img.top_left_y,
            bottom_right_x=img.bottom_right_x,
            bottom_right_y=img.bottom_right_y,
            blobs=blobs,
        )
        image_mapping[img.original_image_id] = image_id

    parse_result = parse_ocr_pages(raw_response_json, image_mapping)

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
        status=OcrProcessingStatus.STORED,
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

    ``status_filter``, when non-empty, is validated against ``OcrJobDerivedStatus``
    (:func:`_validate_status_filter`; raises ``ValueError`` naming the legal values
    on an unknown filter) and applied to each row's *derived* display status — the
    same value ``ocr list`` prints and the same path ``_derive_display_status`` uses
    for the unfiltered listing — never as a SQL predicate on the raw
    ``ocr_job_status.status`` column: that column speaks ``OcrProcessingStatus``
    (submitted/processing/stored/failed), a different vocabulary, so a raw-column
    predicate for "succeeded"/"errored"/"unknown" would silently match zero rows.
    Filtering therefore happens in Python, after every row's status has been
    derived; ``limit`` is applied last, to the *filtered* results, so
    ``--limit 10 --status errored`` returns up to 10 errored rows rather than
    filtering an arbitrary 10 unfiltered ones.
    """
    import sqlalchemy as sa
    from sax_platform.contracts.batch_jobs import batch_jobs as bj

    from ocr.store import OcrJobStatus

    derived_filter = _validate_status_filter(status_filter)

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

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()

    jobs = [
        OcrJobEntry(
            file_path=row["file_path"],
            document_id=row["document_id"],
            status=_derive_display_status(row["ocr_status"], row["provider_status"]),
            created_at=row["created_at"].isoformat() if row["created_at"] else "",
        )
        for row in rows
    ]
    if derived_filter is not None:
        jobs = [job for job in jobs if job.status == derived_filter]
    jobs = jobs[:limit]
    return OcrListJobsResult(jobs=jobs, total=len(jobs))


def _coerce_ocr_status(raw: str) -> OcrProcessingStatus | None:
    """Parse a stored OCR status string, tolerating unknown legacy values (``None``)."""
    try:
        return OcrProcessingStatus(raw)
    except ValueError:
        return None


def _coerce_provider_status(raw: str | None) -> BatchJobStatus | None:
    """Parse a joined provider status, tolerating absent/unknown values (``None``)."""
    if raw is None:
        return None
    try:
        return BatchJobStatus(raw)
    except ValueError:
        return None


_LEGAL_STATUS_FILTERS = frozenset(status.value for status in OcrJobDerivedStatus)


def _validate_status_filter(status_filter: str) -> OcrJobDerivedStatus | None:
    """Validate a raw ``--status`` filter against the derived-status vocabulary (pure).

    Empty string means "no filter" (``None``). Any non-empty value must name one
    of ``OcrJobDerivedStatus``'s members — ``processing``, ``succeeded``,
    ``errored``, or ``unknown`` (the derived enum's own catch-all state is a legal
    filter target: rows the read side couldn't classify are still listable). An
    unrecognized value raises ``ValueError`` naming the legal set, at the
    CLI/activity boundary this function is called from — never silently matching
    zero rows the way filtering the raw ``OcrProcessingStatus`` column did.
    """
    if not status_filter:
        return None
    try:
        return OcrJobDerivedStatus(status_filter)
    except ValueError as exc:
        legal = ", ".join(sorted(_LEGAL_STATUS_FILTERS))
        msg = f"Unknown --status filter {status_filter!r}; must be one of: {legal}"
        raise ValueError(msg) from exc


def _derive_display_status(ocr_status: str, provider_status: str | None) -> OcrJobDerivedStatus:
    """Coerce the raw joined status strings, then derive the display status (pure).

    The read side is tolerant by design: an unrecognized stored OCR status
    (a legacy row) maps straight to ``UNKNOWN`` without entering ``_derive_status``,
    and an unknown provider string is treated as "no provider info" (``None``).
    """
    ocr = _coerce_ocr_status(ocr_status)
    if ocr is None:
        return OcrJobDerivedStatus.UNKNOWN
    return _derive_status(ocr, _coerce_provider_status(provider_status))


def _derive_status(
    ocr_status: OcrProcessingStatus, provider_status: BatchJobStatus | None
) -> OcrJobDerivedStatus:
    """Map the (OCR processing, provider batch) status pair to a display status (pure).

    ``ocr_job_status`` (OCR single-writer) is authoritative for terminal OCR
    outcomes; the platform ``batch_jobs`` provider status only refines the still
    in-flight case. Derivation table (``None`` provider = no ledger row /
    pre-submit failure)::

        OCR \\ provider   None  submitted  processing  ended  failed  expired  missing
        stored            succ  succ       succ        succ   succ    succ     succ
        failed            err   err        err         err    err     err      err
        submitted         proc  proc       proc        proc   err     err      err
        processing        proc  proc       proc        proc   err     err      err

    ``stored``/``failed`` are OCR-terminal and ignore the provider column. While
    OCR is still ``submitted``/``processing`` a provider-terminal failure
    (``failed``/``expired``/``missing``) surfaces as ``errored`` before the OCR
    writer catches up; every other provider state — including ``ended`` (fetch +
    store imminent) and ``None`` — reads as ``processing``.
    """
    match ocr_status:
        case OcrProcessingStatus.STORED:
            return OcrJobDerivedStatus.SUCCEEDED
        case OcrProcessingStatus.FAILED:
            return OcrJobDerivedStatus.ERRORED
        case OcrProcessingStatus.SUBMITTED | OcrProcessingStatus.PROCESSING:
            match provider_status:
                case BatchJobStatus.FAILED | BatchJobStatus.EXPIRED | BatchJobStatus.MISSING:
                    return OcrJobDerivedStatus.ERRORED
                case (
                    BatchJobStatus.SUBMITTED
                    | BatchJobStatus.PROCESSING
                    | BatchJobStatus.ENDED
                    | None
                ):
                    return OcrJobDerivedStatus.PROCESSING
                case _ as unreachable:
                    assert_never(unreachable)
        case _ as unreachable:
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Imperative shell — class-based Temporal activities (T3.6 composition root)
# ---------------------------------------------------------------------------


class OcrStoreActivities:
    """Dependency-carrying OCR activities: store engine + blob client + Mistral OCR.

    Temporal's sanctioned dependency injection (T3.6): the process-wide store
    engine, :class:`S3Blobs`, and :class:`~sax_platform.ocr.MistralOcr` capability
    are built once at worker startup and injected here. OCR requires all three —
    the worker fails fast at startup on an unset S3 bucket or a missing
    ``MISTRAL_API_KEY`` (T4.2 makes OCR poll its own Mistral batches) rather than
    deferring the error to the first activity that needs them.

    Each method is a bare ``@activity.defn`` so its *registered* name equals the
    method ``__name__`` — the exact names the OCR workflows invoke by string.
    Bound methods preserve ``__name__``, so converting the former module-level
    activity functions into methods is invisible to the workflows and to the
    by-name activity mocks in the workflow tests.
    """

    def __init__(self, engine: Engine, blobs: S3Blobs, mistral: MistralOcr) -> None:
        self._engine = engine
        self._blobs = blobs
        self._mistral = mistral

    @activity.defn
    async def read_and_store_file_content(self, file_path: str) -> FileContentRef:
        """Activity: read a file and store its bytes (S3-backed)."""
        logger.info("Reading and storing file: %s", file_path)
        return execute_read_and_store_file(file_path, self._engine, self._blobs)

    @activity.defn
    async def split_file_into_chunks(self, input: OcrSplitInput) -> SplitResult:
        """Activity: split a stored file into OCR chunks."""
        return execute_split_file_into_chunks(
            content_id=input.content_id,
            mime_type=input.mime_type,
            file_size_bytes=input.file_size_bytes,
            engine=self._engine,
            blobs=self._blobs,
        )

    @activity.defn
    async def build_ocr_request_blob(self, input: OcrBuildRequestInput) -> OcrBatchRequestRef:
        """Activity: build the /v1/ocr request and stash it to S3; mint request_id."""
        return execute_build_request_blob(
            content_id=input.content_id,
            mime_type=input.mime_type,
            model_name=input.model_name,
            engine=self._engine,
            blobs=self._blobs,
        )

    @activity.defn
    async def delete_file_content_blob(self, content_id: str) -> None:
        """Activity: delete a stored file-content blob (DB row + S3 object)."""
        from ocr.store import delete_file_content

        delete_file_content(self._engine, content_id, self._blobs)

    @activity.defn
    async def submit_ocr_batch(self, input: OcrSubmitBatchInput) -> str:
        """Activity: fetch the request blob and submit the Mistral /v1/ocr batch.

        Reads the pre-built request list from ocr's own S3 by ``s3_key`` and calls
        ``MistralOcr.submit_batch`` → the provider batch id. Writes nothing (the
        workflow records the batch_jobs ledger row separately — the
        double-submit-safety invariant: a provider submit and a DB write never
        share one re-runnable activity).
        """
        raw = self._blobs.get(input.s3_key)
        requests = json.loads(raw.decode("utf-8"))
        return await self._mistral.submit_batch(requests, input.model, endpoint="/v1/ocr")

    @activity.defn
    async def ocr_batch_status(self, input: OcrBatchStatusInput) -> str:
        """Activity: poll one Mistral batch's status (no download).

        Maps ``MistralOcr.get_batch_status`` onto the ``wait_batch_ended`` state
        vocabulary ("in_progress"/"ended"/"failed"/"expired"/"canceled").
        """
        status = await self._mistral.get_batch_status(input.batch_id)
        return _POLL_STATUS_TO_STATE.get(status, "in_progress")

    @activity.defn
    async def fetch_and_store_ocr_result(self, input: OcrFetchStoreInput) -> OcrStoreResult:
        """Activity: download the finished batch, select this request, store it.

        Fetches all result entries, selects the one whose ``custom_id`` matches
        ``request_id`` (absent or failed → a non-retryable ``ApplicationError``),
        then stores text + images + status. Result bytes never return to the
        workflow — only the small :class:`OcrStoreResult` summary does.
        """
        entries = await self._mistral.fetch_batch_results(input.batch_id)
        entry = next((e for e in entries if e.custom_id == input.request_id), None)
        if entry is None:
            msg = f"No OCR result entry for request {input.request_id} in batch {input.batch_id}"
            raise ApplicationError(msg, non_retryable=True)
        if not entry.succeeded or entry.raw_response_json is None:
            detail = entry.error or "no response body"
            msg = f"OCR batch entry failed for request {input.request_id}: {detail}"
            raise ApplicationError(msg, non_retryable=True)
        logger.info("Storing OCR result: document_id=%s", input.document_id)
        return execute_store_ocr_result(
            request_id=input.request_id,
            document_id=input.document_id,
            file_path=input.file_path,
            batch_id=input.batch_id,
            workflow_id=input.workflow_id,
            raw_response_json=entry.raw_response_json,
            extracted_images=entry.extracted_images,
            engine=self._engine,
            blobs=self._blobs,
        )

    @activity.defn
    async def upsert_ocr_status(self, input: OcrStatusUpsertInput) -> None:
        """Activity: upsert the OCR processing-status row (single-writer)."""
        from ocr.store import upsert_ocr_job_status

        upsert_ocr_job_status(
            self._engine,
            request_id=input.request_id,
            document_id=input.document_id,
            file_path=input.file_path,
            status=input.status,
            error_message=input.error_message,
        )

    @activity.defn
    async def reassemble_ocr_chunks(self, input: OcrReassembleInput) -> OcrStoreResult:
        """Activity: combine OCR results from multiple chunks into one."""
        return execute_reassemble_ocr_chunks(
            document_id=input.document_id,
            chunk_document_ids=input.chunk_document_ids,
            file_path=input.file_path,
            total_pages=input.total_pages,
            engine=self._engine,
        )

    @activity.defn
    async def export_ocr_document(self, input: OcrExportInput) -> OcrExportResult:
        """Activity: export OCR text and images to the filesystem."""
        return execute_export_ocr_document(
            document_id=input.document_id,
            output_dir=input.output_dir,
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
    async def list_ocr_jobs(self, input: OcrListJobsInput) -> OcrListJobsResult:
        """Activity: list OCR submissions (status join)."""
        return execute_list_ocr_jobs(
            self._engine,
            limit=input.limit,
            status_filter=input.status_filter,
        )
