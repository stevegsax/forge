"""The platform's Mistral OCR capability (D88 / T3.3).

Ported from `sax_llm.mistral.MistralProvider`, OCR-only: a sync single-document
call (`MistralOcr.process`) and file-based batch submit (`submit_batch`) plus
two narrow batch-result primitives — `get_batch_status` (a status-only poll,
never a download) and `fetch_batch_results` (download + parse the result files)
— against the `/v1/ocr` endpoint. This mirrors the anthropic lane's
`get_batch_status`/`fetch_batch_result_lines` split in `sax_platform.llm.batch`:
the timer-loop transport asks the cheap status question every tick and pays for
the download exactly once, after the batch has ended. Mistral **chat** support
(`build_request_params`, `call`, message/content translation, the tool-call
output-type registry) was deleted, not ported — verified zero production callers
(all forge tier defaults are Anthropic; see
`development-plans/tasks/T3.3-mistral-ocr-chat-deleted.md`).

Provenance note on `process`: sax_llm's `mistral.py` has no sync OCR call to
port. Its only sync entry point, `call()`, is chat-only
(`client.chat.complete_async`); every OCR code path in that module is batch
(`submit_batch`'s `/v1/ocr` file-upload branch, the batch result-fetch path's
image extraction). `process` below is therefore new code, not a port: a thin
wrapper around the Mistral SDK's own `client.ocr.process_async`, built to
share `extract_images`/`parse_batch_result` with the batch lane so a
sync-call result has the same (body, images) shape as a batch-fetched one.

The Mistral SDK client is a constructor parameter (`MistralOcr.__init__`) —
no module global, no env reading inside the class. Build one with
`make_mistral_client`.

This module keeps the `mistralai` SDK out of module import time (T3.5):
the frozen batch-result models here (`BatchPollStatus`, `BatchResultEntry`,
`ExtractedImage`) are the shared batch-result shapes that consumers import at
module level — including modules that are chain-imported inside the Temporal
workflow sandbox — so `import sax_platform.ocr` must not drag in an HTTP stack.
SDK types are annotation-only (`TYPE_CHECKING` + deferred annotations) and the
two runtime touch points (`make_mistral_client`, `MistralOcr.submit_batch`'s
`UnrecognizedStr`) import locally, mirroring the PEP-562 lazy-export discipline
`sax_platform.llm` uses.
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from datetime import datetime

    from mistralai import Mistral
    from mistralai.models import BatchError, DocumentTypedDict, FileTypedDict

__all__ = [
    "BatchPollStatus",
    "BatchResultEntry",
    "ExtractedImage",
    "MistralOcr",
    "extract_images",
    "make_mistral_client",
]

logger = logging.getLogger(__name__)

_OCR_ENDPOINT = "/v1/ocr"

# Page size for the batch-jobs list sweep. The Mistral list endpoint caps a
# page at 100 rows; requesting that maximum minimizes round-trips.
_LIST_PAGE_SIZE = 100


# ---------------------------------------------------------------------------
# Result models — structurally identical (field names + types) to the
# OCR-relevant subset of sax_llm.models: BatchPollStatus, ExtractedImage,
# BatchResultEntry. A consumer (forge) adapts by import-path swap alone. Frozen
# per this repo's value-type convention (see sax_platform.llm.models) —
# sax_llm's originals were plain, unfrozen BaseModels; freezing here doesn't
# change field names/types/behavior as observed by a caller that only reads
# attributes. (sax_llm's combined status+entries poll-result envelope is gone:
# the status-only and result-fetch primitives return `BatchPollStatus` and
# `list[BatchResultEntry]` directly.)
# ---------------------------------------------------------------------------


class BatchPollStatus(StrEnum):
    """Normalized batch poll statuses."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    FAILED = "failed"
    CANCELED = "canceled"
    EXPIRED = "expired"


class ExtractedImage(BaseModel):
    """An image extracted from an OCR response, before storage."""

    model_config = ConfigDict(frozen=True)

    original_image_id: str
    page_index: int
    image_base64: str
    mime_type: str = "image/jpeg"
    top_left_x: int | None = None
    top_left_y: int | None = None
    bottom_right_x: int | None = None
    bottom_right_y: int | None = None


class BatchResultEntry(BaseModel):
    """A single result entry from a batch response."""

    model_config = ConfigDict(frozen=True)

    custom_id: str
    succeeded: bool
    raw_response_json: str | None = None
    error: str | None = None
    extracted_images: list[ExtractedImage] = Field(default_factory=list)


# Mistral's raw batch-job statuses, mapped onto the platform's normalized set.
# An unrecognized status falls back to IN_PROGRESS — "keep waiting" is the safe
# default for a state the timer loop can't classify as terminal.
_MISTRAL_JOB_STATUS: dict[str, BatchPollStatus] = {
    "QUEUED": BatchPollStatus.PENDING,
    "RUNNING": BatchPollStatus.IN_PROGRESS,
    "SUCCESS": BatchPollStatus.ENDED,
    "FAILED": BatchPollStatus.FAILED,
    "TIMEOUT_EXCEEDED": BatchPollStatus.EXPIRED,
    "CANCELLATION_REQUESTED": BatchPollStatus.CANCELED,
    "CANCELLED": BatchPollStatus.CANCELED,
}


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def make_mistral_client(api_key: str) -> Mistral:
    """Construct a `Mistral` SDK client from an explicit `api_key`.

    `api_key` is required, explicit config: a composition root resolves it from
    `MISTRAL_API_KEY` via `LlmSettings` and passes it in. This factory reads no
    environment itself. (The installed `mistralai` SDK does not fall back to a
    `MISTRAL_API_KEY` env var of its own, so an empty key would otherwise
    construct a client that fails every call later with a 401.)

    Raises `ValueError` if `api_key` is empty — the ocr worker calls this at
    startup, exactly where an operator should learn of a missing key.
    """
    from mistralai import Mistral

    if not api_key:
        msg = "make_mistral_client requires a non-empty api_key (MISTRAL_API_KEY)."
        raise ValueError(msg)
    return Mistral(api_key=api_key)


# ---------------------------------------------------------------------------
# Batch error / file helpers (module-level; ported from sax_llm.mistral)
# ---------------------------------------------------------------------------


def _is_set(value: object) -> bool:
    """Return True only if *value* is a usable, non-empty string.

    Guards against the Mistral SDK's `Unset` sentinel, which is falsy but
    not `None` — `is not None` alone would miss it.
    """
    return bool(value)


def _format_batch_errors(errors: list[BatchError]) -> str:
    """Format a list of BatchError objects into a human-readable string."""
    parts: list[str] = []
    for err in errors:
        message = getattr(err, "message", str(err))
        count = getattr(err, "count", None)
        count = count if isinstance(count, int) else 1
        part = message if count <= 1 else f"{message} (x{count})"
        parts.append(part)
    return "; ".join(parts)


async def _download_file_content(client: Mistral, file_id: str) -> str:
    """Download a Mistral file and return its decoded text content."""
    output_file: object = await client.files.download_async(file_id=file_id)
    aread = getattr(output_file, "aread", None)
    if callable(aread):
        content = await aread()
        return cast("bytes", content).decode("utf-8")
    read = getattr(output_file, "read", None)
    if callable(read):
        content = read()
        return cast("bytes", content).decode("utf-8")
    return str(output_file)


def _parse_error_file_entries(content: str) -> list[BatchResultEntry]:
    """Parse error-file JSONL content into BatchResultEntry objects."""
    entries: list[BatchResultEntry] = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed error-file line: %.120s", line)
            continue

        custom_id = data.get("custom_id", "unknown")
        # `data.get("response", {})` only supplies the {} default when the
        # key is *absent*. The error-file shape can carry an explicit
        # `"response": null`, in which case `.get` returns the key's actual
        # value (None) and a chained `.get("body", ...)` crashes. `(... or
        # {})` normalizes both the missing-key and explicit-null cases.
        response = data.get("response") or {}
        body = response.get("body") or {}
        error_detail = body.get("error") or data.get("error")
        error_str = json.dumps(error_detail) if error_detail else "Unknown error"

        entries.append(
            BatchResultEntry(
                custom_id=custom_id,
                succeeded=False,
                error=error_str,
            )
        )
    return entries


def extract_images(response_body: dict[str, Any]) -> list[ExtractedImage]:
    """Extract images from an OCR response and strip base64 data from the body.

    Mutates `response_body` in place (deletes each image's `image_base64`
    key) — ported from sax_llm's `_extract_images_from_response`, made
    public (it is now one of this module's OCR-relevant exports rather than
    a provider-private helper). The bounding-box fields were assembled
    through an intermediate `top_left`/`bottom_right` tuple pair in the
    original for no behavioral reason (each component was read straight
    back out); this port reads them directly, which is equivalent.
    """
    extracted: list[ExtractedImage] = []
    pages = response_body.get("pages", [])
    for page_index, page in enumerate(pages):
        for img in page.get("images", []):
            image_base64 = img.get("image_base64")
            if not image_base64:
                continue
            original_id = img.get("id", f"img-{page_index}.jpeg")

            mime_type = "image/jpeg"
            if isinstance(image_base64, str) and image_base64.startswith("data:"):
                header = image_base64.split(",", 1)[0]
                mime_type = header.split(":")[1].split(";")[0]

            extracted.append(
                ExtractedImage(
                    original_image_id=original_id,
                    page_index=page_index,
                    image_base64=image_base64,
                    mime_type=mime_type,
                    top_left_x=img.get("top_left_x"),
                    top_left_y=img.get("top_left_y"),
                    bottom_right_x=img.get("bottom_right_x"),
                    bottom_right_y=img.get("bottom_right_y"),
                )
            )
            del img["image_base64"]

    return extracted


# ---------------------------------------------------------------------------
# MistralOcr
# ---------------------------------------------------------------------------


class MistralOcr:
    """Mistral OCR capability: a sync single-document call plus file-based
    batch submit and the two batch-result primitives — `get_batch_status`
    (status only, no download) and `fetch_batch_results` (download + parse) —
    against the `/v1/ocr` endpoint.

    The SDK client is injected (constructor argument) — this class holds no
    other state and reads no environment variables itself. Callers own the
    client's lifecycle; build one with `make_mistral_client`.
    """

    def __init__(self, client: Mistral) -> None:
        self._client = client

    async def process(
        self,
        *,
        document: DocumentTypedDict,
        model: str,
        include_image_base64: bool = True,
    ) -> tuple[dict[str, Any], list[ExtractedImage]]:
        """Synchronous single-document OCR call.

        New code — see the module docstring's provenance note; there is no
        sync OCR call in sax_llm.mistral to port. Thinly wraps the SDK's
        `client.ocr.process_async` and routes the response through
        `parse_batch_result` so the returned shape — `(body, images)` with
        images already stripped out of `body` — matches what a polled
        batch entry's `raw_response_json`/`extracted_images` pair produces.
        """
        response = await self._client.ocr.process_async(
            document=document,
            model=model,
            include_image_base64=include_image_base64,
        )
        raw_json = json.dumps(response.model_dump(mode="json"))
        return self.parse_batch_result(raw_json)

    async def submit_batch(
        self,
        requests: list[dict[str, Any]],
        model: str,
        *,
        endpoint: str = _OCR_ENDPOINT,
    ) -> str:
        """Submit a batch job via file upload.

        Port of sax_llm's `_submit_batch_via_file`. `submit_batch`'s
        original dispatch between inline-JSON submission (default chat
        endpoint) and file-based upload (`/v1/ocr` only) is gone along with
        chat support: this OCR-only class always uploads a JSONL file.
        `endpoint` stays a keyword parameter — defaulted to `/v1/ocr`. ocr's
        own submit activity (T4.2 ST2) is the caller and passes it
        explicitly. `endpoint = endpoint or _OCR_ENDPOINT` normalizes an
        empty string the same as "not supplied," so a caller that forwards
        `endpoint=""` still gets the OCR endpoint.
        """
        from mistralai.types.basemodel import UnrecognizedStr

        endpoint = endpoint or _OCR_ENDPOINT
        lines = [json.dumps(r) for r in requests]
        jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")

        file_payload: FileTypedDict = {"file_name": "batch.jsonl", "content": jsonl_bytes}
        upload_result = await self._client.files.upload_async(
            file=file_payload,
            purpose="batch",
        )

        job = await self._client.batch.jobs.create_async(
            input_files=[upload_result.id],
            model=model,
            endpoint=UnrecognizedStr(endpoint),
        )
        return job.id

    async def get_batch_status(self, batch_id: str) -> BatchPollStatus:
        """Fetch one batch job's normalized lifecycle status. No download, ever.

        A single `batch.jobs.get_async`: Mistral's `jobs.get` returns status
        metadata and file IDs only — result content lives behind the Files
        endpoint — so a status-only poll never touches file storage and is
        natively cheap to call every timer tick. The raw Mistral status maps
        through `_MISTRAL_JOB_STATUS` (an unrecognized status falls back to
        `IN_PROGRESS`, i.e. "keep waiting"). `job.errors` and a non-zero
        `job.failed_requests` are logged as warnings here — once per poll —
        exactly as the old all-in-one poll did before the split.

        Mirrors the anthropic lane's `sax_platform.llm.batch.get_batch_status`:
        a thin status read that classifies no results and downloads nothing.
        The terminal result download is `fetch_batch_results`' job.
        """
        job = await self._client.batch.jobs.get_async(job_id=batch_id)

        if job.errors:
            logger.warning("Batch %s errors: %s", batch_id, _format_batch_errors(job.errors))
        if job.failed_requests > 0:
            logger.warning("Batch %s has %d failed request(s)", batch_id, job.failed_requests)

        return _MISTRAL_JOB_STATUS.get(job.status, BatchPollStatus.IN_PROGRESS)

    async def list_batch_statuses(self, *, created_after: datetime) -> dict[str, BatchPollStatus]:
        """List normalized statuses for every batch job created after a cutoff.

        A stateless broadcast primitive (T4.4): one `batch.jobs.list_async`
        sweep per page, paged with `page_size=_LIST_PAGE_SIZE` until a short
        page (fewer rows than the page size, including an empty one) signals the
        server has no more. `created_after` is a server-side filter, passed
        straight through — the sweep asks only for jobs newer than the cutoff.

        Like `get_batch_status`, this reads status metadata only and downloads
        nothing: each job's `output_file` is a file id left untouched, so the
        sweep never touches file storage. The raw Mistral status maps through
        `_MISTRAL_JOB_STATUS` (an unrecognized status falls back to
        `IN_PROGRESS`, i.e. "keep waiting"). No status filtering is applied — a
        remotely-finished job is reported exactly like a running one, so a
        broadcast can hint completion to a waiter.

        Returns a `{job_id: BatchPollStatus}` mapping across all pages.
        """
        statuses: dict[str, BatchPollStatus] = {}
        page_index = 0
        while True:
            page = await self._client.batch.jobs.list_async(
                page=page_index,
                page_size=_LIST_PAGE_SIZE,
                created_after=created_after,
            )
            jobs = page.data or []
            for job in jobs:
                statuses[job.id] = _MISTRAL_JOB_STATUS.get(job.status, BatchPollStatus.IN_PROGRESS)
            if len(jobs) < _LIST_PAGE_SIZE:
                break
            page_index += 1

        logger.debug(
            "Listed %d batch job status(es) created after %s", len(statuses), created_after
        )
        return statuses

    async def fetch_batch_results(self, batch_id: str) -> list[BatchResultEntry]:
        """Download and parse a finished batch's result files into per-request entries.

        Precondition: call only after `get_batch_status` has reported `ENDED`.
        This method does not re-check status — it goes straight for the files.

        One `batch.jobs.get_async` to obtain the `error_file`/`output_file` IDs,
        then the download/parse logic ported verbatim from sax_llm's old
        all-in-one poll: error-file entries first, then each `output_file` line,
        with an `output_file` entry replacing any same-`custom_id` error-file
        entry (the success record wins). Image data is stripped out of each
        succeeded body into `extracted_images` via `extract_images`. The
        success-branch dispatch narrows to `"pages"` only — `"choices"` was the
        chat-batch shape, and chat support isn't ported.

        A SUCCESS job with no `output_file` is a partial failure: the current
        warning is logged and whatever error-file entries exist are returned
        (possibly none). Callers key results by `custom_id`; a request whose id
        is absent from the returned list surfaces to that one waiter as a
        per-request error, which is the correct signal for a request that
        produced no output.
        """
        job = await self._client.batch.jobs.get_async(job_id=batch_id)

        entries: list[BatchResultEntry] = []
        if _is_set(job.error_file):
            error_content = await _download_file_content(self._client, cast("str", job.error_file))
            entries.extend(_parse_error_file_entries(error_content))

        if not _is_set(job.output_file):
            error_detail = _format_batch_errors(job.errors) if job.errors else "no output file"
            logger.warning(
                "Batch %s succeeded but output_file is not set: %s",
                batch_id,
                error_detail,
            )
            return entries

        content = await _download_file_content(self._client, cast("str", job.output_file))
        error_ids = {e.custom_id for e in entries}
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            entry_data = json.loads(line)
            custom_id = entry_data.get("custom_id", "")
            # `or {}` on both hops: .get's default only applies when the key is
            # absent — a present "response": null would otherwise crash the
            # whole fetch (same defect class as _parse_error_file_entries).
            response_body = (entry_data.get("response") or {}).get("body") or {}
            if custom_id in error_ids:
                entries = [e for e in entries if e.custom_id != custom_id]
                error_ids.discard(custom_id)
            if response_body.get("pages"):
                extracted = extract_images(response_body)
                entries.append(
                    BatchResultEntry(
                        custom_id=custom_id,
                        succeeded=True,
                        raw_response_json=json.dumps(response_body),
                        extracted_images=extracted,
                    )
                )
            else:
                entries.append(
                    BatchResultEntry(
                        custom_id=custom_id,
                        succeeded=False,
                        error=json.dumps(response_body.get("error", "Unknown error")),
                    )
                )

        return entries

    def parse_batch_result(self, raw_json: str) -> tuple[dict[str, Any], list[ExtractedImage]]:
        """Parse a raw OCR response body into `(body, extracted_images)`.

        sax_llm's `parse_batch_result` took an `output_type_name` and
        looked up a registered pydantic model to decode chat tool-call
        arguments — chat-only machinery, not ported. What OCR actually
        needs from a raw response body is the same image extraction
        `fetch_batch_results` does inline for each batch entry; this method
        exposes that as a standalone step so `process` (the sync lane) can
        share it instead of duplicating the body-parsing logic.
        """
        body: dict[str, Any] = json.loads(raw_json)
        images = extract_images(body)
        return body, images
