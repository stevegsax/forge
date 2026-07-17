"""The platform's Mistral OCR capability (D88 / T3.3).

Ported from `sax_llm.mistral.MistralProvider`, OCR-only: sync single-document
calls (`MistralOcr.process`) and file-based batch submit/poll for the
`/v1/ocr` endpoint. Mistral **chat** support (`build_request_params`, `call`,
message/content translation, the tool-call output-type registry) was
deleted, not ported — verified zero production callers (all forge tier
defaults are Anthropic; see `development-plans/tasks/T3.3-mistral-ocr-chat-deleted.md`).

Provenance note on `process`: sax_llm's `mistral.py` has no sync OCR call to
port. Its only sync entry point, `call()`, is chat-only
(`client.chat.complete_async`); every OCR code path in that module is batch
(`submit_batch`'s `/v1/ocr` file-upload branch, `poll_batch`'s image
extraction). `process` below is therefore new code, not a port: a thin
wrapper around the Mistral SDK's own `client.ocr.process_async`, built to
share `extract_images`/`parse_batch_result` with the batch lane so a
sync-call result has the same (body, images) shape as a polled one.

The Mistral SDK client is a constructor parameter (`MistralOcr.__init__`) —
no module global, no env reading inside the class. Build one with
`make_mistral_client`.

This module imports `mistralai` eagerly at the top level: unlike
`sax_platform.llm`, which lazily exports its SDK-importing surfaces via PEP
562 in `__init__.py` so `import sax_platform` / `import sax_platform.llm`
stay safe inside the Temporal workflow sandbox, `sax_platform/ocr.py` is a
standalone top-level module (a sibling of `llm/`, not exported through
either `__init__.py`). A consumer imports it explicitly —
`from sax_platform.ocr import MistralOcr` — which is what keeps the
sandbox-light rule intact: neither `sax_platform/__init__.py` nor
`sax_platform/llm/__init__.py` references this module, so importing them
never drags in the Mistral SDK.
"""

import json
import logging
import os
from enum import StrEnum
from typing import Any, cast

from mistralai import Mistral
from mistralai.models import BatchError, DocumentTypedDict, FileTypedDict
from mistralai.types.basemodel import UnrecognizedStr
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BatchPollResult",
    "BatchPollStatus",
    "BatchResultEntry",
    "ExtractedImage",
    "MistralOcr",
    "extract_images",
    "make_mistral_client",
]

logger = logging.getLogger(__name__)

_OCR_ENDPOINT = "/v1/ocr"


# ---------------------------------------------------------------------------
# Result models — structurally identical (field names + types) to the
# OCR-relevant subset of sax_llm.models: BatchPollStatus, ExtractedImage,
# BatchResultEntry, BatchPollResult. A consumer (forge) adapts by import-path
# swap alone. Frozen per this repo's value-type convention (see
# sax_platform.llm.models) — sax_llm's originals were plain, unfrozen
# BaseModels; freezing here doesn't change field names/types/behavior as
# observed by a caller that only reads attributes.
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


class BatchPollResult(BaseModel):
    """Result of polling a batch job."""

    model_config = ConfigDict(frozen=True)

    status: BatchPollStatus
    entries: list[BatchResultEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def make_mistral_client(api_key: str | None = None) -> Mistral:
    """Construct a `Mistral` SDK client.

    Unlike `sax_platform.llm.client.make_client` (Anthropic), the installed
    `mistralai` SDK (1.12.4) does **not** fall back to a `MISTRAL_API_KEY`
    env var itself when `api_key` is omitted or `None` — there is no
    reference to that variable anywhere in the package. sax_llm's
    `MistralProvider.__init__` read the env var explicitly
    (`os.environ.get("MISTRAL_API_KEY", "")`) before constructing the SDK
    client; this factory preserves that exact fallback rather than relying
    on SDK behavior that doesn't exist.
    """
    resolved_key = api_key if api_key is not None else os.environ.get("MISTRAL_API_KEY", "")
    return Mistral(api_key=resolved_key)


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
        error_detail = data.get("response", {}).get("body", {}).get("error") or data.get("error")
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
    batch submit/poll against the `/v1/ocr` endpoint.

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
        `endpoint` stays a keyword parameter — defaulted to `/v1/ocr` — so
        the call shape matches what forge's opaque-blob SPI already calls:
        `provider.submit_batch(requests, model, endpoint=input.endpoint)`.
        """
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

    async def poll_batch(self, batch_id: str) -> BatchPollResult:
        """Poll a batch job.

        Port of sax_llm's `poll_batch`, including error-file merging (an
        `error_file` entry is dropped in favor of a same-`custom_id` entry
        from `output_file`, matching the original exactly) and the
        `output_file`-missing-on-SUCCESS degrade-to-FAILED path. The
        success-branch dispatch on `response_body.get("choices") or
        response_body.get("pages")` narrows to `"pages"` only — `"choices"`
        was the chat-batch response shape, and chat support isn't ported.
        """
        job = await self._client.batch.jobs.get_async(job_id=batch_id)

        status_map: dict[str, BatchPollStatus] = {
            "QUEUED": BatchPollStatus.PENDING,
            "RUNNING": BatchPollStatus.IN_PROGRESS,
            "SUCCESS": BatchPollStatus.ENDED,
            "FAILED": BatchPollStatus.FAILED,
            "TIMEOUT_EXCEEDED": BatchPollStatus.EXPIRED,
            "CANCELLATION_REQUESTED": BatchPollStatus.CANCELED,
            "CANCELLED": BatchPollStatus.CANCELED,
        }
        poll_status = status_map.get(job.status, BatchPollStatus.IN_PROGRESS)

        if job.errors:
            logger.warning("Batch %s errors: %s", batch_id, _format_batch_errors(job.errors))
        if job.failed_requests > 0:
            logger.warning("Batch %s has %d failed request(s)", batch_id, job.failed_requests)

        if poll_status != BatchPollStatus.ENDED:
            return BatchPollResult(status=poll_status)

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
            return BatchPollResult(status=BatchPollStatus.FAILED, entries=entries)

        content = await _download_file_content(self._client, cast("str", job.output_file))
        error_ids = {e.custom_id for e in entries}
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            entry_data = json.loads(line)
            custom_id = entry_data.get("custom_id", "")
            response_body = entry_data.get("response", {}).get("body", {})
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

        return BatchPollResult(status=BatchPollStatus.ENDED, entries=entries)

    def parse_batch_result(self, raw_json: str) -> tuple[dict[str, Any], list[ExtractedImage]]:
        """Parse a raw OCR response body into `(body, extracted_images)`.

        sax_llm's `parse_batch_result` took an `output_type_name` and
        looked up a registered pydantic model to decode chat tool-call
        arguments — chat-only machinery, not ported. What OCR actually
        needs from a raw response body is the same image extraction
        `poll_batch` already does inline for each batch entry; this method
        exposes that as a standalone step so `process` (the sync lane) can
        share it instead of duplicating the body-parsing logic.
        """
        body: dict[str, Any] = json.loads(raw_json)
        images = extract_images(body)
        return body, images
