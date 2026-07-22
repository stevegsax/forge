"""Tests for sax_platform.ocr — the platform's Mistral OCR capability.

The Mistral SDK client is a constructor seam (`MistralOcr.__init__(self,
client: Mistral)`), not a module-level HTTP client the way
`anthropic.AsyncAnthropic` is for `sax_platform.llm.client` (see
test_client.py's httpx-`MockTransport` tests). So these tests mock the
injected `client` object directly with `MagicMock`/`AsyncMock` — the same
pattern sax_llm's own mistral provider tests used — rather than faking a
transport underneath a real SDK client. There is no wire-format contract to
verify here beyond what the SDK itself already tests; what's under test is
this module's own request shaping and response parsing.
"""

import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from sax_platform.ocr import (
    BatchPollStatus,
    ExtractedImage,
    MistralOcr,
    _download_file_content,
    _format_batch_errors,
    _is_set,
    _parse_error_file_entries,
    extract_images,
    make_mistral_client,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Unset:
    """Mimic the Mistral SDK's `Unset` sentinel: falsy, but not `None` — a
    naive `is not None` check would treat it as present."""

    def __bool__(self) -> bool:
        return False


def _make_mock_batch_job(
    status: str = "SUCCESS",
    *,
    output_file: object = "file-output-default",
    error_file: object = None,
    errors: list[Any] | None = None,
    failed_requests: int = 0,
) -> MagicMock:
    """Build a mock Mistral BatchJobOut with explicit error-related fields.

    Explicit values (rather than leaving MagicMock to auto-create truthy
    stubs) matter for `errors`, `error_file`, and `failed_requests`, whose
    absence/presence this module branches on.
    """
    job = MagicMock()
    job.status = status
    job.output_file = output_file
    job.error_file = error_file
    job.errors = errors or []
    job.failed_requests = failed_requests
    return job


def _make_mock_file(content: str) -> MagicMock:
    """Build a mock Mistral file download response (async httpx-style)."""
    mock_file = MagicMock(spec=[])  # spec=[] prevents auto-attribute creation
    mock_file.aread = AsyncMock(return_value=content.encode("utf-8"))
    return mock_file


def _ocr_body(
    *, pages: list[dict[str, Any]] | None = None, model: str = "mistral-ocr-latest"
) -> dict[str, Any]:
    """A minimal successful OCR response body ('pages', not 'choices')."""
    return {"pages": pages if pages is not None else [{"markdown": "text"}], "model": model}


# `list_batch_statuses` pages by page_size 100; a job carries only id + status.
_LIST_PAGE_SIZE = 100
_CREATED_AFTER = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)


def _make_list_job(job_id: str, status: str = "RUNNING") -> MagicMock:
    """A minimal Mistral BatchJobOut mock for the list endpoint: id + status."""
    job = MagicMock()
    job.id = job_id
    job.status = status
    return job


def _make_list_page(jobs: list[MagicMock]) -> MagicMock:
    """A Mistral BatchJobsOut mock: `.data` is the page's list of jobs."""
    page = MagicMock()
    page.data = jobs
    page.total = len(jobs)
    return page


# ---------------------------------------------------------------------------
# make_mistral_client
# ---------------------------------------------------------------------------


class TestMakeMistralClient:
    def test_explicit_api_key_used_verbatim(self) -> None:
        client = make_mistral_client(api_key="explicit-key")
        assert client.sdk_configuration.security.api_key == "explicit-key"  # type: ignore[union-attr]

    def test_empty_string_raises(self) -> None:
        """An empty key used to resolve silently to "" and construct a client
        that would only fail hours later with 401s. It must now fail loudly at
        construction — the ocr worker calls this at startup, exactly where an
        operator should learn of the misconfiguration."""
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            make_mistral_client(api_key="")

    def test_explicit_arg_used_verbatim_ignoring_ambient_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The factory reads no environment: a non-empty explicit api_key is
        used verbatim regardless of any MISTRAL_API_KEY in the env."""
        monkeypatch.setenv("MISTRAL_API_KEY", "env-key")
        client = make_mistral_client(api_key="explicit-key")
        assert client.sdk_configuration.security.api_key == "explicit-key"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestIsSet:
    def test_none_is_not_set(self) -> None:
        assert _is_set(None) is False

    def test_unset_sentinel_is_not_set(self) -> None:
        assert _is_set(_Unset()) is False

    def test_valid_string_is_set(self) -> None:
        assert _is_set("file-abc-123") is True

    def test_empty_string_is_not_set(self) -> None:
        assert _is_set("") is False


class TestFormatBatchErrors:
    def test_empty_list(self) -> None:
        assert _format_batch_errors([]) == ""

    def test_single_error_count_one(self) -> None:
        err = MagicMock()
        err.message = "rate limit exceeded"
        err.count = 1
        assert _format_batch_errors([err]) == "rate limit exceeded"

    def test_single_error_count_greater_than_one(self) -> None:
        err = MagicMock()
        err.message = "context length exceeded"
        err.count = 5
        assert _format_batch_errors([err]) == "context length exceeded (x5)"

    def test_multiple_errors(self) -> None:
        err1 = MagicMock()
        err1.message = "error A"
        err1.count = 1
        err2 = MagicMock()
        err2.message = "error B"
        err2.count = 3
        assert _format_batch_errors([err1, err2]) == "error A; error B (x3)"

    def test_fallback_to_str_when_no_message_attr(self) -> None:
        assert _format_batch_errors(["plain string error"]) == "plain string error"  # type: ignore[list-item]


class TestDownloadFileContent:
    async def test_aread_file_like_object(self) -> None:
        client = MagicMock()
        client.files.download_async = AsyncMock(return_value=_make_mock_file("hello world"))

        result = await _download_file_content(client, "file-123")

        assert result == "hello world"
        client.files.download_async.assert_called_once_with(file_id="file-123")

    async def test_read_file_like_object(self) -> None:
        client = MagicMock()
        file_obj = BytesIO(b"hello world")
        client.files.download_async = AsyncMock(return_value=file_obj)

        result = await _download_file_content(client, "file-456")

        assert result == "hello world"

    async def test_plain_string_fallback(self) -> None:
        client = MagicMock()
        # spec=[] so no aread/read attributes get auto-created by MagicMock.
        client.files.download_async = AsyncMock(return_value="raw string content")

        result = await _download_file_content(client, "file-789")

        assert result == "raw string content"


class TestParseErrorFileEntries:
    def test_empty_content(self) -> None:
        assert _parse_error_file_entries("") == []

    def test_blank_lines_skipped(self) -> None:
        assert _parse_error_file_entries("\n\n  \n") == []

    def test_standard_error_format(self) -> None:
        line = json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "body": {"error": {"type": "invalid_request", "message": "bad input"}}
                },
            }
        )
        entries = _parse_error_file_entries(line)
        assert len(entries) == 1
        assert entries[0].custom_id == "req-1"
        assert entries[0].succeeded is False
        assert entries[0].error is not None
        assert "bad input" in entries[0].error

    def test_top_level_error_key(self) -> None:
        line = json.dumps({"custom_id": "req-2", "error": {"message": "server error"}})
        entries = _parse_error_file_entries(line)
        assert entries[0].custom_id == "req-2"
        assert entries[0].error is not None
        assert "server error" in entries[0].error

    def test_malformed_json_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        content = "not valid json\n" + json.dumps(
            {"custom_id": "req-ok", "error": {"message": "real error"}}
        )
        with caplog.at_level(logging.WARNING):
            entries = _parse_error_file_entries(content)
        assert len(entries) == 1
        assert entries[0].custom_id == "req-ok"
        assert "malformed" in caplog.text.lower()

    def test_missing_custom_id_defaults_to_unknown(self) -> None:
        entries = _parse_error_file_entries(json.dumps({"error": {"message": "oops"}}))
        assert entries[0].custom_id == "unknown"

    def test_multiple_lines(self) -> None:
        lines = "\n".join(
            [
                json.dumps({"custom_id": "r1", "error": {"message": "e1"}}),
                json.dumps({"custom_id": "r2", "error": {"message": "e2"}}),
            ]
        )
        entries = _parse_error_file_entries(lines)
        assert [e.custom_id for e in entries] == ["r1", "r2"]

    def test_null_response_falls_back_to_top_level_error(self) -> None:
        """Regression test: an error-file line can carry an explicit
        `"response": null` (key present, value None) rather than omitting
        the key entirely. `data.get("response", {})` does NOT apply its {}
        default in that case — it returns the actual value, None — so a
        naive `.get("body", {})` chained off it crashes with
        AttributeError. This must parse into a failed BatchResultEntry
        using the top-level `"error"` instead."""
        line = json.dumps(
            {
                "custom_id": "req-null-response",
                "response": None,
                "error": {"message": "request failed before a response was produced"},
            }
        )

        entries = _parse_error_file_entries(line)

        assert len(entries) == 1
        assert entries[0].custom_id == "req-null-response"
        assert entries[0].succeeded is False
        assert entries[0].error is not None
        assert "request failed before a response was produced" in entries[0].error


class TestExtractImages:
    def test_extracts_images_and_strips_base64(self) -> None:
        response_body = {
            "pages": [
                {
                    "markdown": "Page 1 text",
                    "images": [
                        {
                            "id": "img-0.jpeg",
                            "image_base64": "aW1hZ2UtZGF0YQ==",
                            "top_left_x": 10,
                            "top_left_y": 20,
                            "bottom_right_x": 100,
                            "bottom_right_y": 200,
                        },
                    ],
                },
            ],
        }

        extracted = extract_images(response_body)

        assert len(extracted) == 1
        img = extracted[0]
        assert img == ExtractedImage(
            original_image_id="img-0.jpeg",
            page_index=0,
            image_base64="aW1hZ2UtZGF0YQ==",
            mime_type="image/jpeg",
            top_left_x=10,
            top_left_y=20,
            bottom_right_x=100,
            bottom_right_y=200,
        )
        # base64 stripped from the body in place; id survives.
        assert "image_base64" not in response_body["pages"][0]["images"][0]
        assert response_body["pages"][0]["images"][0]["id"] == "img-0.jpeg"

    def test_multiple_pages_multiple_images(self) -> None:
        response_body = {
            "pages": [
                {
                    "markdown": "Page 1",
                    "images": [
                        {"id": "img-0.jpeg", "image_base64": "data0"},
                        {"id": "img-1.jpeg", "image_base64": "data1"},
                    ],
                },
                {"markdown": "Page 2", "images": [{"id": "img-0.jpeg", "image_base64": "data2"}]},
            ],
        }

        extracted = extract_images(response_body)

        assert [img.page_index for img in extracted] == [0, 0, 1]

    def test_no_images_returns_empty(self) -> None:
        response_body = {"pages": [{"markdown": "No images here", "images": []}]}
        assert extract_images(response_body) == []

    def test_no_pages_returns_empty(self) -> None:
        assert extract_images({"model": "mistral-ocr-latest"}) == []

    def test_skips_images_without_base64(self) -> None:
        response_body = {
            "pages": [
                {
                    "markdown": "Page 1",
                    "images": [
                        {"id": "img-0.jpeg"},
                        {"id": "img-1.jpeg", "image_base64": "data1"},
                    ],
                },
            ],
        }

        extracted = extract_images(response_body)

        assert len(extracted) == 1
        assert extracted[0].original_image_id == "img-1.jpeg"

    def test_no_bounding_box_fields_default_to_none(self) -> None:
        response_body = {
            "pages": [
                {"markdown": "Page 1", "images": [{"id": "img-0.jpeg", "image_base64": "data0"}]}
            ],
        }

        extracted = extract_images(response_body)

        assert extracted[0].top_left_x is None
        assert extracted[0].bottom_right_y is None

    @pytest.mark.parametrize(
        ("data_uri", "expected_mime"),
        [
            ("data:image/png;base64,iVBORw0KGgo=", "image/png"),
            ("data:image/webp;base64,UklGR...", "image/webp"),
        ],
    )
    def test_mime_type_detected_from_data_uri(self, data_uri: str, expected_mime: str) -> None:
        response_body = {
            "pages": [
                {"markdown": "Page 1", "images": [{"id": "img-0", "image_base64": data_uri}]}
            ],
        }

        extracted = extract_images(response_body)

        assert extracted[0].mime_type == expected_mime

    def test_defaults_to_jpeg_without_data_uri_prefix(self) -> None:
        response_body = {
            "pages": [
                {"markdown": "Page 1", "images": [{"id": "img-0.jpeg", "image_base64": "aW1hZ2U="}]}
            ],
        }

        extracted = extract_images(response_body)

        assert extracted[0].mime_type == "image/jpeg"


# ---------------------------------------------------------------------------
# MistralOcr.submit_batch
# ---------------------------------------------------------------------------


class TestSubmitBatch:
    async def test_uploads_jsonl_and_returns_job_id(self) -> None:
        client = MagicMock()
        upload_result = MagicMock()
        upload_result.id = "file-upload-123"
        client.files.upload_async = AsyncMock(return_value=upload_result)
        job = MagicMock()
        job.id = "batch-job-789"
        client.batch.jobs.create_async = AsyncMock(return_value=job)

        ocr = MistralOcr(client)
        requests = [{"custom_id": "r1", "body": {"document": {"type": "document_url"}}}]
        result = await ocr.submit_batch(requests, "mistral-ocr-latest")

        assert result == "batch-job-789"

    async def test_upload_request_shape(self) -> None:
        client = MagicMock()
        upload_result = MagicMock()
        upload_result.id = "file-upload-1"
        client.files.upload_async = AsyncMock(return_value=upload_result)
        client.batch.jobs.create_async = AsyncMock(return_value=MagicMock(id="batch-1"))

        ocr = MistralOcr(client)
        doc_1 = {"type": "document_url", "document_url": "u1"}
        doc_2 = {"type": "document_url", "document_url": "u2"}
        requests = [
            {"custom_id": "r1", "body": {"document": doc_1}},
            {"custom_id": "r2", "body": {"document": doc_2}},
        ]
        await ocr.submit_batch(requests, "mistral-ocr-latest")

        upload_kwargs = client.files.upload_async.call_args.kwargs
        assert upload_kwargs["purpose"] == "batch"
        assert upload_kwargs["file"]["file_name"] == "batch.jsonl"
        lines = upload_kwargs["file"]["content"].decode("utf-8").strip().split("\n")
        assert [json.loads(line)["custom_id"] for line in lines] == ["r1", "r2"]

    async def test_create_job_uses_input_files_not_inline_requests(self) -> None:
        client = MagicMock()
        client.files.upload_async = AsyncMock(return_value=MagicMock(id="file-upload-2"))
        client.batch.jobs.create_async = AsyncMock(return_value=MagicMock(id="batch-2"))

        ocr = MistralOcr(client)
        await ocr.submit_batch([{"custom_id": "r1", "body": {}}], "mistral-ocr-latest")

        create_kwargs = client.batch.jobs.create_async.call_args.kwargs
        assert create_kwargs["input_files"] == ["file-upload-2"]
        assert create_kwargs["model"] == "mistral-ocr-latest"
        assert str(create_kwargs["endpoint"]) == "/v1/ocr"
        assert "requests" not in create_kwargs

    async def test_default_endpoint_is_ocr(self) -> None:
        client = MagicMock()
        client.files.upload_async = AsyncMock(return_value=MagicMock(id="file-upload-3"))
        client.batch.jobs.create_async = AsyncMock(return_value=MagicMock(id="batch-3"))

        ocr = MistralOcr(client)
        await ocr.submit_batch([{"custom_id": "r1", "body": {}}], "mistral-ocr-latest")

        assert str(client.batch.jobs.create_async.call_args.kwargs["endpoint"]) == "/v1/ocr"

    async def test_explicit_endpoint_override_is_honored(self) -> None:
        client = MagicMock()
        client.files.upload_async = AsyncMock(return_value=MagicMock(id="file-upload-4"))
        client.batch.jobs.create_async = AsyncMock(return_value=MagicMock(id="batch-4"))

        ocr = MistralOcr(client)
        await ocr.submit_batch(
            [{"custom_id": "r1", "body": {}}], "mistral-ocr-latest", endpoint="/v1/ocr"
        )

        assert str(client.batch.jobs.create_async.call_args.kwargs["endpoint"]) == "/v1/ocr"

    async def test_empty_string_endpoint_normalizes_to_default(self) -> None:
        """Regression test: a caller that forwards `endpoint=""` (rather than
        omitting it) still gets the default `/v1/ocr` — the keyword default
        alone can't catch an explicit empty string, so `endpoint or
        _OCR_ENDPOINT` normalizes it."""
        client = MagicMock()
        client.files.upload_async = AsyncMock(return_value=MagicMock(id="file-upload-5"))
        client.batch.jobs.create_async = AsyncMock(return_value=MagicMock(id="batch-5"))

        ocr = MistralOcr(client)
        await ocr.submit_batch([{"custom_id": "r1", "body": {}}], "mistral-ocr-latest", endpoint="")

        assert str(client.batch.jobs.create_async.call_args.kwargs["endpoint"]) == "/v1/ocr"


# ---------------------------------------------------------------------------
# MistralOcr.get_batch_status  (status-only poll; never downloads)
# ---------------------------------------------------------------------------


class TestGetBatchStatus:
    @pytest.mark.parametrize(
        ("mistral_status", "expected"),
        [
            ("QUEUED", BatchPollStatus.PENDING),
            ("RUNNING", BatchPollStatus.IN_PROGRESS),
            ("SUCCESS", BatchPollStatus.ENDED),
            ("FAILED", BatchPollStatus.FAILED),
            ("TIMEOUT_EXCEEDED", BatchPollStatus.EXPIRED),
            ("CANCELLATION_REQUESTED", BatchPollStatus.CANCELED),
            ("CANCELLED", BatchPollStatus.CANCELED),
            # An unrecognized status is treated as "keep waiting".
            ("SOME_FUTURE_STATUS", BatchPollStatus.IN_PROGRESS),
        ],
    )
    async def test_status_maps_and_never_downloads(
        self, mistral_status: str, expected: BatchPollStatus
    ) -> None:
        client = MagicMock()
        # An output_file is present even on SUCCESS — proving the status read
        # never reaches for it.
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job(mistral_status, output_file="file-out")
        )
        ocr = MistralOcr(client)

        status = await ocr.get_batch_status("batch-1")

        assert status == expected
        client.files.download_async.assert_not_called()

    async def test_logs_errors_and_failed_requests(self, caplog: pytest.LogCaptureFixture) -> None:
        client = MagicMock()
        err = MagicMock()
        err.message = "context length exceeded"
        err.count = 3
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job(
                "FAILED", output_file=None, errors=[err], failed_requests=3
            )
        )
        ocr = MistralOcr(client)

        with caplog.at_level(logging.WARNING):
            status = await ocr.get_batch_status("batch-1")

        assert status == BatchPollStatus.FAILED
        assert "context length exceeded (x3)" in caplog.text
        assert "3 failed request" in caplog.text
        client.files.download_async.assert_not_called()


# ---------------------------------------------------------------------------
# MistralOcr.list_batch_statuses  (stateless broadcast sweep; never downloads)
# ---------------------------------------------------------------------------


class TestListBatchStatuses:
    async def test_pages_until_short_page_and_returns_all_jobs(self) -> None:
        # Two full pages (100 each) then an empty page terminates the sweep.
        page0 = _make_list_page([_make_list_job(f"job-0-{i}") for i in range(_LIST_PAGE_SIZE)])
        page1 = _make_list_page([_make_list_job(f"job-1-{i}") for i in range(_LIST_PAGE_SIZE)])
        page2 = _make_list_page([])
        client = MagicMock()
        client.batch.jobs.list_async = AsyncMock(side_effect=[page0, page1, page2])
        ocr = MistralOcr(client)

        statuses = await ocr.list_batch_statuses(created_after=_CREATED_AFTER)

        assert len(statuses) == 2 * _LIST_PAGE_SIZE
        assert statuses["job-0-0"] == BatchPollStatus.IN_PROGRESS
        assert statuses["job-1-99"] == BatchPollStatus.IN_PROGRESS

        calls = client.batch.jobs.list_async.call_args_list
        # page advances 0, 1, 2; page_size and created_after are constant.
        assert [c.kwargs["page"] for c in calls] == [0, 1, 2]
        assert [c.kwargs["page_size"] for c in calls] == [_LIST_PAGE_SIZE] * 3
        assert [c.kwargs["created_after"] for c in calls] == [_CREATED_AFTER] * 3

    async def test_partial_page_terminates_without_further_fetch(self) -> None:
        # A full page then a partial page (< page_size) ends the sweep — a third
        # fetch would raise StopIteration off the 2-element side_effect.
        page0 = _make_list_page([_make_list_job(f"a{i}") for i in range(_LIST_PAGE_SIZE)])
        page1 = _make_list_page([_make_list_job(f"b{i}") for i in range(30)])
        client = MagicMock()
        client.batch.jobs.list_async = AsyncMock(side_effect=[page0, page1])
        ocr = MistralOcr(client)

        statuses = await ocr.list_batch_statuses(created_after=_CREATED_AFTER)

        assert len(statuses) == _LIST_PAGE_SIZE + 30
        assert client.batch.jobs.list_async.call_count == 2

    async def test_single_short_page_is_one_fetch(self) -> None:
        client = MagicMock()
        client.batch.jobs.list_async = AsyncMock(
            return_value=_make_list_page([_make_list_job("only-job")])
        )
        ocr = MistralOcr(client)

        statuses = await ocr.list_batch_statuses(created_after=_CREATED_AFTER)

        assert statuses == {"only-job": BatchPollStatus.IN_PROGRESS}
        assert client.batch.jobs.list_async.call_count == 1

    async def test_data_none_is_treated_as_empty(self) -> None:
        client = MagicMock()
        empty_page = MagicMock()
        empty_page.data = None  # the SDK types `data` as list | None
        client.batch.jobs.list_async = AsyncMock(return_value=empty_page)
        ocr = MistralOcr(client)

        statuses = await ocr.list_batch_statuses(created_after=_CREATED_AFTER)

        assert statuses == {}
        assert client.batch.jobs.list_async.call_count == 1

    @pytest.mark.parametrize(
        ("mistral_status", "expected"),
        [
            ("QUEUED", BatchPollStatus.PENDING),
            ("RUNNING", BatchPollStatus.IN_PROGRESS),
            ("SUCCESS", BatchPollStatus.ENDED),
            ("FAILED", BatchPollStatus.FAILED),
            ("TIMEOUT_EXCEEDED", BatchPollStatus.EXPIRED),
            ("CANCELLATION_REQUESTED", BatchPollStatus.CANCELED),
            ("CANCELLED", BatchPollStatus.CANCELED),
            # An unrecognized status is reported as "keep waiting".
            ("SOME_FUTURE_STATUS", BatchPollStatus.IN_PROGRESS),
        ],
    )
    async def test_every_provider_status_maps(
        self, mistral_status: str, expected: BatchPollStatus
    ) -> None:
        client = MagicMock()
        client.batch.jobs.list_async = AsyncMock(
            return_value=_make_list_page([_make_list_job("job-1", status=mistral_status)])
        )
        ocr = MistralOcr(client)

        statuses = await ocr.list_batch_statuses(created_after=_CREATED_AFTER)

        assert statuses == {"job-1": expected}

    async def test_reports_finished_jobs_without_status_filtering(self) -> None:
        # A terminal job (SUCCESS) is returned like any other — the sweep does
        # not filter by remote status, so a completion can be broadcast.
        client = MagicMock()
        client.batch.jobs.list_async = AsyncMock(
            return_value=_make_list_page(
                [
                    _make_list_job("running", status="RUNNING"),
                    _make_list_job("done", status="SUCCESS"),
                ]
            )
        )
        ocr = MistralOcr(client)

        statuses = await ocr.list_batch_statuses(created_after=_CREATED_AFTER)

        assert statuses == {
            "running": BatchPollStatus.IN_PROGRESS,
            "done": BatchPollStatus.ENDED,
        }
        # `status` is never passed — no server-side status narrowing.
        assert "status" not in client.batch.jobs.list_async.call_args.kwargs

    async def test_never_downloads(self) -> None:
        client = MagicMock()
        client.batch.jobs.list_async = AsyncMock(
            return_value=_make_list_page(
                [_make_list_job("job-1", status="SUCCESS")]  # terminal, has an output_file remotely
            )
        )
        ocr = MistralOcr(client)

        await ocr.list_batch_statuses(created_after=_CREATED_AFTER)

        # A status sweep touches no file storage and no single-job get endpoint.
        client.files.download_async.assert_not_called()
        client.batch.jobs.get_async.assert_not_called()


# ---------------------------------------------------------------------------
# MistralOcr.fetch_batch_results  (download + parse; call only after ENDED)
# ---------------------------------------------------------------------------


class TestFetchBatchResults:
    async def test_parses_jsonl_results(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job("SUCCESS", output_file="file-output-123")
        )
        jsonl = "\n".join(
            [
                json.dumps({"custom_id": "req-1", "response": {"body": _ocr_body()}}),
                json.dumps({"custom_id": "req-2", "response": {"body": _ocr_body()}}),
            ]
        )
        client.files.download_async = AsyncMock(return_value=_make_mock_file(jsonl))
        ocr = MistralOcr(client)

        entries = await ocr.fetch_batch_results("batch-1")

        assert len(entries) == 2
        assert [e.custom_id for e in entries] == ["req-1", "req-2"]
        assert all(e.succeeded for e in entries)
        assert all(e.raw_response_json is not None for e in entries)

    async def test_null_response_entry_in_output_file_does_not_crash(self) -> None:
        # Same defect class as the error-file path: "response": null is a
        # present key, so .get's default never applies — must not AttributeError.
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job("SUCCESS", output_file="file-output-null")
        )
        jsonl = "\n".join(
            [
                json.dumps({"custom_id": "req-null", "response": None}),
                json.dumps({"custom_id": "req-ok", "response": {"body": _ocr_body()}}),
            ]
        )
        client.files.download_async = AsyncMock(return_value=_make_mock_file(jsonl))
        ocr = MistralOcr(client)

        entries = await ocr.fetch_batch_results("batch-1")

        by_id = {e.custom_id: e for e in entries}
        assert by_id["req-null"].succeeded is False
        assert by_id["req-ok"].succeeded is True

    async def test_downloads_from_output_file(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job("SUCCESS", output_file="file-abc-789")
        )
        line = json.dumps({"custom_id": "req-1", "response": {"body": _ocr_body()}})
        client.files.download_async = AsyncMock(return_value=_make_mock_file(line))
        ocr = MistralOcr(client)

        await ocr.fetch_batch_results("batch-1")

        client.files.download_async.assert_called_once_with(file_id="file-abc-789")

    async def test_error_entry_within_output_file(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job("SUCCESS", output_file="file-output-456")
        )
        jsonl = json.dumps(
            {
                "custom_id": "req-bad",
                "response": {
                    "body": {"error": {"type": "invalid_request", "message": "bad input"}}
                },
            }
        )
        client.files.download_async = AsyncMock(return_value=_make_mock_file(jsonl))
        ocr = MistralOcr(client)

        entries = await ocr.fetch_batch_results("batch-1")

        assert len(entries) == 1
        assert entries[0].succeeded is False
        assert entries[0].error is not None

    async def test_null_output_file_returns_empty_without_download(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job("SUCCESS", output_file=None)
        )
        ocr = MistralOcr(client)

        entries = await ocr.fetch_batch_results("batch-1")

        # No output_file and no error_file: nothing to download, no entries.
        assert entries == []
        client.files.download_async.assert_not_called()

    async def test_missing_output_file_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job("SUCCESS", output_file=None)
        )
        ocr = MistralOcr(client)

        with caplog.at_level(logging.WARNING):
            await ocr.fetch_batch_results("batch-1")

        assert "output_file is not set" in caplog.text

    async def test_unset_sentinel_output_file_returns_empty(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job("SUCCESS", output_file=_Unset())
        )
        ocr = MistralOcr(client)

        entries = await ocr.fetch_batch_results("batch-1")

        assert entries == []

    async def test_skips_blank_lines(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job("SUCCESS", output_file="file-blank")
        )
        entry_1 = json.dumps({"custom_id": "req-1", "response": {"body": _ocr_body()}})
        entry_2 = json.dumps({"custom_id": "req-2", "response": {"body": _ocr_body()}})
        # A blank line strictly between two entries — content.strip() at the
        # top of the loop only trims the ends, so this is the only shape
        # that exercises the mid-loop blank-line continue.
        client.files.download_async = AsyncMock(
            return_value=_make_mock_file(f"{entry_1}\n\n{entry_2}")
        )
        ocr = MistralOcr(client)

        entries = await ocr.fetch_batch_results("batch-1")

        assert len(entries) == 2

    async def test_extracts_images_from_ocr_response(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job("SUCCESS", output_file="file-out-1")
        )
        image = {"id": "img-0.jpeg", "image_base64": "aW1n"}
        ocr_response = _ocr_body(pages=[{"markdown": "![img](img)", "images": [image]}])
        line = json.dumps({"custom_id": "req-1", "response": {"body": ocr_response}})
        client.files.download_async = AsyncMock(return_value=_make_mock_file(line))
        ocr = MistralOcr(client)

        entries = await ocr.fetch_batch_results("batch-img-1")

        entry = entries[0]
        assert entry.succeeded is True
        assert len(entry.extracted_images) == 1
        assert entry.extracted_images[0].original_image_id == "img-0.jpeg"
        assert entry.raw_response_json is not None
        assert "image_base64" not in entry.raw_response_json


class TestFetchBatchResultsErrorFileMerging:
    async def test_error_file_entries_merge_with_output_file_entries(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job(
                "SUCCESS", output_file="file-output", error_file="file-errors"
            )
        )
        output_jsonl = json.dumps({"custom_id": "req-1", "response": {"body": _ocr_body()}})
        error_jsonl = json.dumps(
            {"custom_id": "req-2", "response": {"body": {"error": {"message": "context too long"}}}}
        )

        async def _download(file_id: str) -> MagicMock:
            return _make_mock_file(output_jsonl if file_id == "file-output" else error_jsonl)

        client.files.download_async = AsyncMock(side_effect=_download)
        ocr = MistralOcr(client)

        entries = await ocr.fetch_batch_results("batch-1")

        by_id = {e.custom_id: e for e in entries}
        assert set(by_id) == {"req-1", "req-2"}
        assert by_id["req-1"].succeeded is True
        assert by_id["req-2"].succeeded is False

    async def test_output_file_entry_wins_on_duplicate_custom_id(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job(
                "SUCCESS", output_file="file-output", error_file="file-errors"
            )
        )
        output_jsonl = json.dumps({"custom_id": "req-1", "response": {"body": _ocr_body()}})
        error_body = {"error": {"message": "should be ignored"}}
        error_jsonl = json.dumps({"custom_id": "req-1", "response": {"body": error_body}})

        async def _download(file_id: str) -> MagicMock:
            return _make_mock_file(output_jsonl if file_id == "file-output" else error_jsonl)

        client.files.download_async = AsyncMock(side_effect=_download)
        ocr = MistralOcr(client)

        entries = await ocr.fetch_batch_results("batch-1")

        assert len(entries) == 1
        assert entries[0].succeeded is True

    async def test_error_file_download_failure_propagates(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job(
                "SUCCESS", output_file="file-output", error_file="file-errors"
            )
        )

        async def _download(file_id: str) -> MagicMock:
            if file_id == "file-errors":
                raise OSError("download failed")
            return _make_mock_file("")  # pragma: no cover

        client.files.download_async = AsyncMock(side_effect=_download)
        ocr = MistralOcr(client)

        with pytest.raises(OSError, match="download failed"):
            await ocr.fetch_batch_results("batch-1")

    async def test_entries_returned_when_output_file_missing_but_error_file_present(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job(
                "SUCCESS", output_file=None, error_file="file-errors", failed_requests=1
            )
        )
        error_body = {"error": {"message": "invalid request body"}}
        error_jsonl = json.dumps({"custom_id": "req-1", "response": {"body": error_body}})
        client.files.download_async = AsyncMock(return_value=_make_mock_file(error_jsonl))
        ocr = MistralOcr(client)

        entries = await ocr.fetch_batch_results("batch-1")

        # Output file missing but error-file entries survive: a waiter whose id
        # is absent here surfaces its own per-request error at the forge seam.
        assert len(entries) == 1
        assert entries[0].succeeded is False
        client.files.download_async.assert_called_once_with(file_id="file-errors")

    async def test_unset_error_file_is_not_downloaded(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job("SUCCESS", output_file="file-ok", error_file=_Unset())
        )
        line = json.dumps({"custom_id": "req-1", "response": {"body": _ocr_body()}})
        client.files.download_async = AsyncMock(return_value=_make_mock_file(line))
        ocr = MistralOcr(client)

        entries = await ocr.fetch_batch_results("batch-1")

        assert len(entries) == 1
        client.files.download_async.assert_called_once_with(file_id="file-ok")

    async def test_none_error_file_is_not_downloaded(self) -> None:
        client = MagicMock()
        client.batch.jobs.get_async = AsyncMock(
            return_value=_make_mock_batch_job("SUCCESS", output_file="file-ok", error_file=None)
        )
        line = json.dumps({"custom_id": "req-1", "response": {"body": _ocr_body()}})
        client.files.download_async = AsyncMock(return_value=_make_mock_file(line))
        ocr = MistralOcr(client)

        await ocr.fetch_batch_results("batch-1")

        client.files.download_async.assert_called_once_with(file_id="file-ok")


# ---------------------------------------------------------------------------
# MistralOcr.parse_batch_result
# ---------------------------------------------------------------------------


class TestParseBatchResult:
    def test_returns_body_and_images(self) -> None:
        client = MagicMock()
        ocr = MistralOcr(client)
        body = _ocr_body(
            pages=[{"markdown": "x", "images": [{"id": "img-0.jpeg", "image_base64": "aW1n"}]}]
        )
        raw_json = json.dumps(body)

        parsed_body, images = ocr.parse_batch_result(raw_json)

        assert parsed_body["pages"][0]["markdown"] == "x"
        assert len(images) == 1
        assert images[0].original_image_id == "img-0.jpeg"
        # Stripped from the returned body, same as the batch lane.
        assert "image_base64" not in parsed_body["pages"][0]["images"][0]

    def test_no_images_returns_empty_list(self) -> None:
        client = MagicMock()
        ocr = MistralOcr(client)

        body, images = ocr.parse_batch_result(json.dumps(_ocr_body()))

        assert images == []
        assert body["pages"] == [{"markdown": "text"}]


# ---------------------------------------------------------------------------
# MistralOcr.process
# ---------------------------------------------------------------------------


class TestProcess:
    async def test_calls_sdk_with_given_document_and_model(self) -> None:
        client = MagicMock()
        response = MagicMock()
        response.model_dump.return_value = _ocr_body()
        client.ocr.process_async = AsyncMock(return_value=response)
        ocr = MistralOcr(client)
        document = {"type": "document_url", "document_url": "https://example.test/doc.pdf"}

        await ocr.process(document=document, model="mistral-ocr-latest")

        client.ocr.process_async.assert_called_once_with(
            document=document, model="mistral-ocr-latest", include_image_base64=True
        )

    async def test_include_image_base64_forwarded(self) -> None:
        client = MagicMock()
        response = MagicMock()
        response.model_dump.return_value = _ocr_body()
        client.ocr.process_async = AsyncMock(return_value=response)
        ocr = MistralOcr(client)

        await ocr.process(
            document={"type": "document_url", "document_url": "u"},
            model="mistral-ocr-latest",
            include_image_base64=False,
        )

        assert client.ocr.process_async.call_args.kwargs["include_image_base64"] is False

    async def test_returns_body_and_extracted_images_like_parse_batch_result(self) -> None:
        client = MagicMock()
        response = MagicMock()
        response.model_dump.return_value = _ocr_body(
            pages=[{"markdown": "x", "images": [{"id": "img-0.jpeg", "image_base64": "aW1n"}]}]
        )
        client.ocr.process_async = AsyncMock(return_value=response)
        ocr = MistralOcr(client)

        body, images = await ocr.process(
            document={"type": "document_url", "document_url": "u"}, model="mistral-ocr-latest"
        )

        assert len(images) == 1
        assert images[0].original_image_id == "img-0.jpeg"
        assert "image_base64" not in body["pages"][0]["images"][0]

    async def test_model_dump_called_in_json_mode(self) -> None:
        """`mode="json"` is required so bytes/enum fields in the SDK's
        response serialize losslessly through `json.dumps`."""
        client = MagicMock()
        response = MagicMock()
        response.model_dump.return_value = _ocr_body()
        client.ocr.process_async = AsyncMock(return_value=response)
        ocr = MistralOcr(client)

        await ocr.process(document={"type": "document_url", "document_url": "u"}, model="m")

        assert response.model_dump.call_args == call(mode="json")


# ---------------------------------------------------------------------------
# Sandbox-light: importing the pure surfaces must never pull in mistralai
# ---------------------------------------------------------------------------


class TestSandboxLight:
    """`sax_platform.ocr` imports `mistralai` eagerly at module scope — but
    neither `sax_platform` nor `sax_platform.llm` may import `ocr` as a
    side effect, which is what keeps both safe to import inside the
    Temporal workflow sandbox (see the `sax_platform.llm` module
    docstring). Verified via a fresh subprocess, not `sys.modules` in this
    test process: this file's own top-level `from sax_platform.ocr import
    ...` above has already pulled `mistralai` into this process before any
    test body runs, so checking `sys.modules` here would prove nothing.
    """

    def test_import_sax_platform_does_not_import_mistralai(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sax_platform, sys; assert 'mistralai' not in sys.modules",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    def test_import_sax_platform_llm_does_not_import_mistralai(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sax_platform.llm, sys; assert 'mistralai' not in sys.modules",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
