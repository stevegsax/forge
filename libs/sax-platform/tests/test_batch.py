"""Tests for sax_platform.llm.batch — the batch lane.

`TestBuildBatchRequest` exercises the pure builder directly: no client, no
I/O. Everything else drives the real `AsyncAnthropic` SDK serialization and
parsing paths against a mocked transport (`httpx.MockTransport`), so the
wire shapes below are exact JSON — not SDK-object mocks — routed by request
path/method. Wire shapes were read from the installed SDK source
(`anthropic/resources/messages/batches.py` and the types it references), not
guessed.
"""

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx
import pytest
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError

from sax_platform.llm.batch import (
    BatchHandle,
    BatchItemResult,
    BatchRequestFailed,
    BatchStatus,
    build_batch_request,
    fetch_batch_results,
    get_batch_status,
    submit_batch,
)
from sax_platform.llm.cache import CacheSpec
from sax_platform.llm.models import Completion, MismatchOutcome, RefusedOutcome, TruncatedOutcome
from sax_platform.llm.schema import to_json_schema

OPUS = "claude-opus-4-8"  # MIN_CACHEABLE_TOKENS["claude-opus-4"] == 4096


class Widget(BaseModel):
    name: str
    count: int


class TestBuildBatchRequest:
    def test_text_lane_has_no_output_config(self) -> None:
        result = build_batch_request(
            "req-1",
            model=OPUS,
            max_tokens=256,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result == {
            "custom_id": "req-1",
            "params": {
                "model": OPUS,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "hi"}],
            },
        }

    def test_custom_id_passthrough(self) -> None:
        result = build_batch_request(
            "my-distinctive-id",
            model=OPUS,
            max_tokens=1,
            messages=[],
        )

        assert result["custom_id"] == "my-distinctive-id"

    def test_output_type_produces_closed_schema(self) -> None:
        result = build_batch_request(
            "req-2",
            model=OPUS,
            max_tokens=256,
            messages=[{"role": "user", "content": "hi"}],
            output_type=Widget,
        )

        expected_schema = to_json_schema(Widget)
        assert result == {
            "custom_id": "req-2",
            "params": {
                "model": OPUS,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "hi"}],
                "output_config": {"format": {"type": "json_schema", "schema": expected_schema}},
            },
        }
        assert expected_schema["additionalProperties"] is False

    def test_raw_output_schema_passed_through_untouched(self) -> None:
        # Deliberately missing `additionalProperties` — to_json_schema would
        # inject it; output_schema must not be run through that closing step.
        raw_schema = {"type": "object", "properties": {"x": {"type": "string"}}}

        result = build_batch_request(
            "req-3",
            model=OPUS,
            max_tokens=256,
            messages=[{"role": "user", "content": "hi"}],
            output_schema=raw_schema,
        )

        assert result == {
            "custom_id": "req-3",
            "params": {
                "model": OPUS,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "hi"}],
                "output_config": {"format": {"type": "json_schema", "schema": raw_schema}},
            },
        }
        assert "additionalProperties" not in raw_schema

    def test_both_output_type_and_output_schema_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            build_batch_request(
                "req-4",
                model=OPUS,
                max_tokens=256,
                messages=[],
                output_type=Widget,
                output_schema={"type": "object"},
            )

    def test_string_system_normalized_to_single_text_block(self) -> None:
        result = build_batch_request(
            "req-5",
            model=OPUS,
            max_tokens=256,
            messages=[],
            system="be concise",
        )

        assert result["params"]["system"] == [{"type": "text", "text": "be concise"}]

    def test_cache_opt_in_applied_above_model_minimum(self) -> None:
        # estimate_tokens is chars // 4; OPUS's threshold is 4096 tokens.
        big_text = "x" * (4096 * 4)

        result = build_batch_request(
            "req-6",
            model=OPUS,
            max_tokens=256,
            messages=[],
            system=[{"type": "text", "text": big_text}],
            cache=CacheSpec(),
        )

        assert result["params"]["system"] == [
            {"type": "text", "text": big_text, "cache_control": {"type": "ephemeral"}}
        ]

    def test_cache_opt_in_omitted_below_model_minimum(self) -> None:
        small_text = "x" * 40  # far below the 4096-token threshold

        result = build_batch_request(
            "req-7",
            model=OPUS,
            max_tokens=256,
            messages=[],
            system=[{"type": "text", "text": small_text}],
            cache=CacheSpec(),
        )

        assert result["params"]["system"] == [{"type": "text", "text": small_text}]

    def test_no_cache_spec_leaves_system_unchanged(self) -> None:
        big_text = "x" * (4096 * 4)

        result = build_batch_request(
            "req-8",
            model=OPUS,
            max_tokens=256,
            messages=[],
            system=[{"type": "text", "text": big_text}],
        )

        assert result["params"]["system"] == [{"type": "text", "text": big_text}]


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> AsyncAnthropic:
    """An `AsyncAnthropic` wired to a mocked httpx transport — the real SDK
    request-building and response-parsing code runs; only the network hop is
    replaced. No `ANTHROPIC_API_KEY` is read from the environment."""
    return AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler)),
    )


def _batch_json(
    *,
    batch_id: str,
    processing_status: str = "in_progress",
    request_counts: dict[str, int] | None = None,
    results_url: str | None = None,
) -> dict[str, Any]:
    """A wire-shaped `MessageBatch` JSON object (see
    `anthropic/types/messages/message_batch.py` for the required fields)."""
    return {
        "id": batch_id,
        "created_at": "2026-07-16T00:00:00Z",
        "expires_at": "2026-07-17T00:00:00Z",
        "processing_status": processing_status,
        "request_counts": request_counts
        or {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0, "processing": 0},
        "results_url": results_url,
        "type": "message_batch",
    }


def _message_json(
    *,
    stop_reason: str,
    content: list[dict[str, Any]],
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> dict[str, Any]:
    """A wire-shaped `Message` JSON object (see `anthropic/types/message.py`
    and `anthropic/types/usage.py`)."""
    return {
        "id": "msg_x",
        "content": content,
        "model": OPUS,
        "role": "assistant",
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "type": "message",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        },
    }


class TestSubmitBatch:
    async def test_returns_handle_from_create_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert request.url.path == "/v1/messages/batches"
            body = json.loads(request.content)
            assert body["requests"][0]["custom_id"] == "req-1"
            return httpx.Response(
                200,
                json=_batch_json(batch_id="msgbatch_01", processing_status="in_progress"),
            )

        client = _make_client(handler)
        requests = [
            build_batch_request(
                "req-1", model=OPUS, max_tokens=100, messages=[{"role": "user", "content": "hi"}]
            )
        ]

        handle = await submit_batch(client, requests)

        assert handle == BatchHandle(batch_id="msgbatch_01", processing_status="in_progress")


class TestGetBatchStatus:
    async def test_maps_request_counts(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET"
            assert request.url.path == "/v1/messages/batches/msgbatch_02"
            return httpx.Response(
                200,
                json=_batch_json(
                    batch_id="msgbatch_02",
                    processing_status="ended",
                    request_counts={
                        "succeeded": 3,
                        "errored": 1,
                        "canceled": 2,
                        "expired": 4,
                        "processing": 5,
                    },
                ),
            )

        client = _make_client(handler)

        status = await get_batch_status(client, "msgbatch_02")

        assert status == BatchStatus(
            batch_id="msgbatch_02",
            processing_status="ended",
            succeeded=3,
            errored=1,
            canceled=2,
            expired=4,
            processing=5,
        )


class TestFetchBatchResults:
    async def test_mixed_batch_classified_by_custom_id(self) -> None:
        batch_id = "msgbatch_03"
        results_url = "https://api.anthropic.com/mock-results/msgbatch_03.jsonl"

        lines = [
            # succeeded + valid JSON, validated via output_types.
            {
                "custom_id": "typed-ok",
                "result": {
                    "type": "succeeded",
                    "message": _message_json(
                        stop_reason="end_turn",
                        content=[
                            {"type": "text", "text": json.dumps({"name": "gadget", "count": 3})}
                        ],
                        cache_read_input_tokens=50,
                    ),
                },
            },
            # refusal (stop_details rides the wire unmodeled on this SDK,
            # surviving as a plain dict — the category must still be read).
            {
                "custom_id": "refused",
                "result": {
                    "type": "succeeded",
                    "message": _message_json(stop_reason="refusal", content=[])
                    | {"stop_details": {"type": "refusal", "category": "cyber"}},
                },
            },
            # max_tokens truncation.
            {
                "custom_id": "truncated",
                "result": {
                    "type": "succeeded",
                    "message": _message_json(
                        stop_reason="max_tokens",
                        content=[{"type": "text", "text": "partial out"}],
                        output_tokens=50,
                    ),
                },
            },
            # succeeded but prose, with an output_types entry -> mismatch.
            {
                "custom_id": "mismatch",
                "result": {
                    "type": "succeeded",
                    "message": _message_json(
                        stop_reason="end_turn",
                        content=[{"type": "text", "text": "This is prose, not JSON."}],
                    ),
                },
            },
            # errored envelope.
            {
                "custom_id": "errored-item",
                "result": {
                    "type": "errored",
                    "error": {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "bad input"},
                    },
                },
            },
            # succeeded, no output_types entry -> plain text Completion.
            {
                "custom_id": "plain-text",
                "result": {
                    "type": "succeeded",
                    "message": _message_json(
                        stop_reason="end_turn",
                        content=[{"type": "text", "text": "plain text answer"}],
                    ),
                },
            },
        ]
        jsonl_body = "\n".join(json.dumps(line) for line in lines).encode() + b"\n"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == f"/v1/messages/batches/{batch_id}":
                return httpx.Response(
                    200,
                    json=_batch_json(
                        batch_id=batch_id,
                        processing_status="ended",
                        request_counts={
                            "succeeded": 4,
                            "errored": 1,
                            "canceled": 0,
                            "expired": 0,
                            "processing": 0,
                        },
                        results_url=results_url,
                    ),
                )
            if request.method == "GET" and str(request.url) == results_url:
                return httpx.Response(200, content=jsonl_body)
            raise AssertionError(f"unexpected request: {request.method} {request.url}")

        client = _make_client(handler)

        results = await fetch_batch_results(
            client, batch_id, output_types={"typed-ok": Widget, "mismatch": Widget}
        )

        by_custom_id = {item.custom_id: item for item in results}
        assert set(by_custom_id) == {
            "typed-ok",
            "refused",
            "truncated",
            "mismatch",
            "errored-item",
            "plain-text",
        }

        typed_ok = by_custom_id["typed-ok"]
        assert isinstance(typed_ok.outcome, Completion)
        assert typed_ok.outcome.output == Widget(name="gadget", count=3)
        assert typed_ok.outcome.cache_read_input_tokens == 50  # acceptance criterion

        refused = by_custom_id["refused"]
        assert isinstance(refused.outcome, RefusedOutcome)
        # stop_details arrives as a plain dict on this SDK (unmodeled field);
        # _refusal_category reads both dict and attribute shapes.
        assert refused.outcome.category == "cyber"

        truncated = by_custom_id["truncated"]
        assert isinstance(truncated.outcome, TruncatedOutcome)
        assert truncated.outcome.partial_text == "partial out"
        assert truncated.outcome.max_tokens == 50

        mismatch = by_custom_id["mismatch"]
        assert isinstance(mismatch.outcome, MismatchOutcome)
        assert mismatch.outcome.raw_text == "This is prose, not JSON."

        errored_item = by_custom_id["errored-item"]
        assert isinstance(errored_item.outcome, BatchRequestFailed)
        assert errored_item.outcome.kind == "errored"
        assert "invalid_request_error" in errored_item.outcome.detail
        assert "bad input" in errored_item.outcome.detail

        plain_text = by_custom_id["plain-text"]
        assert isinstance(plain_text.outcome, Completion)
        assert plain_text.outcome.output == "plain text answer"

    async def test_results_keyed_by_custom_id_not_order(self) -> None:
        batch_id = "msgbatch_04"
        results_url = "https://api.anthropic.com/mock-results/msgbatch_04.jsonl"

        # Results are not guaranteed to come back in request order; return
        # them in a scrambled order and assert lookup is by custom_id.
        lines = [
            {
                "custom_id": "second",
                "result": {
                    "type": "succeeded",
                    "message": _message_json(
                        stop_reason="end_turn", content=[{"type": "text", "text": "b"}]
                    ),
                },
            },
            {
                "custom_id": "first",
                "result": {
                    "type": "succeeded",
                    "message": _message_json(
                        stop_reason="end_turn", content=[{"type": "text", "text": "a"}]
                    ),
                },
            },
        ]
        jsonl_body = "\n".join(json.dumps(line) for line in lines).encode() + b"\n"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == f"/v1/messages/batches/{batch_id}":
                return httpx.Response(
                    200,
                    json=_batch_json(
                        batch_id=batch_id, processing_status="ended", results_url=results_url
                    ),
                )
            if request.method == "GET" and str(request.url) == results_url:
                return httpx.Response(200, content=jsonl_body)
            raise AssertionError(f"unexpected request: {request.method} {request.url}")

        client = _make_client(handler)

        results = await fetch_batch_results(client, batch_id)

        by_custom_id = {item.custom_id: item for item in results}
        assert isinstance(by_custom_id["first"].outcome, Completion)
        assert by_custom_id["first"].outcome.output == "a"
        assert isinstance(by_custom_id["second"].outcome, Completion)
        assert by_custom_id["second"].outcome.output == "b"


class TestBatchItemResultShape:
    def test_is_frozen(self) -> None:
        item = BatchItemResult(
            custom_id="x", outcome=BatchRequestFailed(kind="expired", detail="gone")
        )

        with pytest.raises(ValidationError):
            item.custom_id = "y"  # type: ignore[misc]
