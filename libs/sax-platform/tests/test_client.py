"""Tests for sax_platform.llm.client — the sync-lane Anthropic client.

Transport is mocked at the httpx layer (`httpx.MockTransport`), not at the
SDK-object layer: every test builds a real `AsyncAnthropic` client wired to
a mock transport, so the SDK's actual request serialization and response
deserialization both run for real — only the network hop is faked. This is
what lets a test assert on the exact JSON body the SDK sends over the wire,
and it means these tests exercise the real quirks of the installed SDK
version (see `TestRefusal` below for one such quirk).
"""

import inspect
import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from sax_platform.llm.cache import CacheSpec
from sax_platform.llm.client import AnthropicLLM, make_client
from sax_platform.llm.models import LLMRefused, LLMSchemaMismatch, LLMTruncated

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]

Handler = Callable[[httpx.Request], httpx.Response]


class _Person(BaseModel):
    name: str
    age: int


def _wire_message(
    *,
    text: str | None = "hello",
    stop_reason: str = "end_turn",
    stop_details: dict[str, Any] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    """A Messages API response body shaped exactly like the real wire
    format, for a mock transport to return."""
    body: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}] if text is not None else [],
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        },
    }
    if stop_details is not None:
        body["stop_details"] = stop_details
    return body


def _handler_for(body: dict[str, Any], *, request_id: str | None = None) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"request-id": request_id} if request_id is not None else {}
        return httpx.Response(200, json=body, request=request, headers=headers)

    return handler


def _make_llm(
    handler: Handler,
) -> tuple[AnthropicLLM, anthropic.AsyncAnthropic, list[httpx.Request]]:
    """Build an `AnthropicLLM` wired to a `MockTransport` that records every
    request it receives before delegating to `handler` for the response.

    `api_key="test-key"` is a dummy value passed explicitly so no test
    depends on (or is broken by) an ambient `ANTHROPIC_API_KEY`.
    """
    captured: list[httpx.Request] = []

    def _recording_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx.MockTransport(_recording_handler)
        ),
    )
    return AnthropicLLM(client), client, captured


def _body_of(request: httpx.Request) -> dict[str, Any]:
    payload = json.loads(request.content)
    assert isinstance(payload, dict)
    return payload


class TestCompleteSuccess:
    async def test_valid_json_parses_into_typed_output_with_telemetry(self) -> None:
        body = _wire_message(
            text=json.dumps({"name": "Ada", "age": 36}),
            input_tokens=42,
            output_tokens=8,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=5,
        )
        llm, client, _captured = _make_llm(_handler_for(body, request_id="req_1"))

        completion = await llm.complete(
            MESSAGES, output_type=_Person, model="claude-sonnet-4-5", max_tokens=100
        )

        assert completion.output == _Person(name="Ada", age=36)
        assert completion.model == "claude-sonnet-4-5"
        assert completion.stop_reason == "end_turn"
        assert completion.input_tokens == 42
        assert completion.output_tokens == 8
        assert completion.cache_creation_input_tokens == 3
        assert completion.cache_read_input_tokens == 5
        assert completion.request_id == "req_1"
        assert client.max_retries == 0

    async def test_request_carries_json_schema_output_config(self) -> None:
        body = _wire_message(text=json.dumps({"name": "Ada", "age": 36}))
        llm, _client, captured = _make_llm(_handler_for(body))

        await llm.complete(MESSAGES, output_type=_Person, model="claude-sonnet-4-5", max_tokens=100)

        [request] = captured
        payload = _body_of(request)
        output_format = payload["output_config"]["format"]
        assert output_format["type"] == "json_schema"
        schema = output_format["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["properties"]) == {"name", "age"}


class TestRefusal:
    async def test_raises_llm_refused_with_telemetry(self) -> None:
        body = _wire_message(
            text=None,
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": "cyber"},
            input_tokens=12,
            output_tokens=0,
        )
        llm, _client, captured = _make_llm(_handler_for(body))

        with pytest.raises(LLMRefused) as exc_info:
            await llm.complete(
                MESSAGES, output_type=_Person, model="claude-sonnet-4-5", max_tokens=100
            )

        # On this SDK (anthropic 0.78) `stop_details` is unmodeled, so the
        # wire object survives as a plain dict; `_refusal_category` reads
        # both dict and attribute shapes.
        assert exc_info.value.category == "cyber"
        assert exc_info.value.telemetry.stop_reason == "refusal"
        assert exc_info.value.telemetry.input_tokens == 12
        assert exc_info.value.telemetry.output_tokens == 0

        # No parsing attempted: had `complete` reached
        # `_Person.model_validate_json(...)`, the empty classified text
        # would fail as invalid JSON and raise `LLMSchemaMismatch`, not
        # `LLMRefused`. Getting `LLMRefused` back is itself proof that
        # classification short-circuited before any parsing occurred.
        assert len(captured) == 1


class TestTruncation:
    async def test_raises_llm_truncated_carrying_partial_text(self) -> None:
        partial = '{"name": "Ada", "ag'
        body = _wire_message(text=partial, stop_reason="max_tokens", output_tokens=50)
        llm, _client, _captured = _make_llm(_handler_for(body))

        with pytest.raises(LLMTruncated) as exc_info:
            await llm.complete(
                MESSAGES, output_type=_Person, model="claude-sonnet-4-5", max_tokens=50
            )

        assert exc_info.value.partial_text == partial
        assert exc_info.value.max_tokens == 50
        assert exc_info.value.telemetry.stop_reason == "max_tokens"
        assert exc_info.value.telemetry.output_tokens == 50


class TestCompleteSchemaMismatchOnProse:
    async def test_prose_on_complete_raises_llm_schema_mismatch(self) -> None:
        prose = "Sure, here's a summary of the plan..."
        body = _wire_message(text=prose, stop_reason="end_turn")
        llm, _client, _captured = _make_llm(_handler_for(body))

        with pytest.raises(LLMSchemaMismatch) as exc_info:
            await llm.complete(
                MESSAGES, output_type=_Person, model="claude-sonnet-4-5", max_tokens=100
            )

        assert exc_info.value.raw_text == prose
        assert exc_info.value.error  # a validation-error message was captured
        assert exc_info.value.telemetry.stop_reason == "end_turn"


class TestCompleteSchema:
    async def test_success_returns_dict_completion(self) -> None:
        body = _wire_message(text=json.dumps({"foo": "bar"}))
        llm, _client, captured = _make_llm(_handler_for(body))

        completion = await llm.complete_schema(
            MESSAGES,
            output_schema={"type": "object", "properties": {"foo": {"type": "string"}}},
            model="claude-sonnet-4-5",
            max_tokens=100,
        )

        assert completion.output == {"foo": "bar"}

        [request] = captured
        payload = _body_of(request)
        assert payload["output_config"]["format"] == {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {"foo": {"type": "string"}}},
        }

    async def test_invalid_json_raises_mismatch(self) -> None:
        body = _wire_message(text="not json {")
        llm, _client, _captured = _make_llm(_handler_for(body))

        with pytest.raises(LLMSchemaMismatch) as exc_info:
            await llm.complete_schema(
                MESSAGES,
                output_schema={"type": "object"},
                model="claude-sonnet-4-5",
                max_tokens=100,
            )

        assert exc_info.value.raw_text == "not json {"

    async def test_non_object_json_raises_mismatch(self) -> None:
        body = _wire_message(text=json.dumps([1, 2, 3]))
        llm, _client, _captured = _make_llm(_handler_for(body))

        with pytest.raises(LLMSchemaMismatch) as exc_info:
            await llm.complete_schema(
                MESSAGES,
                output_schema={"type": "array"},
                model="claude-sonnet-4-5",
                max_tokens=100,
            )

        assert "list" in exc_info.value.error


class TestCompleteText:
    async def test_success_returns_raw_text(self) -> None:
        body = _wire_message(text="hello there")
        llm, _client, _captured = _make_llm(_handler_for(body))

        completion = await llm.complete_text(MESSAGES, model="claude-sonnet-4-5", max_tokens=100)

        assert completion.output == "hello there"
        assert completion.stop_reason == "end_turn"

    async def test_request_carries_no_output_config(self) -> None:
        body = _wire_message(text="hello there")
        llm, _client, captured = _make_llm(_handler_for(body))

        await llm.complete_text(MESSAGES, model="claude-sonnet-4-5", max_tokens=100)

        [request] = captured
        payload = _body_of(request)
        assert "output_config" not in payload


class TestCacheOptIn:
    # claude-sonnet-4-5's minimum cacheable prefix is 1024 estimated tokens
    # (`estimate_tokens` is chars // 4), so >4096 chars clears it.
    LONG_SYSTEM = "x" * 5000
    SHORT_SYSTEM = "short system prompt"

    async def test_cache_spec_places_breakpoint_on_last_system_block(self) -> None:
        body = _wire_message(text="ok")
        llm, _client, captured = _make_llm(_handler_for(body))

        await llm.complete_text(
            MESSAGES,
            model="claude-sonnet-4-5",
            max_tokens=100,
            system=self.LONG_SYSTEM,
            cache=CacheSpec(ttl="1h"),
        )

        [request] = captured
        payload = _body_of(request)
        system_blocks = payload["system"]
        assert isinstance(system_blocks, list)
        assert system_blocks[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    async def test_sub_minimum_system_gets_no_cache_control_anywhere(self) -> None:
        body = _wire_message(text="ok")
        llm, _client, captured = _make_llm(_handler_for(body))

        await llm.complete_text(
            MESSAGES,
            model="claude-sonnet-4-5",
            max_tokens=100,
            system=self.SHORT_SYSTEM,
            cache=CacheSpec(ttl="1h"),
        )

        [request] = captured
        payload = _body_of(request)
        assert "system" in payload  # the short prompt is still sent...
        assert "cache_control" not in json.dumps(payload)  # ...just uncached.

    async def test_no_cache_spec_omits_cache_control_even_for_long_system(self) -> None:
        body = _wire_message(text="ok")
        llm, _client, captured = _make_llm(_handler_for(body))

        await llm.complete_text(
            MESSAGES, model="claude-sonnet-4-5", max_tokens=100, system=self.LONG_SYSTEM
        )

        [request] = captured
        payload = _body_of(request)
        assert "cache_control" not in json.dumps(payload)


class TestRepeatedPrefixTelemetry:
    async def test_cache_read_tokens_land_in_completion(self) -> None:
        body = _wire_message(text="hello", cache_read_input_tokens=777)
        llm, _client, _captured = _make_llm(_handler_for(body))

        completion = await llm.complete_text(MESSAGES, model="claude-sonnet-4-5", max_tokens=100)

        assert completion.cache_read_input_tokens == 777


class TestMakeClient:
    def test_default_has_max_retries_zero(self) -> None:
        client = make_client(api_key="dummy-test-key")

        assert client.max_retries == 0

    def test_api_key_passed_through_when_given(self) -> None:
        client = make_client(api_key="dummy-test-key")

        assert client.api_key == "dummy-test-key"


class TestMaxTokensRequired:
    @pytest.mark.parametrize("method_name", ["complete", "complete_schema", "complete_text"])
    def test_no_default_value_for_max_tokens(self, method_name: str) -> None:
        method = getattr(AnthropicLLM, method_name)
        signature = inspect.signature(method)

        assert signature.parameters["max_tokens"].default is inspect.Parameter.empty
