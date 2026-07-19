"""Tests for the shared test-support module ``sax_platform.testing``.

Covers the two recording fakes (``FakeLLM``, ``FakeMistralOcr``), the
``RecordedCall`` record, and the ``temporal_env`` session fixture plus its
``env`` alias (the app-conftest re-export idiom).
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel

from sax_platform.llm import Completion, LLMRefused, Telemetry
from sax_platform.ocr import (
    BatchPollResult,
    BatchPollStatus,
    BatchResultEntry,
    ExtractedImage,
)
from sax_platform.testing import FakeLLM, FakeMistralOcr, RecordedCall, temporal_env

# Exercise the exact app-conftest re-export idiom documented in
# ``sax_platform.testing``: aliasing the session fixture under the name ``env``.
# The smoke test below requests ``env`` and gets the environment, proving the
# idiom resolves in pytest.
env = temporal_env

_USER_MSG = [{"role": "user", "content": "hello"}]


class _Out(BaseModel):
    value: str = "x"


def _telemetry(stop_reason: str = "refusal") -> Telemetry:
    return Telemetry(
        model="claude-x",
        stop_reason=stop_reason,
        input_tokens=1,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        request_id=None,
    )


# ---------------------------------------------------------------------------
# temporal_env fixture + env alias
# ---------------------------------------------------------------------------


class TestTemporalEnv:
    def test_env_alias_is_temporal_env(self) -> None:
        """The exported ``env`` is the very same fixture object as
        ``temporal_env`` — the re-export is an alias, not a copy."""
        assert env is temporal_env

    @pytest.mark.asyncio(loop_scope="session")
    async def test_env_fixture_provides_workflow_environment(self, env) -> None:
        """Smoke test: requesting the aliased session fixture starts a
        time-skipping environment and injects it. Using the ``env`` name proves
        the ``env = temporal_env`` conftest re-export idiom functions in pytest.
        """
        assert env.client is not None


# ---------------------------------------------------------------------------
# RecordedCall
# ---------------------------------------------------------------------------


class TestRecordedCall:
    def test_unpacks_as_tuple_and_by_attribute(self) -> None:
        rc = RecordedCall("complete", (1, 2), {"k": "v"})
        method, args, kwargs = rc
        assert method == "complete"
        assert args == (1, 2)
        assert kwargs == {"k": "v"}
        # Attribute access also works (it's a NamedTuple).
        assert rc.method == "complete"
        assert rc.kwargs["k"] == "v"


# ---------------------------------------------------------------------------
# FakeLLM
# ---------------------------------------------------------------------------


class TestFakeLLM:
    async def test_complete_returns_completion_with_default_telemetry(self) -> None:
        llm = FakeLLM(output=_Out(value="hi"))
        result = await llm.complete(_USER_MSG, output_type=_Out, model="m", max_tokens=10)
        assert isinstance(result, Completion)
        assert result.output == _Out(value="hi")
        assert result.model == "test-model"
        assert result.stop_reason == "end_turn"
        assert result.input_tokens == 100
        assert result.output_tokens == 200
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0
        assert result.request_id is None

    async def test_complete_records_the_call(self) -> None:
        llm = FakeLLM(output=_Out())
        await llm.complete(
            _USER_MSG, output_type=_Out, model="claude-x", max_tokens=512, system="sys"
        )
        call = llm.calls[-1]
        assert call.method == "complete"
        assert call.args == ([{"role": "user", "content": "hello"}],)
        assert call.kwargs["output_type"] is _Out
        assert call.kwargs["model"] == "claude-x"
        assert call.kwargs["max_tokens"] == 512
        assert call.kwargs["system"] == "sys"
        assert call.kwargs["cache"] is None
        assert call.kwargs["thinking"] is None

    async def test_custom_telemetry_knobs_propagate(self) -> None:
        llm = FakeLLM(
            output=_Out(),
            model="claude-opus",
            stop_reason="tool_use",
            input_tokens=11,
            output_tokens=22,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=4,
            request_id="req-9",
        )
        result = await llm.complete(_USER_MSG, output_type=_Out, model="m", max_tokens=1)
        assert result.model == "claude-opus"
        assert result.stop_reason == "tool_use"
        assert result.input_tokens == 11
        assert result.output_tokens == 22
        assert result.cache_creation_input_tokens == 3
        assert result.cache_read_input_tokens == 4
        assert result.request_id == "req-9"

    async def test_error_knob_makes_every_method_raise(self) -> None:
        error = LLMRefused(category="policy", telemetry=_telemetry())
        llm = FakeLLM(error=error)
        with pytest.raises(LLMRefused) as excinfo:
            await llm.complete(_USER_MSG, output_type=_Out, model="m", max_tokens=1)
        assert excinfo.value is error
        # The call is still recorded even though it raised.
        assert llm.calls[-1].method == "complete"

    async def test_complete_schema_returns_dict_output(self) -> None:
        llm = FakeLLM(output={"k": "v"})
        result = await llm.complete_schema(
            _USER_MSG, output_schema={"type": "object"}, model="m", max_tokens=5
        )
        assert result.output == {"k": "v"}
        call = llm.calls[-1]
        assert call.method == "complete_schema"
        assert call.kwargs["output_schema"] == {"type": "object"}

    async def test_complete_text_returns_str_output(self) -> None:
        llm = FakeLLM(output="hello there")
        result = await llm.complete_text(_USER_MSG, model="m", max_tokens=5)
        assert result.output == "hello there"
        assert llm.calls[-1].method == "complete_text"

    async def test_output_sequencing_consumes_in_order(self) -> None:
        llm = FakeLLM(outputs=[_Out(value="a"), _Out(value="b")])
        first = await llm.complete(_USER_MSG, output_type=_Out, model="m", max_tokens=1)
        second = await llm.complete(_USER_MSG, output_type=_Out, model="m", max_tokens=1)
        assert first.output == _Out(value="a")
        assert second.output == _Out(value="b")

    async def test_output_sequencing_exhausted_raises(self) -> None:
        llm = FakeLLM(outputs=[_Out(value="only")])
        await llm.complete(_USER_MSG, output_type=_Out, model="m", max_tokens=1)
        with pytest.raises(RuntimeError, match="exhausted"):
            await llm.complete(_USER_MSG, output_type=_Out, model="m", max_tokens=1)

    def test_output_and_outputs_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            FakeLLM(output=_Out(), outputs=[_Out()])

    def test_exposes_the_anthropic_llm_async_method_surface(self) -> None:
        """FakeLLM is a duck-typed drop-in for AnthropicLLM: all three public
        methods exist and are coroutine functions."""
        llm = FakeLLM()
        for name in ("complete", "complete_schema", "complete_text"):
            assert inspect.iscoroutinefunction(getattr(llm, name))


# ---------------------------------------------------------------------------
# FakeMistralOcr
# ---------------------------------------------------------------------------


class TestFakeMistralOcr:
    async def test_submit_then_poll_round_trip(self) -> None:
        poll = BatchPollResult(
            status=BatchPollStatus.ENDED,
            entries=[BatchResultEntry(custom_id="c1", succeeded=True, raw_response_json="{}")],
        )
        ocr = FakeMistralOcr(submit_batch_id="batch-7", poll_result=poll)

        batch_id = await ocr.submit_batch([{"custom_id": "c1"}], "mistral-ocr-latest")
        assert batch_id == "batch-7"

        result = await ocr.poll_batch(batch_id)
        assert result is poll
        assert result.status is BatchPollStatus.ENDED
        assert result.entries[0].custom_id == "c1"

        assert [c.method for c in ocr.calls] == ["submit_batch", "poll_batch"]

    async def test_submit_batch_records_args_and_endpoint(self) -> None:
        ocr = FakeMistralOcr()
        await ocr.submit_batch([{"a": 1}], "m", endpoint="/v1/custom")
        call = ocr.calls[-1]
        assert call.method == "submit_batch"
        assert call.args == ([{"a": 1}], "m")
        assert call.kwargs == {"endpoint": "/v1/custom"}

    async def test_submit_batch_default_endpoint_is_ocr(self) -> None:
        ocr = FakeMistralOcr()
        await ocr.submit_batch([{"a": 1}], "m")
        assert ocr.calls[-1].kwargs["endpoint"] == "/v1/ocr"

    async def test_process_returns_canned_body_and_images(self) -> None:
        body = {"pages": [{"index": 0}]}
        images = [ExtractedImage(original_image_id="i0", page_index=0, image_base64="AAA")]
        ocr = FakeMistralOcr(process_result=(body, images))

        got_body, got_images = await ocr.process(
            document={"type": "document_url", "document_url": "https://x"}, model="m"
        )
        assert got_body == body
        assert got_images == images
        call = ocr.calls[-1]
        assert call.method == "process"
        assert call.kwargs["document"] == {"type": "document_url", "document_url": "https://x"}
        assert call.kwargs["model"] == "m"
        assert call.kwargs["include_image_base64"] is True

    async def test_poll_batch_records_batch_id(self) -> None:
        ocr = FakeMistralOcr()
        await ocr.poll_batch("batch-xyz")
        assert ocr.calls[-1] == RecordedCall("poll_batch", ("batch-xyz",), {})

    def test_parse_batch_result_falls_back_to_process_result(self) -> None:
        body = {"pages": []}
        images: list[ExtractedImage] = []
        ocr = FakeMistralOcr(process_result=(body, images))
        assert ocr.parse_batch_result('{"any": "json"}') == (body, images)
        assert ocr.calls[-1] == RecordedCall("parse_batch_result", ('{"any": "json"}',), {})

    def test_parse_batch_result_uses_explicit_parse_result(self) -> None:
        parse_body = {"parsed": True}
        parse_images = [ExtractedImage(original_image_id="p", page_index=1, image_base64="B")]
        ocr = FakeMistralOcr(
            process_result=({"unused": True}, []),
            parse_result=(parse_body, parse_images),
        )
        assert ocr.parse_batch_result("{}") == (parse_body, parse_images)

    async def test_defaults_are_usable_with_no_arguments(self) -> None:
        ocr = FakeMistralOcr()
        poll = await ocr.poll_batch("b")
        assert poll.status is BatchPollStatus.ENDED
        assert poll.entries == []
        assert await ocr.submit_batch([], "m") == "batch-fake"
        assert await ocr.process(document={}, model="m") == ({}, [])
        assert ocr.parse_batch_result("{}") == ({}, [])

    def test_exposes_the_mistral_ocr_method_surface(self) -> None:
        """FakeMistralOcr is a duck-typed drop-in for MistralOcr: three async
        methods plus the sync ``parse_batch_result``."""
        ocr = FakeMistralOcr()
        for name in ("process", "submit_batch", "poll_batch"):
            assert inspect.iscoroutinefunction(getattr(ocr, name))
        assert not inspect.iscoroutinefunction(ocr.parse_batch_result)
