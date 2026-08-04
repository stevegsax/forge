"""Tests for the shared test-support module ``sax_platform.testing``.

Covers the two recording fakes (``FakeLLM``, ``FakeMistralOcr``), the
``RecordedCall`` record, the ``temporal_env`` session fixture plus its
``env`` alias (the app-conftest re-export idiom), and the Postgres
test-database helpers — including the property that matters most about them:
an unreachable database is a named error, never a skip.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel
from sqlalchemy.engine import make_url

from sax_platform.llm import Completion, LLMRefused, Telemetry
from sax_platform.ocr import (
    BatchPollStatus,
    BatchResultEntry,
    ExtractedImage,
)
from sax_platform.testing import (
    FORGE_TEST_DB_URL_ENV,
    FORGE_TRUST_TEST_DB_URL,
    PBOOK_TEST_DB_URL_ENV,
    PBOOK_TRUST_TEST_DB_URL,
    FakeLLM,
    FakeMistralOcr,
    RecordedCall,
    UnreachableTestDatabaseError,
    require_reachable_test_database,
    reset_public_schema,
    resolve_test_database_url,
    temporal_env,
    unreachable_test_database_message,
)

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
    async def test_submit_status_then_fetch_round_trip(self) -> None:
        entry = BatchResultEntry(custom_id="c1", succeeded=True, raw_response_json="{}")
        ocr = FakeMistralOcr(
            submit_batch_id="batch-7", status=BatchPollStatus.ENDED, entries=[entry]
        )

        batch_id = await ocr.submit_batch([{"custom_id": "c1"}], "mistral-ocr-latest")
        assert batch_id == "batch-7"

        status = await ocr.get_batch_status(batch_id)
        assert status is BatchPollStatus.ENDED

        entries = await ocr.fetch_batch_results(batch_id)
        assert entries == [entry]
        assert entries[0].custom_id == "c1"

        assert [c.method for c in ocr.calls] == [
            "submit_batch",
            "get_batch_status",
            "fetch_batch_results",
        ]

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

    async def test_get_batch_status_records_batch_id(self) -> None:
        ocr = FakeMistralOcr(status=BatchPollStatus.IN_PROGRESS)
        status = await ocr.get_batch_status("batch-xyz")
        assert status is BatchPollStatus.IN_PROGRESS
        assert ocr.calls[-1] == RecordedCall("get_batch_status", ("batch-xyz",), {})

    async def test_fetch_batch_results_records_batch_id(self) -> None:
        entry = BatchResultEntry(custom_id="c1", succeeded=True, raw_response_json="{}")
        ocr = FakeMistralOcr(entries=[entry])
        entries = await ocr.fetch_batch_results("batch-xyz")
        assert entries == [entry]
        assert ocr.calls[-1] == RecordedCall("fetch_batch_results", ("batch-xyz",), {})

    async def test_get_batch_status_never_calls_fetch(self) -> None:
        """The status path is independent of the download path: polling status
        records only ``get_batch_status`` and never ``fetch_batch_results`` —
        the fake's proof that a status check performs no download."""
        ocr = FakeMistralOcr()
        await ocr.get_batch_status("b")
        assert [c.method for c in ocr.calls] == ["get_batch_status"]

    async def test_list_batch_statuses_returns_canned_map_and_records_cutoff(self) -> None:
        cutoff = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)
        canned = {"job-1": BatchPollStatus.ENDED, "job-2": BatchPollStatus.IN_PROGRESS}
        ocr = FakeMistralOcr(list_statuses=canned)

        result = await ocr.list_batch_statuses(created_after=cutoff)

        assert result == canned
        assert ocr.calls[-1] == RecordedCall("list_batch_statuses", (), {"created_after": cutoff})

    async def test_list_batch_statuses_defaults_to_empty_map(self) -> None:
        ocr = FakeMistralOcr()
        assert await ocr.list_batch_statuses(created_after=datetime(2026, 1, 1, tzinfo=UTC)) == {}

    async def test_list_endpoint_uninvoked_is_provable_from_calls(self) -> None:
        """A caller can prove the list endpoint was never hit by inspecting
        ``calls`` — the sanctioned way to assert a poll path took no sweep."""
        ocr = FakeMistralOcr()
        await ocr.get_batch_status("b")
        assert "list_batch_statuses" not in [c.method for c in ocr.calls]

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
        assert await ocr.get_batch_status("b") is BatchPollStatus.ENDED
        assert await ocr.fetch_batch_results("b") == []
        assert await ocr.list_batch_statuses(created_after=datetime(2026, 1, 1, tzinfo=UTC)) == {}
        assert await ocr.submit_batch([], "m") == "batch-fake"
        assert await ocr.process(document={}, model="m") == ({}, [])
        assert ocr.parse_batch_result("{}") == ({}, [])

    def test_exposes_the_mistral_ocr_method_surface(self) -> None:
        """FakeMistralOcr is a duck-typed drop-in for MistralOcr: five async
        methods plus the sync ``parse_batch_result``."""
        ocr = FakeMistralOcr()
        for name in (
            "process",
            "submit_batch",
            "get_batch_status",
            "list_batch_statuses",
            "fetch_batch_results",
        ):
            assert inspect.iscoroutinefunction(getattr(ocr, name))
        assert not inspect.iscoroutinefunction(ocr.parse_batch_result)


class TestResolveTestDatabaseUrl:
    """The override wins; empty is unset; otherwise the trust default."""

    def test_returns_the_override_when_set(self) -> None:
        url = resolve_test_database_url(
            {"FORGE_TEST_DATABASE_URL": "postgresql+psycopg2://ci@localhost/x"},
            env_var=FORGE_TEST_DB_URL_ENV,
            default=FORGE_TRUST_TEST_DB_URL,
        )
        assert url == "postgresql+psycopg2://ci@localhost/x"

    def test_falls_back_to_the_trust_url_when_unset(self) -> None:
        url = resolve_test_database_url(
            {},
            env_var=FORGE_TEST_DB_URL_ENV,
            default=FORGE_TRUST_TEST_DB_URL,
        )
        assert url == FORGE_TRUST_TEST_DB_URL

    def test_treats_an_empty_override_as_unset(self) -> None:
        url = resolve_test_database_url(
            {PBOOK_TEST_DB_URL_ENV: ""},
            env_var=PBOOK_TEST_DB_URL_ENV,
            default=PBOOK_TRUST_TEST_DB_URL,
        )
        assert url == PBOOK_TRUST_TEST_DB_URL


class TestTrustUrls:
    def test_carry_no_credential(self) -> None:
        """The whole point of the trust path: no password exists to leak."""
        for url in (FORGE_TRUST_TEST_DB_URL, PBOOK_TRUST_TEST_DB_URL):
            assert make_url(url).password is None

    def test_name_the_dev_stack_and_their_own_database(self) -> None:
        """pg_hba matches one (user, database) pair per row — so they must agree."""
        for url in (FORGE_TRUST_TEST_DB_URL, PBOOK_TRUST_TEST_DB_URL):
            parsed = make_url(url)
            assert (parsed.host, parsed.port) == ("127.0.0.1", 5432)
            assert parsed.username == parsed.database
            assert parsed.database is not None
            assert parsed.database.endswith("_test")


class TestUnreachableTestDatabaseMessage:
    def test_names_the_mechanism_and_both_fixes(self) -> None:
        message = unreachable_test_database_message(
            FORGE_TRUST_TEST_DB_URL,
            env_var=FORGE_TEST_DB_URL_ENV,
            cause=OSError("connection refused"),
        )
        assert "unreachable" in message.lower()
        assert "sax-datastores" in message
        assert "make dev-up" in message
        assert FORGE_TEST_DB_URL_ENV in message
        assert "connection refused" in message

    def test_hides_a_password(self) -> None:
        """CI's override carries one; an error text must not print it."""
        message = unreachable_test_database_message(
            "postgresql+psycopg2://postgres:hunter2@localhost:5432/forge_test",
            env_var=FORGE_TEST_DB_URL_ENV,
            cause="boom",
        )
        assert "hunter2" not in message
        assert "***" in message


class TestUnreachableIsAnErrorNotASkip:
    """A dead port must raise by name — the silent-skip trap this replaces."""

    DEAD_URL = "postgresql+psycopg2://forge_test@127.0.0.1:1/forge_test"

    def test_require_reachable_raises(self) -> None:
        with pytest.raises(UnreachableTestDatabaseError, match="Test database unreachable"):
            require_reachable_test_database(self.DEAD_URL, env_var=FORGE_TEST_DB_URL_ENV)

    def test_reset_public_schema_raises(self) -> None:
        with pytest.raises(UnreachableTestDatabaseError, match=FORGE_TEST_DB_URL_ENV):
            reset_public_schema(self.DEAD_URL, env_var=FORGE_TEST_DB_URL_ENV)

    def test_the_error_is_not_a_skip_exception(self) -> None:
        # pytest.skip raises Skipped (a BaseException); this must be a plain
        # RuntimeError so a suite fails rather than reports green.
        assert issubclass(UnreachableTestDatabaseError, RuntimeError)
