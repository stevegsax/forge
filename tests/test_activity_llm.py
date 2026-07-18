"""Tests for forge.activities.llm — LLM call activity."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sax_platform.llm import LLMRefused, LLMTruncated, Telemetry

from forge.activities.llm import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, execute_llm_call
from forge.models import AssembledContext, FileOutput, LLMResponse
from tests.conftest import build_mock_llm


def _telemetry(stop_reason: str = "end_turn") -> Telemetry:
    return Telemetry(
        model="test-model",
        stop_reason=stop_reason,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        request_id=None,
    )


# ---------------------------------------------------------------------------
# execute_llm_call
# ---------------------------------------------------------------------------


class TestExecuteLlmCall:
    def _make_context(self) -> AssembledContext:
        return AssembledContext(
            task_id="llm-task",
            system_prompt="You are a code generator.",
            user_prompt="Generate the code.",
        )

    def _make_llm(
        self,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        *,
        stop_reason: str = "end_turn",
    ) -> MagicMock:
        return build_mock_llm(
            output=LLMResponse(
                files=[FileOutput(file_path="out.py", content="print('hi')")],
                explanation="Created output file.",
            ),
            model="claude-sonnet-4-5-20250929",
            input_tokens=100,
            output_tokens=200,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            stop_reason=stop_reason,
        )

    @pytest.mark.asyncio
    async def test_returns_llm_call_result(self) -> None:
        context = self._make_context()
        llm = self._make_llm()

        result = await execute_llm_call(context, llm)

        assert result.task_id == "llm-task"
        assert result.model_name == "claude-sonnet-4-5-20250929"
        assert len(result.response.files) == 1
        assert result.response.files[0].file_path == "out.py"

    @pytest.mark.asyncio
    async def test_extracts_usage(self) -> None:
        context = self._make_context()
        llm = self._make_llm()

        result = await execute_llm_call(context, llm)

        assert result.input_tokens == 100
        assert result.output_tokens == 200

    @pytest.mark.asyncio
    async def test_latency_is_positive(self) -> None:
        context = self._make_context()
        llm = self._make_llm()

        result = await execute_llm_call(context, llm)

        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_propagates_task_id(self) -> None:
        context = AssembledContext(
            task_id="custom-id",
            system_prompt="sys",
            user_prompt="usr",
        )
        llm = self._make_llm()

        result = await execute_llm_call(context, llm)

        assert result.task_id == "custom-id"

    @pytest.mark.asyncio
    async def test_calls_llm_complete_with_expected_kwargs(self) -> None:
        context = self._make_context()
        llm = self._make_llm()

        await execute_llm_call(context, llm)

        llm.complete.assert_awaited_once()
        call = llm.complete.await_args
        assert call.args[0] == [{"role": "user", "content": "Generate the code."}]
        assert call.kwargs["output_type"] is LLMResponse
        # No model_name on the context -> the GENERATION-tier default, provider stripped.
        assert call.kwargs["model"] == "claude-sonnet-5"
        assert call.kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
        assert call.kwargs["system"] == "You are a code generator."
        # llm.py attaches no thinking policy (matches the pre-migration behavior).
        assert call.kwargs.get("thinking") is None

    @pytest.mark.asyncio
    async def test_explanation_preserved(self) -> None:
        context = self._make_context()
        llm = self._make_llm()

        result = await execute_llm_call(context, llm)

        assert result.response.explanation == "Created output file."

    @pytest.mark.asyncio
    async def test_stop_reason_passthrough(self) -> None:
        context = self._make_context()
        llm = self._make_llm(stop_reason="end_turn")

        result = await execute_llm_call(context, llm)

        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_stop_reason_passthrough_non_default(self) -> None:
        context = self._make_context()
        llm = self._make_llm(stop_reason="stop_sequence")

        result = await execute_llm_call(context, llm)

        assert result.stop_reason == "stop_sequence"

    @pytest.mark.asyncio
    async def test_refusal_propagates(self) -> None:
        context = self._make_context()
        llm = build_mock_llm(
            error=LLMRefused(category=None, telemetry=_telemetry(stop_reason="refusal"))
        )

        with pytest.raises(LLMRefused):
            await execute_llm_call(context, llm)

    @pytest.mark.asyncio
    async def test_truncation_propagates(self) -> None:
        context = self._make_context()
        llm = build_mock_llm(
            error=LLMTruncated(
                partial_text="partial",
                max_tokens=DEFAULT_MAX_TOKENS,
                telemetry=_telemetry(stop_reason="max_tokens"),
            )
        )

        with pytest.raises(LLMTruncated):
            await execute_llm_call(context, llm)


# ---------------------------------------------------------------------------
# Phase 9: cache stats extraction
# ---------------------------------------------------------------------------


class TestCacheStatsExtraction:
    def _make_context(self) -> AssembledContext:
        return AssembledContext(
            task_id="cache-task",
            system_prompt="sys",
            user_prompt="usr",
        )

    @pytest.mark.asyncio
    async def test_extracts_cache_tokens(self) -> None:
        context = self._make_context()
        llm = build_mock_llm(
            output=LLMResponse(
                files=[FileOutput(file_path="out.py", content="pass")],
                explanation="Done.",
            ),
            cache_creation_input_tokens=500,
            cache_read_input_tokens=1000,
        )
        result = await execute_llm_call(context, llm)
        assert result.cache_creation_input_tokens == 500
        assert result.cache_read_input_tokens == 1000

    @pytest.mark.asyncio
    async def test_zero_cache_tokens(self) -> None:
        context = self._make_context()
        llm = build_mock_llm(
            output=LLMResponse(explanation="Done."),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        result = await execute_llm_call(context, llm)
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0


# ---------------------------------------------------------------------------
# Phase 11: model_name threading via call_llm activity
# ---------------------------------------------------------------------------


def _mock_tracer() -> MagicMock:
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span
    return mock_tracer


class TestCallLlmModelNameThreading:
    @pytest.mark.asyncio
    async def test_threads_model_name_to_client(self) -> None:
        from forge.activities.llm import call_llm

        llm = build_mock_llm(
            output=LLMResponse(explanation="done"),
            model="custom-model",
            input_tokens=10,
            output_tokens=20,
        )

        with (
            patch("forge.llm_client.get_llm", return_value=llm),
            patch("forge.tracing.get_tracer", return_value=_mock_tracer()),
        ):
            context = AssembledContext(
                task_id="t1",
                system_prompt="sys",
                user_prompt="usr",
                model_name="custom-model",
            )
            llm_result = await call_llm(context)

        assert llm_result.model_name == "custom-model"
        assert llm.complete.await_args.kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_uses_default_when_model_name_empty(self) -> None:
        from forge.activities.llm import call_llm

        llm = build_mock_llm(
            output=LLMResponse(explanation="done"),
            model="whatever-the-server-returns",
            input_tokens=10,
            output_tokens=20,
        )

        with (
            patch("forge.llm_client.get_llm", return_value=llm),
            patch("forge.tracing.get_tracer", return_value=_mock_tracer()),
        ):
            context = AssembledContext(
                task_id="t1",
                system_prompt="sys",
                user_prompt="usr",
            )
            await call_llm(context)

        _, default_model = DEFAULT_MODEL.split(":", 1)
        assert llm.complete.await_args.kwargs["model"] == default_model
