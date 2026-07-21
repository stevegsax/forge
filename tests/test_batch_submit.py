"""Tests for forge.activities.batch_submit — batch submit activity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sax_platform.llm.batch import BatchHandle

from forge.activities.batch_submit import execute_batch_submit
from forge.models import AssembledContext, BatchSubmitInput, ThinkingPolicy


def _make_input(
    *,
    model_name: str = "",
    output_type_name: str = "LLMResponse",
    thinking: ThinkingPolicy | None = None,
    max_tokens: int = 4096,
    request_id: str = "req-default",
) -> BatchSubmitInput:
    """Build a BatchSubmitInput with sensible defaults."""
    context = AssembledContext(
        task_id="test-task",
        system_prompt="You are a helpful assistant.",
        user_prompt="Do something.",
        model_name=model_name,
    )
    return BatchSubmitInput(
        context=context,
        output_type_name=output_type_name,
        workflow_id="wf-test-123",
        thinking=thinking or ThinkingPolicy(),
        max_tokens=max_tokens,
        request_id=request_id,
    )


def _submit_mock(batch_id: str = "msgbatch_test123") -> AsyncMock:
    """AsyncMock standing in for the platform ``submit_batch`` helper."""
    return AsyncMock(return_value=BatchHandle(batch_id=batch_id, processing_status="in_progress"))


def _submitted_params(submit: AsyncMock) -> dict:
    """Return the ``params`` of the single request handed to ``submit_batch``."""
    _client, requests = submit.await_args.args
    assert len(requests) == 1
    return requests[0]["params"]


# ---------------------------------------------------------------------------
# execute_batch_submit
# ---------------------------------------------------------------------------


class TestExecuteBatchSubmit:
    @pytest.mark.asyncio
    async def test_returns_batch_submit_result(self) -> None:
        client = MagicMock()
        submit = _submit_mock()
        with patch("sax_platform.llm.batch.submit_batch", submit):
            result = await execute_batch_submit(_make_input(), client)

        assert result.batch_id == "msgbatch_test123"
        assert result.request_id == "req-default"  # the workflow-minted id, echoed
        called_client, requests = submit.await_args.args
        assert called_client is client
        assert requests[0]["custom_id"] == result.request_id

    @pytest.mark.asyncio
    async def test_resolves_output_type_into_structured_format(self) -> None:
        client = MagicMock()
        submit = _submit_mock()
        with patch("sax_platform.llm.batch.submit_batch", submit):
            await execute_batch_submit(_make_input(output_type_name="LLMResponse"), client)

        params = _submitted_params(submit)
        assert params["output_config"]["format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_text_mode_omits_structured_format(self) -> None:
        client = MagicMock()
        submit = _submit_mock()
        with patch("sax_platform.llm.batch.submit_batch", submit):
            # Empty output_type_name -> text lane, no structured-output format.
            await execute_batch_submit(_make_input(output_type_name=""), client)

        params = _submitted_params(submit)
        assert "format" not in params.get("output_config", {})

    @pytest.mark.asyncio
    async def test_passes_thinking_through_for_non_haiku(self) -> None:
        client = MagicMock()
        submit = _submit_mock()
        input_data = _make_input(
            model_name="anthropic:claude-sonnet-5",
            thinking=ThinkingPolicy(enabled=True, effort="high"),
        )
        with patch("sax_platform.llm.batch.submit_batch", submit):
            await execute_batch_submit(input_data, client)

        params = _submitted_params(submit)
        assert params["model"] == "claude-sonnet-5"
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"]["effort"] == "high"

    @pytest.mark.asyncio
    async def test_haiku_guard_drops_thinking(self) -> None:
        # Old sax_llm silently dropped thinking for haiku; the platform builder
        # would otherwise emit a shape haiku 400s on, so batch_submit passes None.
        client = MagicMock()
        submit = _submit_mock()
        input_data = _make_input(
            model_name="anthropic:claude-haiku-4-5",
            thinking=ThinkingPolicy(enabled=True, effort="high"),
        )
        with patch("sax_platform.llm.batch.submit_batch", submit):
            await execute_batch_submit(input_data, client)

        params = _submitted_params(submit)
        assert "thinking" not in params
        assert "effort" not in params.get("output_config", {})

    @pytest.mark.asyncio
    async def test_caller_request_id_used_verbatim(self) -> None:
        # D88: the workflow mints the custom_id (workflow.uuid4()) and threads it in.
        # A non-empty request_id is used verbatim as the provider custom_id.
        client = MagicMock()
        submit = _submit_mock()
        with patch("sax_platform.llm.batch.submit_batch", submit):
            result = await execute_batch_submit(_make_input(request_id="wf-minted-id"), client)

        assert result.request_id == "wf-minted-id"
        _client, requests = submit.await_args.args
        assert requests[0]["custom_id"] == "wf-minted-id"

    @pytest.mark.asyncio
    async def test_duplicate_submit_retry_reuses_custom_id(self) -> None:
        # AC (duplicate-submit-retry): two submits with the same workflow-minted
        # request_id produce the same provider custom_id — one paid batch identity,
        # so a retried submit cannot orphan a second batch.
        client = MagicMock()
        submit = _submit_mock()
        input_data = _make_input(request_id="stable-id")
        with patch("sax_platform.llm.batch.submit_batch", submit):
            first = await execute_batch_submit(input_data, client)
            second = await execute_batch_submit(input_data, client)

        assert first.request_id == second.request_id == "stable-id"

    @pytest.mark.asyncio
    async def test_passes_max_tokens_and_prompts_through(self) -> None:
        client = MagicMock()
        submit = _submit_mock()
        with patch("sax_platform.llm.batch.submit_batch", submit):
            await execute_batch_submit(_make_input(max_tokens=8192), client)

        params = _submitted_params(submit)
        assert params["max_tokens"] == 8192
        assert params["messages"] == [{"role": "user", "content": "Do something."}]
        # system is normalized to a text block by the builder.
        assert params["system"][0]["text"] == "You are a helpful assistant."
