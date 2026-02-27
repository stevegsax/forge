"""Tests for MistralProvider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.llm_providers.mistral import MistralProvider
from forge.llm_providers.models import BatchPollStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(
    tool_input: dict,
    *,
    model: str = "mistral-large-latest",
    prompt_tokens: int = 80,
    completion_tokens: int = 150,
) -> MagicMock:
    """Build a mock Mistral ChatCompletionResponse."""
    func = MagicMock()
    func.arguments = json.dumps(tool_input)

    tool_call = MagicMock()
    tool_call.function = func

    message = MagicMock()
    message.tool_calls = [tool_call]

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    response.model_dump = MagicMock(return_value={"model": model, "choices": []})

    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildRequestParams:
    """Tests for MistralProvider.build_request_params."""

    def test_builds_mistral_format(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = provider.build_request_params(
            system_prompt="You are helpful.",
            user_prompt="Do something.",
            output_type=TestOutput,
            model="mistral-large-latest",
            max_tokens=1024,
        )

        assert params["model"] == "mistral-large-latest"
        assert params["max_tokens"] == 1024
        assert params["tool_choice"] == "any"

        # Messages: system + user
        assert len(params["messages"]) == 2
        assert params["messages"][0]["role"] == "system"
        assert params["messages"][1]["role"] == "user"

        # Tool definition in function format
        tool = params["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "test_output"

    def test_no_cache_control(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = provider.build_request_params(
            system_prompt="sys",
            user_prompt="user",
            output_type=TestOutput,
            model="mistral-large-latest",
            max_tokens=1024,
            cache_instructions=True,
            cache_tool_definitions=True,
        )

        # No cache_control on messages or tools
        for msg in params["messages"]:
            assert "cache_control" not in msg
        assert "cache_control" not in params["tools"][0]

    def test_thinking_budget_silently_ignored(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = provider.build_request_params(
            system_prompt="sys",
            user_prompt="user",
            output_type=TestOutput,
            model="mistral-large-latest",
            max_tokens=1024,
            thinking_budget_tokens=5000,
        )

        # No thinking param in output
        assert "thinking" not in params


class TestCall:
    """Tests for MistralProvider.call."""

    @pytest.mark.asyncio
    async def test_extracts_tool_input(self) -> None:
        tool_input = {"value": "hello"}
        mock_response = _make_mock_response(tool_input)

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call({"model": "mistral-large-latest"})

        assert result.tool_input == tool_input

    @pytest.mark.asyncio
    async def test_maps_usage_tokens(self) -> None:
        mock_response = _make_mock_response(
            {"value": "ok"}, prompt_tokens=120, completion_tokens=300
        )

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call({"model": "test"})

        assert result.input_tokens == 120
        assert result.output_tokens == 300

    @pytest.mark.asyncio
    async def test_cache_tokens_are_zero(self) -> None:
        mock_response = _make_mock_response({"value": "ok"})

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call({"model": "test"})

        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0

    @pytest.mark.asyncio
    async def test_model_name_from_response(self) -> None:
        mock_response = _make_mock_response({"value": "ok"}, model="mistral-large-2")

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call({"model": "mistral-large-latest"})

        assert result.model_name == "mistral-large-2"


class TestSupportsBatch:
    """Tests for batch support flag."""

    def test_supports_batch_is_true(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        assert provider.supports_batch is True


class TestBuildBatchRequest:
    """Tests for MistralProvider.build_batch_request."""

    def test_wraps_params_with_custom_id_and_body(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        result = provider.build_batch_request("req-456", {"model": "test"})

        assert result["custom_id"] == "req-456"
        assert result["body"]["model"] == "test"


def _make_batch_choice(arguments: str = "{}") -> dict:
    """Build a minimal successful Mistral batch response body with a tool call."""
    return {
        "choices": [
            {"message": {"tool_calls": [{"function": {"arguments": arguments}}]}}
        ],
    }


# ---------------------------------------------------------------------------
# submit_batch
# ---------------------------------------------------------------------------


class TestSubmitBatch:
    """Tests for MistralProvider.submit_batch."""

    @pytest.mark.asyncio
    async def test_returns_job_id(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.id = "batch-job-123"
        provider._client.batch.jobs.create_async = AsyncMock(return_value=mock_job)

        requests = [{"custom_id": "r1", "body": {"model": "mistral-large-latest"}}]
        result = await provider.submit_batch(requests, "mistral-large-latest")

        assert result == "batch-job-123"

    @pytest.mark.asyncio
    async def test_passes_correct_args(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.id = "batch-job-456"
        provider._client.batch.jobs.create_async = AsyncMock(return_value=mock_job)

        requests = [{"custom_id": "r1", "body": {}}]
        await provider.submit_batch(requests, "codestral-latest")

        provider._client.batch.jobs.create_async.assert_called_once_with(
            input_data=requests,
            model="codestral-latest",
            endpoint="/v1/chat/completions",
        )


# ---------------------------------------------------------------------------
# poll_batch
# ---------------------------------------------------------------------------


class TestPollBatch:
    """Tests for MistralProvider.poll_batch."""

    @pytest.mark.asyncio
    async def test_queued_returns_pending(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.status = "QUEUED"
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.PENDING
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_running_returns_in_progress(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.status = "RUNNING"
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_failed_returns_failed(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.status = "FAILED"
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.FAILED
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_timeout_exceeded_returns_expired(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.status = "TIMEOUT_EXCEEDED"
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_cancelled_returns_canceled(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.status = "CANCELLED"
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.CANCELED

    @pytest.mark.asyncio
    async def test_success_parses_jsonl_results(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.status = "SUCCESS"
        mock_job.output_file = "file-output-123"
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        body_1 = {
            **_make_batch_choice(),
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model": "mistral-large-latest",
        }
        body_2 = {
            **_make_batch_choice(),
            "usage": {"prompt_tokens": 15, "completion_tokens": 25},
            "model": "mistral-large-latest",
        }
        jsonl = "\n".join([
            json.dumps({"custom_id": "req-1", "response": {"body": body_1}}),
            json.dumps({"custom_id": "req-2", "response": {"body": body_2}}),
        ])
        mock_file = MagicMock()
        mock_file.read.return_value = jsonl.encode("utf-8")
        provider._client.files.download_async = AsyncMock(return_value=mock_file)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.ENDED
        assert len(result.entries) == 2
        assert result.entries[0].custom_id == "req-1"
        assert result.entries[0].succeeded is True
        assert result.entries[0].raw_response_json is not None
        assert result.entries[1].custom_id == "req-2"
        assert result.entries[1].succeeded is True

    @pytest.mark.asyncio
    async def test_success_with_errored_entry(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.status = "SUCCESS"
        mock_job.output_file = "file-output-456"
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        jsonl = json.dumps({
            "custom_id": "req-bad",
            "response": {
                "body": {
                    "error": {"type": "invalid_request", "message": "bad input"},
                }
            },
        })
        mock_file = MagicMock()
        mock_file.read.return_value = jsonl.encode("utf-8")
        provider._client.files.download_async = AsyncMock(return_value=mock_file)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.ENDED
        assert len(result.entries) == 1
        assert result.entries[0].custom_id == "req-bad"
        assert result.entries[0].succeeded is False
        assert result.entries[0].error is not None

    @pytest.mark.asyncio
    async def test_success_downloads_from_output_file(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.status = "SUCCESS"
        mock_job.output_file = "file-abc-789"
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        jsonl = json.dumps({
            "custom_id": "req-1",
            "response": {"body": {"choices": [{"message": {"tool_calls": []}}]}},
        })
        mock_file = MagicMock()
        mock_file.read.return_value = jsonl.encode("utf-8")
        provider._client.files.download_async = AsyncMock(return_value=mock_file)

        await provider.poll_batch("batch-1")

        provider._client.files.download_async.assert_called_once_with(
            file_id="file-abc-789"
        )

    @pytest.mark.asyncio
    async def test_success_with_string_output_file(self) -> None:
        """When download returns a string instead of a file-like object."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.status = "SUCCESS"
        mock_job.output_file = "file-str-1"
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        jsonl = json.dumps({
            "custom_id": "req-1",
            "response": {
                "body": {
                    "choices": [{"message": {"tool_calls": [{"function": {"arguments": "{}"}}]}}],
                    "model": "mistral-large-latest",
                }
            },
        })
        # Return a plain string (no .read method)
        provider._client.files.download_async = AsyncMock(return_value=jsonl)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.ENDED
        assert len(result.entries) == 1
        assert result.entries[0].succeeded is True

    @pytest.mark.asyncio
    async def test_skips_blank_lines_in_jsonl(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.status = "SUCCESS"
        mock_job.output_file = "file-blank"
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        entry = {
            "custom_id": "req-1",
            "response": {"body": _make_batch_choice()},
        }
        jsonl = "\n" + json.dumps(entry) + "\n\n"
        mock_file = MagicMock()
        mock_file.read.return_value = jsonl.encode("utf-8")
        provider._client.files.download_async = AsyncMock(return_value=mock_file)

        result = await provider.poll_batch("batch-1")

        assert len(result.entries) == 1


# ---------------------------------------------------------------------------
# parse_batch_result
# ---------------------------------------------------------------------------


class TestParseBatchResult:
    """Tests for MistralProvider.parse_batch_result."""

    def test_parses_successful_response(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        raw = json.dumps({
            "model": "mistral-large-latest",
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "llm_response",
                            "arguments": json.dumps({
                                "files": [],
                                "edits": [],
                                "explanation": "Done.",
                            }),
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
            },
        })

        result = provider.parse_batch_result(raw, "LLMResponse")

        assert result.tool_input["explanation"] == "Done."
        assert result.model_name == "mistral-large-latest"
        assert result.input_tokens == 100
        assert result.output_tokens == 200
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0
        assert result.raw_response_json == raw

    def test_parses_plan(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        plan_data = {
            "task_id": "t1",
            "steps": [{"step_id": "s1", "description": "Do it.", "target_files": ["a.py"]}],
            "explanation": "Single step.",
        }
        raw = json.dumps({
            "model": "mistral-large-latest",
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "plan",
                            "arguments": json.dumps(plan_data),
                        }
                    }]
                }
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 100},
        })

        result = provider.parse_batch_result(raw, "Plan")

        assert result.tool_input["task_id"] == "t1"
        assert len(result.tool_input["steps"]) == 1

    def test_raises_for_unknown_output_type(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        with pytest.raises(KeyError, match="Unknown output type"):
            provider.parse_batch_result("{}", "NonExistentType")

    def test_missing_tool_calls_returns_empty_dict(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        raw = json.dumps({
            "model": "mistral-large-latest",
            "choices": [{"message": {"content": "text only"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

        result = provider.parse_batch_result(raw, "LLMResponse")

        assert result.tool_input == {}

    def test_missing_usage_defaults_to_zero(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        raw = json.dumps({
            "model": "mistral-large-latest",
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {"arguments": "{}"}
                    }]
                }
            }],
        })

        result = provider.parse_batch_result(raw, "LLMResponse")

        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_dict_arguments_not_double_parsed(self) -> None:
        """If arguments is already a dict (not a string), it should pass through."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        raw = json.dumps({
            "model": "mistral-large-latest",
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {"arguments": {"key": "value"}}
                    }]
                }
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        })

        result = provider.parse_batch_result(raw, "LLMResponse")

        assert result.tool_input == {"key": "value"}
