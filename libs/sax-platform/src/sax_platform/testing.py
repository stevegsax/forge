"""Shared test-support for the workspace: one Temporal fixture and two
recording fakes, imported EXPLICITLY by each app's ``conftest.py`` (D93).

This module is **test-only** even though it lives under ``src/``: it imports
``pytest_asyncio`` at module scope to define the ``temporal_env`` fixture, and
``pytest`` is a dev-group dependency, never a runtime one. **Production code
must never import ``sax_platform.testing``.** There is deliberately **no
pytest11 entry point and no plugin registration** — an app opts in by importing
the names it wants into its own ``conftest.py`` (D93).

What it replaces:

- ``temporal_env`` — the byte-identical session-scoped
  ``start_time_skipping(data_converter=pydantic_data_converter)`` fixture that
  forge and ocr each copied into their conftests. pbook's per-file copies
  omitted the pydantic converter (a test/prod mismatch — pbook's worker
  connects *with* it); unifying on this fixture fixes that.
- ``FakeLLM`` — a recording fake structurally compatible with
  ``sax_platform.llm.AnthropicLLM`` and with pbook's ``SupportsComplete``
  protocol, replacing forge's ``build_mock_llm`` (a ``MagicMock``) and pbook's
  ``_StubProvider``.
- ``FakeMistralOcr`` — an async recording fake mirroring
  ``sax_platform.ocr.MistralOcr``'s public surface, replacing bare
  ``MagicMock``/``AsyncMock`` OCR stubs.

The fakes are real classes: they record every call on ``self.calls`` so tests
assert against captured kwargs directly, with no ``MagicMock`` in sight.

Recommended app-conftest usage::

    # apps/<app>/tests/conftest.py
    from sax_platform.testing import FakeLLM, FakeMistralOcr, temporal_env

    # Re-export the session Temporal fixture under the name the suite uses:
    env = temporal_env

Assigning ``env = temporal_env`` registers the *same* fixture object under a
second name; pytest resolves it exactly like any imported fixture (verified by
this module's own test suite). Because pytest caches a session-scoped fixture
per fixture *name*, request only ONE of the two names within a session (use
``env`` OR ``temporal_env``, not both) — otherwise the time-skipping test
server starts once per requested name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, cast

import pytest_asyncio
from pydantic import BaseModel

from sax_platform.llm import Completion
from sax_platform.ocr import BatchPollStatus, BatchResultEntry, ExtractedImage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence
    from typing import Protocol

    from anthropic.types import MessageParam
    from mistralai.models import DocumentTypedDict
    from temporalio.testing import WorkflowEnvironment

    from sax_platform.llm import CacheSpec, ThinkingPolicy

__all__ = [
    "FakeLLM",
    "FakeMistralOcr",
    "RecordedCall",
    "env",
    "temporal_env",
]

# Mirrors sax_platform.ocr's module-private default; restated here so this
# module stays self-contained rather than importing a private name.
_OCR_ENDPOINT = "/v1/ocr"


class RecordedCall(NamedTuple):
    """One recorded call to a fake: ``(method, args, kwargs)``.

    ``args`` holds positional arguments (with any consumed iterable already
    materialized to a ``list``); ``kwargs`` holds keyword arguments. Being a
    tuple, it unpacks as ``method, args, kwargs`` while also allowing
    attribute access (``call.kwargs["max_tokens"]``).
    """

    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


# ---------------------------------------------------------------------------
# Temporal environment fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    """Session-scoped time-skipping Temporal environment, pydantic-aware.

    Byte-for-byte the shape forge and ocr each defined inline: a
    ``start_time_skipping`` environment whose data converter is
    ``pydantic_data_converter`` so workflow/activity payloads that are pydantic
    models round-trip. Session-scoped (with ``loop_scope="session"``) because
    starting the test-server binary is expensive; one per session is enough.
    """
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import WorkflowEnvironment

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as environment:
        yield environment


# Alias so an app conftest can ``from sax_platform.testing import env``. It is
# the SAME fixture object; see the module docstring on requesting one name per
# session.
env = temporal_env


# ---------------------------------------------------------------------------
# FakeLLM — recording structured-outputs client
# ---------------------------------------------------------------------------


class FakeLLM:
    """Recording fake for ``sax_platform.llm.AnthropicLLM``.

    Structurally matches the three public methods of ``AnthropicLLM``
    (``complete``, ``complete_schema``, ``complete_text``), so it drops in
    anywhere one is expected — including anywhere pbook's narrower
    ``SupportsComplete`` protocol is required.

    Construct it with the same knobs as forge's old ``build_mock_llm``: pass
    ``output`` (the parsed value each call should return) for the success path,
    or ``error`` to make every call raise it (e.g. an
    ``LLMRefused``/``LLMTruncated``/``LLMSchemaMismatch``). Telemetry fields
    (``model``, ``stop_reason``, token counts, ``request_id``) are knobs too and
    ride along on the returned ``Completion``.

    Per-call output sequencing: pass ``outputs`` (a sequence) instead of
    ``output`` and each call returns the next element in order; a call past the
    end raises ``RuntimeError``. ``output`` and ``outputs`` are mutually
    exclusive.

    Every call is appended to ``self.calls`` as a :class:`RecordedCall`, so a
    test can assert exactly what was forwarded (``self.calls[-1].kwargs[...]``)
    without a ``MagicMock``.
    """

    def __init__(
        self,
        output: Any = None,
        *,
        outputs: Sequence[Any] | None = None,
        error: Exception | None = None,
        model: str = "test-model",
        stop_reason: str = "end_turn",
        input_tokens: int = 100,
        output_tokens: int = 200,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        request_id: str | None = None,
    ) -> None:
        if output is not None and outputs is not None:
            msg = "pass either `output` or `outputs`, not both"
            raise ValueError(msg)
        self._output = output
        self._outputs_queue: list[Any] | None = list(outputs) if outputs is not None else None
        self._error = error
        self._model = model
        self._stop_reason = stop_reason
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cache_creation_input_tokens = cache_creation_input_tokens
        self._cache_read_input_tokens = cache_read_input_tokens
        self._request_id = request_id
        self.calls: list[RecordedCall] = []

    def _take_output(self) -> Any:
        """Return the next canned output, advancing the sequence if any."""
        if self._outputs_queue is None:
            return self._output
        if not self._outputs_queue:
            msg = "FakeLLM outputs exhausted: more calls than canned outputs"
            raise RuntimeError(msg)
        return self._outputs_queue.pop(0)

    def _next_completion(self) -> Completion[Any]:
        """Raise the canned error, or build a ``Completion`` from the knobs."""
        if self._error is not None:
            raise self._error
        return Completion(
            output=self._take_output(),
            model=self._model,
            stop_reason=self._stop_reason,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cache_creation_input_tokens=self._cache_creation_input_tokens,
            cache_read_input_tokens=self._cache_read_input_tokens,
            request_id=self._request_id,
        )

    async def complete[T: BaseModel](
        self,
        messages: Iterable[MessageParam],
        *,
        output_type: type[T],
        model: str,
        max_tokens: int,
        system: str | list[dict[str, Any]] | None = None,
        cache: CacheSpec | None = None,
        thinking: ThinkingPolicy | None = None,
    ) -> Completion[T]:
        """Record the call and return (or raise) the canned result."""
        self.calls.append(
            RecordedCall(
                "complete",
                (list(messages),),
                {
                    "output_type": output_type,
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "cache": cache,
                    "thinking": thinking,
                },
            )
        )
        return cast("Completion[T]", self._next_completion())

    async def complete_schema(
        self,
        messages: Iterable[MessageParam],
        *,
        output_schema: dict[str, Any],
        model: str,
        max_tokens: int,
        system: str | list[dict[str, Any]] | None = None,
        cache: CacheSpec | None = None,
        thinking: ThinkingPolicy | None = None,
    ) -> Completion[dict[str, Any]]:
        """Record the call and return (or raise) the canned result."""
        self.calls.append(
            RecordedCall(
                "complete_schema",
                (list(messages),),
                {
                    "output_schema": output_schema,
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "cache": cache,
                    "thinking": thinking,
                },
            )
        )
        return cast("Completion[dict[str, Any]]", self._next_completion())

    async def complete_text(
        self,
        messages: Iterable[MessageParam],
        *,
        model: str,
        max_tokens: int,
        system: str | list[dict[str, Any]] | None = None,
        cache: CacheSpec | None = None,
        thinking: ThinkingPolicy | None = None,
    ) -> Completion[str]:
        """Record the call and return (or raise) the canned result."""
        self.calls.append(
            RecordedCall(
                "complete_text",
                (list(messages),),
                {
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "cache": cache,
                    "thinking": thinking,
                },
            )
        )
        return cast("Completion[str]", self._next_completion())


# ---------------------------------------------------------------------------
# FakeMistralOcr — recording OCR client
# ---------------------------------------------------------------------------


class FakeMistralOcr:
    """Recording fake for ``sax_platform.ocr.MistralOcr``.

    Mirrors ``MistralOcr``'s public surface exactly — ``process``,
    ``submit_batch``, ``get_batch_status``, ``fetch_batch_results`` (all async)
    and ``parse_batch_result`` (sync) — so it drops in anywhere a ``MistralOcr``
    instance is expected. It holds no SDK client; instead it returns the canned
    values passed to the constructor and records every call on ``self.calls``.

    The batch-result surface is split the same way the real class is: ``status``
    is the canned ``BatchPollStatus`` returned by ``get_batch_status`` and
    ``entries`` is the canned ``list[BatchResultEntry]`` returned by
    ``fetch_batch_results``. Because they are separate methods (and separate
    recorded calls), a test can prove the status path performs no download by
    asserting ``fetch_batch_results`` was never called.

    Defaults keep it usable with no arguments: ``get_batch_status`` returns
    ``ENDED``, ``fetch_batch_results`` returns ``[]``, ``submit_batch`` returns
    ``"batch-fake"``, and ``process``/``parse_batch_result`` return ``({}, [])``.
    """

    def __init__(
        self,
        *,
        status: BatchPollStatus = BatchPollStatus.ENDED,
        entries: list[BatchResultEntry] | None = None,
        submit_batch_id: str = "batch-fake",
        process_result: tuple[dict[str, Any], list[ExtractedImage]] | None = None,
        parse_result: tuple[dict[str, Any], list[ExtractedImage]] | None = None,
    ) -> None:
        self._status = status
        self._entries: list[BatchResultEntry] = entries if entries is not None else []
        self._submit_batch_id = submit_batch_id
        self._process_result: tuple[dict[str, Any], list[ExtractedImage]] = (
            process_result if process_result is not None else ({}, [])
        )
        # Fall back to the process result so a caller that only sets one gets
        # a consistent (body, images) shape from either sync entry point.
        self._parse_result: tuple[dict[str, Any], list[ExtractedImage]] = (
            parse_result if parse_result is not None else self._process_result
        )
        self.calls: list[RecordedCall] = []

    async def process(
        self,
        *,
        document: DocumentTypedDict,
        model: str,
        include_image_base64: bool = True,
    ) -> tuple[dict[str, Any], list[ExtractedImage]]:
        """Record the call and return the canned ``(body, images)`` result."""
        self.calls.append(
            RecordedCall(
                "process",
                (),
                {
                    "document": document,
                    "model": model,
                    "include_image_base64": include_image_base64,
                },
            )
        )
        return self._process_result

    async def submit_batch(
        self,
        requests: list[dict[str, Any]],
        model: str,
        *,
        endpoint: str = _OCR_ENDPOINT,
    ) -> str:
        """Record the call and return the canned batch id."""
        self.calls.append(RecordedCall("submit_batch", (requests, model), {"endpoint": endpoint}))
        return self._submit_batch_id

    async def get_batch_status(self, batch_id: str) -> BatchPollStatus:
        """Record the call and return the canned status. No download."""
        self.calls.append(RecordedCall("get_batch_status", (batch_id,), {}))
        return self._status

    async def fetch_batch_results(self, batch_id: str) -> list[BatchResultEntry]:
        """Record the call and return the canned result entries."""
        self.calls.append(RecordedCall("fetch_batch_results", (batch_id,), {}))
        return self._entries

    def parse_batch_result(self, raw_json: str) -> tuple[dict[str, Any], list[ExtractedImage]]:
        """Record the call and return the canned ``(body, images)`` result."""
        self.calls.append(RecordedCall("parse_batch_result", (raw_json,), {}))
        return self._parse_result


if TYPE_CHECKING:
    # Static proof that FakeLLM satisfies the shape pbook's llm_chat requires.
    # pbook.llm.SupportsComplete cannot be imported here — sax-platform is a
    # library and the dependency graph forbids a lib importing an app — so its
    # `complete` shape is restated and FakeLLM is assigned to it. mypy checks
    # this assignment; a drift in either signature breaks the build.
    class _SupportsComplete(Protocol):
        async def complete(
            self,
            messages: Iterable[MessageParam],
            *,
            output_type: type[BaseModel],
            model: str,
            max_tokens: int,
            system: str | list[dict[str, Any]] | None = ...,
            cache: CacheSpec | None = ...,
            thinking: ThinkingPolicy | None = ...,
        ) -> Completion[Any]: ...

    _supports_complete_check: _SupportsComplete = FakeLLM()
