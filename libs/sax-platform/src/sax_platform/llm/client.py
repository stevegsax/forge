"""Sync-lane LLM client: one request in, one classified + parsed `Completion`
out.

``messages.create`` + ``output_config.format`` is used deliberately instead
of the SDK's ``messages.parse`` helper. ``messages.parse`` couples transport
and parsing into one call, so a refusal or truncation can surface as (or
hide behind) a downstream parsing failure before ``stop_reason`` is ever
classified. ``output_config.format`` is the same structured-outputs feature
under the hood (verified against the live API 2026-07-16) — sending it via
plain ``messages.create`` keeps the two concerns separate: this module
always classifies ``stop_reason`` first (via
`sax_platform.llm.models.classify_message`, raising `LLMRefused` /
`LLMTruncated`) and only then attempts to parse the classified text, so a
parse failure is reported as the distinct, typed `LLMSchemaMismatch` rather
than an opaque exception from inside the SDK's own parsing path.

Three request shapes share one transport helper (`AnthropicLLM._complete_raw`):

- `AnthropicLLM.complete` — structured output validated into a caller-given
  pydantic model (schema derived via `sax_platform.llm.schema.to_json_schema`).
- `AnthropicLLM.complete_schema` — structured output validated as a raw JSON
  object against a caller-supplied JSON schema (the caller owns the schema;
  this module does not close its objects the way `to_json_schema` does for
  the pydantic-model case).
- `AnthropicLLM.complete_text` — plain text, no ``output_config`` sent at
  all.

`make_client` fixes ``max_retries=0``: this client runs inside Temporal
activities, and the activity's own `RetryPolicy` already owns retries.
Stacking the SDK's built-in retry loop underneath Temporal's would multiply
retries (each Temporal attempt could itself retry several times inside the
SDK), turning one configured retry budget into two compounding ones.
"""

import json
from collections.abc import Iterable
from typing import Any, cast

from anthropic import AsyncAnthropic, Omit, omit
from anthropic.types import (
    JSONOutputFormatParam,
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
)
from pydantic import BaseModel, ValidationError

from sax_platform.llm.cache import CacheSpec, apply_cache_control
from sax_platform.llm.models import (
    ClassifiedMessage,
    Completion,
    LLMSchemaMismatch,
    classify_message,
)
from sax_platform.llm.schema import to_json_schema, to_output_format

__all__ = ["AnthropicLLM", "make_client"]


def make_client(api_key: str | None = None) -> AsyncAnthropic:
    """Construct an `AsyncAnthropic` client with SDK-level retries disabled.

    ``max_retries=0`` is load-bearing, not a default we happen to pick: this
    client is called from Temporal activities, and the activity's
    `RetryPolicy` already owns retry behavior. Leaving the SDK's own retry
    loop enabled would stack retries under retries — each Temporal attempt
    could itself silently retry several times inside the SDK before
    Temporal ever sees a failure to schedule its own retry against,
    multiplying total attempts and defeating the activity's configured
    backoff/attempt budget.

    `api_key` is passed through to the SDK only when given; when omitted,
    the SDK falls back to its own default behavior (reading
    ``ANTHROPIC_API_KEY`` from the environment).
    """
    if api_key is not None:
        return AsyncAnthropic(api_key=api_key, max_retries=0)
    return AsyncAnthropic(max_retries=0)


def _normalize_system(
    system: str | list[dict[str, Any]] | None,
    *,
    model: str,
    cache: CacheSpec | None,
) -> list[TextBlockParam] | Omit:
    """Normalize the caller's `system` into wire-ready blocks, with the
    cache breakpoint policy applied.

    A bare string becomes a single text block; `None` (or a resulting empty
    block list) becomes `anthropic.omit` so the request carries no
    ``system`` key at all, rather than an explicit empty value.
    """
    if system is None:
        blocks: list[dict[str, Any]] = []
    elif isinstance(system, str):
        blocks = [{"type": "text", "text": system}]
    else:
        blocks = list(system)

    cached_blocks = apply_cache_control(blocks, model=model, spec=cache)
    if not cached_blocks:
        return omit
    # Each block was built above (or passed in) as a `{"type": "text",
    # "text": ...}` shape, optionally with `cache_control` added by
    # `apply_cache_control` — exactly `TextBlockParam`'s contract. The cast
    # documents that proof; nothing here has an unvalidated shape.
    return cast("list[TextBlockParam]", cached_blocks)


def _completion_from_classified[T](classified: ClassifiedMessage, output: T) -> Completion[T]:
    """Build a `Completion[T]` from a classified message plus the caller's
    already-parsed `output`. Shared tail of all three `AnthropicLLM` methods."""
    telemetry = classified.telemetry
    return Completion[T](
        output=output,
        model=telemetry.model,
        stop_reason=telemetry.stop_reason,
        input_tokens=telemetry.input_tokens,
        output_tokens=telemetry.output_tokens,
        cache_creation_input_tokens=telemetry.cache_creation_input_tokens,
        cache_read_input_tokens=telemetry.cache_read_input_tokens,
        request_id=telemetry.request_id,
    )


class AnthropicLLM:
    """Sync-lane LLM client wrapping a caller-supplied `AsyncAnthropic`.

    The client is a constructor parameter, not built internally — callers
    (the Temporal activity shell) own the client's lifecycle and construct
    it via `make_client`. This class holds no other state.
    """

    def __init__(self, client: AsyncAnthropic) -> None:
        self._client = client

    async def complete[T: BaseModel](
        self,
        messages: Iterable[MessageParam],
        *,
        output_type: type[T],
        model: str,
        max_tokens: int,
        system: str | list[dict[str, Any]] | None = None,
        cache: CacheSpec | None = None,
    ) -> Completion[T]:
        """Structured output validated into `output_type`.

        The JSON schema sent to the API is derived from `output_type` via
        `sax_platform.llm.schema.to_json_schema`. A response whose
        classified text fails `output_type.model_validate_json` raises
        `LLMSchemaMismatch` — a pydantic `ValidationError` never escapes
        this method directly.
        """
        # `to_output_format` always returns the `{"type": "json_schema",
        # "schema": ...}` shape `JSONOutputFormatParam` requires; the cast
        # documents that proof rather than widening the return type to
        # `dict[str, Any]` at the call site.
        output_config: OutputConfigParam = {
            "format": cast("JSONOutputFormatParam", to_output_format(to_json_schema(output_type)))
        }
        classified = await self._complete_raw(
            messages,
            model=model,
            max_tokens=max_tokens,
            system=system,
            cache=cache,
            output_config=output_config,
        )
        try:
            parsed = output_type.model_validate_json(classified.text)
        except ValidationError as exc:
            raise LLMSchemaMismatch(
                raw_text=classified.text, error=str(exc), telemetry=classified.telemetry
            ) from exc
        return _completion_from_classified(classified, parsed)

    async def complete_schema(
        self,
        messages: Iterable[MessageParam],
        *,
        output_schema: dict[str, Any],
        model: str,
        max_tokens: int,
        system: str | list[dict[str, Any]] | None = None,
        cache: CacheSpec | None = None,
    ) -> Completion[dict[str, Any]]:
        """Structured output validated as a raw JSON object against a
        caller-supplied `output_schema`.

        `output_schema` is sent to the API exactly as given — unlike
        `complete`, this method does not close its object nodes with
        ``additionalProperties: false``; the caller owns that schema and
        its validity. A response whose classified text fails to parse as
        JSON, or parses to something other than a JSON object, raises
        `LLMSchemaMismatch`.
        """
        output_config: OutputConfigParam = {
            "format": cast("JSONOutputFormatParam", to_output_format(output_schema))
        }
        classified = await self._complete_raw(
            messages,
            model=model,
            max_tokens=max_tokens,
            system=system,
            cache=cache,
            output_config=output_config,
        )
        try:
            parsed = json.loads(classified.text)
        except json.JSONDecodeError as exc:
            raise LLMSchemaMismatch(
                raw_text=classified.text, error=str(exc), telemetry=classified.telemetry
            ) from exc
        if not isinstance(parsed, dict):
            raise LLMSchemaMismatch(
                raw_text=classified.text,
                error=f"expected a JSON object at the top level, got {type(parsed).__name__}",
                telemetry=classified.telemetry,
            )
        return _completion_from_classified(classified, parsed)

    async def complete_text(
        self,
        messages: Iterable[MessageParam],
        *,
        model: str,
        max_tokens: int,
        system: str | list[dict[str, Any]] | None = None,
        cache: CacheSpec | None = None,
    ) -> Completion[str]:
        """Plain text completion. No ``output_config`` is sent — this is
        the one variant with no structured-output request at all."""
        classified = await self._complete_raw(
            messages,
            model=model,
            max_tokens=max_tokens,
            system=system,
            cache=cache,
            output_config=omit,
        )
        return _completion_from_classified(classified, classified.text)

    async def _complete_raw(
        self,
        messages: Iterable[MessageParam],
        *,
        model: str,
        max_tokens: int,
        system: str | list[dict[str, Any]] | None,
        cache: CacheSpec | None,
        output_config: OutputConfigParam | Omit,
    ) -> ClassifiedMessage:
        """Shared transport + classification step for all three public
        methods: normalize `system`, apply the cache placement policy, send
        the request, and classify the response by `stop_reason`.

        Raises `LLMRefused` / `LLMTruncated` (via `classify_message`) for
        the two terminal stop reasons; returns a `ClassifiedMessage` for
        every other stop reason, leaving parsing to the caller.
        """
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=_normalize_system(system, model=model, cache=cache),
            output_config=output_config,
        )
        return classify_message(response, max_tokens=max_tokens)
