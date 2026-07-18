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
    ThinkingConfigAdaptiveParam,
    ThinkingConfigDisabledParam,
    ThinkingConfigParam,
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
from sax_platform.llm.tiers import Effort, ThinkingPolicy

__all__ = ["AnthropicLLM", "make_client"]


def thinking_request_parts(
    thinking: ThinkingPolicy | None,
) -> tuple[ThinkingConfigParam | None, Effort | None]:
    """Translate a `ThinkingPolicy` into its two request contributions.

    Both lanes (`sax_platform.llm.client` and `sax_platform.llm.batch`) build
    requests from the same pair, so this is the single place a `ThinkingPolicy`
    becomes wire shapes — the successor to `sax_llm.client.build_thinking_param`,
    minus the forced-tool-use machinery that path carried (there are no tools
    on the structured-outputs path). Returns:

    - the ``thinking`` param value, or `None` to omit the param entirely; and
    - the ``effort`` to merge into ``output_config``, or `None`.

    Semantics ported from `sax_llm` (D94):

    - `None` — no `ThinkingPolicy` given: emit no ``thinking`` param and no
      ``effort`` key, leaving the request wire-identical to one built without
      any thinking argument.
    - `ThinkingPolicy(enabled=True)` — adaptive thinking: the API runs
      adaptive thinking whenever the ``thinking`` field is present as
      ``{"type": "adaptive"}``, and the policy's `effort` rides along in
      ``output_config``.
    - `ThinkingPolicy(enabled=False)` — thinking explicitly OFF: on the
      current model generation, *omitting* the ``thinking`` field runs
      adaptive thinking BY DEFAULT, so disabling it requires the explicit
      ``{"type": "disabled"}`` shape rather than leaving the field out. No
      `effort` accompanies a disabled policy.

    Unlike `sax_llm.client.build_thinking_param`, this does not gate on the
    model name (no Haiku/pre-adaptive special-casing, no `PRE_ADAPTIVE_HINTS`
    warning): thinking is now opt-in per call via an explicit `ThinkingPolicy`,
    and the platform's single tier registry only pins adaptive-generation
    models (D94), so a pre-adaptive pin cannot reach this through the supported
    path.
    """
    if thinking is None:
        return None, None
    if thinking.enabled:
        adaptive: ThinkingConfigAdaptiveParam = {"type": "adaptive"}
        return adaptive, thinking.effort
    disabled: ThinkingConfigDisabledParam = {"type": "disabled"}
    return disabled, None


def _with_effort(
    output_config: OutputConfigParam | Omit, effort: Effort | None
) -> OutputConfigParam | Omit:
    """Merge `effort` into `output_config`, leaving `format` (if any) intact.

    When `effort` is `None`, `output_config` is returned unchanged — including
    the `omit` sentinel, so the text lane with no thinking sends no
    ``output_config`` key at all. Otherwise `effort` is added alongside any
    existing keys (e.g. `format`): starting from an empty dict when the base
    was `omit`, or from a copy of the base config when structured output is
    requested.

    The `cast` widens the platform's `Effort` vocabulary — which includes
    ``"xhigh"`` — onto `OutputConfigParam`, whose installed SDK stub models a
    narrower ``Literal`` for the field. `sax_llm` likewise sent `effort` as a
    free string; the platform tier vocabulary is the contract, not this SDK
    version's generated literal.
    """
    if effort is None:
        return output_config
    base: dict[str, Any] = {} if isinstance(output_config, Omit) else dict(output_config)
    base["effort"] = effort
    return cast("OutputConfigParam", base)


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
        thinking: ThinkingPolicy | None = None,
    ) -> Completion[T]:
        """Structured output validated into `output_type`.

        The JSON schema sent to the API is derived from `output_type` via
        `sax_platform.llm.schema.to_json_schema`. A response whose
        classified text fails `output_type.model_validate_json` raises
        `LLMSchemaMismatch` — a pydantic `ValidationError` never escapes
        this method directly.

        `thinking` opts into extended thinking (see `thinking_request_parts`):
        `None` (the default) leaves the request wire-identical to the
        no-thinking path; a `ThinkingPolicy` adds the ``thinking`` param and,
        when enabled, its `effort` alongside the ``format`` in ``output_config``.
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
            thinking=thinking,
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
        thinking: ThinkingPolicy | None = None,
    ) -> Completion[dict[str, Any]]:
        """Structured output validated as a raw JSON object against a
        caller-supplied `output_schema`.

        `output_schema` is sent to the API exactly as given — unlike
        `complete`, this method does not close its object nodes with
        ``additionalProperties: false``; the caller owns that schema and
        its validity. A response whose classified text fails to parse as
        JSON, or parses to something other than a JSON object, raises
        `LLMSchemaMismatch`.

        `thinking` behaves as documented on `complete`.
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
            thinking=thinking,
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
        thinking: ThinkingPolicy | None = None,
    ) -> Completion[str]:
        """Plain text completion. No structured-output ``format`` is sent —
        this is the one variant with no structured-output request at all.

        `thinking` still applies: when it enables thinking, its `effort` is
        the *only* key in ``output_config`` (there is no ``format`` here). With
        `thinking=None` (the default) no ``output_config`` is sent at all.
        """
        classified = await self._complete_raw(
            messages,
            model=model,
            max_tokens=max_tokens,
            system=system,
            cache=cache,
            output_config=omit,
            thinking=thinking,
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
        thinking: ThinkingPolicy | None = None,
    ) -> ClassifiedMessage:
        """Shared transport + classification step for all three public
        methods: normalize `system`, apply the cache placement policy, apply
        the thinking policy, send the request, and classify the response by
        `stop_reason`.

        Raises `LLMRefused` / `LLMTruncated` (via `classify_message`) for
        the two terminal stop reasons; returns a `ClassifiedMessage` for
        every other stop reason, leaving parsing to the caller.

        `thinking` is translated by `thinking_request_parts` into the
        ``thinking`` param (omitted entirely when `None`) and an `effort`
        merged into `output_config` via `_with_effort`.
        """
        thinking_param, effort = thinking_request_parts(thinking)
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=_normalize_system(system, model=model, cache=cache),
            output_config=_with_effort(output_config, effort),
            thinking=thinking_param if thinking_param is not None else omit,
        )
        return classify_message(response, max_tokens=max_tokens)
