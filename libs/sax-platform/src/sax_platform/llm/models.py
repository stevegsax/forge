"""Shared result/failure surface for the LLM client's sync and batch lanes.

Both lanes need the same three things: a typed success envelope
(`Completion`), a common vocabulary for what went wrong when a call didn't
produce a usable result (the `LLMOutcomeError` family for the sync lane, and
the value-shaped `*Outcome` types for the batch lane, which has no single
in-flight call to raise from), and one classifier (`classify_message`) that
both lanes call so refusal/truncation detection is defined exactly once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    # Annotation-only: classify/telemetry read Message attributes duck-typed,
    # so the SDK never has to be imported at runtime — keeping this module
    # importable inside the Temporal workflow sandbox without an HTTP stack.
    import anthropic.types

__all__ = [
    "ClassifiedMessage",
    "Completion",
    "LLMOutcomeError",
    "LLMRefused",
    "LLMSchemaMismatch",
    "LLMTruncated",
    "MismatchOutcome",
    "RefusedOutcome",
    "Telemetry",
    "TruncatedOutcome",
    "classify_message",
    "outcome_from_error",
    "telemetry_from_message",
]


class Telemetry(BaseModel):
    """Billing/rate-limit telemetry for one LLM call, independent of output.

    Same fields as `Completion` minus `output`. The failure types below
    embed a `Telemetry` so a refusal or truncation still carries full
    usage/billing detail even though there is no parsed `output` to report.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    request_id: str | None


class Completion[T](BaseModel):
    """A successfully completed LLM call: the caller's parsed output plus
    full telemetry.

    Generic over the caller's expected output type `T` — a plain string, a
    structured-output model instance, or whatever the caller's document
    completion schema decodes to.
    """

    model_config = ConfigDict(frozen=True)

    output: T
    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    request_id: str | None


class LLMOutcomeError(Exception):
    """Common base for the three classification-time LLM outcome failures.

    Catch this to handle refusal/truncation/mismatch uniformly; catch a
    specific subclass when the caller needs to branch on which occurred.
    """

    def __init__(self, message: str, *, telemetry: Telemetry) -> None:
        super().__init__(message)
        self.telemetry = telemetry


@final
class LLMRefused(LLMOutcomeError):  # noqa: N818 — name is a fixed public API contract
    """Raised when `stop_reason` was `"refusal"`.

    `category` is the policy category from `stop_details.category` when the
    SDK response carries one; `None` when absent (older SDKs don't model
    `stop_details` at all) or unset.
    """

    def __init__(self, *, category: str | None, telemetry: Telemetry) -> None:
        self.category = category
        detail = f" (category={category})" if category is not None else ""
        super().__init__(f"LLM call was refused{detail}", telemetry=telemetry)


@final
class LLMTruncated(LLMOutcomeError):  # noqa: N818 — name is a fixed public API contract
    """Raised when `stop_reason` was `"max_tokens"`.

    `partial_text` is whatever text had been produced before the cutoff.
    """

    def __init__(self, *, partial_text: str, max_tokens: int, telemetry: Telemetry) -> None:
        self.partial_text = partial_text
        self.max_tokens = max_tokens
        super().__init__(
            f"LLM output truncated at max_tokens={max_tokens} "
            f"({len(partial_text)} chars produced before truncation)",
            telemetry=telemetry,
        )


@final
class LLMSchemaMismatch(LLMOutcomeError):  # noqa: N818 — name is a fixed public API contract
    """Raised by the caller when a completed response (`stop_reason` was NOT
    `"refusal"` or `"max_tokens"`) failed to parse into the expected shape —
    including text-only prose where JSON was expected.

    `classify_message` never raises this itself; it only classifies
    stop_reason. Parsing the classified text is the caller's job, and a
    parse failure is the caller's failure to report.
    """

    def __init__(self, *, raw_text: str, error: str, telemetry: Telemetry) -> None:
        self.raw_text = raw_text
        self.error = error
        super().__init__(f"LLM output did not match expected schema: {error}", telemetry=telemetry)


class ClassifiedMessage(BaseModel):
    """A completed message that was neither refused nor truncated, ready
    for the caller to parse."""

    model_config = ConfigDict(frozen=True)

    text: str
    stop_reason: str
    telemetry: Telemetry


@final
class RefusedOutcome(BaseModel):
    """Value-shaped counterpart to `LLMRefused`, for the batch lane."""

    model_config = ConfigDict(frozen=True)

    category: str | None
    telemetry: Telemetry


@final
class TruncatedOutcome(BaseModel):
    """Value-shaped counterpart to `LLMTruncated`, for the batch lane."""

    model_config = ConfigDict(frozen=True)

    partial_text: str
    max_tokens: int
    telemetry: Telemetry


@final
class MismatchOutcome(BaseModel):
    """Value-shaped counterpart to `LLMSchemaMismatch`, for the batch lane."""

    model_config = ConfigDict(frozen=True)

    raw_text: str
    error: str
    telemetry: Telemetry


def _first_text(message: anthropic.types.Message) -> str:
    """Return the first text block's text, or `""` if the message has none
    (e.g. a tool-use-only or refusal response)."""
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


def _refusal_category(message: anthropic.types.Message) -> str | None:
    """Read `stop_details.category` defensively.

    `stop_details` is documented as GA on Opus 4.7+ but is not a field
    every installed SDK version models on `Message`. On such SDKs
    (anthropic 0.78 included) the wire field survives only via pydantic's
    `extra="allow"`, which stores it as a plain **dict** — while a newer
    SDK would model it as an attribute-bearing object. Read both shapes;
    anything else degrades to `None` instead of raising.
    """
    stop_details = getattr(message, "stop_details", None)
    if stop_details is None:
        return None
    if isinstance(stop_details, dict):
        category = stop_details.get("category")
    else:
        category = getattr(stop_details, "category", None)
    return category if isinstance(category, str) else None


def telemetry_from_message(message: anthropic.types.Message) -> Telemetry:
    """Extract billing/rate-limit telemetry from a completed `Message`.

    `request_id` is read via `getattr` with a `None` default: the SDK
    attaches `_request_id` dynamically to responses parsed from a live HTTP
    call (it is not a declared pydantic field), so a hand-constructed
    `Message` — as in tests, or a message replayed from storage — won't
    carry it.
    """
    usage = message.usage
    return Telemetry(
        model=message.model,
        stop_reason=message.stop_reason or "",
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        request_id=getattr(message, "_request_id", None),
    )


def classify_message(message: anthropic.types.Message, *, max_tokens: int) -> ClassifiedMessage:
    """Classify a completed Anthropic `Message` by `stop_reason`, before any
    parsing of its content.

    `stop_reason` is checked first — and exclusively — so a refusal or a
    truncation can never masquerade as a downstream `ValidationError`: this
    function raises `LLMRefused` / `LLMTruncated` for those two terminal
    stop reasons and returns a plain `ClassifiedMessage` for every other
    stop reason (`end_turn`, `tool_use`, `stop_sequence`, `pause_turn`, ...).
    It does not parse JSON or validate shape — that is the caller's job,
    and a caller-side parse failure is reported as `LLMSchemaMismatch`, not
    raised from here.
    """
    telemetry = telemetry_from_message(message)
    text = _first_text(message)

    if message.stop_reason == "refusal":
        raise LLMRefused(category=_refusal_category(message), telemetry=telemetry)
    if message.stop_reason == "max_tokens":
        raise LLMTruncated(partial_text=text, max_tokens=max_tokens, telemetry=telemetry)

    return ClassifiedMessage(text=text, stop_reason=message.stop_reason or "", telemetry=telemetry)


def outcome_from_error(err: LLMOutcomeError) -> RefusedOutcome | TruncatedOutcome | MismatchOutcome:
    """Convert a raised classification failure into its value-shaped
    counterpart.

    This is what lets the batch lane — which reconciles many results at
    once rather than catching one raised exception per call — share the
    exact same classification core as the sync lane: both ultimately
    produce one of these three shapes, just via `raise` vs. `return`.
    """
    if isinstance(err, LLMRefused):
        return RefusedOutcome(category=err.category, telemetry=err.telemetry)
    if isinstance(err, LLMTruncated):
        return TruncatedOutcome(
            partial_text=err.partial_text, max_tokens=err.max_tokens, telemetry=err.telemetry
        )
    if isinstance(err, LLMSchemaMismatch):
        return MismatchOutcome(raw_text=err.raw_text, error=err.error, telemetry=err.telemetry)
    raise TypeError(f"unrecognized LLMOutcomeError subclass: {type(err).__name__}")
