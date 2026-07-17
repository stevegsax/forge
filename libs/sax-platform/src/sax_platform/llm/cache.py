"""Pure prompt-cache placement policy.

Production telemetry showed zero recorded prompt-cache interactions on the
prior client, so caching defaults OFF at the client layer (later agents wire
that default; this module has no opinion on it). This module only computes
*where* a cache breakpoint would go when a caller explicitly opts in via a
`CacheSpec` — it performs no I/O and makes no request itself.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "MIN_CACHEABLE_TOKENS",
    "CacheSpec",
    "apply_cache_control",
    "estimate_tokens",
    "min_cacheable_tokens",
]


class CacheSpec(BaseModel):
    """Explicit caller opt-in for prompt-cache placement."""

    model_config = ConfigDict(frozen=True)

    ttl: Literal["5m", "1h"] = "5m"


# Per-model minimum cacheable prefix, in (estimated) tokens. Keyed by model
# *prefix* — see `min_cacheable_tokens` for the longest-prefix lookup.
MIN_CACHEABLE_TOKENS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "claude-opus-4": 4096,
        "claude-haiku-4-5": 4096,
        "claude-fable-5": 2048,
        "claude-sonnet-4-6": 2048,
        "claude-sonnet-4-5": 1024,
    }
)

# Unknown models get the most conservative (highest) known threshold, so an
# unrecognized model never gets a breakpoint it hasn't earned.
_DEFAULT_MIN_CACHEABLE_TOKENS: Final[int] = 4096


def min_cacheable_tokens(model: str) -> int:
    """Minimum cacheable prefix length for `model`, in estimated tokens.

    Looks up `MIN_CACHEABLE_TOKENS` by longest-prefix match (so a dated
    snapshot ID like `claude-sonnet-4-5-20250929` resolves the same as its
    bare alias). A model matching no known prefix returns the conservative
    default, `_DEFAULT_MIN_CACHEABLE_TOKENS`.
    """
    matches = (
        (prefix, threshold)
        for prefix, threshold in MIN_CACHEABLE_TOKENS.items()
        if model.startswith(prefix)
    )
    best = max(matches, key=lambda item: len(item[0]), default=None)
    return best[1] if best is not None else _DEFAULT_MIN_CACHEABLE_TOKENS


def estimate_tokens(text: str) -> int:
    """Crude, deliberately conservative token estimate: `len(text) // 4`.

    This is not a tokenizer — it is a cheap heuristic used only to decide
    whether a system prefix clears a model's minimum cacheable size. It
    overestimates token-dense text (code, non-English scripts, which
    average fewer characters per token) and underestimates token-sparse
    text (long runs of whitespace), but integer-division-by-4 on character
    count is deliberately biased toward the conservative side of "don't
    place a breakpoint that costs more than it saves" for typical English
    prose, which is the dominant case for system prompts.
    """
    return len(text) // 4


def _block_estimated_tokens(block: dict[str, Any]) -> int:
    text = block.get("text")
    return estimate_tokens(text) if isinstance(text, str) else 0


def apply_cache_control(
    system_blocks: list[dict[str, Any]],
    *,
    model: str,
    spec: CacheSpec | None,
) -> list[dict[str, Any]]:
    """Return a NEW list of system content blocks with a cache breakpoint
    placed on the last block, or an unchanged copy when no breakpoint
    should be placed.

    Pure: never mutates `system_blocks` or any block within it.

    A breakpoint is placed only when:
      - the caller opted in (`spec` is not `None`), and
      - the estimated cumulative token count of all system blocks meets or
        exceeds `min_cacheable_tokens(model)`.

    Below that threshold the breakpoint is silently omitted rather than
    placed anyway. This is the load-bearing rule: a cache write costs 1.25x
    the base input price at the default 5-minute TTL, and 2x at the 1-hour
    TTL — a breakpoint on a prefix too short to be re-read enough times to
    recoup that premium is a net loss, not a neutral no-op. See
    `shared/prompt-caching.md` in the claude-api skill for the underlying
    economics.

    When a breakpoint is placed, the last block gets
    `cache_control: {"type": "ephemeral"}`, plus `"ttl": "1h"` only when
    `spec.ttl == "1h"` — the 5-minute default omits the `ttl` key entirely
    (matching the API's own default-omission convention).
    """
    if spec is None or not system_blocks:
        return list(system_blocks)

    total_tokens = sum(_block_estimated_tokens(block) for block in system_blocks)
    if total_tokens < min_cacheable_tokens(model):
        return list(system_blocks)

    control: dict[str, Any] = {"type": "ephemeral"}
    if spec.ttl == "1h":
        control["ttl"] = "1h"

    *head, last = system_blocks
    new_last = {**last, "cache_control": control}
    return [*head, new_last]
