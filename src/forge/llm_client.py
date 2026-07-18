"""Interim AnthropicLLM acquisition for forge's imperative shells (T3.5).

Module-global cache mirroring the seam the retired ``sax_llm.get_provider``
provided: activities' shell functions call ``get_llm()`` to reuse one client
across calls, and tests swap it via ``reset_llm()`` plus patching. Deleted in
T3.6 (D93) when composition roots construct the client once in worker/CLI
mains and thread it through class-based activities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sax_platform.llm import AnthropicLLM

_llm: AnthropicLLM | None = None


def get_llm() -> AnthropicLLM:
    """Return the process-wide AnthropicLLM, building it on first use."""
    global _llm
    if _llm is None:
        from sax_platform.llm import AnthropicLLM, make_client

        _llm = AnthropicLLM(make_client())
    return _llm


def reset_llm() -> None:
    """Clear the cached client (tests only)."""
    global _llm
    _llm = None
