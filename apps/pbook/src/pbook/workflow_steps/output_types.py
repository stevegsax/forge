"""Frozen mapping from output-type names to Pydantic model classes.

The :func:`llm_chat` activity in this package resolves a class from a
string at activity-time so callers can request structured output by name
(the activity input is JSON-serializable and a class reference can't cross
Temporal's JSON boundary). The set of output types pbook uses is fixed —
extraction, review, and consolidation — so the mapping is a frozen,
module-level constant rather than a registry populated at worker startup.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pbook.llm import ConsolidationResult, ExtractionResult, ReviewResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import BaseModel

OUTPUT_TYPES: Final[Mapping[str, type[BaseModel]]] = MappingProxyType(
    {
        "ExtractionResult": ExtractionResult,
        "ReviewResult": ReviewResult,
        "ConsolidationResult": ConsolidationResult,
    }
)


def resolve_output_type(name: str) -> type[BaseModel]:
    """Return the Pydantic class registered under ``name``.

    Raises ``KeyError`` with a clear message listing the known output types
    so an unrecognized name is obvious at the call site.
    """
    try:
        return OUTPUT_TYPES[name]
    except KeyError:
        known = ", ".join(sorted(OUTPUT_TYPES))
        msg = f"Output type {name!r} is not a known pbook output type. Known types: {known}."
        raise KeyError(msg) from None
