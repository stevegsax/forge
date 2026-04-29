"""Registry mapping output-type names to Pydantic model classes.

The :func:`llm_chat` activity in this package needs to resolve a class
from a string at activity-time so that callers can request structured
output by name (the activity input is JSON-serializable). The registry
is local to pbook and intentionally separate from
``sax_llm.register_output_type`` (which forge uses for batch parsing
and which already holds entries with overlapping names).

Workflows / workers register types at startup; activities resolve them
by name. Tests can ``reset_registry()`` between runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}


def register_output_type(name: str, cls: type[BaseModel]) -> None:
    """Associate ``name`` with the given Pydantic class.

    Re-registering the same name overwrites the prior entry, which keeps
    test setup simple and lets a worker re-initialize cleanly.
    """
    _REGISTRY[name] = cls


def resolve_output_type(name: str) -> type[BaseModel]:
    """Return the Pydantic class registered under ``name``.

    Raises ``KeyError`` with a clear message pointing at the registration
    helper so the failure mode of "forgot to register" is obvious.
    """
    if name not in _REGISTRY:
        msg = (
            f"Output type {name!r} is not registered. Call "
            "pbook.workflow_steps.output_types.register_output_type() "
            "at worker startup before invoking llm_chat with this name."
        )
        raise KeyError(msg)
    return _REGISTRY[name]


def reset_registry() -> None:
    """Clear the registry (for tests)."""
    _REGISTRY.clear()
