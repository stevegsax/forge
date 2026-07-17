"""Dependency-injection seam for the OCR app's Mistral OCR capability (T3.3).

`MistralOcr` (from `sax_platform.ocr`, D88) is injected here at worker startup
rather than imported as a module-level global by whichever code needs it:
consumers call `get_mistral_ocr()` and fail loudly with a `RuntimeError` if
the worker never registered one, instead of constructing a client implicitly
and hiding a missing `MISTRAL_API_KEY` until the first real call. Phase 4's
self-polling activities (ocr polling its own Mistral batches through this
capability, D88) are the seam's first consumer.

Mirrors `pbook.llm`'s `set_provider`/`get_provider`/`reset_provider` registry
pattern.
"""

# `from __future__ import annotations` is otherwise dropped on this repo's
# pinned 3.14 (PEP 649 defers annotation evaluation by default) — kept here
# per the narrow exception: `requires-python` still floors at >=3.12, this
# module isn't introspected at runtime (no pydantic/dataclass consumer of
# `__annotations__`), and the `MistralOcr` name below is TYPE_CHECKING-only,
# which needs the string-deferral this import provides on pre-3.14 runtimes.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sax_platform.ocr import MistralOcr

__all__ = ["get_mistral_ocr", "reset_mistral_ocr", "set_mistral_ocr"]

_mistral_ocr: MistralOcr | None = None


def set_mistral_ocr(instance: MistralOcr) -> None:
    """Register the Mistral OCR capability for OCR activities."""
    global _mistral_ocr
    _mistral_ocr = instance


def get_mistral_ocr() -> MistralOcr:
    """Get the registered Mistral OCR capability.

    Raises ``RuntimeError`` if none has been registered.
    """
    if _mistral_ocr is None:
        msg = (
            "No Mistral OCR capability registered. Call ocr.deps.set_mistral_ocr() "
            "before running OCR activities that need it."
        )
        raise RuntimeError(msg)
    return _mistral_ocr


def reset_mistral_ocr() -> None:
    """Clear the registered Mistral OCR capability (for testing)."""
    global _mistral_ocr
    _mistral_ocr = None
