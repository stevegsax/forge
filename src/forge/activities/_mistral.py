"""Shared, lazily-cached Mistral OCR resolver for the batch submit/poll activities.

Both ``submit_batch_blob``/``_resolve_blob_submit_provider`` (batch_submit.py) and
``_poll_batch_for`` (batch_poll.py) route the ``"mistral"`` provider through
``sax_platform.ocr.MistralOcr`` instead of sax_llm's registry (T3.3: mistral's
OCR capability moved to the platform library; sax_llm carries no provider
entry for it anymore). Each activity previously built its own
``MistralOcr(make_mistral_client())`` per call — a fresh Mistral SDK client on
every poll cycle. This module caches the single pair at module scope
(mirroring the ``_temporal_client`` cache in ``batch_poll.py`` and sax_llm's
own ``_provider_cache``) so successive calls within a worker process reuse it.

``make_mistral_client()`` may raise ``ValueError`` when ``MISTRAL_API_KEY`` is
unset (2026-07 Phase 3 code review, item 5): construction happens lazily,
inside :func:`get_mistral_ocr`, on first call, so a worker that never touches
Mistral never pays for it. A raise there must propagate to the caller on
every call until construction actually succeeds — the module-level variable
is only assigned *after* both calls return, so a failed attempt leaves it
``None`` and the next call retries construction rather than replaying a
cached failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sax_platform.ocr import MistralOcr

_mistral_ocr: MistralOcr | None = None


def get_mistral_ocr() -> MistralOcr:
    """Return the process-wide cached ``MistralOcr``, building it on first call."""
    global _mistral_ocr
    if _mistral_ocr is None:
        from sax_platform.ocr import MistralOcr, make_mistral_client

        _mistral_ocr = MistralOcr(make_mistral_client())
    return _mistral_ocr


def reset_mistral_ocr_cache() -> None:
    """Clear the cached ``MistralOcr`` instance. Intended for testing."""
    global _mistral_ocr
    _mistral_ocr = None
