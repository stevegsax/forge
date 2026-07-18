"""Frozen output-type registry for Forge's batch structured-output types (D90/D93).

Replaces the former string→type plugin registry. The mapping is a module-level
frozen constant, imported directly wherever a batch submit or parse needs to turn
an output-type *name* (carried on the wire as ``output_type_name``) into a
concrete Pydantic model — there is no worker-startup registration step anymore.

Keys are class names; values are the model classes. ``TranscriptAnalysisResult``
(pbook) is optional: when pbook is not installed the key is simply absent,
mirroring the former conditional registration that worker startup once performed.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from forge.eval.models import JudgeVerdict
from forge.models import (
    ConflictResolutionResponse,
    ExplorationResponse,
    ExtractionResult,
    LLMResponse,
    Plan,
    SanityCheckResponse,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Final

    from pydantic import BaseModel

_OUTPUT_TYPES: dict[str, type[BaseModel]] = {
    "LLMResponse": LLMResponse,
    "Plan": Plan,
    "ExplorationResponse": ExplorationResponse,
    "SanityCheckResponse": SanityCheckResponse,
    "ConflictResolutionResponse": ConflictResolutionResponse,
    "ExtractionResult": ExtractionResult,
    "JudgeVerdict": JudgeVerdict,
}

try:
    from pbook.ingestion_prompts import TranscriptAnalysisResult

    _OUTPUT_TYPES["TranscriptAnalysisResult"] = TranscriptAnalysisResult
except ImportError:
    # pbook not installed — ingestion's output type is unavailable, exactly as
    # worker.py's former conditional registration handled it.
    pass

OUTPUT_TYPES: Final[Mapping[str, type[BaseModel]]] = MappingProxyType(_OUTPUT_TYPES)


def resolve_output_type(name: str) -> type[BaseModel]:
    """Resolve an output-type name to its Pydantic model class.

    Raises ``KeyError`` naming the unknown type (and the known ones) — the same
    failure the old registry raised on a missing key, kept clear for operators.
    """
    try:
        return OUTPUT_TYPES[name]
    except KeyError:
        known = ", ".join(sorted(OUTPUT_TYPES))
        raise KeyError(f"Unknown output type {name!r}. Known types: {known}") from None
