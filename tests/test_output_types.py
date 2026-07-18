"""Tests for forge.output_types — the frozen output-type registry (D90/D93)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from forge.output_types import OUTPUT_TYPES, resolve_output_type

_CORE_NAMES = [
    "LLMResponse",
    "Plan",
    "ExplorationResponse",
    "SanityCheckResponse",
    "ConflictResolutionResponse",
    "ExtractionResult",
    "JudgeVerdict",
]


class TestOutputTypes:
    @pytest.mark.parametrize("name", _CORE_NAMES)
    def test_core_type_present_and_is_model(self, name: str) -> None:
        assert name in OUTPUT_TYPES
        assert issubclass(OUTPUT_TYPES[name], BaseModel)

    def test_keys_match_class_names(self) -> None:
        for name, cls in OUTPUT_TYPES.items():
            assert cls.__name__ == name

    def test_mapping_is_read_only(self) -> None:
        # MappingProxyType rejects mutation — the registry is frozen.
        with pytest.raises(TypeError):
            OUTPUT_TYPES["Nope"] = BaseModel  # type: ignore[index]

    def test_resolve_returns_the_class(self) -> None:
        from forge.models import LLMResponse

        assert resolve_output_type("LLMResponse") is LLMResponse

    def test_resolve_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown output type"):
            resolve_output_type("NopeType")

    def test_transcript_analysis_result_present_when_pbook_installed(self) -> None:
        # pbook is a required workspace dependency; the optional key is present,
        # mirroring worker.py's former conditional registration.
        try:
            from pbook.ingestion_prompts import TranscriptAnalysisResult
        except ImportError:
            pytest.skip("pbook not installed")

        assert OUTPUT_TYPES["TranscriptAnalysisResult"] is TranscriptAnalysisResult
