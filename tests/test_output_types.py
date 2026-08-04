"""Tests for forge.output_types — the frozen output-type registry (D90/D93)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from sax_platform.llm.schema import to_json_schema

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

    def test_registry_is_not_empty(self) -> None:
        # Guards the sweep below: a parametrized sweep over an empty mapping
        # would report "passed" while checking nothing.
        assert set(OUTPUT_TYPES) >= set(_CORE_NAMES)

    @pytest.mark.parametrize("name", sorted(OUTPUT_TYPES))
    def test_every_registered_type_is_representable(self, name: str) -> None:
        """Every wire model must survive schema derivation.

        Issue #47: ``ExplorationResponse`` carried a ``dict[str, str]``, which
        renders an object-valued ``additionalProperties`` the API rejects at
        submit time. Deriving the schema here turns that class of defect into a
        test failure — the registry is the full set of models that reach the
        wire, pbook's optional entry included.
        """
        schema = to_json_schema(OUTPUT_TYPES[name])
        assert _open_keyed_paths(schema, name) == []


def _open_keyed_paths(node: object, path: str) -> list[str]:
    """Every path under *node* whose ``additionalProperties`` is not ``False``.

    Belt-and-braces against the derivation itself: ``to_json_schema`` raises on
    these, so a non-empty result here would mean the guard had a hole.
    """
    if isinstance(node, dict):
        found = [
            f"{path}.additionalProperties"
            for key, value in node.items()
            if key == "additionalProperties" and value is not False
        ]
        return found + [p for k, v in node.items() for p in _open_keyed_paths(v, f"{path}.{k}")]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _open_keyed_paths(v, f"{path}[{i}]")]
    return []


class TestPbookOptionalEntry:
    def test_transcript_analysis_result_present_when_pbook_installed(self) -> None:
        # pbook is a required workspace dependency; the optional key is present,
        # mirroring worker.py's former conditional registration.
        try:
            from pbook.ingestion_prompts import TranscriptAnalysisResult
        except ImportError:
            pytest.skip("pbook not installed")

        assert OUTPUT_TYPES["TranscriptAnalysisResult"] is TranscriptAnalysisResult
