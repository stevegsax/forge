"""Tests for structured-output schema derivation."""

from typing import Any

import pytest
from pydantic import BaseModel

from sax_platform.llm.schema import (
    UnrepresentableSchemaError,
    to_json_schema,
    to_output_format,
)


class Inner(BaseModel):
    name: str


class Outer(BaseModel):
    count: int
    inner: Inner
    tags: list[str]


class OpenKeyed(BaseModel):
    """The shape the API rejects: dict renders an object-valued additionalProperties."""

    params: dict[str, str]


class OpenKeyedAny(BaseModel):
    """``dict[str, Any]`` renders ``additionalProperties: true`` — equally unrepresentable."""

    payload: dict[str, Any]


class NestedOpenKeyed(BaseModel):
    """The offending node is one level down, inside a ``$defs`` entry."""

    items: list[OpenKeyed]


class TestToJsonSchema:
    def test_root_object_closed(self) -> None:
        schema = to_json_schema(Outer)
        assert schema["additionalProperties"] is False

    def test_nested_defs_closed(self) -> None:
        schema = to_json_schema(Outer)
        assert schema["$defs"]["Inner"]["additionalProperties"] is False

    def test_required_fields_preserved(self) -> None:
        schema = to_json_schema(Outer)
        assert set(schema["required"]) == {"count", "inner", "tags"}

    def test_non_object_nodes_untouched(self) -> None:
        schema = to_json_schema(Outer)
        assert schema["properties"]["tags"] == {
            "items": {"type": "string"},
            "title": "Tags",
            "type": "array",
        }


class TestUnrepresentableModels:
    """A dict-typed field is a build-time error, not a production API rejection.

    Closing such a node with ``additionalProperties: false`` would leave the
    schema valid while making ``{}`` the only value the LLM could return, so
    the derivation refuses rather than silently destroying the field.
    """

    def test_object_valued_additional_properties_raises(self) -> None:
        with pytest.raises(UnrepresentableSchemaError) as exc_info:
            to_json_schema(OpenKeyed)
        assert exc_info.value.path == "OpenKeyed.properties.params.additionalProperties"
        assert exc_info.value.value == {"type": "string"}

    def test_true_valued_additional_properties_raises(self) -> None:
        with pytest.raises(UnrepresentableSchemaError) as exc_info:
            to_json_schema(OpenKeyedAny)
        assert exc_info.value.path == "OpenKeyedAny.properties.payload.additionalProperties"
        assert exc_info.value.value is True

    def test_message_names_the_path_and_the_fix(self) -> None:
        with pytest.raises(
            UnrepresentableSchemaError,
            match=r"OpenKeyed\.properties\.params\.additionalProperties",
        ) as exc_info:
            to_json_schema(OpenKeyed)
        assert "list of key/value pair models" in str(exc_info.value)

    def test_nested_definition_is_reached(self) -> None:
        with pytest.raises(UnrepresentableSchemaError) as exc_info:
            to_json_schema(NestedOpenKeyed)
        assert exc_info.value.path == (
            "NestedOpenKeyed.$defs.OpenKeyed.properties.params.additionalProperties"
        )

    def test_pair_list_replacement_is_representable(self) -> None:
        """The prescribed fix produces a schema every node of which is closed."""

        class Param(BaseModel):
            name: str
            value: str

        class PairList(BaseModel):
            params: list[Param]

        schema = to_json_schema(PairList)
        assert schema["$defs"]["Param"]["additionalProperties"] is False
        assert schema["properties"]["params"]["type"] == "array"


class TestToOutputFormat:
    def test_wraps_as_json_schema_format(self) -> None:
        schema = to_json_schema(Inner)
        fmt = to_output_format(schema)
        assert fmt == {"type": "json_schema", "schema": schema}
