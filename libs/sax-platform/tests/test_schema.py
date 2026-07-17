"""Tests for structured-output schema derivation."""

from pydantic import BaseModel

from sax_platform.llm.schema import to_json_schema, to_output_format


class Inner(BaseModel):
    name: str


class Outer(BaseModel):
    count: int
    inner: Inner
    tags: list[str]


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


class TestToOutputFormat:
    def test_wraps_as_json_schema_format(self) -> None:
        schema = to_json_schema(Inner)
        fmt = to_output_format(schema)
        assert fmt == {"type": "json_schema", "schema": schema}
