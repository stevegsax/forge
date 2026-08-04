"""Pure derivation of structured-output schemas from Pydantic models.

Both lanes send the same ``output_config.format`` payload; this module is
the single place that shape is produced. ``messages.parse`` is deliberately
not used on the sync lane: it couples transport and parsing, so a refusal
or truncation would surface as (or hide behind) a parsing failure before
``stop_reason`` could be classified. Deriving the schema ourselves and
validating after classification keeps the failure surface typed.

The API requires ``additionalProperties: false`` on every object node and
rejects recursive schemas; wire models here are expected to be simple
(flat-ish, no recursion). We add ``additionalProperties`` recursively and
leave everything else to the API's own validation.

One shape cannot be repaired by closing it: a dict-typed field. Pydantic
renders ``dict[str, str]`` as ``{"type": "object", "additionalProperties":
{"type": "string"}}``, and the API rejects a non-``false``
``additionalProperties`` outright. Closing it would be worse than the
rejection — the model would stay schema-valid while the LLM could only ever
return ``{}``. So such a node raises :class:`UnrepresentableSchemaError`
here, turning a production API rejection into an import/test-time failure.
"""

from typing import Any

from pydantic import BaseModel

__all__ = [
    "UnrepresentableSchemaError",
    "to_json_schema",
    "to_output_format",
]


class UnrepresentableSchemaError(ValueError):
    """A wire model has a field structured outputs cannot express.

    Attributes:
        path: Dotted path to the offending node, rooted at the model name.
        value: The ``additionalProperties`` value found there.
    """

    def __init__(self, *, path: str, value: object) -> None:
        self.path = path
        self.value = value
        super().__init__(
            f"{path}: additionalProperties is {value!r}, but Anthropic structured "
            "outputs accept only 'additionalProperties: false'. Dict-typed fields "
            "(dict[str, str], dict[str, Any], ...) are unrepresentable in structured "
            "outputs; use a list of key/value pair models instead — e.g. replace "
            "'params: dict[str, str]' with 'params: list[Param]' where Param has "
            "'name' and 'value' fields, and collapse it to a dict at the consumer."
        )


def _close_objects(node: Any, path: str) -> Any:
    if isinstance(node, dict):
        declared = node.get("additionalProperties", False)
        if declared is not False:
            raise UnrepresentableSchemaError(path=f"{path}.additionalProperties", value=declared)
        out = {k: _close_objects(v, f"{path}.{k}") for k, v in node.items()}
        if out.get("type") == "object":
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [_close_objects(item, f"{path}[{i}]") for i, item in enumerate(node)]
    return node


def to_json_schema(output_type: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for ``output_type`` with every object node closed.

    Raises:
        UnrepresentableSchemaError: the model contains a dict-typed (open-keyed)
            object node, which structured outputs cannot express.
    """
    schema = _close_objects(output_type.model_json_schema(), output_type.__name__)
    if not isinstance(schema, dict):  # pragma: no cover - model_json_schema returns dict
        raise TypeError(f"unexpected schema shape for {output_type!r}")
    return schema


def to_output_format(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON schema as the ``output_config.format`` payload."""
    return {"type": "json_schema", "schema": schema}
