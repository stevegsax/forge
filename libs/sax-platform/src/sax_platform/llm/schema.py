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
"""

from typing import Any

from pydantic import BaseModel


def _close_objects(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _close_objects(v) for k, v in node.items()}
        if out.get("type") == "object" and "additionalProperties" not in out:
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [_close_objects(item) for item in node]
    return node


def to_json_schema(output_type: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for ``output_type`` with every object node closed."""
    schema = _close_objects(output_type.model_json_schema())
    if not isinstance(schema, dict):  # pragma: no cover - model_json_schema returns dict
        raise TypeError(f"unexpected schema shape for {output_type!r}")
    return schema


def to_output_format(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON schema as the ``output_config.format`` payload."""
    return {"type": "json_schema", "schema": schema}
