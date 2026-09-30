"""Schema customization must preserve custom envelope requirements and non-object schemas."""

from __future__ import annotations

import pytest
from pydantic import GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from pydantic_ai._event_registry import unknown_event_json_schema


@pytest.mark.parametrize(
    'schema,expected',
    [
        pytest.param(
            {
                'type': 'object',
                'properties': {'name': {'type': 'string'}, 'data': dict[str, object]()},
                'required': ['name'],
            },
            {
                'allOf': [
                    {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']},
                    {'type': 'object', 'additionalProperties': dict[str, object]()},
                ]
            },
        ),
        pytest.param({'type': 'string'}, {'type': 'string'}),
    ],
    ids=['required-envelope-field', 'non-object-custom-schema'],
)
def test_unknown_event_custom_schema(schema: JsonSchemaValue, expected: JsonSchemaValue):
    class Handler(GetJsonSchemaHandler):
        def __call__(self, schema_or_field: object) -> JsonSchemaValue:
            return schema

        def resolve_ref_schema(self, maybe_ref_json_schema: JsonSchemaValue) -> JsonSchemaValue:
            return maybe_ref_json_schema

    assert unknown_event_json_schema(core_schema.any_schema(), Handler()) == expected
