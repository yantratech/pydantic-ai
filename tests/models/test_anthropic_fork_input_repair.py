# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip('anthropic')

from pydantic_ai.messages import INVALID_JSON_KEY
from pydantic_ai.models.anthropic import (
    _coerce_anthropic_code_execution_tool_input,
    _repair_anthropic_code_execution_tool_input,
)


@pytest.mark.parametrize(
    ('input_value', 'expected'),
    [
        ('{"command":"printf \\"ok\\""', {'command': 'printf "ok"'}),
        ({INVALID_JSON_KEY: '{"command":[]]'}, {INVALID_JSON_KEY: '{"command":[]]'}),
        ({INVALID_JSON_KEY: '{"command": bare'}, {INVALID_JSON_KEY: '{"command": bare'}),
        (None, None),
    ],
)
def test_anthropic_code_execution_input_parser_edges(input_value: Any, expected: Any):
    assert _coerce_anthropic_code_execution_tool_input('bash_code_execution', input_value) == expected


@pytest.mark.parametrize(
    ('input_value', 'expected'),
    [
        ({INVALID_JSON_KEY: '{"command":"create"'}, {'command': 'create'}),
        ({'command': 'create'}, None),
        (None, None),
    ],
)
def test_anthropic_streaming_input_repair_defensive_shapes(input_value: Any, expected: dict[str, Any] | None):
    assert _repair_anthropic_code_execution_tool_input(input_value) == expected
