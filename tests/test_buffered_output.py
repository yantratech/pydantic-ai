# pyright: reportPrivateUsage=false

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from pydantic import TypeAdapter, ValidationError

from pydantic_ai import (
    ModelRequest,
    ModelRetry,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    _output,
    _tool_execution,
)
from pydantic_ai._run_context import OutputBufferState
from pydantic_ai.exceptions import ToolRetryError


def test_restore_output_buffers_filters_and_restores_validation_errors():
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(int).validate_python('not-an-integer')
    error_details = exc_info.value.errors(include_url=False, include_context=False)

    message = ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='invalid_revision',
                content={
                    'status': 'valid',
                    'tool_name': 'invalid_revision',
                    'revision': True,
                    'buffer': {'value': 'ignored'},
                },
            ),
            ToolReturnPart(
                tool_name='string_error',
                content={
                    'status': 'invalid',
                    'tool_name': 'string_error',
                    'revision': 2,
                    'buffer': {'value': 'latest'},
                    'errors': 'plain retry',
                },
            ),
            ToolReturnPart(
                tool_name='string_error',
                content={
                    'status': 'valid',
                    'tool_name': 'string_error',
                    'revision': 1,
                    'buffer': {'value': 'stale'},
                },
            ),
            ToolReturnPart(
                tool_name='structured_error',
                content={
                    'status': 'invalid',
                    'tool_name': 'structured_error',
                    'revision': 1,
                    'buffer': {'value': 'structured'},
                    'errors': error_details,
                },
            ),
            ToolReturnPart(
                tool_name='malformed_error',
                content={
                    'status': 'invalid',
                    'tool_name': 'malformed_error',
                    'revision': 1,
                    'buffer': {'value': 'kept'},
                    'errors': [{}],
                },
            ),
        ]
    )

    buffers = _output.restore_output_buffers(
        [message], frozenset({'invalid_revision', 'string_error', 'structured_error', 'malformed_error'})
    )

    assert 'invalid_revision' not in buffers
    assert buffers['string_error'].raw_args == {'value': 'latest'}
    assert buffers['string_error'].revision == 2
    assert buffers['string_error'].validation_error is not None
    assert buffers['string_error'].validation_error.tool_name == 'string_error'
    assert buffers['string_error'].validation_error.content == 'plain retry'
    assert buffers['structured_error'].raw_args == {'value': 'structured'}
    assert buffers['structured_error'].validation_error is not None
    assert buffers['structured_error'].validation_error.tool_name == 'structured_error'
    assert buffers['structured_error'].validation_error.content == error_details
    assert buffers['malformed_error'] == OutputBufferState(
        raw_args={'value': 'kept'}, validation_error=None, revision=1
    )


def test_buffered_output_json_patch_applies_supported_operations():
    replacement = {'replacement': {'value': 1}}
    replaced = _output._apply_json_patch({'old': True}, [{'op': 'replace', 'path': '', 'value': replacement}])
    assert replaced == replacement
    assert replaced is not replacement
    assert _output._apply_json_patch({'old': True}, [{'op': 'remove', 'path': ''}]) == {}

    patched = _output._apply_json_patch(
        {
            'details': {'replace': 'old', 'remove': True},
            'items': ['a', 'b'],
            'nested': [{'value': 1}],
        },
        [
            {'op': 'replace', 'path': '/details/replace', 'value': 'new'},
            {'op': 'remove', 'path': '/details/remove'},
            {'op': 'add', 'path': '/items/-', 'value': 'tail'},
            {'op': 'add', 'path': '/items/1', 'value': 'inserted'},
            {'op': 'replace', 'path': '/items/0', 'value': 'first'},
            {'op': 'remove', 'path': '/items/2'},
            {'op': 'replace', 'path': '/nested/0/value', 'value': 2},
        ],
    )

    assert patched == {
        'details': {'replace': 'new'},
        'items': ['first', 'inserted', 'tail'],
        'nested': [{'value': 2}],
    }


@pytest.mark.parametrize(
    ('document', 'operations', 'message'),
    cast(
        list[tuple[dict[str, Any], Any, str]],
        [
            ({}, {}, '`operations` must be a list of JSON Patch operations.'),
            ({}, [None], 'Each JSON Patch operation must be an object.'),
            (
                {},
                [{'op': 1, 'path': '/value'}],
                'Each JSON Patch operation must include string `op` and `path` fields.',
            ),
            (
                {},
                [{'op': 'copy', 'path': '/value'}],
                "Unsupported JSON Patch operation 'copy'; expected `add`, `replace`, or `remove`.",
            ),
            ({}, [{'op': 'add', 'path': '/value'}], "JSON Patch operation 'add' requires a `value` field."),
            (
                {},
                [{'op': 'add', 'path': 'value', 'value': 1}],
                "JSON Patch path must be empty or start with `/`, got 'value'.",
            ),
            (
                {},
                [{'op': 'replace', 'path': '', 'value': []}],
                'Root JSON Patch replacement must be an object.',
            ),
            (
                {},
                [{'op': 'replace', 'path': '/missing/value', 'value': 1}],
                "JSON Patch path '/missing/value' does not exist.",
            ),
            (
                {'items': []},
                [{'op': 'replace', 'path': '/items/not-an-index/value', 'value': 1}],
                "JSON Patch path '/items/not-an-index/value' does not exist.",
            ),
            (
                {'items': []},
                [{'op': 'replace', 'path': '/items/0/value', 'value': 1}],
                "JSON Patch path '/items/0/value' does not exist.",
            ),
            (
                {'value': 1},
                [{'op': 'replace', 'path': '/value/nested/key', 'value': 2}],
                "JSON Patch path '/value/nested/key' cannot traverse a non-container value.",
            ),
            ({}, [{'op': 'remove', 'path': '/missing'}], "JSON Patch path '/missing' does not exist."),
            ({}, [{'op': 'replace', 'path': '/missing', 'value': 1}], "JSON Patch path '/missing' does not exist."),
            (
                {'items': []},
                [{'op': 'add', 'path': '/items/not-an-index', 'value': 1}],
                "JSON Patch path '/items/not-an-index' does not identify a list index.",
            ),
            (
                {'items': []},
                [{'op': 'add', 'path': '/items/1', 'value': 1}],
                "JSON Patch path '/items/1' does not identify an insertion point.",
            ),
            (
                {'items': []},
                [{'op': 'replace', 'path': '/items/0', 'value': 1}],
                "JSON Patch path '/items/0' does not exist.",
            ),
            ({'items': []}, [{'op': 'remove', 'path': '/items/0'}], "JSON Patch path '/items/0' does not exist."),
        ],
    ),
)
def test_buffered_output_json_patch_rejects_invalid_operations(document: dict[str, Any], operations: Any, message: str):
    with pytest.raises(ModelRetry) as exc_info:
        _output._apply_json_patch(document, operations)

    assert exc_info.value.message == message


def test_buffered_output_json_patch_rejects_scalar_parent(monkeypatch: pytest.MonkeyPatch):
    def scalar_parent(_document: dict[str, Any], _parts: list[str], _path: str) -> int:
        return 1

    monkeypatch.setattr(_output, '_json_pointer_parent', scalar_parent)

    with pytest.raises(ModelRetry, match='cannot update a non-container value'):
        _output._apply_json_patch({}, [{'op': 'add', 'path': '/value', 'value': 1}])


def test_buffered_output_json_patch_rejects_non_object_internal_result(monkeypatch: pytest.MonkeyPatch):
    def non_object_result(_document: dict[str, Any], _operation: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], [])

    monkeypatch.setattr(_output, '_apply_json_patch_operation', non_object_result)

    with pytest.raises(ModelRetry, match='Buffered output arguments must remain an object after patching'):
        _output._apply_json_patch({}, [{}])


async def test_buffered_output_editor_context_and_validation_errors():
    toolset = _output.BufferedOutputEditorToolset(('final_result',), cast(Any, object()))
    tool = cast(Any, None)

    assert toolset.label == "the agent's buffered output editor tools"

    missing_state_context = cast(Any, SimpleNamespace(_output_buffers=None))
    with pytest.raises(ModelRetry, match='Buffered output state is not available for this run'):
        await toolset.call_tool('read_final_result_buffer', {}, missing_state_context, tool)

    buffers: dict[str, OutputBufferState] = {}
    run_context = cast(Any, SimpleNamespace(_output_buffers=buffers))
    assert await toolset.call_tool('read_final_result_buffer', {}, run_context, tool) == {
        'status': 'missing',
        'tool_name': 'final_result',
        'revision': 0,
        'buffer': None,
    }

    buffers['final_result'] = OutputBufferState(
        raw_args={'value': 1},
        validation_error=RetryPromptPart(tool_name='final_result', content='invalid buffer'),
        revision=2,
    )
    assert await toolset.call_tool('read_final_result_buffer', {}, run_context, tool) == {
        'status': 'current',
        'tool_name': 'final_result',
        'revision': 2,
        'buffer': {'value': 1},
        'errors': 'invalid buffer',
    }

    buffer = OutputBufferState(raw_args={'value': 'wrong'})
    managerless_context = cast(Any, SimpleNamespace(tool_manager=None))
    with pytest.raises(ModelRetry, match='Tool manager is not available for buffered output validation'):
        await toolset._validate_buffer('final_result', managerless_context, buffer)

    retry = RetryPromptPart(tool_name='final_result', content='tool retry')
    validate_output_tool_call = AsyncMock(side_effect=ToolRetryError(retry))
    manager_context = cast(
        Any, SimpleNamespace(tool_manager=SimpleNamespace(validate_output_tool_call=validate_output_tool_call))
    )
    await toolset._validate_buffer('final_result', manager_context, buffer)
    assert buffer.validation_error is retry

    validate_output_tool_call.side_effect = ModelRetry('model retry')
    await toolset._validate_buffer('final_result', manager_context, buffer)
    assert buffer.validation_error is not None
    assert buffer.validation_error.tool_name == 'final_result'
    assert buffer.validation_error.content == 'model retry'

    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(int).validate_python('not-an-integer')
    validate_output_tool_call.side_effect = exc_info.value
    await toolset._validate_buffer('final_result', manager_context, buffer)
    assert buffer.validation_error is not None
    assert buffer.validation_error.tool_name == 'final_result'
    assert buffer.validation_error.content == exc_info.value.errors(include_url=False, include_context=False)

    validate_output_tool_call.side_effect = None
    await toolset._validate_buffer('final_result', manager_context, buffer)
    assert buffer.validation_error is None


def test_tool_execution_buffer_validation_error_preserves_call_identity():
    call = ToolCallPart(tool_name='final_result', args={}, tool_call_id='call-id')
    retry = RetryPromptPart(tool_name='other', content='tool retry', tool_call_id='other-id')

    tool_retry_part = _tool_execution._buffer_validation_error(call, ToolRetryError(retry))
    assert tool_retry_part.tool_name == 'final_result'
    assert tool_retry_part.tool_call_id == 'call-id'
    assert tool_retry_part.content == 'tool retry'

    model_retry_part = _tool_execution._buffer_validation_error(call, ModelRetry('model retry'))
    assert model_retry_part.tool_name == 'final_result'
    assert model_retry_part.tool_call_id == 'call-id'
    assert model_retry_part.content == 'model retry'

    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(int).validate_python('not-an-integer')
    validation_retry_part = _tool_execution._buffer_validation_error(call, exc_info.value)
    assert validation_retry_part.tool_name == 'final_result'
    assert validation_retry_part.tool_call_id == 'call-id'
    assert validation_retry_part.content == exc_info.value.errors(include_url=False, include_context=False)
