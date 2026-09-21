from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from pydantic_ai import Agent, RunContext, ToolOutput
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')]


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize(
    'settings',
    [
        {'thinking': True},
        {'bedrock_additional_model_requests_fields': {'thinking': {'type': 'adaptive'}}},
    ],
    ids=['unified', 'provider'],
)
async def test_buffered_adaptive_thinking(
    allow_model_requests: None, bedrock_provider: BedrockProvider, mocker: MockerFixture, settings: BedrockModelSettings
):
    model = BedrockConverseModel('eu.anthropic.claude-opus-4-7', provider=bedrock_provider, settings=settings)
    converse = mocker.patch.object(model.client, 'converse')
    converse.side_effect = [
        {
            'output': {'message': {'role': 'assistant', 'content': content}},
            'stopReason': stop,
            'usage': {'inputTokens': 1, 'outputTokens': 1},
        }
        for content, stop in [
            ([{'text': 'Done already.'}], 'end_turn'),
            ([{'toolUse': {'toolUseId': 'draft', 'name': 'final_result', 'input': {'value': 'invalid'}}}], 'tool_use'),
            (
                [
                    {
                        'toolUse': {
                            'toolUseId': 'edit',
                            'name': 'patch_final_result_buffer',
                            'input': {
                                'operations': [{'op': 'replace', 'path': '/value', 'value': 42}],
                            },
                        }
                    }
                ],
                'tool_use',
            ),
            (
                [{'toolUse': {'toolUseId': 'submit', 'name': 'final_result', 'input': {'submit_as_final': True}}}],
                'tool_use',
            ),
        ]
    ]
    agent = Agent(model, output_type=ToolOutput(Answer, buffered=True))
    result = await agent.run('Return the answer using the buffer.')
    assert result.output == Answer(value=42)
    assert converse.call_count == 4
    for call in converse.call_args_list:
        request = call.kwargs
        assert request['additionalModelRequestFields']['thinking'] == {'type': 'adaptive'}
        assert request['toolConfig']['toolChoice'] == {'auto': {}}
        assert {tool['toolSpec']['name'] for tool in request['toolConfig']['tools']} == {
            'final_result',
            'read_final_result_buffer',
            'patch_final_result_buffer',
        }
    messages = converse.call_args.kwargs['messages']
    assert json.loads(messages[4]['content'][0]['toolResult']['content'][0]['text'])['status'] == 'invalid'
    assert json.loads(messages[6]['content'][0]['toolResult']['content'][0]['text'])['status'] == 'valid'


async def test_buffered_adaptive_thinking_text_exhaustion(
    allow_model_requests: None, bedrock_provider: BedrockProvider, mocker: MockerFixture
):
    model = BedrockConverseModel('eu.anthropic.claude-opus-4-7', provider=bedrock_provider, settings={'thinking': True})
    converse = mocker.patch.object(
        model.client,
        'converse',
        return_value={
            'output': {'message': {'role': 'assistant', 'content': [{'text': 'Unstructured answer'}]}},
            'stopReason': 'end_turn',
            'usage': {'inputTokens': 1, 'outputTokens': 1},
        },
    )
    agent = Agent(model, output_type=ToolOutput(Answer, buffered=True), retries=1)
    with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries'):
        await agent.run('Return the answer.')
    assert converse.call_count == 2


async def test_buffered_adaptive_thinking_stream(
    allow_model_requests: None, bedrock_provider: BedrockProvider, mocker: MockerFixture
):
    model = BedrockConverseModel('eu.anthropic.claude-opus-4-7', provider=bedrock_provider, settings={'thinking': True})
    converse = mocker.patch.object(model.client, 'converse_stream')
    converse.side_effect = [
        {
            'stream': iter(
                [
                    {'messageStart': {'role': 'assistant'}},
                    {
                        'contentBlockStart': {
                            'contentBlockIndex': 0,
                            'start': {
                                'toolUse': {'toolUseId': tool_id, 'name': 'final_result'},
                            },
                        }
                    },
                    {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'toolUse': {'input': json.dumps(args)}}}},
                    {'contentBlockStop': {'contentBlockIndex': 0}},
                    {'messageStop': {'stopReason': 'tool_use'}},
                ]
            )
        }
        for tool_id, args in [('draft', {'value': 42}), ('submit', {'submit_as_final': True})]
    ]
    agent = Agent(model, output_type=ToolOutput(Answer, buffered=True))
    async with agent.run_stream('Return the answer using the buffer.') as result:
        assert await result.get_output() == Answer(value=42)
    assert converse.call_count == 2
    for call in converse.call_args_list:
        assert call.kwargs['additionalModelRequestFields']['thinking'] == {'type': 'adaptive'}
        assert call.kwargs['toolConfig']['toolChoice'] == {'auto': {}}


@pytest.mark.parametrize('buffered', [False, True], ids=['ordinary', 'buffered'])
async def test_adaptive_output_token_count(
    allow_model_requests: None, bedrock_provider: BedrockProvider, mocker: MockerFixture, buffered: bool
):

    model = BedrockConverseModel('eu.anthropic.claude-opus-4-7', provider=bedrock_provider, settings={'thinking': True})
    counter = mocker.patch.object(model.client, 'count_tokens', return_value={'inputTokens': 12})
    mocker.patch.object(
        model.client,
        'converse',
        return_value={
            'output': {
                'message': {
                    'role': 'assistant',
                    'content': [
                        {
                            'toolUse': {
                                'toolUseId': 'submit',
                                'name': 'final_result',
                                'input': {'value': 42, 'submit_as_final': True} if buffered else {'value': 42},
                            }
                        }
                    ],
                }
            },
            'stopReason': 'tool_use',
            'usage': {'inputTokens': 12, 'outputTokens': 1},
        },
    )
    hooks = Hooks()

    @hooks.on.before_model_request
    async def count(ctx: RunContext[None], request: ModelRequestContext) -> ModelRequestContext:
        usage = await model.count_tokens(request.messages, request.model_settings, request.model_request_parameters)
        assert usage.input_tokens == 12
        return request

    result = await Agent(model, output_type=ToolOutput(Answer, buffered=buffered), capabilities=[hooks]).run('Answer')
    assert result.output == Answer(value=42)
    payload = counter.call_args.kwargs['input']['converse']
    assert payload['additionalModelRequestFields']['thinking'] == {'type': 'adaptive'}
    assert payload['toolConfig']['toolChoice'] == {'auto': {}}


@pytest.mark.parametrize('provider_type', ['enabled', 'disabled'])
def test_provider_thinking_overrides_unified(bedrock_provider: BedrockProvider, provider_type: str):

    model = BedrockConverseModel('eu.anthropic.claude-opus-4-7', provider=bedrock_provider)
    parameters = ModelRequestParameters(
        output_mode='tool',
        allow_text_output=False,
        output_tools=[ToolDefinition(name='final_result', parameters_json_schema={'type': 'object'})],
    )
    # Preserve the existing rejection whenever the effective request is not adaptive.
    with pytest.raises(UserError, match='Bedrock does not support thinking and output tools'):
        model.prepare_request(
            BedrockModelSettings(
                thinking=True,
                bedrock_additional_model_requests_fields={'thinking': {'type': provider_type}},
            ),
            parameters,
        )


@pytest.mark.parametrize('choice', ['required', ['edit']])
async def test_adaptive_thinking_rejects_explicit_forcing(
    allow_model_requests: None, bedrock_provider: BedrockProvider, choice: Literal['required'] | list[str]
):
    model = BedrockConverseModel('eu.anthropic.claude-opus-4-7', provider=bedrock_provider)
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='edit', parameters_json_schema={'type': 'object'})],
    )
    with pytest.raises(UserError, match='Bedrock does not support forcing specific tools with thinking mode'):
        await model.request(
            [],
            BedrockModelSettings(thinking=True, tool_choice=choice),
            parameters,
        )
