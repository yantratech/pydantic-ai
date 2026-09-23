"""GPT-6 Sol and Luna request-wire coverage."""

from __future__ import annotations

import pytest

pytest.importorskip('openai')

from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.openai import OpenAIProvider

from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

pytestmark = pytest.mark.anyio


class Answer(BaseModel):
    answer: str


@pytest.mark.parametrize('model_name', ['gpt-6-sol', 'gpt-6-luna'])
async def test_gpt6_request_wire_with_tool_and_reasoning(allow_model_requests: None, model_name: str) -> None:
    response = response_message(
        [
            ResponseOutputMessage.model_validate(
                {
                    'id': 'output-1',
                    'content': [{'text': 'done', 'type': 'output_text', 'annotations': []}],
                    'role': 'assistant',
                    'status': 'completed',
                    'type': 'message',
                }
            )
        ]
    )
    client = MockOpenAIResponses.create_mock(response)
    model = OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=client))
    agent = Agent(
        model,
        model_settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort='max',
            openai_reasoning_mode='standard',
            openai_reasoning_context='all_turns',
        ),
    )

    @agent.tool_plain
    def lookup(query: str) -> str:
        return query

    await agent.run('hello')
    request = get_mock_responses_kwargs(client)[0]
    assert request['model'] == model_name
    assert request['reasoning'] == {'effort': 'max', 'mode': 'standard'}
    assert any(tool.get('name') == 'lookup' for tool in request['tools'])


@pytest.mark.parametrize('model_name', ['gpt-6-sol', 'gpt-6-luna'])
async def test_gpt6_function_tool_round_trip(allow_model_requests: None, model_name: str) -> None:
    replies = [
        response_message(
            [
                ResponseFunctionToolCall(
                    arguments='{"query":"hello"}',
                    call_id='call_1',
                    name='lookup',
                    type='function_call',
                )
            ]
        ),
        response_message(
            [
                ResponseOutputMessage.model_validate(
                    {
                        'id': 'output-2',
                        'content': [{'text': 'done', 'type': 'output_text', 'annotations': []}],
                        'role': 'assistant',
                        'status': 'completed',
                        'type': 'message',
                    }
                )
            ]
        ),
    ]
    client = MockOpenAIResponses.create_mock(replies)
    model = OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=client))
    agent = Agent(model)
    calls: list[str] = []

    @agent.tool_plain
    def lookup(query: str) -> str:
        calls.append(query)
        return 'found'

    result = await agent.run('hello')
    assert result.output == 'done'
    assert calls == ['hello']
    requests = get_mock_responses_kwargs(client)
    assert [request['model'] for request in requests] == [model_name, model_name]
    assert {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'found'} in requests[1]['input']


@pytest.mark.parametrize('model_name', ['gpt-6-sol', 'gpt-6-luna'])
async def test_gpt6_native_output_request(allow_model_requests: None, model_name: str) -> None:
    response = response_message(
        [
            ResponseOutputMessage.model_validate(
                {
                    'id': 'output-1',
                    'content': [{'text': '{"answer":"ok"}', 'type': 'output_text', 'annotations': []}],
                    'role': 'assistant',
                    'status': 'completed',
                    'type': 'message',
                }
            )
        ]
    )
    client = MockOpenAIResponses.create_mock(response)
    model = OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=client))
    result = await Agent(model, output_type=NativeOutput(Answer)).run('answer')
    assert result.output.answer == 'ok'
    request = get_mock_responses_kwargs(client)[0]
    assert request['model'] == model_name
    assert request['text']['format']['type'] == 'json_schema'


@pytest.mark.parametrize('model_name', ['gpt-6-sol', 'gpt-6-luna'])
async def test_gpt6_stream_request(allow_model_requests: None, model_name: str) -> None:
    from openai.types import responses as resp

    base = resp.Response(
        id='resp_1',
        model=model_name,
        object='response',
        created_at=1704067200,
        output=[],
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
    )
    stream: list[resp.ResponseStreamEvent] = [
        resp.ResponseCreatedEvent(response=base, type='response.created', sequence_number=0),
        resp.ResponseOutputItemAddedEvent(
            item=ResponseOutputMessage(
                id='msg_1',
                content=[],
                role='assistant',
                status='in_progress',
                type='message',
            ),
            output_index=0,
            type='response.output_item.added',
            sequence_number=1,
        ),
        resp.ResponseTextDeltaEvent(
            content_index=0,
            delta='done',
            item_id='msg_1',
            output_index=0,
            type='response.output_text.delta',
            sequence_number=2,
            logprobs=[],
        ),
        resp.ResponseTextDoneEvent(
            content_index=0,
            item_id='msg_1',
            output_index=0,
            text='done',
            type='response.output_text.done',
            sequence_number=3,
            logprobs=[],
        ),
        resp.ResponseCompletedEvent(
            response=base.model_copy(update={'status': 'completed'}),
            type='response.completed',
            sequence_number=4,
        ),
    ]
    client = MockOpenAIResponses.create_mock_stream(stream)
    model = OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=client))
    agent = Agent(model, model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort='high'))
    async with agent.run_stream('hello') as result:
        assert await result.get_output() == 'done'
    request = get_mock_responses_kwargs(client)[0]
    assert request['model'] == model_name
    assert request['reasoning'] == {'effort': 'high'}
