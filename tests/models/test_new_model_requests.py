"""Request-wire coverage for the newly catalogued GPT-6 and Claude models."""

from __future__ import annotations

import pytest
from anthropic import omit
from anthropic.types.beta import BetaTextBlock, BetaUsage
from openai.types.responses import ResponseOutputMessage
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message
from .test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

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


@pytest.mark.parametrize('model_name', ['claude-opus-5-5', 'claude-fable-5-1'])
async def test_claude_request_wire_with_tool_and_effort(allow_model_requests: None, model_name: str) -> None:
    response = completion_message([BetaTextBlock(text='done', type='text')], BetaUsage(input_tokens=5, output_tokens=2))
    client = MockAnthropic.create_mock(response)
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))
    agent = Agent(
        model,
        model_settings=AnthropicModelSettings(
            anthropic_thinking={'type': 'adaptive'},
            anthropic_effort='high',
        ),
    )

    @agent.tool_plain
    def lookup(query: str) -> str:
        return query

    await agent.run('hello')
    request = get_mock_chat_completion_kwargs(client)[0]
    assert request['model'] == model_name
    assert request['thinking'] == {'type': 'adaptive'}
    assert request['output_config']['effort'] == 'high'
    assert request['tool_choice'] == {'type': 'auto'}
    assert any(tool['name'] == 'lookup' for tool in request['tools'])


@pytest.mark.parametrize('model_name', ['claude-opus-5-5', 'claude-fable-5-1'])
async def test_claude_native_output_request(allow_model_requests: None, model_name: str) -> None:
    response = completion_message(
        [BetaTextBlock(text='{"answer":"ok"}', type='text')], BetaUsage(input_tokens=5, output_tokens=2)
    )
    client = MockAnthropic.create_mock(response)
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))
    result = await Agent(model, output_type=NativeOutput(Answer)).run('answer')
    assert result.output.answer == 'ok'
    request = get_mock_chat_completion_kwargs(client)[0]
    assert request['model'] == model_name
    assert request['output_config']['format']['type'] == 'json_schema'
    assert request['tool_choice'] is omit


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


@pytest.mark.parametrize('model_name', ['claude-opus-5-5', 'claude-fable-5-1'])
async def test_claude_stream_request(allow_model_requests: None, model_name: str) -> None:
    from anthropic.types.beta import (
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
    )
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    stream = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_1',
                model=model_name,
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=5, output_tokens=0),
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextBlock(type='text', text='done'),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(input_tokens=5, output_tokens=1),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]
    client = MockAnthropic.create_stream_mock(stream)
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))
    agent = Agent(model, model_settings=AnthropicModelSettings(anthropic_effort='medium'))
    async with agent.run_stream('hello') as result:
        assert await result.get_output() == 'done'
    request = get_mock_chat_completion_kwargs(client)[0]
    assert request['model'] == model_name
    assert request['output_config']['effort'] == 'medium'
