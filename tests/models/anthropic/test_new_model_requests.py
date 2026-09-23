"""Claude Opus 5.5 and Fable 5.1 request-wire coverage."""

from __future__ import annotations

import pytest

pytest.importorskip('anthropic')

from anthropic import omit
from anthropic.types.beta import BetaTextBlock, BetaUsage
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.anthropic import AnthropicProvider

from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

pytestmark = pytest.mark.anyio


class Answer(BaseModel):
    answer: str


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
