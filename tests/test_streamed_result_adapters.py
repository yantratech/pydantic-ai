"""Public result adapters retain their contract independently of Agent.run_stream."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from types import NoneType
from typing import Literal

import pytest

from pydantic_ai import Agent, UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import StreamedRunResult, StreamedRunResultSync


@pytest.mark.parametrize('source', ['live', 'settled'])
@pytest.mark.parametrize('method', ['get_output', 'stream_output', 'stream_text', 'stream_response'])
async def test_public_result_adapter_consumes_and_settles_once(
    source: Literal['live', 'settled'],
    method: Literal['get_output', 'stream_output', 'stream_text', 'stream_response'],
):
    agent = Agent(TestModel(custom_output_text='Adapter answer'), deps_type=NoneType)
    completed: list[str] = []
    async with AsyncExitStack() as stack:
        if source == 'live':
            run = await stack.enter_async_context(agent.iter('Hello', metadata={'adapter': 'live'}))
            first_node = run.next_node
            assert Agent.is_user_prompt_node(first_node)
            node = await run.next(first_node)
            assert Agent.is_model_request_node(node)
            native = await stack.enter_async_context(node.stream(run.ctx))

            async def on_complete() -> None:
                completed.append(await native.get_output())

            streamed = StreamedRunResult(list(run.all_messages()), 0, native, on_complete)
            with pytest.raises(UserError, match='still streaming'):
                _ = streamed.result
        else:
            settled = await agent.run('Hello', metadata={'adapter': 'settled'})
            streamed = StreamedRunResult(settled.all_messages(), 0, run_result=settled)

        workspace = streamed.workspace
        timestamp = streamed.timestamp
        assert not streamed.is_complete
        assert streamed.metadata == {'adapter': source}
        assert not streamed.cancelled
        if method == 'get_output':
            answer = await streamed.get_output()
        elif method == 'stream_output':
            answer = [value async for value in streamed.stream_output(debounce_by=None)][-1]
        elif method == 'stream_text':
            answer = [value async for value in streamed.stream_text(debounce_by=None)][-1]
        else:
            response = [value async for value in streamed.stream_response(debounce_by=None)][-1]
            answer = response.text
            assert response.run_id == streamed.run_id
            assert response.conversation_id == streamed.conversation_id

        assert answer == 'Adapter answer'
        assert streamed.is_complete
        assert await streamed.get_output() == answer
        assert [value async for value in streamed.stream_output()] == [answer]
        assert [value async for value in streamed.stream_text()] == [answer]
        assert await streamed.validate_response_output(streamed.response) == answer
        assert streamed.result.output == answer
        assert streamed.result.run_id == streamed.run_id
        assert streamed.result.conversation_id == streamed.conversation_id
        assert streamed.workspace is workspace
        assert streamed.timestamp >= timestamp
        assert streamed.timestamp == streamed.response.timestamp
        assert streamed.usage.input_tokens > 0
        assert completed == (['Adapter answer'] if source == 'live' else [])
        await streamed.cancel()
        assert streamed.is_complete


async def test_live_result_adapter_cancellation_preserves_history_without_an_output():
    agent = Agent(TestModel(custom_output_text='Unfinished answer'), deps_type=NoneType)
    async with agent.iter('Hello') as run:
        first_node = run.next_node
        assert Agent.is_user_prompt_node(first_node)
        node = await run.next(first_node)
        assert Agent.is_model_request_node(node)
        async with node.stream(run.ctx) as native:
            streamed = StreamedRunResult(list(run.all_messages()), 0, native, None)
            assert not streamed.cancelled
            await streamed.cancel()
            assert streamed.cancelled
            assert streamed.is_complete
            recorded = streamed.all_messages()[-1]
            assert recorded.parts == streamed.response.parts
            assert recorded.state == 'interrupted'
            assert recorded.run_id == streamed.run_id
            assert recorded.conversation_id == streamed.conversation_id
            assert streamed.response.state == 'interrupted'
            with pytest.raises(UserError, match='cancelled before it produced an output'):
                _ = streamed.result


def test_sync_result_adapter_preserves_its_context_owner():
    owner = ContextVar[str]('adapter_owner', default='caller')
    seen: list[str] = []
    agent = Agent(TestModel(custom_output_text='Owned answer'), deps_type=NoneType)

    @agent.output_validator
    def validate(value: str) -> str:
        seen.append(owner.get())
        return value

    @asynccontextmanager
    async def adapter() -> AsyncGenerator[StreamedRunResult[None, str]]:
        token = owner.set('adapter')
        try:
            async with agent.run_stream('Hello') as result:
                yield result
        finally:
            owner.reset(token)

    with StreamedRunResultSync(adapter()) as result:
        assert owner.get() == 'caller'
        assert result.get_output() == 'Owned answer'
    assert seen == ['adapter']
    assert owner.get() == 'caller'


async def test_settled_result_adapter_rejects_text_consumption_of_structured_output():
    agent = Agent(TestModel(custom_output_args=[1, 2]), output_type=list[int])
    settled = await agent.run('Hello')
    streamed = StreamedRunResult(settled.all_messages(), 0, run_result=settled)
    with pytest.raises(UserError, match=r'stream_text.*text responses'):
        _ = [text async for text in streamed.stream_text()]
