"""Node cancellation requires deterministic held streams, not provider recordings."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import NoneType
from typing import Literal

import anyio
import pytest

from pydantic_ai import Agent, CallToolsNode, ModelRequestNode, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import AgentNode, NodeResult, WrapNodeRunHandler
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel


@dataclass
class InterruptNode(AbstractCapability[None]):
    tools: bool = False
    scope: anyio.CancelScope | None = field(default=None, init=False)
    recovered: anyio.Event = field(default_factory=anyio.Event, init=False)

    async def wrap_node_run(
        self, ctx: RunContext[None], *, node: AgentNode[None], handler: WrapNodeRunHandler[None]
    ) -> NodeResult[None]:
        target = CallToolsNode if self.tools else ModelRequestNode
        if not isinstance(node, target) or self.scope is not None:
            return await handler(node)
        with anyio.CancelScope() as scope:
            self.scope = scope
            return await handler(node)
        self.recovered.set()
        return ModelRequestNode(ModelRequest(parts=[UserPromptPart('Finish with the available results.')]))


@pytest.mark.parametrize('delta', [True, False], ids=['append-only-preview', 'cumulative-replacement'])
async def test_native_stream_recovers_a_cancelled_node_without_cancelling_its_consumer(delta: bool):
    closed = anyio.Event()
    requests = 0

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal requests
        requests += 1
        if requests == 1:
            try:
                yield 'Draft'
                await anyio.sleep_forever()
            finally:
                closed.set()
        else:
            yield 'Final answer'

    capability = InterruptNode()
    agent = Agent(FunctionModel(stream_function=model), deps_type=NoneType, capabilities=[capability])
    with anyio.fail_after(3):
        async with agent.run_stream('Start') as result:
            chunks = result.stream_text(delta=delta, debounce_by=None)
            assert await anext(chunks) == 'Draft'
            assert capability.scope is not None
            capability.scope.cancel()
            # No consumer pull is active here. Recovery must own its provider
            # iterator and must not cancel this unrelated caller-side wait.
            await capability.recovered.wait()
            assert closed.is_set()
            assert [chunk async for chunk in chunks] == ([] if delta else ['Final answer'])
            assert await result.get_output() == 'Final answer'
            assert result.result.output == 'Final answer'
    assert requests == 2


@pytest.mark.parametrize('strategy', ['early', 'graceful', 'exhaustive'])
async def test_native_stream_recovery_retains_a_completed_tool_result(
    strategy: Literal['early', 'graceful', 'exhaustive'],
):
    capability = InterruptNode(tools=True)
    completed: list[str] = []
    requests = 0

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        nonlocal requests
        requests += 1
        if requests == 1:
            yield {0: DeltaToolCall(name='publish', json_args='{}', tool_call_id='publish-1')}
        else:
            results = [
                part
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, ToolReturnPart)
            ]
            assert [(part.tool_call_id, part.content) for part in results] == [('publish-1', 'Published')]
            yield 'Publication confirmed'

    async def events(_ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            if isinstance(event, FunctionToolResultEvent):
                assert capability.scope is not None
                capability.scope.cancel()
                await anyio.lowlevel.checkpoint()

    agent = Agent(
        FunctionModel(stream_function=model), deps_type=NoneType, capabilities=[capability], end_strategy=strategy
    )

    @agent.tool_plain
    async def publish() -> str:
        completed.append('published')
        return 'Published'

    with anyio.fail_after(3):
        async with agent.run_stream('Publish', event_stream_handler=events) as result:
            assert await result.get_output() == 'Publication confirmed'
    assert capability.recovered.is_set()
    assert completed == ['published']
    assert requests == 2


async def test_native_stream_validator_context_does_not_retain_a_reset_caller_value():
    caller = ContextVar[str]('caller')
    observed: list[str] = []

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield 'a'
        yield 'b'

    agent = Agent(FunctionModel(stream_function=model))

    @agent.output_validator
    async def validate(value: str) -> str:
        observed.append(caller.get('unset'))
        return value

    async with agent.run_stream('Start') as result:
        chunks = result.stream_text(debounce_by=None)
        token = caller.set('first')
        assert await anext(chunks) == 'a'
        caller.reset(token)
        assert [chunk async for chunk in chunks] == ['ab']
    assert observed == ['first', 'unset', 'unset']
