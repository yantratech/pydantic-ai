"""Node cancellation requires deterministic held streams, not provider recordings."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import NoneType
from typing import Literal

import anyio
import pytest

from pydantic_ai import Agent, CallToolsNode, ModelRequestNode, RunContext, UserError
from pydantic_ai._agent_graph import SetFinalResult
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import AgentNode, NodeResult, WrapNodeRunHandler, WrapRunHandler
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import FinalResult
from pydantic_ai.run import AgentRunResult
from pydantic_graph import End


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


async def test_native_stream_final_handoff_preserves_an_already_validated_answer():
    requests = 0
    handoffs = 0

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal requests
        requests += 1
        yield 'Validated answer'

    class ExpireFinalHandoff(AbstractCapability[None]):
        async def wrap_node_run(
            self, ctx: RunContext[None], *, node: AgentNode[None], handler: WrapNodeRunHandler[None]
        ) -> NodeResult[None]:
            nonlocal handoffs
            if isinstance(node, SetFinalResult):
                handoffs += 1
                # Publishing an already validated answer has no asynchronous work.
                # An expired node budget must not interrupt a graph-worker handoff.
                with anyio.fail_after(0):
                    return await handler(node)
            return await handler(node)

    agent = Agent(FunctionModel(stream_function=model), deps_type=NoneType, capabilities=[ExpireFinalHandoff()])
    async with agent.run_stream('Start') as result:
        assert await result.get_output() == 'Validated answer'
    assert requests == 1
    assert handoffs == 1


async def test_native_stream_cancellation_during_partial_validation_discards_pending_output():
    validating = anyio.Event()
    release = anyio.Event()
    observed: list[str] = []
    agent = Agent(TestModel(custom_output_text='Draft answer'), deps_type=NoneType)

    @agent.output_validator
    async def validate(ctx: RunContext[None], value: str) -> str:
        assert ctx.partial_output
        validating.set()
        await release.wait()
        return value

    async with agent.run_stream('Start') as result:

        async def consume() -> None:
            observed.extend([value async for value in result.stream_output(debounce_by=None)])

        with anyio.fail_after(3):
            async with anyio.create_task_group() as group:
                group.start_soon(consume)
                await validating.wait()
                group.start_soon(result.cancel)
                await anyio.lowlevel.checkpoint()
                assert result.cancelled
                release.set()
        assert observed == []
        assert result.is_complete
        assert result.response.state == 'interrupted'
        assert [response async for response in result.stream_response()] == []
        with pytest.raises(UserError, match='cancelled before it produced an output'):
            await result.get_output()


async def test_native_stream_switches_from_text_preview_to_validated_output():
    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield 'Draft '
        yield 'answer'

    agent = Agent(FunctionModel(stream_function=model))
    async with agent.run_stream('Start') as result:
        preview = result.stream_text(debounce_by=None)
        assert await anext(preview) == 'Draft '
        assert [value async for value in result.stream_output(debounce_by=None)] == ['Draft answer']
        assert [text async for text in preview] == []
        assert result.result.output == 'Draft answer'


async def test_native_stream_rejects_text_consumption_of_settled_structured_output():
    agent = Agent(TestModel(custom_output_args=[1, 2]), output_type=list[int])
    async with agent.run_stream('Start') as result:
        assert await result.get_output() == [1, 2]
        with pytest.raises(UserError, match=r'stream_text.*text responses'):
            _ = [text async for text in result.stream_text()]


async def test_native_stream_can_cancel_a_short_circuited_run():
    class ShortCircuit(AbstractCapability[None]):
        async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
            return AgentRunResult(output='Cached answer')

    agent = Agent(TestModel(), deps_type=NoneType, capabilities=[ShortCircuit()])
    async with agent.run_stream('Start') as result:
        assert not result.is_complete
        await result.cancel()
        assert result.cancelled
        assert await result.get_output() == 'Cached answer'


async def test_native_final_response_recovery_retains_completed_exhaustive_tools():
    capability = InterruptNode()
    requests = 0

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        nonlocal requests
        requests += 1
        if requests == 1:
            yield {0: DeltaToolCall(name='final_result', json_args='{"response": ["Draft"]}')}
            yield {1: DeltaToolCall(name='publish', json_args='{}', tool_call_id='published')}
            yield {2: DeltaToolCall(name='interrupt', json_args='{}', tool_call_id='interrupted')}
        else:
            returns = [
                part
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, ToolReturnPart) and part.tool_name == 'publish'
            ]
            assert [(part.tool_call_id, part.content) for part in returns] == [('published', 'Published')]
            yield {0: DeltaToolCall(name='final_result', json_args='{"response": ["Final answer"]}')}

    agent = Agent(
        FunctionModel(stream_function=model),
        deps_type=NoneType,
        output_type=list[str],
        end_strategy='exhaustive',
        capabilities=[capability],
    )

    @agent.tool_plain(sequential=True)
    async def publish() -> str:
        return 'Published'

    @agent.tool_plain(sequential=True)
    async def interrupt() -> str:
        assert capability.scope is not None
        capability.scope.cancel()
        await anyio.lowlevel.checkpoint()
        assert False, 'Node cancellation must interrupt the pending tool'

    async with agent.run_stream('Start') as result:
        assert await result.get_output() == ['Final answer']
    assert requests == 2


async def test_native_stream_context_exit_settles_recovery_while_consumer_is_idle():
    scope: anyio.CancelScope | None = None

    class RecoverDirectly(AbstractCapability[None]):
        async def wrap_node_run(
            self, ctx: RunContext[None], *, node: AgentNode[None], handler: WrapNodeRunHandler[None]
        ) -> NodeResult[None]:
            nonlocal scope
            if not isinstance(node, ModelRequestNode):
                return await handler(node)
            with anyio.CancelScope() as scope:
                return await handler(node)
            return End(FinalResult('Recovered answer'))

    agent = Agent(TestModel(custom_output_text='Draft answer'), deps_type=NoneType, capabilities=[RecoverDirectly()])
    async with agent.run_stream('Start') as result:
        preview = result.stream_text(debounce_by=None)
        assert await anext(preview) == 'Draft '
        assert scope is not None
        scope.cancel()
        await anyio.wait_all_tasks_blocked()
    assert result.is_complete
    assert result.result.output == 'Recovered answer'
    assert [text async for text in preview] == []
