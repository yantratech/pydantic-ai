"""Native output streaming with one owner for node execution and recovery.

The producer owns the agent context, provider iterators and node hooks. The
public result consumes bounded snapshots, so a recoverable node cancellation
never cancels caller code or closes an iterator being advanced in another task.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Generator
from contextlib import AbstractAsyncContextManager, aclosing, asynccontextmanager, contextmanager
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, Literal

import anyio

from pydantic_graph import End

from . import _agent_graph, _utils, exceptions, messages
from ._run_context import AgentDepsT
from .agent.abstract import EventStreamHandler
from .output import OutputDataT
from .result import AgentStream, FinalResult, StreamedRunResult
from .run import AgentRun, AgentRunResult


@dataclass
class _Snapshot(Generic[AgentDepsT, OutputDataT]):
    stream: AgentStream[AgentDepsT, OutputDataT]
    response: messages.ModelResponse
    output: OutputDataT | _utils.Unset = _utils.UNSET
    text: str = ''


class _NodeStreamResult(StreamedRunResult[AgentDepsT, OutputDataT], Generic[AgentDepsT, OutputDataT]):
    def __init__(self) -> None:
        super().__init__([], 0, None, None)
        self.ready = anyio.Event()
        self.done = anyio.Event()
        self.demand = anyio.Semaphore(0)
        self.debounce_by: float | None = None
        self.mode: Literal['drain', 'output', 'text', 'delta', 'response'] = 'drain'
        self.updates, self.receiver = anyio.create_memory_object_stream[_Snapshot[AgentDepsT, OutputDataT]](1)
        self.finished_result: AgentRunResult[OutputDataT] | None = None
        self.error: BaseException | None = None
        self.snapshot: messages.ModelResponse | None = None
        self.locally_cancelled = False
        self.error_retrieved = False
        self.consumer_exit_error: BaseException | None = None
        self.teardown_scope: anyio.CancelScope | None = None
        self.caller_context = copy_context()
        self.consumer_context = self.caller_context
        self.owned_context = copy_context()

    async def publish(self, update: _Snapshot[AgentDepsT, OutputDataT]) -> None:
        if self.locally_cancelled:
            return
        await self.updates.send(update)
        await self.demand.acquire()

    async def consume(self, stream: AgentStream[AgentDepsT, OutputDataT]) -> None:
        # Reuse the native consumer operations, including their batching and
        # validation. Only their ownership changes: all run in the producer.
        async def updates() -> AsyncGenerator[_Snapshot[AgentDepsT, OutputDataT]]:
            if self.mode == 'output':
                async with aclosing(stream.stream_output(debounce_by=self.debounce_by)) as values:
                    async for value in values:
                        # The final validated value is delivered after lifecycle
                        # hooks settle the run, below the partial snapshots.
                        if stream._cached_output is None:  # pyright: ignore[reportPrivateUsage]
                            yield _Snapshot(stream, stream.response, output=value)
            elif self.mode in ('text', 'delta'):
                async with aclosing(
                    stream.stream_text(delta=self.mode == 'delta', debounce_by=self.debounce_by)
                ) as values:
                    async for value in values:
                        yield _Snapshot(stream, stream.response, text=value)
            else:
                async with aclosing(stream.stream_response(debounce_by=self.debounce_by)) as responses:
                    async for response in responses:
                        yield _Snapshot(stream, response)

        async with aclosing(updates()) as iterator:
            while not self.locally_cancelled:
                with self.consumer_context_scope():
                    try:
                        update = await anext(iterator)
                    except StopAsyncIteration:
                        return
                await self.publish(update)

    @contextmanager
    def consumer_context_scope(self) -> Generator[None]:
        tokens = [
            var.set(value)
            for var, value in self.consumer_context.items()
            if var not in self.owned_context or self.owned_context[var] is self.caller_context.get(var)
        ]
        try:
            yield
        finally:
            for token in reversed(tokens):
                token.var.reset(token)

    @asynccontextmanager
    async def forward_consumer_exit(self) -> AsyncGenerator[None]:
        try:
            yield
        except anyio.get_cancelled_exc_class():
            if self.consumer_exit_error is not None:
                # The consumer's exit cancels the producer to wake its current
                # wait. Preserve the actual exit reason for background-job
                # detach/cancel policy, then let owned teardown finish once.
                assert self.teardown_scope is not None
                self.teardown_scope.shield = True
                raise self.consumer_exit_error
            raise

    async def wait_result(self) -> AgentRunResult[OutputDataT] | None:
        self.error_retrieved = True
        await self.done.wait()
        if self.error is not None:
            raise self.error
        return self.finished_result

    async def _snapshots(self) -> AsyncIterator[_Snapshot[AgentDepsT, OutputDataT]]:
        while not self.locally_cancelled and not self.is_complete:
            self.consumer_context = copy_context()
            self.demand.release()
            try:
                update = await self.receiver.receive()
            except anyio.EndOfStream:
                self.settle(await self.wait_result())
                return
            self.snapshot = update.response
            yield update

    def settle(self, result: AgentRunResult[OutputDataT] | None) -> None:
        if result is not None:
            self._run_result = result
            self._all_messages = result.all_messages()
            self._new_message_index = len(self._all_messages) - len(result.new_messages())
            response = next(
                (message for message in reversed(self._all_messages) if isinstance(message, messages.ModelResponse)),
                None,
            )
            if response is not None:
                if self.snapshot is None or self.snapshot.parts != response.parts:
                    self.snapshot = response
            self.is_complete = True

    async def get_output(self) -> OutputDataT:
        self.mode = 'drain'
        async for _ in self._snapshots():
            pass
        if self._run_result is None:
            raise exceptions.UserError('The stream was cancelled before it produced an output.')
        return deepcopy(self._run_result.output)

    async def stream_output(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
        self.mode = 'output'
        self.debounce_by = debounce_by
        async for update in self._snapshots():
            if not isinstance(update.output, _utils.Unset):
                yield update.output
        if self._run_result is not None:
            yield self._run_result.output

    async def stream_response(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[messages.ModelResponse]:
        self.mode = 'response'
        self.debounce_by = debounce_by
        last_response: messages.ModelResponse | None = None
        async for update in self._snapshots():
            last_response = update.response
            yield last_response
        if self._run_result is not None:
            response = self.response
            if last_response is None or last_response.parts != response.parts:
                yield response
            else:
                last_response.run_id = self._run_result.response.run_id
                last_response.conversation_id = self._run_result.response.conversation_id

    async def stream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> AsyncIterator[str]:
        self.mode = 'delta' if delta else 'text'
        first_stream: AgentStream[AgentDepsT, OutputDataT] | None = None
        self.debounce_by = debounce_by
        async for update in self._snapshots():
            if first_stream is None:
                first_stream = update.stream
            if delta:
                # Append-only deltas stay on the first response. The settled
                # result, or cumulative streaming, can represent a replacement.
                if update.stream is first_stream:
                    yield update.text
            else:
                yield update.text
        if self._run_result is not None:
            if not isinstance(self._run_result.output, str):
                raise exceptions.UserError('stream_text() can only be used with text responses')
            recovered = (
                self._stream_response is None or self._run_result.response.parts != self._stream_response.response.parts
            )
            if first_stream is None or (not delta and recovered):
                yield self._run_result.output

    @property
    def response(self) -> messages.ModelResponse:
        assert self.snapshot is not None
        return self.snapshot

    async def cancel(self) -> None:
        self.locally_cancelled = True
        if self.is_complete:
            return
        if self._stream_response is not None:
            await self._stream_response.cancel()
            self.snapshot = self._stream_response.response
        self.demand.release()
        await self.wait_result()
        self.is_complete = True

    @property
    def cancelled(self) -> bool:
        return self.locally_cancelled


@asynccontextmanager
async def stream_run(  # noqa: C901
    run_context: AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]],
    event_stream_handler: EventStreamHandler[AgentDepsT] | None,
) -> AsyncGenerator[StreamedRunResult[AgentDepsT, OutputDataT]]:
    output = _NodeStreamResult[AgentDepsT, OutputDataT]()

    async def drive() -> AgentRunResult[OutputDataT] | None:  # noqa: C901
        async with run_context as run, output.forward_consumer_exit():
            output._all_messages = run.all_messages()  # pyright: ignore[reportPrivateUsage]
            output._new_message_index = len(run.all_messages()) - len(run.new_messages())  # pyright: ignore[reportPrivateUsage]

            async def stream_node(  # noqa: C901
                node: _agent_graph.AgentNode[AgentDepsT, OutputDataT],
            ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
                graph_ctx = run.ctx
                run_ctx = _agent_graph.build_run_context(graph_ctx)
                if isinstance(node, _agent_graph.ModelRequestNode):
                    final: FinalResult[OutputDataT] | None = None
                    parts: list[messages.ModelRequestPart] = []
                    try:
                        async with node.stream(graph_ctx) as stream, output.forward_consumer_exit():
                            final_event: messages.FinalResultEvent | None = None

                            async def to_final() -> AsyncIterator[messages.AgentStreamEvent]:
                                nonlocal final_event
                                async for event in stream:
                                    yield event
                                    if isinstance(event, messages.FinalResultEvent):
                                        final_event = event
                                        return

                            truncated = to_final()
                            if event_stream_handler is not None:
                                await event_stream_handler(run_ctx, truncated)
                            async for _ in truncated:
                                pass
                            if final_event is not None:
                                output._all_messages = run.all_messages()  # pyright: ignore[reportPrivateUsage]
                                output._new_message_index = len(run.all_messages()) - len(run.new_messages())  # pyright: ignore[reportPrivateUsage]
                                output.owned_context = copy_context()
                                output._stream_response = stream  # pyright: ignore[reportPrivateUsage]
                                if output.snapshot is None:
                                    output.snapshot = stream.response
                                    output.ready.set()
                                    await output.demand.acquire()
                                if output.locally_cancelled:
                                    return node
                                await output.consume(stream)
                                if output.locally_cancelled:
                                    return node
                                with output.consumer_context_scope():
                                    final = FinalResult(
                                        await stream.get_output(), final_event.tool_name, final_event.tool_call_id
                                    )
                                async for _ in _agent_graph.process_tool_calls(
                                    tool_manager=graph_ctx.deps.tool_manager,
                                    tool_calls=stream.response.tool_calls,
                                    tool_call_results=None,
                                    tool_call_metadata=None,
                                    final_result=final,
                                    ctx=graph_ctx,
                                    output_parts=parts,
                                ):
                                    pass
                                # Keep the raw provider snapshot separate from the
                                # settled history's attribution and derived costs.
                                output.snapshot = deepcopy(stream.response)
                    except BaseException:
                        if parts:
                            graph_ctx.state.message_history.append(
                                messages.ModelRequest(
                                    parts,
                                    state='interrupted',
                                    run_id=run.run_id,
                                    conversation_id=run.conversation_id,
                                    timestamp=_utils.now_utc(),
                                )
                            )
                        raise
                    if final is not None:
                        if parts:
                            graph_ctx.state.message_history.append(
                                messages.ModelRequest(
                                    parts,
                                    run_id=run.run_id,
                                    conversation_id=run.conversation_id,
                                    timestamp=_utils.now_utc(),
                                )
                            )
                        return _agent_graph.SetFinalResult(final)
                elif isinstance(node, _agent_graph.CallToolsNode):
                    async with node.stream(graph_ctx) as stream, output.forward_consumer_exit():
                        if event_stream_handler is not None:
                            await event_stream_handler(run_ctx, stream)
                        async for _ in stream:
                            pass
                return await run._advance_graph(node)  # pyright: ignore[reportPrivateUsage]

            node = run.next_node
            try:
                while not isinstance(node, End) and run.result is None and not output.locally_cancelled:
                    node = await run._run_node_with_hooks(node, stream_node)  # pyright: ignore[reportPrivateUsage]
            except GeneratorExit as error:
                if error is not output.consumer_exit_error:
                    raise
                # The model stream has recorded a resumable detach. The enclosing
                # run now exits normally, as it did when the consumer owned it.
                return None
            output._all_messages = run.all_messages()  # pyright: ignore[reportPrivateUsage]
        if output.locally_cancelled:
            return None
        assert run.result is not None, 'The graph run did not finish properly'
        return run.result

    async def produce() -> None:
        with anyio.CancelScope() as teardown_scope:
            output.teardown_scope = teardown_scope
            try:
                output.finished_result = await drive()
            except BaseException as error:
                output.error = error
            finally:
                if output.finished_result is None and output._stream_response is not None:  # pyright: ignore[reportPrivateUsage]
                    output.snapshot = deepcopy(output._stream_response.response)  # pyright: ignore[reportPrivateUsage]
                output.updates.close()
                output.done.set()
                output.ready.set()

    caller_error: BaseException | None = None
    async with anyio.create_task_group() as group:
        group.start_soon(produce, name='pydantic-ai-run-stream')
        try:
            await output.ready.wait()
            if output.done.is_set():
                output.settle(await output.wait_result())
                output.is_complete = False
            yield output
            if output.done.is_set() and not output.locally_cancelled and not output.error_retrieved:
                output.settle(await output.wait_result())
        except BaseException as error:
            # Keep the caller's exception intact rather than wrapping it in the
            # task group's ExceptionGroup. Producer errors are retrieved above.
            caller_error = error
        finally:
            output.consumer_exit_error = caller_error or GeneratorExit()
            group.cancel_scope.cancel()
    output.receiver.close()
    if caller_error is not None:
        raise caller_error
