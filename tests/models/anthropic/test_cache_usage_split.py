"""Preserve Anthropic cache-write durations through ordinary and streamed usage."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import pytest
from anthropic.types.beta import BetaMessage, BetaMessageDeltaUsage, BetaUsage
from anthropic.types.beta.beta_raw_message_delta_event import BetaRawMessageDeltaEvent
from anthropic.types.beta.beta_raw_message_start_event import BetaRawMessageStartEvent

from pydantic_ai.models.anthropic import _map_usage


@pytest.fixture
def response_usage() -> BetaUsage:
    return BetaUsage.model_validate(
        {
            'input_tokens': 100,
            'output_tokens': 20,
            'cache_creation_input_tokens': 30,
            'cache_read_input_tokens': 0,
            'cache_creation': {'ephemeral_5m_input_tokens': 10, 'ephemeral_1h_input_tokens': 20},
        }
    )


@pytest.mark.parametrize('streamed', [False, True], ids=['ordinary', 'streamed'])
def test_anthropic_cache_write_duration_survives_usage_mapping(response_usage: BetaUsage, streamed: bool) -> None:
    message = BetaMessage.model_construct(usage=response_usage)
    if streamed:
        start = BetaRawMessageStartEvent.model_construct(message=message)
        usage = _map_usage(start, 'anthropic', 'https://api.anthropic.com', 'claude-opus-5-5')
        delta_usage = BetaMessageDeltaUsage.model_validate(
            {'input_tokens': 100, 'output_tokens': 25, 'cache_creation_input_tokens': 30, 'cache_read_input_tokens': 0}
        )
        delta = BetaRawMessageDeltaEvent.model_construct(usage=delta_usage)
        usage = _map_usage(delta, 'anthropic', 'https://api.anthropic.com', 'claude-opus-5-5', usage)
    else:
        usage = _map_usage(message, 'anthropic', 'https://api.anthropic.com', 'claude-opus-5-5')

    assert usage.input_tokens == 130
    assert usage.cache_write_tokens == 30
    assert usage.details['cache_write_5m_tokens'] == 10
    assert usage.details['cache_write_1h_tokens'] == 20
    assert usage.opentelemetry_attributes()['gen_ai.usage.details.cache_write_1h_tokens'] == 20
    assert usage.output_tokens == (25 if streamed else 20)
