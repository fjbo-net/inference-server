import asyncio
from collections.abc import AsyncIterator

import pytest

from inference_server.engines.base import GenerationChunk, ModelNotFoundError
from inference_server.engines.echo import ECHO_MODEL_ID, EchoEngine
from tests.factories import (
    make_chat_completion_request,
    make_user_message_payload,
)


def _collect_chunks(
    stream: AsyncIterator[GenerationChunk]
) -> list[GenerationChunk]:
    async def _consume() -> list[GenerationChunk]:
        return [chunk async for chunk in stream]

    return asyncio.run(_consume())


def test_echo_engine_lists_echo_model() -> None:
    # Arrange
    expected_models = [ECHO_MODEL_ID]

    engine = EchoEngine()


    # Act
    models = engine.list_models()


    # Assert
    assert models == expected_models


def test_echo_engine_echoes_last_user_message() -> None:
    # Arrange
    expected_text = "repeat after me"
    expected_finish_reason = "stop"

    engine = EchoEngine()
    request = make_chat_completion_request(
        model=ECHO_MODEL_ID,
        messages=[make_user_message_payload(content=expected_text)]
    )


    # Act
    chunks = _collect_chunks(engine.generate(request))


    # Assert
    tokens = [chunk.token for chunk in chunks if chunk.token is not None]
    assert "".join(tokens) == expected_text
    assert chunks[-1].finish_reason == expected_finish_reason


def test_echo_engine_truncates_when_max_tokens_is_reached() -> None:
    # Arrange
    expected_completion_tokens = 2
    expected_finish_reason = "length"

    engine = EchoEngine()
    request = make_chat_completion_request(
        model=ECHO_MODEL_ID,
        messages=[make_user_message_payload(content="one two three four")],
        max_tokens=expected_completion_tokens
    )


    # Act
    chunks = _collect_chunks(engine.generate(request))


    # Assert
    tokens = [chunk.token for chunk in chunks if chunk.token is not None]
    assert len(tokens) == expected_completion_tokens
    assert chunks[-1].finish_reason == expected_finish_reason


def test_echo_engine_reports_usage_when_generation_completes() -> None:
    # Arrange
    expected_prompt_tokens = 3
    expected_completion_tokens = 3

    engine = EchoEngine()
    request = make_chat_completion_request(
        model=ECHO_MODEL_ID,
        messages=[make_user_message_payload(content="echo this back")]
    )


    # Act
    chunks = _collect_chunks(engine.generate(request))


    # Assert
    usage = chunks[-1].usage
    assert usage is not None
    assert usage.prompt_tokens == expected_prompt_tokens
    assert usage.completion_tokens == expected_completion_tokens
    assert usage.total_tokens == expected_prompt_tokens + expected_completion_tokens


def test_echo_engine_raises_model_not_found_when_model_is_unknown() -> None:
    # Arrange
    engine = EchoEngine()
    request = make_chat_completion_request(model="missing-model")


    # Act & Assert
    with pytest.raises(ModelNotFoundError):
        engine.generate(request)
