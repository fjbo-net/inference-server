"""Compatibility tests driving the server through the official `openai` SDK.

The SDK talks to the ASGI app over httpx's ASGI transport, so these
tests prove unmodified OpenAI clients can use the server without a
running socket.
"""

import asyncio
from typing import Any

import httpx
import openai
import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from inference_server.app import create_app
from inference_server.config import Settings


def _make_sdk_client(http_client: httpx.AsyncClient) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="http://testserver/v1",
        api_key="test-key",
        http_client=http_client
    )


def _make_http_client() -> httpx.AsyncClient:
    settings = Settings(
        _env_file=None,
        engine="echo"
    )
    app = create_app(settings)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver"
    )


async def _create_chat_completion(
    content: str,
    **params: Any
) -> ChatCompletion:
    async with _make_http_client() as http_client:
        client = _make_sdk_client(http_client)
        return await client.chat.completions.create(
            model=params.pop(
                "model",
                "echo"
            ),
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            **params
        )


async def _stream_chat_completion(content: str) -> list[ChatCompletionChunk]:
    async with _make_http_client() as http_client:
        client = _make_sdk_client(http_client)
        stream = await client.chat.completions.create(
            model="echo",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            stream=True
        )
        return [chunk async for chunk in stream]


async def _list_model_ids() -> list[str]:
    async with _make_http_client() as http_client:
        client = _make_sdk_client(http_client)
        models = await client.models.list()
        return [model.id for model in models.data]


def test_openai_sdk_receives_completion_when_payload_is_valid() -> None:
    # Arrange
    expected_content = "hello from the sdk"
    expected_finish_reason = "stop"


    # Act
    completion = asyncio.run(_create_chat_completion(expected_content))


    # Assert
    choice = completion.choices[0]
    assert choice.message.role == "assistant"
    assert choice.message.content == expected_content
    assert choice.finish_reason == expected_finish_reason
    assert completion.usage is not None
    assert completion.usage.completion_tokens == len(expected_content.split())


def test_openai_sdk_streams_chunks_when_stream_is_enabled() -> None:
    # Arrange
    expected_content = "stream through the sdk"
    expected_finish_reason = "stop"


    # Act
    chunks = asyncio.run(_stream_chat_completion(expected_content))


    # Assert
    deltas = [chunk.choices[0].delta for chunk in chunks]
    streamed_content = "".join(
        delta.content
        for delta in deltas
        if delta.content is not None
    )
    assert deltas[0].role == "assistant"
    assert streamed_content == expected_content
    assert chunks[-1].choices[0].finish_reason == expected_finish_reason


def test_openai_sdk_lists_models() -> None:
    # Arrange
    expected_model_ids = ["echo"]


    # Act
    model_ids = asyncio.run(_list_model_ids())


    # Assert
    assert model_ids == expected_model_ids


def test_openai_sdk_raises_not_found_when_model_is_unknown() -> None:
    # Arrange
    unknown_model = "missing-model"


    # Act & Assert
    with pytest.raises(openai.NotFoundError):
        asyncio.run(
            _create_chat_completion(
                "anyone there?",
                model=unknown_model
            )
        )
