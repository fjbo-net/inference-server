import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from inference_server.app import create_app
from inference_server.config import Settings
from tests.factories import (
    make_chat_completion_request_payload,
    make_user_message_payload,
)


@pytest.fixture
def client() -> TestClient:
    settings = Settings(
        _env_file=None,
        engine="echo"
    )
    return TestClient(create_app(settings))


def _data_events(sse_body: str) -> list[str]:
    return [
        line.removeprefix("data: ")
        for line in sse_body.split("\n\n")
        if line.startswith("data: ")
    ]


def test_create_chat_completion_returns_completion_when_payload_is_valid(
    client: TestClient
) -> None:
    # Arrange
    expected_status_code = 200
    expected_content = "echo this back"
    expected_finish_reason = "stop"

    payload = make_chat_completion_request_payload(
        model="echo",
        messages=[make_user_message_payload(content=expected_content)]
    )


    # Act
    response = client.post(
        "/v1/chat/completions",
        json=payload
    )


    # Assert
    body = response.json()
    choice = body["choices"][0]
    assert response.status_code == expected_status_code
    assert body["object"] == "chat.completion"
    assert body["model"] == "echo"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == expected_content
    assert choice["finish_reason"] == expected_finish_reason
    assert body["usage"]["completion_tokens"] == len(expected_content.split())


def test_create_chat_completion_streams_chunks_when_stream_is_enabled(
    client: TestClient
) -> None:
    # Arrange
    expected_content = "streaming echo test"
    expected_finish_reason = "stop"

    payload = make_chat_completion_request_payload(
        model="echo",
        messages=[make_user_message_payload(content=expected_content)],
        stream=True
    )


    # Act
    response = client.post(
        "/v1/chat/completions",
        json=payload
    )


    # Assert
    events = _data_events(response.text)
    chunks: list[dict[str, Any]] = [
        json.loads(event)
        for event in events
        if event != "[DONE]"
    ]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    streamed_content = "".join(
        delta["content"]
        for delta in deltas
        if "content" in delta
    )
    finish_reasons = [
        chunk["choices"][0]["finish_reason"]
        for chunk in chunks
        if "finish_reason" in chunk["choices"][0]
    ]
    assert response.headers["content-type"].startswith("text/event-stream")
    assert deltas[0]["role"] == "assistant"
    assert streamed_content == expected_content
    assert finish_reasons[-1] == expected_finish_reason
    assert events[-1] == "[DONE]"


def test_create_chat_completion_returns_error_envelope_when_model_is_unknown(
    client: TestClient
) -> None:
    # Arrange
    expected_status_code = 404
    expected_error_type = "invalid_request_error"
    expected_error_code = "model_not_found"

    payload = make_chat_completion_request_payload(model="missing-model")


    # Act
    response = client.post(
        "/v1/chat/completions",
        json=payload
    )


    # Assert
    error = response.json()["error"]
    assert response.status_code == expected_status_code
    assert error["type"] == expected_error_type
    assert error["code"] == expected_error_code
    assert error["param"] == "model"


def test_create_chat_completion_returns_error_envelope_when_messages_are_missing(
    client: TestClient
) -> None:
    # Arrange
    expected_status_code = 422
    expected_error_type = "invalid_request_error"

    payload = make_chat_completion_request_payload(model="echo")
    del payload["messages"]


    # Act
    response = client.post(
        "/v1/chat/completions",
        json=payload
    )


    # Assert
    error = response.json()["error"]
    assert response.status_code == expected_status_code
    assert error["type"] == expected_error_type
    assert error["param"] == "messages"
