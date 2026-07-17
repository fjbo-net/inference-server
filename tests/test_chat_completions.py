from inference_server.schemas.openai import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceDelta,
    SystemMessage,
    Usage,
    UserMessage,
)


def test_chat_completion_request_parses_fields_when_payload_is_valid() -> None:
    # Arrange
    expected_model = "qwen2.5-0.5b-instruct"
    expected_temperature = 0.7
    expected_message_types = [
        SystemMessage,
        UserMessage
    ]

    payload = {
        "model": expected_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hi!"
            }
        ],
        "temperature": expected_temperature
    }


    # Act
    request = ChatCompletionRequest.model_validate(payload)


    # Assert
    assert request.model == expected_model
    assert request.temperature == expected_temperature
    assert request.stream is False
    assert [type(message) for message in request.messages] == expected_message_types


def test_chat_completion_request_ignores_unknown_fields() -> None:
    # Arrange
    payload = {
        "model": "qwen2.5-0.5b-instruct",
        "messages": [
            {
                "role": "user",
                "content": "Hi!"
            }
        ],
        "logit_bias": {"50256": -100}
    }


    # Act
    request = ChatCompletionRequest.model_validate(payload)


    # Assert
    assert not hasattr(request, "logit_bias")


def test_chat_completion_response_serializes_to_openai_shape() -> None:
    # Arrange
    expected_payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1752710400,
        "model": "qwen2.5-0.5b-instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help?",
                    "name": None
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19
        }
    }

    message = AssistantMessage(content="Hello! How can I help?")
    choice = ChatCompletionChoice(
        index=0,
        message=message,
        finish_reason="stop"
    )
    usage = Usage(
        prompt_tokens=12,
        completion_tokens=7,
        total_tokens=19
    )
    response = ChatCompletionResponse(
        id="chatcmpl-1",
        created=1752710400,
        model="qwen2.5-0.5b-instruct",
        choices=[choice],
        usage=usage
    )


    # Act
    payload = response.model_dump()


    # Assert
    assert payload == expected_payload


def test_chat_completion_chunk_serializes_to_openai_shape() -> None:
    # Arrange
    expected_payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1752710400,
        "model": "qwen2.5-0.5b-instruct",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": None,
                    "content": "Hello"
                },
                "finish_reason": None
            }
        ]
    }

    delta = ChoiceDelta(content="Hello")
    choice = ChatCompletionChunkChoice(
        index=0,
        delta=delta
    )
    chunk = ChatCompletionChunk(
        id="chatcmpl-1",
        created=1752710400,
        model="qwen2.5-0.5b-instruct",
        choices=[choice]
    )


    # Act
    payload = chunk.model_dump()


    # Assert
    assert payload == expected_payload


def test_chat_completion_chunk_carries_finish_reason_when_final() -> None:
    # Arrange
    expected_finish_reason = "stop"

    delta = ChoiceDelta()
    choice = ChatCompletionChunkChoice(
        index=0,
        delta=delta,
        finish_reason=expected_finish_reason
    )
    chunk = ChatCompletionChunk(
        id="chatcmpl-1",
        created=1752710400,
        model="qwen2.5-0.5b-instruct",
        choices=[choice]
    )


    # Act
    finish_reason = chunk.choices[0].finish_reason


    # Assert
    assert finish_reason == expected_finish_reason
