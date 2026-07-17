from inference_server.schemas.openai import (
    ChatCompletionRequest,
    SystemMessage,
    UserMessage,
)


def test_chat_completion_request_parsing() -> None:
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
