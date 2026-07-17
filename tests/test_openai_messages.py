import pytest
from pydantic import TypeAdapter, ValidationError

from inference_server.schemas.openai import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


def test_user_message_parses_fields_when_payload_is_valid() -> None:
    # Arrange
    expected_role = "user"
    expected_content = "Hello there!"

    payload = {
        "role": expected_role,
        "content": expected_content
    }


    # Act
    message = UserMessage.model_validate(payload)


    # Assert
    assert message.role == expected_role
    assert message.content == expected_content
    assert message.name is None


def test_assistant_message_parses_when_content_is_null() -> None:
    # Arrange
    payload = {
        "role": "assistant",
        "content": None
    }


    # Act
    message = AssistantMessage.model_validate(payload)


    # Assert
    assert message.content is None


def test_tool_message_raises_validation_error_when_tool_call_id_is_missing() -> None:
    # Arrange
    payload = {
        "role": "tool",
        "content": "42"
    }


    # Act & Assert
    with pytest.raises(ValidationError):
        ToolMessage.model_validate(payload)


def test_chat_message_discriminates_on_role() -> None:
    # Arrange
    expected_types = [
        SystemMessage,
        UserMessage,
        AssistantMessage,
        ToolMessage
    ]

    payload = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hi!"
        },
        {
            "role": "assistant",
            "content": "Hello! How can I help?"
        },
        {
            "role": "tool",
            "content": "42",
            "tool_call_id": "call_1"
        }
    ]

    adapter = TypeAdapter(list[ChatMessage])


    # Act
    messages = adapter.validate_python(payload)


    # Assert
    assert [type(message) for message in messages] == expected_types
