"""Deterministic test data factories.

Each factory returns a valid baseline; tests override the fields they
assert on, so the Arrange block stays explicit about what matters.
"""

from typing import Any

DEFAULT_MODEL_ID = "qwen2.5-0.5b-instruct"


def make_user_message_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "user",
        "content": "Hi!"
    }
    payload.update(overrides)
    return payload


def make_chat_completion_request_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": DEFAULT_MODEL_ID,
        "messages": [make_user_message_payload()]
    }
    payload.update(overrides)
    return payload
