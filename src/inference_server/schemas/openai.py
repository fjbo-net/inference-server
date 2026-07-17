"""OpenAI-compatible Chat Completions API schemas.

Mirrors the subset of the OpenAI API surface served by this project, so
that unmodified OpenAI clients can talk to the server.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str
    name: str | None = None


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str
    name: str | None = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    name: str | None = None


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


ChatMessage = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role")
]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    frequency_penalty: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None


FinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter"
]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: AssistantMessage
    finish_reason: FinishReason


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
