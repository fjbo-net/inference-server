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
