"""Engine interface every backend (echo, OpenVINO, ROCm) implements."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from inference_server.schemas.openai import (
    ChatCompletionRequest,
    FinishReason,
    Usage,
)


class EngineError(Exception):
    """Raised when an engine fails to load a model or generate."""


class ModelNotFoundError(EngineError):
    """Raised when a request names a model the engine does not serve."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"The model `{model_id}` does not exist.")


@dataclass
class GenerationChunk:
    """One step of a token stream.

    The final chunk carries `finish_reason` (and `usage` when the
    engine tracks token counts) with `token` set to None. Engines
    that fail after streaming has started emit a final chunk with
    `error` set instead of raising, so an in-flight SSE response can
    report the failure rather than being severed.
    """

    token: str | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    error: str | None = None


class BaseInferenceEngine(ABC):
    """Interface for edge inference engine backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g. "echo", "openvino", "rocm")."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return the ids of the models this engine can serve."""

    @abstractmethod
    def generate(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[GenerationChunk]:
        """Stream generation chunks for the request.

        Raises `ModelNotFoundError` when `request.model` is not served.
        """
