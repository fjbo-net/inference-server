"""Mock engine that echoes the last user message back token by token.

Used by tests and local development; exercises the full engine
contract (streaming, truncation, usage, model lookup) without any
inference runtime.
"""

from collections.abc import AsyncIterator

from inference_server.engines.base import (
    BaseInferenceEngine,
    GenerationChunk,
    ModelNotFoundError,
)
from inference_server.schemas.openai import (
    ChatCompletionRequest,
    FinishReason,
    Usage,
    UserMessage,
)

ECHO_MODEL_ID = "echo"


class EchoEngine(BaseInferenceEngine):
    @property
    def name(self) -> str:
        return "echo"

    def list_models(self) -> list[str]:
        return [ECHO_MODEL_ID]

    def generate(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[GenerationChunk]:
        # Validate eagerly so callers get errors before streaming starts.
        if request.model not in self.list_models():
            raise ModelNotFoundError(request.model)
        return self._stream(request)

    async def _stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[GenerationChunk]:
        words = self._last_user_content(request).split()
        limit = request.max_tokens if request.max_tokens is not None else len(words)
        emitted = words[:limit]
        finish_reason: FinishReason = "length" if len(words) > limit else "stop"

        for index, word in enumerate(emitted):
            token = word if index == 0 else f" {word}"
            yield GenerationChunk(token=token)

        prompt_tokens = sum(
            len((message.content or "").split())
            for message in request.messages
        )
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(emitted),
            total_tokens=prompt_tokens + len(emitted)
        )
        yield GenerationChunk(
            finish_reason=finish_reason,
            usage=usage
        )

    def _last_user_content(self, request: ChatCompletionRequest) -> str:
        for message in reversed(request.messages):
            if isinstance(message, UserMessage):
                return message.content
        return ""
