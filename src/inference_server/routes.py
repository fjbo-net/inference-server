"""OpenAI-compatible API routes."""

import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from inference_server.engines.base import (
    BaseInferenceEngine,
    EngineError,
    GenerationChunk,
)
from inference_server.schemas.openai import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceDelta,
    ErrorDetail,
    ErrorResponse,
    FinishReason,
    Model,
    ModelList,
    Usage,
)

router = APIRouter(prefix="/v1")


def _engine(request: Request) -> BaseInferenceEngine:
    engine: BaseInferenceEngine = request.app.state.engine
    return engine


@router.get("/models")
def list_models(request: Request) -> ModelList:
    engine = _engine(request)
    created = int(time.time())
    models = [
        Model(
            id=model_id,
            created=created,
            owned_by="inference-server"
        )
        for model_id in engine.list_models()
    ]
    return ModelList(data=models)


@router.post("/chat/completions")
async def create_chat_completion(
    request: Request,
    body: ChatCompletionRequest
) -> Response:
    engine = _engine(request)
    stream = engine.generate(body)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if body.stream:
        return StreamingResponse(
            _sse_events(
                stream,
                completion_id,
                created,
                body.model
            ),
            media_type="text/event-stream"
        )

    response = await _build_response(
        stream,
        completion_id,
        created,
        body.model
    )
    return JSONResponse(content=response.model_dump(exclude_none=True))


async def _build_response(
    stream: AsyncIterator[GenerationChunk],
    completion_id: str,
    created: int,
    model: str
) -> ChatCompletionResponse:
    tokens: list[str] = []
    finish_reason: FinishReason = "stop"
    usage = Usage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0
    )

    async for chunk in stream:
        if chunk.error is not None:
            # No bytes are on the wire yet, so the EngineError handler
            # can still turn this into a proper 500 envelope.
            raise EngineError(chunk.error)
        if chunk.token is not None:
            tokens.append(chunk.token)
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
        if chunk.usage is not None:
            usage = chunk.usage

    message = AssistantMessage(content="".join(tokens))
    choice = ChatCompletionChoice(
        index=0,
        message=message,
        finish_reason=finish_reason
    )
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[choice],
        usage=usage
    )


async def _sse_events(
    stream: AsyncIterator[GenerationChunk],
    completion_id: str,
    created: int,
    model: str
) -> AsyncIterator[str]:
    def event(
        delta: ChoiceDelta,
        finish_reason: FinishReason | None = None
    ) -> str:
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason
                )
            ]
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    yield event(
        ChoiceDelta(
            role="assistant",
            content=""
        )
    )

    async for chunk in stream:
        if chunk.error is not None:
            error = ErrorResponse(
                error=ErrorDetail(
                    message=chunk.error,
                    type="server_error",
                    code="engine_error"
                )
            )
            yield f"data: {error.model_dump_json(exclude_none=True)}\n\n"
            break
        if chunk.token is not None:
            yield event(ChoiceDelta(content=chunk.token))
        if chunk.finish_reason is not None:
            yield event(
                ChoiceDelta(),
                chunk.finish_reason
            )

    yield "data: [DONE]\n\n"
