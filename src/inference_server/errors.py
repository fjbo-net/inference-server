"""OpenAI-shaped error envelopes for API failures.

FastAPI's default validation response is not OpenAI-compatible, so
every error surface is translated into the `ErrorResponse` schema
that OpenAI client SDKs know how to raise.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from inference_server.engines.base import EngineError, ModelNotFoundError
from inference_server.schemas.openai import ErrorDetail, ErrorResponse


def _error_json(status_code: int, detail: ErrorDetail) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=detail).model_dump()
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        error: RequestValidationError
    ) -> JSONResponse:
        first = error.errors()[0]
        param = ".".join(
            str(part)
            for part in first["loc"]
            if part != "body"
        )
        detail = ErrorDetail(
            message=str(first["msg"]),
            type="invalid_request_error",
            param=param or None
        )
        return _error_json(422, detail)

    @app.exception_handler(ModelNotFoundError)
    async def handle_model_not_found(
        request: Request,
        error: ModelNotFoundError
    ) -> JSONResponse:
        detail = ErrorDetail(
            message=str(error),
            type="invalid_request_error",
            param="model",
            code="model_not_found"
        )
        return _error_json(404, detail)

    @app.exception_handler(EngineError)
    async def handle_engine_error(
        request: Request,
        error: EngineError
    ) -> JSONResponse:
        detail = ErrorDetail(
            message=str(error),
            type="server_error",
            code="engine_error"
        )
        return _error_json(500, detail)
