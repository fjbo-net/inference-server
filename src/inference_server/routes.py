"""OpenAI-compatible API routes."""

import time

from fastapi import APIRouter, Request

from inference_server.engines.base import BaseInferenceEngine
from inference_server.schemas.openai import Model, ModelList

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
