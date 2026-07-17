"""FastAPI application factory."""

from fastapi import FastAPI

from inference_server import __version__
from inference_server.config import Settings, get_settings
from inference_server.engines.factory import get_inference_engine
from inference_server.errors import register_error_handlers
from inference_server.routes import router


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(title="Inference Server", version=__version__)
    app.state.settings = settings
    app.state.engine = get_inference_engine(settings)
    app.include_router(router)
    register_error_handlers(app)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app
