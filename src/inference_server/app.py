"""FastAPI application factory."""

from fastapi import FastAPI

from inference_server import __version__


def create_app() -> FastAPI:
    app = FastAPI(title="Inference Server", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app
