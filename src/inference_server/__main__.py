"""Entrypoint: python -m inference_server."""

import uvicorn

from inference_server.app import create_app
from inference_server.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
