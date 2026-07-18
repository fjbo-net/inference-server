import pytest

from inference_server.app import create_app
from inference_server.config import Settings
from inference_server.engines.echo import EchoEngine


def test_create_app_builds_engine_from_settings() -> None:
    # Arrange
    settings = Settings(
        _env_file=None,
        engine="echo"
    )


    # Act
    app = create_app(settings)


    # Assert
    assert isinstance(app.state.engine, EchoEngine)


def test_create_app_raises_value_error_when_engine_is_unknown() -> None:
    # Arrange
    settings = Settings(
        _env_file=None,
        engine="warp-drive"
    )


    # Act & Assert
    with pytest.raises(ValueError):
        create_app(settings)
