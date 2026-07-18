import pytest

from inference_server.config import Settings
from inference_server.engines.echo import EchoEngine
from inference_server.engines.factory import get_inference_engine
from inference_server.engines.onnx import OnnxEngine


def test_get_inference_engine_returns_onnx_engine_when_engine_is_onnx() -> None:
    # Arrange
    settings = Settings(
        _env_file=None,
        engine="onnx"
    )


    # Act
    engine = get_inference_engine(settings)


    # Assert
    assert isinstance(engine, OnnxEngine)


def test_get_inference_engine_returns_echo_engine_when_engine_is_echo() -> None:
    # Arrange
    settings = Settings(
        _env_file=None,
        engine="echo"
    )


    # Act
    engine = get_inference_engine(settings)


    # Assert
    assert isinstance(engine, EchoEngine)


def test_get_inference_engine_raises_value_error_when_engine_is_unknown() -> None:
    # Arrange
    settings = Settings(
        _env_file=None,
        engine="warp-drive"
    )


    # Act & Assert
    with pytest.raises(ValueError):
        get_inference_engine(settings)
